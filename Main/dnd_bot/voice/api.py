"""FastAPI backend for D&D voice game management.

Provides REST endpoints for the web UI to manage game sessions,
characters, campaigns, and player actions. Serves alongside the
LiveKit voice agent.

Endpoints:
    GET  /api/campaigns                     - List available campaigns
    POST /api/campaigns                     - Create a new campaign
    GET  /api/campaigns/{id}/characters     - List characters in a campaign
    GET  /api/races                         - List SRD races
    GET  /api/classes                       - List SRD classes
    POST /api/characters/create             - Create a new character
    POST /api/game/start                    - Start a game session
    POST /api/game/join                     - Join with a character
    GET  /api/game/status                   - Get current session state
    POST /api/game/action                   - Process a player action
    POST /api/game/end                      - End the session
    GET  /api/token                         - Generate a LiveKit room token
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import structlog
import io
import struct

logger = structlog.get_logger()

app = FastAPI(title="D&D 5e Voice Game")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve web UI static files
WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "web")


# -- Request/Response Models -------------------------------------------------

class CreateCampaignRequest(BaseModel):
    name: str
    description: str = ""
    world_setting: str = "A classic high fantasy world filled with magic, monsters, and adventure."

class StartGameRequest(BaseModel):
    campaign_id: str
    room_name: str = "dnd-session"
    player_name: str = "Player 1"

class JoinGameRequest(BaseModel):
    session_key: str
    player_name: str
    character_id: str

class PlayerActionRequest(BaseModel):
    session_key: str
    player_name: str
    action: str

class CreateCharacterRequest(BaseModel):
    campaign_id: str
    player_name: str
    name: str
    race_index: str
    class_index: str
    ability_method: str = "standard_array"  # standard_array | roll

class CampaignResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    world_setting: str

class CharacterResponse(BaseModel):
    id: str
    name: str
    race: str
    class_name: str
    level: int
    hp_current: int
    hp_max: int
    ac: int = 10
    abilities: Optional[dict] = None

class SessionResponse(BaseModel):
    session_key: str
    campaign_name: str
    state: str
    players: list[dict]
    room_name: str

class TokenResponse(BaseModel):
    token: str
    url: str
    room: str


# -- Active voice session tracking -------------------------------------------

_active_voice_session: Optional[dict] = None


# -- Endpoints ---------------------------------------------------------------

@app.get("/")
async def serve_index():
    """Serve the web UI."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(404, "Web UI not found")


# -- Campaigns ---------------------------------------------------------------

@app.get("/api/campaigns")
async def list_campaigns():
    """List all available campaigns."""
    from ..data.repositories import get_campaign_repo
    repo = await get_campaign_repo()
    # Use guild_id=0 for voice sessions (no Discord guild)
    campaigns = await repo.get_all_by_guild(0)

    # Also check if there are any Discord campaigns we can use
    from ..data.database import get_database
    db = await get_database()
    rows = await db.fetch_all("SELECT DISTINCT guild_id FROM campaign")
    for row in rows:
        if row[0] != 0:
            guild_campaigns = await repo.get_all_by_guild(row[0])
            campaigns.extend(guild_campaigns)

    return [
        CampaignResponse(
            id=c.id,
            name=c.name,
            description=c.description,
            world_setting=c.world_setting,
        )
        for c in campaigns
    ]


@app.post("/api/campaigns")
async def create_campaign(req: CreateCampaignRequest):
    """Create a new campaign for voice play."""
    from ..data.repositories import get_campaign_repo
    from ..models import Campaign

    campaign = Campaign(
        id=str(uuid.uuid4()),
        guild_id=0,  # Voice campaigns use guild_id=0
        name=req.name,
        description=req.description,
        world_setting=req.world_setting,
        dm_user_id=0,  # AI is the DM
    )

    repo = await get_campaign_repo()
    await repo.create(campaign)

    return CampaignResponse(
        id=campaign.id,
        name=campaign.name,
        description=campaign.description,
        world_setting=campaign.world_setting,
    )


@app.get("/api/campaigns/{campaign_id}/characters")
async def list_characters(campaign_id: str):
    """List characters in a campaign."""
    from ..data.repositories import get_character_repo
    repo = await get_character_repo()
    characters = await repo.get_all_by_campaign(campaign_id)

    return [
        CharacterResponse(
            id=char.id,
            name=char.name,
            race=char.race_index,
            class_name=char.class_index,
            level=char.level,
            hp_current=char.hp.current,
            hp_max=char.hp.maximum,
            ac=char.armor_class,
            abilities={
                "str": char.abilities.strength,
                "dex": char.abilities.dexterity,
                "con": char.abilities.constitution,
                "int": char.abilities.intelligence,
                "wis": char.abilities.wisdom,
                "cha": char.abilities.charisma,
            } if char.abilities else None,
        )
        for char in characters
    ]


# -- Character Creation (SRD data + create) ----------------------------------

@app.get("/api/races")
async def list_races():
    """List available SRD races for character creation."""
    from ..data.srd import get_srd
    srd = get_srd()
    races = srd.get_all_races()

    result = []
    for race in races:
        bonuses = []
        for bonus in race.get("ability_bonuses", []):
            ability = bonus.get("ability_score", {}).get("index", "")
            amount = bonus.get("bonus", 0)
            if ability and amount:
                bonuses.append({"ability": ability, "bonus": amount})

        subraces = []
        for sub in race.get("subraces", []):
            subraces.append({
                "index": sub.get("index", ""),
                "name": sub.get("name", ""),
            })

        result.append({
            "index": race.get("index", ""),
            "name": race.get("name", ""),
            "speed": race.get("speed", 30),
            "ability_bonuses": bonuses,
            "size": race.get("size", "Medium"),
            "subraces": subraces,
        })

    return result


@app.get("/api/classes")
async def list_classes():
    """List available SRD classes for character creation."""
    from ..data.srd import get_srd
    srd = get_srd()
    classes = srd.get_all_classes()

    result = []
    for cls in classes:
        hit_die = cls.get("hit_die", 8)

        # Saving throw proficiencies
        saves = [s.get("index", "") for s in cls.get("saving_throws", [])]

        result.append({
            "index": cls.get("index", ""),
            "name": cls.get("name", ""),
            "hit_die": hit_die,
            "saving_throws": saves,
        })

    return result


@app.post("/api/characters/create")
async def create_character(req: CreateCharacterRequest):
    """Create a new character with auto-generated stats."""
    from ..game.character.creation import (
        CharacterCreator,
        CharacterCreationState,
        AbilityScoreMethod,
        STANDARD_ARRAY,
    )
    from ..models import AbilityScore
    from ..data.repositories import get_character_repo
    from ..data.srd import get_srd

    srd = get_srd()
    creator = CharacterCreator()

    # Validate race and class
    race_data = srd.get_race(req.race_index)
    if not race_data:
        raise HTTPException(400, f"Unknown race: {req.race_index}")

    class_data = srd.get_class(req.class_index)
    if not class_data:
        raise HTTPException(400, f"Unknown class: {req.class_index}")

    # Generate ability scores
    if req.ability_method == "roll":
        rolls = creator.roll_ability_scores()
        totals = rolls.get_sorted_totals()
    else:
        totals = STANDARD_ARRAY.copy()

    # Auto-assign scores optimally for the class
    assignments = _auto_assign_abilities(req.class_index, totals)

    from ..models import AbilityScores
    base_abilities = AbilityScores(
        strength=assignments[AbilityScore.STRENGTH],
        dexterity=assignments[AbilityScore.DEXTERITY],
        constitution=assignments[AbilityScore.CONSTITUTION],
        intelligence=assignments[AbilityScore.INTELLIGENCE],
        wisdom=assignments[AbilityScore.WISDOM],
        charisma=assignments[AbilityScore.CHARISMA],
    )

    # Apply racial bonuses
    final_abilities = creator.apply_racial_bonuses(
        base_abilities, req.race_index
    )

    # Build creation state
    user_id = hash(req.player_name) & 0x7FFFFFFF
    state = CharacterCreationState(
        user_id=user_id,
        campaign_id=req.campaign_id,
        name=req.name,
        race_index=req.race_index,
        class_index=req.class_index,
        final_abilities=final_abilities,
    )

    # Build character
    character = creator.build_character(state)

    # Save to database
    repo = await get_character_repo()
    await repo.create(character)

    logger.info(
        "character_created_via_web",
        name=character.name,
        race=character.race_index,
        class_=character.class_index,
        hp=character.hp.maximum,
    )

    return CharacterResponse(
        id=character.id,
        name=character.name,
        race=character.race_index,
        class_name=character.class_index,
        level=character.level,
        hp_current=character.hp.current,
        hp_max=character.hp.maximum,
        ac=character.armor_class,
        abilities={
            "str": final_abilities.strength,
            "dex": final_abilities.dexterity,
            "con": final_abilities.constitution,
            "int": final_abilities.intelligence,
            "wis": final_abilities.wisdom,
            "cha": final_abilities.charisma,
        },
    )


def _auto_assign_abilities(class_index: str, scores: list[int]) -> dict:
    """Auto-assign ability scores optimally for a class.

    Puts highest scores in the class's primary abilities.
    """
    from ..models import AbilityScore

    # Primary abilities by class (highest score first)
    CLASS_PRIORITIES = {
        "barbarian": [AbilityScore.STRENGTH, AbilityScore.CONSTITUTION, AbilityScore.DEXTERITY, AbilityScore.WISDOM, AbilityScore.CHARISMA, AbilityScore.INTELLIGENCE],
        "bard": [AbilityScore.CHARISMA, AbilityScore.DEXTERITY, AbilityScore.CONSTITUTION, AbilityScore.WISDOM, AbilityScore.INTELLIGENCE, AbilityScore.STRENGTH],
        "cleric": [AbilityScore.WISDOM, AbilityScore.CONSTITUTION, AbilityScore.STRENGTH, AbilityScore.CHARISMA, AbilityScore.DEXTERITY, AbilityScore.INTELLIGENCE],
        "druid": [AbilityScore.WISDOM, AbilityScore.CONSTITUTION, AbilityScore.DEXTERITY, AbilityScore.INTELLIGENCE, AbilityScore.CHARISMA, AbilityScore.STRENGTH],
        "fighter": [AbilityScore.STRENGTH, AbilityScore.CONSTITUTION, AbilityScore.DEXTERITY, AbilityScore.WISDOM, AbilityScore.CHARISMA, AbilityScore.INTELLIGENCE],
        "monk": [AbilityScore.DEXTERITY, AbilityScore.WISDOM, AbilityScore.CONSTITUTION, AbilityScore.STRENGTH, AbilityScore.CHARISMA, AbilityScore.INTELLIGENCE],
        "paladin": [AbilityScore.STRENGTH, AbilityScore.CHARISMA, AbilityScore.CONSTITUTION, AbilityScore.WISDOM, AbilityScore.DEXTERITY, AbilityScore.INTELLIGENCE],
        "ranger": [AbilityScore.DEXTERITY, AbilityScore.WISDOM, AbilityScore.CONSTITUTION, AbilityScore.STRENGTH, AbilityScore.INTELLIGENCE, AbilityScore.CHARISMA],
        "rogue": [AbilityScore.DEXTERITY, AbilityScore.CONSTITUTION, AbilityScore.CHARISMA, AbilityScore.INTELLIGENCE, AbilityScore.WISDOM, AbilityScore.STRENGTH],
        "sorcerer": [AbilityScore.CHARISMA, AbilityScore.CONSTITUTION, AbilityScore.DEXTERITY, AbilityScore.WISDOM, AbilityScore.INTELLIGENCE, AbilityScore.STRENGTH],
        "warlock": [AbilityScore.CHARISMA, AbilityScore.CONSTITUTION, AbilityScore.DEXTERITY, AbilityScore.WISDOM, AbilityScore.INTELLIGENCE, AbilityScore.STRENGTH],
        "wizard": [AbilityScore.INTELLIGENCE, AbilityScore.CONSTITUTION, AbilityScore.DEXTERITY, AbilityScore.WISDOM, AbilityScore.CHARISMA, AbilityScore.STRENGTH],
    }

    # Default: STR, DEX, CON, INT, WIS, CHA
    priority = CLASS_PRIORITIES.get(class_index, list(AbilityScore))

    sorted_scores = sorted(scores, reverse=True)
    return {ability: score for ability, score in zip(priority, sorted_scores)}


# -- Game Session Management -------------------------------------------------

@app.post("/api/game/start")
async def start_game(req: StartGameRequest):
    """Start a voice game session."""
    global _active_voice_session

    from ..game.session import get_session_manager

    session_manager = get_session_manager()
    session_key = f"voice:{req.room_name}"

    # Check if session already active
    existing = session_manager.get_session_by_key(session_key)
    if existing:
        raise HTTPException(400, "A game session is already active in this room.")

    # Start the session
    session = await session_manager.start_session(
        channel_id=0,  # Not a Discord channel
        guild_id=0,
        campaign_id=req.campaign_id,
    )

    # Override the session key to be voice-based
    old_key = session.session_key
    session.session_key = session_key
    # Re-register under the correct key
    session_manager._sessions.pop(old_key, None)
    session_manager._sessions[session_key] = session

    _active_voice_session = {
        "session_key": session_key,
        "campaign_id": req.campaign_id,
        "room_name": req.room_name,
        "player_name": req.player_name,
    }

    # Get campaign info
    from ..data.repositories import get_campaign_repo
    campaign_repo = await get_campaign_repo()
    campaign = await campaign_repo.get_by_id(req.campaign_id)
    campaign_name = campaign.name if campaign else "Unknown"

    logger.info("voice_game_started", session_key=session_key, campaign=campaign_name)

    # Generate opening narration
    opening_narrative = ""
    try:
        opening_narrative = await _generate_opening(session, session_manager, campaign)
    except Exception as e:
        logger.error("opening_narration_failed", error=str(e))
        opening_narrative = f"Welcome to {campaign_name}. The adventure begins..."

    return {
        "session_key": session_key,
        "campaign_name": campaign_name,
        "state": "active",
        "players": [],
        "room_name": req.room_name,
        "opening_narrative": opening_narrative,
    }


@app.post("/api/game/join")
async def join_game(req: JoinGameRequest):
    """Join the active voice game with a character."""
    from ..game.session import get_session_manager
    from ..data.repositories import get_character_repo

    session_manager = get_session_manager()
    session = session_manager.get_session_by_key(req.session_key)
    if not session:
        raise HTTPException(404, "No active game session.")

    # Load the character
    char_repo = await get_character_repo()
    character = await char_repo.get_by_id(req.character_id)
    if not character:
        raise HTTPException(404, "Character not found.")

    # Use a voice-specific user ID (hash of player name)
    user_id = hash(req.player_name) & 0x7FFFFFFF

    player = await session_manager.join_session(
        channel_id=0,
        user_id=user_id,
        user_name=req.player_name,
        character=character,
        session_key=req.session_key,
    )

    if not player:
        raise HTTPException(500, "Failed to join session.")

    logger.info("voice_player_joined", player=req.player_name, character=character.name)

    return {
        "player_name": req.player_name,
        "character": CharacterResponse(
            id=character.id,
            name=character.name,
            race=character.race_index,
            class_name=character.class_index,
            level=character.level,
            hp_current=character.hp.current,
            hp_max=character.hp.maximum,
            ac=character.armor_class,
            abilities={
                "str": character.abilities.strength,
                "dex": character.abilities.dexterity,
                "con": character.abilities.constitution,
                "int": character.abilities.intelligence,
                "wis": character.abilities.wisdom,
                "cha": character.abilities.charisma,
            } if character.abilities else None,
        ),
    }


@app.post("/api/game/action")
async def player_action(req: PlayerActionRequest):
    """Process a player action through the game engine.

    Takes the player's text input, routes through the DM orchestrator,
    and returns narrative + mechanics as JSON.
    """
    from ..game.session import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.get_session_by_key(req.session_key)
    if not session:
        raise HTTPException(404, "No active game session.")

    # Find the player by name
    user_id = hash(req.player_name) & 0x7FFFFFFF
    player = session.get_player(user_id)
    if not player:
        raise HTTPException(400, "Player not in session. Join first.")

    # Collect events for the response
    events = []

    class WebFrontend:
        """Lightweight frontend that collects events for JSON response."""

        @property
        def frontend_type(self):
            return "web"

        async def on_event(self, event):
            events.append({
                "type": event.type.value,
                "data": _serialize_event_data(event.data),
            })

        async def get_combat_action(self, turn_context):
            # Combat actions come via separate endpoint or buttons
            raise NotImplementedError("Use combat action buttons")

    try:
        response = await session_manager.process_message(
            channel_id=0,
            user_id=user_id,
            user_name=req.player_name,
            content=req.action,
            frontend=WebFrontend(),
            session_key=req.session_key,
        )
    except Exception as e:
        logger.error("action_processing_failed", error=str(e))
        raise HTTPException(500, f"Failed to process action: {str(e)[:200]}")

    if not response:
        raise HTTPException(400, "No response from game engine.")

    # Build response
    result = {
        "narrative": response.narrative or "",
        "mechanical_result": response.mechanical_result,
        "dice_rolls": _serialize_dice_rolls(response.dice_rolls),
        "combat_triggered": response.combat_triggered,
        "events": events,
    }

    # Include character state after action
    if player.character:
        char = player.character
        result["character_state"] = {
            "hp_current": char.hp.current,
            "hp_max": char.hp.maximum,
            "ac": char.armor_class,
            "conditions": [c.condition.value for c in char.conditions] if char.conditions else [],
            "spell_slots": _serialize_spell_slots(char.spell_slots),
        }

    return result


@app.get("/api/game/status")
async def game_status():
    """Get current voice game session status."""
    global _active_voice_session

    if not _active_voice_session:
        return {"active": False}

    from ..game.session import get_session_manager
    session_manager = get_session_manager()
    session = session_manager.get_session_by_key(_active_voice_session["session_key"])

    if not session:
        _active_voice_session = None
        return {"active": False}

    players = []
    for p in session.players.values():
        player_data = {
            "name": p.user_name,
            "character_name": p.character.name if p.character else None,
            "character_class": p.character.class_index if p.character else None,
            "character_level": p.character.level if p.character else None,
        }
        if p.character:
            player_data["hp_current"] = p.character.hp.current
            player_data["hp_max"] = p.character.hp.maximum
            player_data["ac"] = p.character.armor_class
            player_data["abilities"] = {
                "str": p.character.abilities.strength,
                "dex": p.character.abilities.dexterity,
                "con": p.character.abilities.constitution,
                "int": p.character.abilities.intelligence,
                "wis": p.character.abilities.wisdom,
                "cha": p.character.abilities.charisma,
            } if p.character.abilities else None
        players.append(player_data)

    from ..data.repositories import get_campaign_repo
    campaign_repo = await get_campaign_repo()
    campaign = await campaign_repo.get_by_id(session.campaign_id)

    return {
        "active": True,
        "session_key": session.session_key,
        "campaign_name": campaign.name if campaign else "Unknown",
        "state": session.state.value,
        "players": players,
        "room_name": _active_voice_session.get("room_name", "dnd-session"),
    }


@app.post("/api/game/end")
async def end_game():
    """End the current voice game session."""
    global _active_voice_session

    if not _active_voice_session:
        raise HTTPException(404, "No active game session.")

    from ..game.session import get_session_manager
    session_manager = get_session_manager()
    session_key = _active_voice_session["session_key"]
    session = session_manager.get_session_by_key(session_key)

    if session:
        session_manager._sessions.pop(session_key, None)

    _active_voice_session = None
    return {"status": "ended"}


@app.get("/api/token")
async def get_livekit_token(
    identity: str = "player1",
    room: str = "dnd-session",
    name: str = "Player 1",
):
    """Generate a LiveKit room token."""
    try:
        from livekit.api import AccessToken, VideoGrants

        api_key = os.getenv("LIVEKIT_API_KEY", "devkey")
        api_secret = os.getenv("LIVEKIT_API_SECRET", "secret")
        livekit_url = os.getenv("LIVEKIT_URL", "ws://localhost:7880")

        token = AccessToken(api_key, api_secret)
        token.with_identity(identity)
        token.with_name(name)
        token.with_grants(VideoGrants(
            room_join=True,
            room=room,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        ))

        return TokenResponse(
            token=token.to_jwt(),
            url=livekit_url,
            room=room,
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to generate token: {e}")


# -- Helpers -----------------------------------------------------------------

async def _generate_opening(session, session_manager, campaign) -> str:
    """Generate opening narration for a new game session.

    Uses the orchestrator to produce an atmospheric opening scene
    based on the campaign's world setting.
    """
    if not campaign:
        return "The adventure begins..."

    # Build a simple opening prompt
    world_setting = campaign.world_setting or "a mysterious fantasy world"
    opening_prompt = (
        f"[OPENING SCENE] The players have gathered. Set the scene for a new "
        f"adventure in {world_setting}. Describe the atmosphere, the setting, "
        f"and hint at the adventure ahead. Keep it to 2-3 paragraphs."
    )

    try:
        orchestrator = session_manager.orchestrator

        # Use the narrator directly for opening scene
        from ..llm.brains.base import BrainContext
        context = BrainContext(
            campaign_id=session.campaign_id,
            session_id=session.id,
            party_members="No adventurers have arrived yet.",
            current_scene=world_setting,
            active_quests="",
            in_combat=False,
            combat_state="",
            memory_context="",
            message_history=[],
            character_stats="",
            world_state_yaml="",
            last_turn_trace="",
        )

        response = await orchestrator.process_action(
            action=opening_prompt,
            player_name="System",
            context=context,
        )

        return response.narrative or f"Welcome to {campaign.name}. The adventure begins..."

    except Exception as e:
        logger.error("opening_generation_failed", error=str(e))
        return (
            f"Welcome to {campaign.name}. {world_setting} "
            f"Your adventure is about to begin..."
        )


def _serialize_event_data(data: dict) -> dict:
    """Serialize event data for JSON response, handling non-serializable types."""
    result = {}
    for key, value in data.items():
        if hasattr(value, "__dict__"):
            # Dataclass or object - convert to dict
            result[key] = {
                k: v for k, v in value.__dict__.items()
                if not k.startswith("_") and _is_json_serializable(v)
            }
        elif isinstance(value, (str, int, float, bool, type(None))):
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = [
                _serialize_event_data({"v": item})["v"]
                if isinstance(item, dict) else
                str(item) if not isinstance(item, (str, int, float, bool, type(None))) else item
                for item in value
            ]
        elif isinstance(value, dict):
            result[key] = _serialize_event_data(value)
        else:
            result[key] = str(value)
    return result


def _is_json_serializable(value) -> bool:
    """Check if a value is JSON-serializable."""
    return isinstance(value, (str, int, float, bool, type(None), list, dict))


def _serialize_dice_rolls(dice_rolls: list) -> list:
    """Serialize dice rolls for JSON response."""
    if not dice_rolls:
        return []

    result = []
    for roll in dice_rolls:
        result.append({
            "dice": roll.kept_dice if hasattr(roll, "kept_dice") else [],
            "total": roll.total if hasattr(roll, "total") else 0,
            "modifier": roll.modifier if hasattr(roll, "modifier") else 0,
            "reason": roll.reason if hasattr(roll, "reason") else "",
        })
    return result


def _serialize_spell_slots(spell_slots) -> dict:
    """Serialize spell slots for JSON response."""
    slots = {}
    for level in range(1, 10):
        current, maximum = spell_slots.get_slots(level)
        if maximum > 0:
            slots[f"level_{level}"] = {"current": current, "max": maximum}
    return slots


# -- TTS Endpoint (provider-agnostic) ----------------------------------------

class TTSRequest(BaseModel):
    text: str


@app.post("/api/tts")
async def synthesize_speech(req: TTSRequest):
    """Synthesize text to speech via the active TTS provider, return WAV audio.

    Provider is determined by the active profile's tts.provider setting.
    Riva calls are serialized with a lock (gRPC concurrency issue).
    Browser provider returns 503 (JS fallback handles it).
    """
    from .tts_factory import get_tts, get_tts_lock, needs_lock, is_browser_tts

    tts = get_tts()
    if is_browser_tts(tts):
        raise HTTPException(503, "TTS handled client-side (browser provider active)")

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        if needs_lock(tts):
            lock = get_tts_lock()
            async with lock:
                audio = await loop.run_in_executor(None, tts.synthesize, req.text)
        else:
            audio = await loop.run_in_executor(None, tts.synthesize, req.text)

        if len(audio) == 0:
            raise HTTPException(500, "TTS produced no audio")

        # Build WAV in memory — works for any provider's sample rate
        audio_bytes = audio.tobytes()
        wav_buf = io.BytesIO()
        sample_rate = tts.sample_rate
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(audio_bytes)
        wav_buf.write(b'RIFF')
        wav_buf.write(struct.pack('<I', 36 + data_size))
        wav_buf.write(b'WAVE')
        wav_buf.write(b'fmt ')
        wav_buf.write(struct.pack('<IHHIIHH', 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample))
        wav_buf.write(b'data')
        wav_buf.write(struct.pack('<I', data_size))
        wav_buf.write(audio_bytes)
        wav_buf.seek(0)

        return StreamingResponse(wav_buf, media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("tts_synthesis_failed", error=str(e))
        raise HTTPException(500, f"TTS failed: {str(e)[:100]}")


# Mount static files (CSS, JS) after API routes
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
