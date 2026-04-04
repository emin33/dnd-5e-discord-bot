"""Game session management - the core game loop."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional
import asyncio
import structlog

from ..models import Character, GameState, Combat
from ..memory import MemoryManager, get_memory_manager_sync
from ..llm.orchestrator import DMOrchestrator, get_orchestrator, DMResponse
from ..llm.brains.base import BrainContext
from ..data.repositories import get_character_repo, get_session_repo
from ..data.repositories.npc_repo import get_npc_repo
from ..models.npc import EntityType, Disposition, SceneEntity
from .combat.manager import CombatManager, get_combat_for_channel
from .scene.registry import get_scene_registry, clear_scene_registry
from .world_state import WorldState

logger = structlog.get_logger()


class SessionState(str, Enum):
    """States of a game session."""
    STARTING = "starting"  # Session being set up
    ACTIVE = "active"  # Normal play
    COMBAT = "combat"  # In combat encounter
    PAUSED = "paused"  # Temporarily paused
    ENDED = "ended"  # Session over


@dataclass
class PlayerInfo:
    """Information about a player in the session."""
    user_id: int
    user_name: str
    character: Optional[Character] = None
    joined_at: datetime = field(default_factory=datetime.utcnow)
    is_dm: bool = False


@dataclass
class GameSession:
    """
    An active game session in a Discord channel.

    This is the central coordinator for:
    - Player management
    - Message routing through the LLM
    - Combat state
    - Memory/context management
    """

    id: str
    channel_id: int
    guild_id: int
    campaign_id: str

    # State
    state: SessionState = SessionState.STARTING

    # Players
    players: dict[int, PlayerInfo] = field(default_factory=dict)
    dm_user_id: Optional[int] = None

    # Timestamps
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

    # Combat reference (if in combat)
    combat_manager: Optional[CombatManager] = None

    # Authoritative world state (narrator reads, Python writes)
    world_state: Optional[WorldState] = None

    def add_player(self, user_id: int, user_name: str, character: Character) -> PlayerInfo:
        """Add a player to the session."""
        player = PlayerInfo(
            user_id=user_id,
            user_name=user_name,
            character=character,
        )
        self.players[user_id] = player
        self.last_activity = datetime.utcnow()

        logger.info(
            "player_joined_session",
            session_id=self.id,
            user_id=user_id,
            character=character.name,
        )

        return player

    def remove_player(self, user_id: int) -> Optional[PlayerInfo]:
        """Remove a player from the session."""
        player = self.players.pop(user_id, None)
        if player:
            logger.info(
                "player_left_session",
                session_id=self.id,
                user_id=user_id,
            )
        return player

    def get_player(self, user_id: int) -> Optional[PlayerInfo]:
        """Get a player by user ID."""
        return self.players.get(user_id)

    def get_player_character(self, user_id: int) -> Optional[Character]:
        """Get a player's character."""
        player = self.get_player(user_id)
        return player.character if player else None

    def get_all_characters(self) -> list[Character]:
        """Get all player characters in the session."""
        return [p.character for p in self.players.values() if p.character]

    def set_dm(self, user_id: int) -> None:
        """Set the DM for this session."""
        self.dm_user_id = user_id
        if user_id in self.players:
            self.players[user_id].is_dm = True

    def is_dm(self, user_id: int) -> bool:
        """Check if a user is the DM."""
        return user_id == self.dm_user_id

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()


class GameSessionManager:
    """
    Manages game sessions and routes messages through the LLM.

    This is the main entry point for the game loop:
    1. Player sends message in game channel
    2. SessionManager receives it
    3. Builds context from memory + current state
    4. Routes through DMOrchestrator
    5. Returns narrated response
    """

    def __init__(self):
        self.orchestrator = get_orchestrator()
        self._sessions: dict[int, GameSession] = {}  # channel_id -> session
        self._memory_managers: dict[str, MemoryManager] = {}  # campaign_id -> memory
        self._processing_lock = asyncio.Lock()

    def get_session(self, channel_id: int) -> Optional[GameSession]:
        """Get the active session for a channel."""
        return self._sessions.get(channel_id)

    def has_active_session(self, channel_id: int) -> bool:
        """Check if channel has an active session."""
        session = self._sessions.get(channel_id)
        return session is not None and session.state in (SessionState.ACTIVE, SessionState.COMBAT)

    async def start_session(
        self,
        channel_id: int,
        guild_id: int,
        campaign_id: str,
        dm_user_id: Optional[int] = None,
    ) -> GameSession:
        """Start a new game session in a channel."""
        import uuid

        # End any existing session
        if channel_id in self._sessions:
            await self.end_session(channel_id)

        # Get session number for this campaign
        session_repo = await get_session_repo()
        session_number = await session_repo.get_session_number(campaign_id)

        session = GameSession(
            id=str(uuid.uuid4()),
            channel_id=channel_id,
            guild_id=guild_id,
            campaign_id=campaign_id,
            state=SessionState.ACTIVE,
            dm_user_id=dm_user_id,
        )

        # Initialize authoritative world state
        session.world_state = WorldState()

        self._sessions[channel_id] = session

        # Initialize memory manager for this campaign
        if campaign_id not in self._memory_managers:
            self._memory_managers[campaign_id] = get_memory_manager_sync(campaign_id)

        # Persist to database
        await session_repo.save_session(
            session_id=session.id,
            campaign_id=campaign_id,
            channel_id=channel_id,
            session_number=session_number,
            state=self._state_to_db(session.state),
        )

        # Pre-load known NPCs from database into scene registry
        scene_registry = get_scene_registry(campaign_id, channel_id)
        try:
            npc_repo = await get_npc_repo()
            campaign_npcs = await npc_repo.get_alive_by_campaign(campaign_id)
            for npc in campaign_npcs:
                entity = SceneEntity(
                    name=npc.name,
                    npc_id=npc.id,
                    entity_type=EntityType.NPC,
                    description=npc.description or "",
                    monster_index=npc.monster_index,
                    disposition=npc.base_disposition if isinstance(npc.base_disposition, Disposition) else Disposition.NEUTRAL,
                )
                scene_registry.register_entity(entity)
            if campaign_npcs:
                logger.info("npcs_loaded_from_db", count=len(campaign_npcs), campaign_id=campaign_id)
        except Exception as e:
            logger.warning("npc_preload_failed", error=str(e))

        logger.info(
            "session_started",
            session_id=session.id,
            channel_id=channel_id,
            campaign_id=campaign_id,
            session_number=session_number,
        )

        return session

    async def end_session(self, channel_id: int) -> Optional[GameSession]:
        """End a game session."""
        session = self._sessions.pop(channel_id, None)
        if session:
            session.state = SessionState.ENDED

            # Persist final state and sync characters
            await self._sync_session_characters(session)

            # Sync NPCs to repository and clear scene registry
            try:
                scene_registry = get_scene_registry(session.campaign_id, channel_id)
                await scene_registry.sync_to_npc_repo()
                clear_scene_registry(channel_id)
            except Exception as e:
                logger.error(
                    "scene_registry_cleanup_failed",
                    session_id=session.id,
                    error=str(e),
                )

            # Generate final summary
            memory = self._memory_managers.get(session.campaign_id)
            if memory:
                await memory.end_session()

            # Mark session as ended in database
            session_repo = await get_session_repo()
            await session_repo.end_session(session.id)

            logger.info(
                "session_ended",
                session_id=session.id,
                channel_id=channel_id,
                duration_minutes=(datetime.utcnow() - session.started_at).seconds // 60,
            )

        return session

    async def _sync_session_characters(self, session: GameSession) -> None:
        """Sync all character states from session to database."""
        char_repo = await get_character_repo()
        for player in session.players.values():
            if player.character:
                try:
                    await char_repo.update(player.character)
                    logger.debug(
                        "character_synced",
                        character=player.character.name,
                        session_id=session.id,
                    )
                except Exception as e:
                    logger.error(
                        "character_sync_failed",
                        character=player.character.name,
                        error=str(e),
                    )

    def _state_to_db(self, state: SessionState) -> str:
        """Convert SessionState enum to database string."""
        state_map = {
            SessionState.STARTING: "lobby",
            SessionState.ACTIVE: "exploration",
            SessionState.COMBAT: "combat",
            SessionState.PAUSED: "paused",
            SessionState.ENDED: "ended",
        }
        return state_map.get(state, "exploration")

    def _db_to_state(self, db_state: str) -> SessionState:
        """Convert database string to SessionState enum."""
        state_map = {
            "lobby": SessionState.STARTING,
            "exploration": SessionState.ACTIVE,
            "combat": SessionState.COMBAT,
            "social": SessionState.ACTIVE,
            "resting": SessionState.ACTIVE,
            "paused": SessionState.PAUSED,
            "ended": SessionState.ENDED,
        }
        return state_map.get(db_state, SessionState.ACTIVE)

    async def recover_sessions(self) -> int:
        """
        Recover active sessions from database after bot restart.

        Returns the number of sessions recovered.
        """
        session_repo = await get_session_repo()
        char_repo = await get_character_repo()
        active_sessions = await session_repo.load_active_sessions()

        recovered = 0
        for session_data in active_sessions:
            try:
                # Create session object
                session = GameSession(
                    id=session_data["id"],
                    channel_id=session_data["channel_id"],
                    guild_id=0,  # Will be resolved when bot reconnects
                    campaign_id=session_data["campaign_id"],
                    state=self._db_to_state(session_data["state"]),
                )

                # Load players with characters
                characters = await char_repo.get_all_by_campaign(session_data["campaign_id"])
                for char in characters:
                    session.players[char.discord_user_id] = PlayerInfo(
                        user_id=char.discord_user_id,
                        user_name="",  # Will be resolved when user interacts
                        character=char,
                    )

                self._sessions[session_data["channel_id"]] = session

                # Initialize memory manager
                if session_data["campaign_id"] not in self._memory_managers:
                    self._memory_managers[session_data["campaign_id"]] = get_memory_manager_sync(
                        session_data["campaign_id"]
                    )

                recovered += 1
                logger.info(
                    "session_recovered",
                    session_id=session.id,
                    channel_id=session_data["channel_id"],
                    players=len(session.players),
                )
            except Exception as e:
                logger.error(
                    "session_recovery_failed",
                    session_id=session_data["id"],
                    error=str(e),
                )

        return recovered

    async def join_session(
        self,
        channel_id: int,
        user_id: int,
        user_name: str,
        character: Character,
    ) -> Optional[PlayerInfo]:
        """Add a player to an active session."""
        session = self.get_session(channel_id)
        if not session or session.state == SessionState.ENDED:
            return None

        player = session.add_player(user_id, user_name, character)

        # Update party status in memory
        memory = self._memory_managers.get(session.campaign_id)
        if memory:
            party_text = self._build_party_status(session)
            memory.update_party_status(party_text)

        return player

    async def leave_session(self, channel_id: int, user_id: int) -> Optional[PlayerInfo]:
        """Remove a player from a session."""
        session = self.get_session(channel_id)
        if not session:
            return None

        player = session.remove_player(user_id)

        # Update party status in memory
        if player:
            memory = self._memory_managers.get(session.campaign_id)
            if memory:
                party_text = self._build_party_status(session)
                memory.update_party_status(party_text)

        return player

    async def process_message(
        self,
        channel_id: int,
        user_id: int,
        user_name: str,
        content: str,
        on_mechanics_ready: Optional[Callable] = None,
        on_narrative_token: Optional[Callable] = None,
    ) -> Optional[DMResponse]:
        """
        Process a player message through the LLM DM.

        This is the core game loop entry point.
        """
        session = self.get_session(channel_id)
        if not session or session.state not in (SessionState.ACTIVE, SessionState.COMBAT):
            return None

        # Get player info
        player = session.get_player(user_id)
        if not player:
            # Player hasn't joined with a character
            return None

        session.update_activity()

        # Get memory manager
        memory = self._memory_managers.get(session.campaign_id)
        if not memory:
            memory = get_memory_manager_sync(session.campaign_id)
            self._memory_managers[session.campaign_id] = memory

        # Update combat state for gated memory consolidation
        memory.set_combat_state(session.state == SessionState.COMBAT)

        # Add message to memory buffer
        await memory.add_player_message(
            content=content,
            author_name=player.character.name if player.character else user_name,
        )

        # Sync player snapshots into world state before building context
        if session.world_state:
            session.world_state.increment_turn()
            for p in session.players.values():
                if p.character:
                    conditions = [
                        c.condition.value for c in p.character.conditions
                    ] if p.character.conditions else []
                    session.world_state.sync_player(
                        name=p.character.name,
                        hp=p.character.hp.current,
                        max_hp=p.character.hp.maximum,
                        conditions=conditions,
                        concentration=p.character.concentration_spell_id or "",
                    )
            # Sync phase from session state
            if session.state == SessionState.COMBAT:
                session.world_state.phase = "combat"
            elif session.world_state.phase == "combat" and session.state != SessionState.COMBAT:
                session.world_state.phase = "exploration"

        # Build context with acting player's character data
        context = self._build_context(
            session, memory, content,
            player_character=player.character,
        )

        # Get or create scene registry for entity tracking
        scene_registry = get_scene_registry(session.campaign_id, channel_id)

        # Process through orchestrator
        async with self._processing_lock:
            try:
                # Set session and scene registry context
                self.orchestrator.set_session(session)
                self.orchestrator.set_scene_registry(scene_registry)

                response = await self.orchestrator.process_action(
                    action=content,
                    player_name=player.character.name if player.character else user_name,
                    context=context,
                    on_mechanics_ready=on_mechanics_ready,
                    on_narrative_token=on_narrative_token,
                )

                # Add response to memory
                await memory.add_dm_response(
                    content=response.narrative,
                    is_narration=True,
                )

                # Update scene memory block from registry
                if scene_registry.has_entities():
                    scene_summary = scene_registry.get_scene_summary()
                    if scene_summary:
                        memory.update_scene(scene_summary)

                # Sync pinned facts from memory into world state
                if session.world_state and memory.buffer.pinned_facts:
                    for fact in memory.buffer.pinned_facts:
                        if fact not in session.world_state.established_facts:
                            session.world_state.established_facts.append(fact)

                # Handle auto-combat trigger
                if response.combat_triggered:
                    session.state = SessionState.COMBAT
                    logger.warning(
                        "session_transitioned_to_combat",
                        session_id=session.id,
                        reason="hostility_threshold_crossed",
                    )

                logger.info(
                    "message_processed",
                    session_id=session.id,
                    user=user_name,
                    action_preview=content[:50],
                    has_mechanics=response.mechanical_result is not None,
                    combat_triggered=response.combat_triggered,
                    scene_entities=scene_registry.get_entity_count(),
                )

                return response

            except Exception as e:
                logger.error(
                    "message_processing_error",
                    session_id=session.id,
                    error=str(e),
                )
                raise
            finally:
                # Clear session context (but keep scene registry alive)
                self.orchestrator.set_session(None)
                self.orchestrator.set_scene_registry(None)

    def _build_context(
        self,
        session: GameSession,
        memory: MemoryManager,
        current_input: str,
        player_character: Optional[Character] = None,
    ) -> BrainContext:
        """Build the context for the LLM brains.

        Populates all BrainContext fields so the narrator has full visibility
        into game state: party status, character details, scene, combat, quests, NPCs.
        """
        # Get party info — include conditions and key stats
        characters = session.get_all_characters()
        party_lines = []
        for c in characters:
            line = f"- {c.name}: Level {c.level} {c.race_index} {c.class_index}, HP {c.hp.current}/{c.hp.maximum}, AC {c.armor_class}"
            if c.conditions:
                conds = ", ".join(cond.condition.value for cond in c.conditions)
                line += f" [{conds}]"
            if c.is_concentrating:
                line += f" (concentrating on {c.concentration_spell_id})"
            party_lines.append(line)
        party_summary = "\n".join(party_lines)

        # Build character stats for the acting player
        character_context = ""
        if player_character:
            character_context = self._build_character_context(player_character)

        # Get combat context if in combat — richer detail
        combat_context = ""
        is_in_combat = session.state == SessionState.COMBAT
        if is_in_combat:
            combat = get_combat_for_channel(session.channel_id)
            if combat:
                current = combat.combat.get_current_combatant()
                combatant_lines = []
                for c in combat.combat.get_sorted_combatants():
                    marker = ">>>" if c == current else "   "
                    status = f"HP {c.hp_current}/{c.hp_max}" if c.is_player else f"{'healthy' if c.hp_current > c.hp_max // 2 else 'wounded' if c.hp_current > 0 else 'down'}"
                    combatant_lines.append(f"{marker} {c.name} ({status})")
                combat_context = (
                    f"COMBAT ACTIVE - Round {combat.combat.current_round}\n"
                    f"Current turn: {current.name if current else 'None'}\n"
                    f"Initiative order:\n" + "\n".join(combatant_lines)
                )

        # Get active quests from memory
        quests_block = memory.core.get_block("quests")
        active_quests = quests_block.content if quests_block else ""

        # Get scene entities (NPCs, creatures, objects) from registry
        scene_context = memory.core.get_block("scene").content if memory.core.get_block("scene") else ""
        scene_registry = get_scene_registry(session.campaign_id, session.channel_id)
        if scene_registry and scene_registry.has_entities():
            entity_context = scene_registry.get_triage_context()
            if entity_context:
                scene_context += f"\n\n### Entities Present\n{entity_context}"
            # Authoritative NPC roster for narrator consistency
            roster = scene_registry.get_narrator_roster()
            if roster:
                scene_context += f"\n\n{roster}"

        # Build full context from memory
        memory_context = memory.build_context(current_input)

        # Get recent messages
        # Feed more history to narrator — models support 32K-200K context,
        # we were only using ~3-5K. More verbatim history = better grounding.
        message_history = memory.get_message_history(limit=30)

        # Serialize world state for narrator bookend injection
        world_state_yaml = ""
        last_turn_trace = ""
        if session.world_state:
            world_state_yaml = session.world_state.to_yaml()
            # Build previous-turn trace from recent events
            if session.world_state.recent_events:
                last_turn_trace = session.world_state.recent_events[-1]

        return BrainContext(
            campaign_id=session.campaign_id,
            session_id=session.id,
            party_members=party_summary,
            current_scene=scene_context,
            active_quests=active_quests,
            in_combat=is_in_combat,
            combat_state=combat_context,
            memory_context=memory_context,
            message_history=message_history,
            character_stats=character_context,
            world_state_yaml=world_state_yaml,
            last_turn_trace=last_turn_trace,
        )

    def _build_character_context(self, character: Character) -> str:
        """Build detailed character context for the narrator.

        Gives the narrator visibility into the acting character's key stats
        so narration accurately reflects their capabilities and condition.
        """
        lines = [
            f"**{character.name}** — Level {character.level} {character.race_index} {character.class_index}",
            f"HP: {character.hp.current}/{character.hp.maximum} | AC: {character.armor_class}",
            f"STR {character.abilities.strength} ({character.abilities.str_mod:+d}) | "
            f"DEX {character.abilities.dexterity} ({character.abilities.dex_mod:+d}) | "
            f"CON {character.abilities.constitution} ({character.abilities.con_mod:+d})",
            f"INT {character.abilities.intelligence} ({character.abilities.int_mod:+d}) | "
            f"WIS {character.abilities.wisdom} ({character.abilities.wis_mod:+d}) | "
            f"CHA {character.abilities.charisma} ({character.abilities.cha_mod:+d})",
        ]

        # Conditions
        if character.conditions:
            conds = ", ".join(f"{c.condition.value}" for c in character.conditions)
            lines.append(f"Conditions: {conds}")

        # Concentration
        if character.is_concentrating:
            lines.append(f"Concentrating on: {character.concentration_spell_id}")

        # Spell slots summary
        slot_parts = []
        for level in range(1, 10):
            current, maximum = character.spell_slots.get_slots(level)
            if maximum > 0:
                slot_parts.append(f"L{level}: {current}/{maximum}")
        if slot_parts:
            lines.append(f"Spell slots: {', '.join(slot_parts)}")

        return "\n".join(lines)

    def _build_party_status(self, session: GameSession) -> str:
        """Build party status text for memory."""
        characters = session.get_all_characters()
        if not characters:
            return "No adventurers have joined yet."

        lines = ["The adventuring party consists of:"]
        for char in characters:
            lines.append(
                f"- {char.name}, a level {char.level} {char.race_index} {char.class_index} "
                f"(HP: {char.hp.current}/{char.hp.maximum})"
            )
        return "\n".join(lines)

    def enter_combat(self, channel_id: int, combat_manager: CombatManager) -> bool:
        """Transition session to combat state."""
        session = self.get_session(channel_id)
        if not session:
            return False

        session.state = SessionState.COMBAT
        session.combat_manager = combat_manager

        # Update memory
        memory = self._memory_managers.get(session.campaign_id)
        if memory:
            memory.update_scene(
                f"COMBAT: {combat_manager.combat.encounter_name or 'Battle'} - "
                f"Round {combat_manager.combat.current_round}"
            )

        return True

    def exit_combat(self, channel_id: int) -> bool:
        """Transition session out of combat."""
        session = self.get_session(channel_id)
        if not session:
            return False

        session.state = SessionState.ACTIVE
        session.combat_manager = None

        return True


# Singleton instance
_manager: Optional[GameSessionManager] = None


def get_session_manager() -> GameSessionManager:
    """Get the singleton session manager."""
    global _manager
    if _manager is None:
        _manager = GameSessionManager()
    return _manager
