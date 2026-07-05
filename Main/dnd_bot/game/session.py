"""Game session management - the core game loop."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
import asyncio
import structlog

from ..models import Character, CombatState
from ..memory import (
    MemoryManager,
    get_memory_manager,
    save_memory_state,
)
from ..memory.manager import peek_memory_manager
from ..llm.orchestrator import get_orchestrator, DMResponse
from ..llm.brains.base import BrainContext
from .frontend import GameFrontend, GameEvent
from ..data.repositories import get_character_repo, get_session_repo
from ..data.repositories.npc_repo import get_npc_repo
from ..models.npc import EntityType, Disposition, SceneEntity
from .combat.manager import CombatManager, get_combat_by_key, clear_combat_by_key
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
    An active game session.

    This is the central coordinator for:
    - Player management
    - Message routing through the LLM
    - Combat state
    - Memory/context management

    Works with any frontend (Discord text, voice, web) via session_key.
    """

    id: str
    channel_id: int
    guild_id: int
    campaign_id: str

    # Generic session identifier. Defaults to f"discord:{channel_id}" in __post_init__.
    # Voice/web frontends use their own keys (e.g., "voice:{room_id}").
    session_key: str = ""

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

    # Knowledge graph (persistent entity relationships)
    knowledge_graph: Optional[Any] = None  # KnowledgeGraph, Optional to avoid circular import

    # Last narrative text (for /imagine command)
    last_narrative: Optional[str] = None

    def __post_init__(self):
        if not self.session_key:
            self.session_key = f"discord:{self.channel_id}"

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

    def get_user_id_for_character(self, character_id: str) -> Optional[int]:
        """Reverse-map a character id to its controlling player's user id.

        Used by combat views' interaction_check so only the acting player
        can drive their turn's buttons (audit P0-6).
        """
        for user_id, player in self.players.items():
            if player.character and player.character.id == character_id:
                return user_id
        return None

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
        self._sessions: dict[str, GameSession] = {}  # session_key -> session
        self._processing_lock = asyncio.Lock()

    @property
    def orchestrator(self):
        """Always read the current orchestrator singleton.

        Using a property instead of caching at __init__ ensures
        /profile switches take effect immediately — the old
        orchestrator (with stale clients) gets dropped.
        """
        return get_orchestrator()

    def get_session_by_key(self, session_key: str) -> Optional[GameSession]:
        """Get a session by its generic session key."""
        return self._sessions.get(session_key)

    def get_session(self, channel_id: int) -> Optional[GameSession]:
        """Get the active session for a Discord channel."""
        return self._sessions.get(f"discord:{channel_id}")

    def has_active_session(self, channel_id: int) -> bool:
        """Check if channel has an active session."""
        session = self.get_session(channel_id)
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
        if self.get_session(channel_id):
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

        # Load knowledge graph for persistent entity relationships
        try:
            from .knowledge import KnowledgeGraph, get_kg_repo
            kg_repo = await get_kg_repo()
            session.knowledge_graph = KnowledgeGraph(campaign_id, kg_repo)
            await session.knowledge_graph.load()

            # Sync entity descriptions to ChromaDB for vector matching
            if session.knowledge_graph.node_count() > 0:
                try:
                    from ..memory import get_vector_store
                    vs = get_vector_store()
                    for entity in session.knowledge_graph.get_entities_for_indexing():
                        vs.add_entity_description(
                            campaign_id=campaign_id,
                            node_id=entity.node_id,
                            entity_type=entity.entity_type.value,
                            name=entity.name,
                            description=entity.properties.get("description", ""),
                            aliases=entity.aliases,
                        )
                except Exception as e:
                    logger.warning("kg_entity_description_sync_failed", error=str(e))

            logger.info(
                "knowledge_graph_loaded",
                campaign_id=campaign_id,
                nodes=session.knowledge_graph.node_count(),
                edges=session.knowledge_graph.edge_count(),
            )
        except Exception as e:
            logger.warning("knowledge_graph_load_failed", error=str(e))
            session.knowledge_graph = None

        self._sessions[session.session_key] = session

        # Warm the memory cache for this campaign — async getter so persisted
        # memory tiers (pinned facts, summaries) load from the DB. The module
        # LRU in memory.manager is the single owner; a session-level copy
        # used to go split-brain with cogs fetching via get_memory_manager.
        await get_memory_manager(campaign_id)

        # Persist to database
        await session_repo.save_session(
            session_id=session.id,
            campaign_id=campaign_id,
            channel_id=channel_id,
            session_number=session_number,
            state=self._state_to_db(session.state),
        )

        # Pre-load known NPCs from database into scene registry. Key by
        # `session.session_key` (audit #8) so concurrent voice/web sessions
        # — which all set `channel_id=0` — don't collide on one shared registry.
        scene_registry = get_scene_registry(campaign_id, session.session_key)
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
                    voice_id=npc.voice_id,
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
        session = self._sessions.get(f"discord:{channel_id}")
        if session is not None and (
            session.state == SessionState.COMBAT
            or session.combat_manager is not None
        ):
            # /game end mid-combat: unwind through the single teardown owner
            # (audit P0-3) first — otherwise the combat-manager, coordinator,
            # and turn-lock registry entries stay keyed to this channel and
            # the NEXT session inherits them (adversarial review, nit).
            await self.end_combat(channel_id, session_key=session.session_key)

        session = self._sessions.pop(f"discord:{channel_id}", None)
        if session:
            session.state = SessionState.ENDED

            # Persist final state and sync characters
            await self._sync_session_characters(session)

            # Sync NPCs to repository and clear scene registry
            try:
                scene_registry = get_scene_registry(session.campaign_id, session.session_key)
                await scene_registry.sync_to_npc_repo()
                clear_scene_registry(session.session_key)
            except Exception as e:
                logger.error(
                    "scene_registry_cleanup_failed",
                    session_id=session.id,
                    error=str(e),
                )

            # Generate final summary, then persist memory tiers so pinned
            # facts / summaries survive a restart. Resolve via the async
            # getter (reloads the eviction snapshot if the LRU dropped this
            # campaign) and persist THIS instance — a campaign_id-only save
            # would silently no-op after eviction.
            memory = await get_memory_manager(session.campaign_id)
            await memory.end_session()
            await save_memory_state(session.campaign_id, manager=memory)

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
        """Reconcile character state at session end.

        DF-1 guard: `player.character` is the live object loaded at join and
        never refreshed, while per-action paths (narrated update_player, combat
        sync, the /character cog) persist FRESHLY-FETCHED Character objects. The
        old code wrote this stale object back, so a graceful end_session
        overwrote the DB with join-time HP/slots/conditions — silently undoing
        every mid-session change (ironically a crash preserved them). The DB is
        already authoritative from the per-action writes, so we re-fetch and
        refresh the in-memory object rather than clobber. Only persist if the
        character somehow isn't in the DB yet.

        NOTE: this stops the end-of-session clobber only. Mid-turn staleness in
        the world_state party snapshot (DF-11, same root) and the true fix —
        one session-owned Character instance — are the Option-B / #6-7 refactor.
        """
        char_repo = await get_character_repo()
        for player in session.players.values():
            if not player.character:
                continue
            try:
                fresh = await char_repo.get_by_id(player.character.id)
                if fresh is not None:
                    # DB holds the authoritative per-action state; refresh the
                    # in-memory ref, do NOT write the stale object back.
                    player.character = fresh
                else:
                    # Not yet persisted (edge case) — write what we have.
                    await char_repo.update(player.character)
                logger.debug(
                    "character_reconciled",
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

    async def join_session(
        self,
        channel_id: int,
        user_id: int,
        user_name: str,
        character: Character,
        session_key: Optional[str] = None,
    ) -> Optional[PlayerInfo]:
        """Add a player to an active session.

        Args:
            session_key: If provided, look up session by key instead of channel_id.
                Used by voice/web frontends where sessions aren't keyed by Discord channel.
        """
        if session_key:
            session = self.get_session_by_key(session_key)
        else:
            session = self.get_session(channel_id)
        if not session or session.state == SessionState.ENDED:
            return None

        player = session.add_player(user_id, user_name, character)

        # Update party status in memory (peek: only touch a manager that is
        # already live in the module cache)
        memory = peek_memory_manager(session.campaign_id)
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
            memory = peek_memory_manager(session.campaign_id)
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
        frontend: Optional[GameFrontend] = None,
        session_key: Optional[str] = None,
    ) -> Optional[DMResponse]:
        """
        Process a player message through the LLM DM.

        This is the core game loop entry point.

        Args:
            frontend: If provided, events are emitted via frontend.on_event()
                instead of using the raw callbacks. The callbacks are kept for
                backward compatibility but frontend takes precedence.
            session_key: If provided, look up session by key instead of channel_id.
                Used by voice/web frontends.
        """
        if session_key:
            session = self.get_session_by_key(session_key)
        else:
            session = self.get_session(channel_id)
        if not session or session.state not in (SessionState.ACTIVE, SessionState.COMBAT):
            return None

        # Get player info
        player = session.get_player(user_id)
        if not player:
            # Player hasn't joined with a character
            return None

        session.update_activity()

        # Get memory manager — the module LRU is the single owner; the async
        # getter reloads persisted tiers if this campaign was evicted.
        memory = await get_memory_manager(session.campaign_id)

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

        # Get or create scene registry for entity tracking (per-session keying).
        scene_registry = get_scene_registry(session.campaign_id, session.session_key)

        # Process through orchestrator
        async with self._processing_lock:
            try:
                # Set session and scene registry context
                self.orchestrator.set_session(session)
                self.orchestrator.set_scene_registry(scene_registry)

                # If frontend provided, bridge events to callbacks for orchestrator
                eff_mechanics_cb = on_mechanics_ready
                eff_narrative_cb = on_narrative_token
                if frontend is not None:
                    async def _fe_mechanics(mechanical_result, dice_rolls):
                        await frontend.on_event(
                            GameEvent.mechanics_ready(mechanical_result, dice_rolls)
                        )
                    async def _fe_narrative(token: str):
                        await frontend.on_event(GameEvent.narrative_token(token))
                    eff_mechanics_cb = _fe_mechanics
                    eff_narrative_cb = _fe_narrative

                response = await self.orchestrator.process_action(
                    action=content,
                    player_name=player.character.name if player.character else user_name,
                    context=context,
                    on_mechanics_ready=eff_mechanics_cb,
                    on_narrative_token=eff_narrative_cb,
                )

                # Store last narrative for /imagine command
                session.last_narrative = response.narrative

                # Emit narrative complete event to frontend (with immersion data)
                if frontend is not None:
                    # Gather context for immersion pipelines (TTS + images)
                    _player_chars = [
                        p.character for p in session.players.values()
                        if p.character
                    ]
                    await frontend.on_event(
                        GameEvent.narrative_complete(
                            narrative=response.narrative,
                            proposed_effects=getattr(response, 'proposed_effects', []),
                            scene_entities=scene_registry.get_all() if scene_registry else [],
                            player_characters=_player_chars,
                        )
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
            combat = get_combat_by_key(session.session_key)
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
        scene_registry = get_scene_registry(session.campaign_id, session.session_key)
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

        # Update memory (peek: sync path, never create a fresh manager here)
        memory = peek_memory_manager(session.campaign_id)
        if memory:
            memory.update_scene(
                f"COMBAT: {combat_manager.combat.encounter_name or 'Battle'} - "
                f"Round {combat_manager.combat.current_round}"
            )

        return True

    async def end_combat(self, channel_id: int, session_key: Optional[str] = None) -> bool:
        """Single owner of combat teardown (audit P0-3).

        Every bot-layer path that ends combat — slash commands, the /game
        turn loop, auto-detected combat-over — must call this instead of
        hand-rolling ``manager.end_combat()`` + registry clears (previously
        copy-pasted at 9 cog sites with drifted contents, none of which
        returned ``session.state`` to ACTIVE, so a session reported COMBAT
        forever after its first fight).

        In order:
        1. Persist player combatants to their Characters via the
           coordinator's single implementation (a transient coordinator is
           built for the manual /combat paths that never created one).
           Persistence failures are logged but never block teardown — a
           session wedged in COMBAT is worse than one missed sync.
        2. Finalize the manager (skipped if a turn advance already ended it,
           so ``ended_at`` is stamped exactly once).
        3. Clear BOTH module registries (combat manager + turn coordinator).
        4. Reset ``session.state`` to ACTIVE and drop
           ``session.combat_manager``.

        The whole sequence runs under the same per-channel turn lock the
        coordinator's mutation methods hold (audit P0-6), so teardown can
        never interleave with a half-finished turn. Callers must therefore
        not invoke this while holding that lock. Clearing the coordinator
        registry below also drops the lock entry itself.

        Idempotent: a second call finds nothing to do and returns False.
        Also works with no GameSession (the /combat commands run standalone)
        and heals a COMBAT session whose manager was never created.
        """
        from .combat.coordinator import (
            CombatTurnCoordinator,
            clear_coordinator_by_key,
            get_coordinator_by_key,
            get_turn_lock,
        )

        key = session_key or f"discord:{channel_id}"
        did_work = False

        async with get_turn_lock(key):
            session = self._sessions.get(key)

            manager = get_combat_by_key(key)
            if manager is None and session is not None:
                manager = session.combat_manager

            if manager is not None:
                coordinator = get_coordinator_by_key(key)
                if coordinator is None:
                    coordinator = CombatTurnCoordinator(manager, session)
                elif coordinator.session is None:
                    # /combat cog paths register coordinators without a session;
                    # bind it so persistence resolves the session-owned Character
                    # instances (Stage A.2 single authority).
                    coordinator.session = session
                try:
                    await coordinator.persist_player_characters()
                except Exception as e:
                    logger.warning(
                        "combat_end_persist_failed",
                        session_key=key,
                        error=str(e),
                    )
                if manager.combat.state != CombatState.COMBAT_END:
                    manager.end_combat()
                did_work = True

            clear_combat_by_key(key)
            clear_coordinator_by_key(key)

            if session is not None:
                if session.state == SessionState.COMBAT:
                    session.state = SessionState.ACTIVE
                    did_work = True
                session.combat_manager = None

        if did_work:
            logger.info(
                "combat_teardown_complete",
                session_key=key,
                session_id=session.id if session else None,
            )
        return did_work


# Singleton instance
_manager: Optional[GameSessionManager] = None


def get_session_manager() -> GameSessionManager:
    """Get the singleton session manager."""
    global _manager
    if _manager is None:
        _manager = GameSessionManager()
    return _manager
