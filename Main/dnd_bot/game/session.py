"""Game session management - the core game loop."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
import asyncio
import json
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
from ..data.repositories import get_campaign_repo, get_character_repo, get_session_repo
from ..data.repositories.npc_repo import get_npc_repo
from ..models.npc import EntityType, Disposition, SceneEntity
from .combat.manager import CombatManager, get_combat_by_key, clear_combat_by_key
from .modes import GameMode, ModeMachine
from .scene.registry import get_scene_registry, clear_scene_registry
from .world_state import WorldState
from .world_store import WorldStateStore

logger = structlog.get_logger()

# Wire-format version of the session_snapshot envelope (ROOT-3). Bump on
# any shape change; recovery refuses versions it doesn't know rather than
# resuming from a misread world.
_SNAPSHOT_VERSION = 1

# WorldState stores disposition as a str; the registry/SceneEntity use the
# Disposition enum. This maps the former back to the latter when the F4
# preload seeds registry entities from recovered NPCStates (Stage C).
_DISPOSITION_BY_VALUE = {d.value: d for d in Disposition}


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

    # Mode pushdown machine (REFACTOR_PLAN Step 3): combat pushes onto
    # exploration and pops back. enter_combat_mode/exit_combat_mode below
    # are the only writers of the mode flip and its derived surfaces.
    modes: ModeMachine = field(default_factory=ModeMachine)

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

    @property
    def world_store(self) -> Optional[WorldStateStore]:
        """Write authority over this session's world state (Step 4).

        Every WorldState mutation goes through the store's apply methods.
        Derived per access (the store is stateless beyond the wrapped
        reference), so reassigning ``world_state`` — tests, resets — can
        never orphan a stale wrapper. None while the session has no world
        state.
        """
        if self.world_state is None:
            return None
        return WorldStateStore(self.world_state)

    def enter_combat_mode(self, combat_manager: Optional[CombatManager] = None) -> None:
        """Push COMBAT mode — the single combat-entry flip (Step 3).

        One owner for every derived surface of the mode: ``state`` flips to
        COMBAT, the manager is stored, and ``world_state.phase`` follows
        immediately (previously it lagged until the next turn's phase sync).

        Args:
            combat_manager: the encounter just built; None keeps whatever
                the entry path already stored on the session.
        """
        self.modes.push(GameMode.COMBAT)
        self.state = SessionState.COMBAT
        if combat_manager is not None:
            self.combat_manager = combat_manager
        store = self.world_store
        if store is not None:
            store.reconcile_phase(in_combat=True)
        logger.warning(
            "session_transitioned_to_combat",
            session_id=self.id,
        )

    def exit_combat_mode(self) -> bool:
        """Pop combat mode — the teardown owner's pop transition (Step 3).

        Returns True when the session was actually in combat mode (the
        ``did_work`` accounting ``end_combat`` keys on). The manager
        reference is dropped unconditionally and ``world_state.phase``
        returns to exploration only if it still reads combat — a narrative
        phase the delta extractor set (dialogue, rest, …) is preserved.
        """
        self.modes.pop()
        was_combat = self.state == SessionState.COMBAT
        if was_combat:
            self.state = SessionState.ACTIVE
        self.combat_manager = None
        store = self.world_store
        if store is not None:
            store.reconcile_phase(in_combat=False)
        return was_combat


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
        await self._load_knowledge_graph(session)

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

        # Pre-load known NPCs from database into scene registry.
        await self._preload_scene_npcs(session)

        logger.info(
            "session_started",
            session_id=session.id,
            channel_id=channel_id,
            campaign_id=campaign_id,
            session_number=session_number,
        )

        return session

    async def _load_knowledge_graph(self, session: GameSession) -> None:
        """Load the campaign KG + sync entity descriptions to ChromaDB.

        Shared by start_session AND recover_sessions (audit DF-16/N10:
        recovery previously skipped the Chroma sync entirely). Failure
        isolation is the caller's contract: a KG failure leaves
        ``knowledge_graph = None``, a Chroma failure keeps the loaded KG;
        neither stops the session.
        """
        campaign_id = session.campaign_id
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
                    logger.warning("kg_entity_description_sync_failed", error=str(e), exc_info=True)

            logger.info(
                "knowledge_graph_loaded",
                campaign_id=campaign_id,
                nodes=session.knowledge_graph.node_count(),
                edges=session.knowledge_graph.edge_count(),
            )
        except Exception as e:
            logger.warning("knowledge_graph_load_failed", error=str(e), exc_info=True)
            session.knowledge_graph = None

    async def _preload_scene_npcs(self, session: GameSession) -> None:
        """Seed the scene registry (and world) with the campaign's NPCs.

        Shared by start_session AND recover_sessions (audit DF-16: recovery
        previously skipped this, so a recovered narrator lost "who is in
        the room"). Keyed by ``session.session_key`` (audit #8) so
        concurrent voice/web sessions — which all set ``channel_id=0`` —
        don't collide on one shared registry. Failures never stop the
        session.

        Stage C makes this the id-convergence point across the session
        boundary. It seeds from the UNION of two rosters, both keyed on the
        canonical NPCState UUID:

        - The durable DB roster (``get_alive_by_campaign``): each row seeds
          a registry SceneEntity AND — the new part — a WorldState NPCState
          under the SAME id (== the row id == the KG node id), so a
          returning NPC keeps its canonical id instead of the first mention
          minting a fresh divergent UUID. A row the recovered WorldState
          knows is DEAD is skipped: WorldState is authoritative for the live
          session, and the DB row is just stale (death that never reached it
          before a crash) — seeding it would resurrect the corpse.
        - The recovered WorldState's own NPCs (F4): extractor-minted NPCs
          live only in the snapshot (the per-turn DB sync is dead; a crash
          beat end_session), so they seed the registry here — otherwise the
          narrator roster forgets someone the world remembers. Dead ones and
          those a DB row already covered are skipped (no duplicate
          SceneEntity — the naive-seed hazard the ROOT-3 wave flagged).
        """
        campaign_id = session.campaign_id
        scene_registry = get_scene_registry(campaign_id, session.session_key)
        world = session.world_state
        store = session.world_store
        try:
            npc_repo = await get_npc_repo()
            campaign_npcs = await npc_repo.get_alive_by_campaign(campaign_id)
            seen_ids: set[str] = set()

            # 1. Durable DB roster — registry + world, under the row id.
            for npc in campaign_npcs:
                ws_state = world.npcs.get(npc.id) if world else None
                if ws_state is not None and not ws_state.alive:
                    continue  # world says dead; don't resurrect the corpse
                disposition = (
                    npc.base_disposition
                    if isinstance(npc.base_disposition, Disposition)
                    else Disposition.NEUTRAL
                )
                scene_registry.register_entity(SceneEntity(
                    name=npc.name,
                    npc_id=npc.id,
                    entity_type=EntityType.NPC,
                    description=npc.description or "",
                    monster_index=npc.monster_index,
                    disposition=disposition,
                    voice_id=npc.voice_id,
                ))
                seen_ids.add(npc.id)
                if store is not None and npc.id not in (world.npcs if world else {}):
                    store.seed_npc(
                        npc.id,
                        npc.name,
                        location=npc.location or "",
                        disposition=disposition.value,
                        description=npc.description or "",
                    )

            # 2. WorldState-only NPCs (recovery, F4) — registry only.
            if world is not None:
                for npc_state in list(world.npcs.values()):
                    if not npc_state.alive or npc_state.id in seen_ids:
                        continue
                    scene_registry.register_entity(SceneEntity(
                        name=npc_state.name,
                        npc_id=npc_state.id,
                        entity_type=EntityType.NPC,
                        description=npc_state.description or "",
                        disposition=_DISPOSITION_BY_VALUE.get(
                            npc_state.disposition, Disposition.NEUTRAL
                        ),
                    ))
                    seen_ids.add(npc_state.id)

            if seen_ids:
                logger.info(
                    "npcs_loaded",
                    count=len(seen_ids),
                    db_count=len(campaign_npcs),
                    campaign_id=campaign_id,
                )
        except Exception as e:
            logger.warning("npc_preload_failed", error=str(e), exc_info=True)

    async def _persist_world_snapshot(self, session: GameSession) -> None:
        """Write the session's snapshot envelope (ROOT-3, DF-5).

        Called once per processed turn. The envelope carries the world
        state (via the store's owned format) plus the session-level facts
        a restart loses: membership, DM, guild, session key. Characters
        are stored by id only — the DB row is authoritative for their
        state (the DF-1 lesson), so recovery re-fetches fresh.

        Persistence failures are logged and never break the turn (the
        standing persist_failed policy): the player keeps playing, the
        snapshot retries next turn.
        """
        try:
            store = session.world_store
            envelope = {
                "version": _SNAPSHOT_VERSION,
                "session_key": session.session_key,
                "guild_id": session.guild_id,
                "dm_user_id": session.dm_user_id,
                "players": [
                    {
                        "user_id": p.user_id,
                        "user_name": p.user_name,
                        "character_id": p.character.id if p.character else None,
                        "is_dm": p.is_dm,
                    }
                    for p in session.players.values()
                ],
                "world_state": store.to_snapshot() if store is not None else None,
            }
            session_repo = await get_session_repo()
            await session_repo.save_world_snapshot(session.id, json.dumps(envelope))
        except Exception as e:
            logger.error(
                "persist_failed",
                entity="world_snapshot",
                session_id=session.id,
                error=str(e),
                exc_info=True,
            )

    async def recover_sessions(self) -> list[GameSession]:
        """Rebuild live sessions from persisted snapshots at startup (ROOT-3).

        For every non-ended ``game_session`` row: rebuild the session from
        its snapshot envelope and run the SAME init helpers as
        start_session (KG + Chroma sync, scene-NPC preload, memory warm —
        the DF-16/N10 parity the old deleted recover_sessions never had).
        Any row that cannot be rebuilt — no snapshot (predates this
        feature, or a voice/web husk), unknown envelope version, missing
        campaign, corrupt payload, duplicate session key — is marked ended
        right there, which replaces the old blanket end_stale_sessions
        sweep with a per-row verdict.

        Declared non-goals of this slice (each logged where it bites):
        - Mid-combat resume: a session that died in combat recovers as
          ACTIVE/exploration (combat state is 100% in-memory bot-layer
          machinery — manager registry, turn coordinator, Discord views —
          that the suite cannot hold a net for).
        - Voice/web sessions: this process has no frontend that can serve
          a ``voice:*``/``web:*`` key (the voice API runs as its own
          process and never calls recover_sessions), and nothing could
          ever END such a recovered session, so those rows are ended —
          the same bound the pre-recovery startup sweep gave them. A real
          voice-resume story needs its own owner.
        - Memory tiers: they persist only at graceful end_session, so a
          crash loses the message buffer / running summary accumulated
          since the last graceful end (pinned facts partially survive via
          the per-turn established_facts sync). The recovered WORLD is
          current; the narrator's conversational memory is as of the last
          graceful end.
        - Scene-registry roster: recovery re-seeds from the npc DB (same
          as start_session). NPCs the extractor minted mid-session live
          in the recovered WorldState but re-enter the registry only when
          next referenced (the per-turn registry sync) — making them
          durable in the registry is Stage-C canonical-id work; seeding
          them naively here would mint duplicate SceneEntities.

        Returns the recovered sessions; the bot layer rebuilds the
        active-campaign map from them (DF-7).
        """
        session_repo = await get_session_repo()
        char_repo = await get_character_repo()
        campaign_repo = await get_campaign_repo()

        rows = await session_repo.load_active_sessions()
        recovered: list[GameSession] = []
        for row in rows:
            session: Optional[GameSession] = None
            try:
                session = await self._recover_one(row, session_repo, char_repo, campaign_repo)
            except Exception as e:
                logger.error(
                    "session_recovery_failed",
                    session_id=row["id"],
                    error=str(e),
                    exc_info=True,
                )

            if session is None:
                # Unrecoverable: end the row so it never zombie-recurs.
                await session_repo.end_session(row["id"])
                continue

            self._sessions[session.session_key] = session
            recovered.append(session)
            logger.info(
                "session_recovered",
                session_id=session.id,
                session_key=session.session_key,
                campaign_id=session.campaign_id,
                players=len(session.players),
                world_turn=session.world_state.turn if session.world_state else None,
            )

        return recovered

    async def _recover_one(
        self,
        row: dict,
        session_repo,
        char_repo,
        campaign_repo,
    ) -> Optional[GameSession]:
        """Rebuild one session from its snapshot; None = unrecoverable."""
        session_id = row["id"]

        payload = await session_repo.get_latest_snapshot(session_id)
        if payload is None:
            logger.info("session_recovery_no_snapshot", session_id=session_id)
            return None

        envelope = json.loads(payload)
        if envelope.get("version") != _SNAPSHOT_VERSION:
            logger.warning(
                "session_recovery_unknown_version",
                session_id=session_id,
                version=envelope.get("version"),
            )
            return None

        # Frontends this process cannot serve (review F1): a recovered
        # voice/web session would be immortal — end_session only reaches
        # discord: keys, the voice API process never recovers, and every
        # boot would rebuild it (KG + Chroma + registry warm) forever.
        # Ending the row restores the bound the old startup sweep gave
        # these sessions.
        snapshot_key = envelope.get("session_key") or ""
        if snapshot_key and not snapshot_key.startswith("discord:"):
            logger.info(
                "session_recovery_foreign_frontend",
                session_id=session_id,
                session_key=snapshot_key,
            )
            return None

        # The campaign row is the authoritative guild_id source (DF-7:
        # the old recover stored guild_id=0, so every slash command's
        # get_active_campaign_id lookup missed and the session was a
        # zombie). No campaign row -> nothing to route to -> unrecoverable.
        campaign = await campaign_repo.get_by_id(row["campaign_id"])
        if campaign is None:
            logger.warning(
                "session_recovery_campaign_missing",
                session_id=session_id,
                campaign_id=row["campaign_id"],
            )
            return None

        world_data = envelope.get("world_state")
        world_state = (
            WorldStateStore.state_from_snapshot(world_data)
            if world_data is not None
            else WorldState()
        )

        session = GameSession(
            id=session_id,
            channel_id=row["channel_id"],
            guild_id=campaign.guild_id,
            campaign_id=row["campaign_id"],
            session_key=envelope.get("session_key") or "",
            state=SessionState.ACTIVE,
            dm_user_id=envelope.get("dm_user_id"),
        )
        session.world_state = world_state

        if session.session_key in self._sessions:
            # Two non-ended rows for one key: rows arrive newest-first,
            # so the already-registered one is newer. End this husk.
            logger.warning(
                "session_recovery_duplicate_key",
                session_id=session_id,
                session_key=session.session_key,
            )
            return None

        # Combat never survives a restart (declared non-goal): return the
        # frozen phase to exploration through the store's reconcile seam.
        if row["state"] == "combat" or world_state.phase == "combat":
            logger.warning(
                "combat_not_recovered",
                session_id=session_id,
                reason="mid-combat resume is out of scope; session resumes in exploration",
            )
        store = session.world_store
        if store is not None:
            store.reconcile_phase(in_combat=False)

        # Membership: ids from the envelope, Character state fresh from
        # the DB (authoritative per the DF-1 lesson).
        for entry in envelope.get("players", []):
            character_id = entry.get("character_id")
            if not character_id:
                continue
            character = await char_repo.get_by_id(character_id)
            if character is None:
                logger.warning(
                    "session_recovery_character_missing",
                    session_id=session_id,
                    character_id=character_id,
                )
                continue
            player = PlayerInfo(
                user_id=entry["user_id"],
                user_name=entry.get("user_name", ""),
                character=character,
                is_dm=bool(entry.get("is_dm", False)),
            )
            session.players[player.user_id] = player

        # The same post-load init start_session runs (DF-16 / N10 parity).
        await self._load_knowledge_graph(session)
        await self._preload_scene_npcs(session)
        await get_memory_manager(session.campaign_id)

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

            # Sync NPCs to repository and clear scene registry. Pass the
            # world's current location so the npc rows get a real place
            # (DF-19), not the old scene-description slice.
            try:
                scene_registry = get_scene_registry(session.campaign_id, session.session_key)
                current_location = (
                    session.world_state.current_location
                    if session.world_state else None
                )
                await scene_registry.sync_to_npc_repo(current_location=current_location)
                clear_scene_registry(session.session_key)
            except Exception as e:
                logger.error(
                    "scene_registry_cleanup_failed",
                    session_id=session.id,
                    error=str(e),
                    exc_info=True,
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
                    "persist_failed",
                    entity="character",
                    character_id=player.character.id,
                    character=player.character.name,
                    error=str(e),
                    exc_info=True,
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

        # Turn bookkeeping through the single-writer store (Step 4):
        # advance the counter, refresh party snapshots, and reconcile the
        # narrative phase with the session mode (the OTHER phase writer is
        # the delta extractor).
        store = session.world_store
        if store is not None:
            store.begin_turn(
                p.character for p in session.players.values() if p.character
            )
            store.reconcile_phase(session.state == SessionState.COMBAT)

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
                fact_store = session.world_store
                if fact_store is not None and memory.buffer.pinned_facts:
                    for fact in memory.buffer.pinned_facts:
                        fact_store.add_established_fact(fact)

                # Persist the live world so a restart stops silently
                # dropping it (ROOT-3, DF-5). All of this turn's writes
                # — extractor delta, effect sync, facts — have landed by
                # now. Failure-isolated: never breaks the turn.
                await self._persist_world_snapshot(session)

                # Combat entry needs no handling here: every entry signal
                # funnels through game.combat.encounter.start_encounter,
                # which pushes combat mode itself (Step 3 single decision
                # point). combat_triggered stays on the response for the
                # bot layer's combat-UI handoff.

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
                    logger.error(
                        "persist_failed",
                        entity="character",
                        session_key=key,
                        error=str(e),
                        exc_info=True,
                    )
                if manager.combat.state != CombatState.COMBAT_END:
                    manager.end_combat()
                did_work = True

            clear_combat_by_key(key)
            clear_coordinator_by_key(key)

            if session is not None:
                # The ModeMachine pop: state back to ACTIVE, manager
                # dropped, world_state.phase returned to exploration.
                if session.exit_combat_mode():
                    did_work = True

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
