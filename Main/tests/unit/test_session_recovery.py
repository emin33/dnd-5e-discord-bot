"""ROOT-3: per-turn world persistence + recover_sessions.

The core deliverable: a restart stops silently dropping the live world
(audit DF-5), recovery runs the SAME init as start_session (DF-16 scene
roster, N10 Chroma sync — via the shared helpers), recovered sessions are
reachable (DF-7: real guild_id from the campaign row; the bot layer
rebuilds the active-campaign map from the returned sessions), and rows
that cannot be rebuilt are ended per-row (replacing the blanket
end_stale_sessions sweep).

Declared non-goal, pinned here: mid-combat resume. A session that died
in combat recovers as ACTIVE/exploration — combat state is in-memory
bot-layer machinery (manager registry, coordinator, Discord views) the
suite cannot hold a net for.

Collaborator seams: tests/unit/conftest.py (shared with the
start_session pins — same fakes on both paths is the parity point).
"""

import json

from dnd_bot.game.scene.registry import get_scene_registry
from dnd_bot.game.session import GameSession, SessionState
from dnd_bot.game.world_state import NPCState, WorldState
from dnd_bot.game.world_store import WorldStateStore
from dnd_bot.llm.orchestrator import DMResponse
from dnd_bot.models.campaign import Campaign
from dnd_bot.models.npc import NPC
from tests.session_fakes import FakeMemoryManager, kg_entity

CAMPAIGN_ID = "recovery-campaign"
GUILD_ID = 777


def _campaign() -> Campaign:
    return Campaign(
        id=CAMPAIGN_ID,
        guild_id=GUILD_ID,
        name="Recovery Campaign",
        dm_user_id=111,
    )


def _world(phase: str = "dialogue") -> WorldState:
    ws = WorldState(
        turn=17,
        phase=phase,
        time_of_day="dusk",
        current_location="The Gilded Flagon",
    )
    fred = NPCState(name="Fred", location="The Gilded Flagon", aliases=["the innkeeper"])
    ws.npcs = {fred.id: fred}
    ws.established_facts = ["The mayor is missing"]
    return ws


def _row(session_id: str, channel_id: int, state: str = "exploration") -> dict:
    return {
        "id": session_id,
        "campaign_id": CAMPAIGN_ID,
        "channel_id": channel_id,
        "session_number": 1,
        "state": state,
        "active_combat_id": None,
        "started_at": "2026-07-05 20:00:00",
    }


def _envelope(
    world: WorldState | None,
    players: list[dict] | None = None,
    session_key: str = "",
    dm_user_id: int | None = 111,
    guild_id: int = GUILD_ID,
    version: int = 1,
) -> str:
    return json.dumps(
        {
            "version": version,
            "session_key": session_key,
            "guild_id": guild_id,
            "dm_user_id": dm_user_id,
            "players": players or [],
            "world_state": (
                WorldStateStore(world).to_snapshot() if world is not None else None
            ),
        }
    )


class TestRecoverSessions:
    async def test_restores_world_and_membership(
        self,
        manager,
        fake_session_repo,
        fake_character_repo,
        fake_campaign_repo,
        fake_npc_repo,
        fake_kg,
        chroma_calls,
        memory_warm_calls,
        unique_channel_id,
        registry_cleanup,
        mock_character,
        mock_wizard,
    ):
        fake_campaign_repo.campaigns[CAMPAIGN_ID] = _campaign()
        fake_character_repo.characters = {
            mock_character.id: mock_character,
            mock_wizard.id: mock_wizard,
        }
        world = _world()
        key = f"discord:{unique_channel_id}"
        registry_cleanup.append(key)
        fake_session_repo.active_rows = [_row("s-1", unique_channel_id)]
        fake_session_repo.snapshots["s-1"] = _envelope(
            world,
            players=[
                {
                    "user_id": 12345,
                    "user_name": "Alice",
                    "character_id": mock_character.id,
                    "is_dm": True,
                },
                {
                    "user_id": 12346,
                    "user_name": "Bob",
                    "character_id": mock_wizard.id,
                    "is_dm": False,
                },
            ],
            session_key=key,
        )

        recovered = await manager.recover_sessions()

        assert len(recovered) == 1
        session = recovered[0]
        assert manager.get_session_by_key(key) is session
        assert session.id == "s-1"
        assert session.state == SessionState.ACTIVE
        assert session.campaign_id == CAMPAIGN_ID
        # DF-7: guild_id comes from the campaign row, not the old 0.
        assert session.guild_id == GUILD_ID
        assert session.dm_user_id == 111

        # The world survived: exact state, not a fresh husk.
        assert session.world_state is not None
        assert session.world_state.model_dump() == world.model_dump()

        # Membership rebuilt; Character objects are the repo's FRESH ones
        # (DF-1 lesson: DB is authoritative for character state).
        assert set(session.players) == {12345, 12346}
        alice = session.players[12345]
        assert alice.user_name == "Alice"
        assert alice.character is mock_character
        assert alice.is_dm is True
        assert session.players[12346].character is mock_wizard

        # Nothing was swept.
        assert fake_session_repo.ended == []

    async def test_runs_same_init_as_start_session(
        self,
        manager,
        fake_session_repo,
        fake_character_repo,
        fake_campaign_repo,
        fake_npc_repo,
        fake_kg,
        chroma_calls,
        memory_warm_calls,
        unique_channel_id,
        registry_cleanup,
    ):
        """DF-16 + N10: recovery runs the shared post-load init helpers."""
        fake_campaign_repo.campaigns[CAMPAIGN_ID] = _campaign()
        fake_kg.entities = [kg_entity("uuid-1", "Fred", "the innkeeper")]
        fake_npc_repo.npcs = [NPC(id="npc-1", campaign_id=CAMPAIGN_ID, name="Grokk")]
        key = f"discord:{unique_channel_id}"
        registry_cleanup.append(key)
        fake_session_repo.active_rows = [_row("s-1", unique_channel_id)]
        fake_session_repo.snapshots["s-1"] = _envelope(_world(), session_key=key)

        recovered = await manager.recover_sessions()

        assert len(recovered) == 1
        session = recovered[0]

        # KG loaded (same helper as start_session).
        assert session.knowledge_graph is fake_kg.built[0]
        assert session.knowledge_graph.loaded is True

        # N10: the ChromaDB entity sync recovery used to skip.
        assert chroma_calls == [
            {
                "campaign_id": CAMPAIGN_ID,
                "node_id": "uuid-1",
                "entity_type": "npc",
                "name": "Fred",
                "description": "the innkeeper",
                "aliases": [],
            }
        ]

        # DF-16: the scene-registry roster recovery used to skip.
        registry = get_scene_registry(CAMPAIGN_ID, key)
        assert [e.name for e in registry.get_all()] == ["Grokk"]

        # Memory tiers warmed.
        assert memory_warm_calls == [CAMPAIGN_ID]

    async def test_row_without_snapshot_is_ended(
        self,
        manager,
        fake_session_repo,
        fake_character_repo,
        fake_campaign_repo,
        fake_npc_repo,
        fake_kg,
        chroma_calls,
        memory_warm_calls,
        unique_channel_id,
    ):
        """The per-row sweep that replaces end_stale_sessions."""
        fake_campaign_repo.campaigns[CAMPAIGN_ID] = _campaign()
        fake_session_repo.active_rows = [_row("s-husk", unique_channel_id)]

        recovered = await manager.recover_sessions()

        assert recovered == []
        assert fake_session_repo.ended == ["s-husk"]
        assert manager.get_session(unique_channel_id) is None

    async def test_combat_session_downgrades_to_exploration(
        self,
        manager,
        fake_session_repo,
        fake_character_repo,
        fake_campaign_repo,
        fake_npc_repo,
        fake_kg,
        chroma_calls,
        memory_warm_calls,
        unique_channel_id,
        registry_cleanup,
    ):
        """Mid-combat resume is a DECLARED NON-GOAL: honest downgrade."""
        fake_campaign_repo.campaigns[CAMPAIGN_ID] = _campaign()
        key = f"discord:{unique_channel_id}"
        registry_cleanup.append(key)
        fake_session_repo.active_rows = [_row("s-1", unique_channel_id, state="combat")]
        fake_session_repo.snapshots["s-1"] = _envelope(
            _world(phase="combat"), session_key=key
        )

        recovered = await manager.recover_sessions()

        assert len(recovered) == 1
        session = recovered[0]
        assert session.state == SessionState.ACTIVE
        assert session.combat_manager is None
        # Phase reconciled through the store's seam, not left frozen.
        assert session.world_state.phase == "exploration"
        # The rest of the world still survived.
        assert session.world_state.turn == 17

    async def test_missing_campaign_row_is_ended(
        self,
        manager,
        fake_session_repo,
        fake_character_repo,
        fake_campaign_repo,
        fake_npc_repo,
        fake_kg,
        chroma_calls,
        memory_warm_calls,
        unique_channel_id,
    ):
        fake_session_repo.active_rows = [_row("s-orphan", unique_channel_id)]
        fake_session_repo.snapshots["s-orphan"] = _envelope(_world())

        recovered = await manager.recover_sessions()

        assert recovered == []
        assert fake_session_repo.ended == ["s-orphan"]

    async def test_corrupt_snapshot_is_ended_and_loop_continues(
        self,
        manager,
        fake_session_repo,
        fake_character_repo,
        fake_campaign_repo,
        fake_npc_repo,
        fake_kg,
        chroma_calls,
        memory_warm_calls,
        unique_channel_ids,
        registry_cleanup,
    ):
        """One bad row must not abort recovery of the others."""
        fake_campaign_repo.campaigns[CAMPAIGN_ID] = _campaign()
        bad_channel, good_channel = unique_channel_ids(), unique_channel_ids()
        good_key = f"discord:{good_channel}"
        registry_cleanup.append(good_key)
        fake_session_repo.active_rows = [
            _row("s-bad", bad_channel),
            _row("s-good", good_channel),
        ]
        fake_session_repo.snapshots["s-bad"] = "{not json"
        fake_session_repo.snapshots["s-good"] = _envelope(_world(), session_key=good_key)

        recovered = await manager.recover_sessions()

        assert [s.id for s in recovered] == ["s-good"]
        assert fake_session_repo.ended == ["s-bad"]

    async def test_unknown_envelope_version_is_ended(
        self,
        manager,
        fake_session_repo,
        fake_character_repo,
        fake_campaign_repo,
        fake_npc_repo,
        fake_kg,
        chroma_calls,
        memory_warm_calls,
        unique_channel_id,
    ):
        fake_campaign_repo.campaigns[CAMPAIGN_ID] = _campaign()
        fake_session_repo.active_rows = [_row("s-future", unique_channel_id)]
        fake_session_repo.snapshots["s-future"] = _envelope(_world(), version=2)

        recovered = await manager.recover_sessions()

        assert recovered == []
        assert fake_session_repo.ended == ["s-future"]

    async def test_missing_character_skips_player_not_session(
        self,
        manager,
        fake_session_repo,
        fake_character_repo,
        fake_campaign_repo,
        fake_npc_repo,
        fake_kg,
        chroma_calls,
        memory_warm_calls,
        unique_channel_id,
        registry_cleanup,
        mock_character,
    ):
        fake_campaign_repo.campaigns[CAMPAIGN_ID] = _campaign()
        fake_character_repo.characters = {mock_character.id: mock_character}
        key = f"discord:{unique_channel_id}"
        registry_cleanup.append(key)
        fake_session_repo.active_rows = [_row("s-1", unique_channel_id)]
        fake_session_repo.snapshots["s-1"] = _envelope(
            _world(),
            players=[
                {"user_id": 12345, "user_name": "Alice",
                 "character_id": mock_character.id, "is_dm": False},
                {"user_id": 99999, "user_name": "Ghost",
                 "character_id": "deleted-char", "is_dm": False},
            ],
            session_key=key,
        )

        recovered = await manager.recover_sessions()

        assert len(recovered) == 1
        assert set(recovered[0].players) == {12345}
        assert fake_session_repo.ended == []

    async def test_duplicate_session_key_keeps_newest_row(
        self,
        manager,
        fake_session_repo,
        fake_character_repo,
        fake_campaign_repo,
        fake_npc_repo,
        fake_kg,
        chroma_calls,
        memory_warm_calls,
        unique_channel_id,
        registry_cleanup,
    ):
        """Rows arrive newest-first; a second row for the same key is a husk."""
        fake_campaign_repo.campaigns[CAMPAIGN_ID] = _campaign()
        key = f"discord:{unique_channel_id}"
        registry_cleanup.append(key)
        fake_session_repo.active_rows = [
            _row("s-newer", unique_channel_id),
            _row("s-older", unique_channel_id),
        ]
        fake_session_repo.snapshots["s-newer"] = _envelope(
            _world(), session_key=key
        )
        fake_session_repo.snapshots["s-older"] = _envelope(
            WorldState(), session_key=key
        )

        recovered = await manager.recover_sessions()

        assert [s.id for s in recovered] == ["s-newer"]
        assert manager.get_session_by_key(key).id == "s-newer"
        assert fake_session_repo.ended == ["s-older"]

    async def test_foreign_frontend_session_is_ended_not_recovered(
        self,
        manager,
        fake_session_repo,
        fake_character_repo,
        fake_campaign_repo,
        fake_npc_repo,
        fake_kg,
        chroma_calls,
        memory_warm_calls,
    ):
        """Review F1: this process cannot serve a voice/web key — nothing
        could ever END the recovered session (end_session reaches only
        discord: keys; the voice API runs as its own process), so every
        boot would rebuild it forever. Ending the row restores the bound
        the old startup sweep gave these sessions.

        Flipped from test_custom_session_key_is_preserved, which pinned
        voice recovery as intended without an owner for its lifecycle.
        """
        fake_campaign_repo.campaigns[CAMPAIGN_ID] = _campaign()
        fake_session_repo.active_rows = [_row("s-voice", 0)]
        fake_session_repo.snapshots["s-voice"] = _envelope(
            _world(), session_key="voice:room-1"
        )

        recovered = await manager.recover_sessions()

        assert recovered == []
        assert fake_session_repo.ended == ["s-voice"]
        assert manager.get_session_by_key("voice:room-1") is None


class FakeOrchestrator:
    """Records process_action turns; returns a canned DMResponse."""

    def __init__(self):
        self.session = None
        self.registry = None
        self.actions: list[str] = []

    def set_session(self, session):
        self.session = session

    def set_scene_registry(self, registry):
        self.registry = registry

    async def process_action(
        self, action, player_name, context,
        on_mechanics_ready=None, on_narrative_token=None,
    ):
        self.actions.append(action)
        return DMResponse(narrative="You proceed carefully.")


class TestPerTurnPersistence:
    async def _session_with_player(
        self, manager, channel_id, registry_cleanup, character
    ) -> GameSession:
        session = await manager.start_session(
            channel_id=channel_id, guild_id=GUILD_ID, campaign_id=CAMPAIGN_ID
        )
        registry_cleanup.append(session.session_key)
        session.add_player(12345, "Alice", character)
        return session

    async def test_turn_persists_world_snapshot(
        self,
        manager,
        fake_session_repo,
        fake_npc_repo,
        fake_kg,
        chroma_calls,
        unique_channel_id,
        registry_cleanup,
        mock_character,
        monkeypatch,
    ):
        """After a processed turn the envelope is on disk (DF-5's fix)."""
        memory = FakeMemoryManager()
        memory.buffer.pinned_facts = ["The mayor is missing"]

        async def _get_memory(campaign_id):
            return memory

        monkeypatch.setattr("dnd_bot.game.session.get_memory_manager", _get_memory)
        orch = FakeOrchestrator()
        monkeypatch.setattr("dnd_bot.game.session.get_orchestrator", lambda: orch)

        session = await self._session_with_player(
            manager, unique_channel_id, registry_cleanup, mock_character
        )

        response = await manager.process_message(
            channel_id=unique_channel_id,
            user_id=12345,
            user_name="Alice",
            content="I search the room",
        )

        assert response is not None
        assert orch.actions == ["I search the room"]

        envelope = json.loads(fake_session_repo.snapshots[session.id])
        assert envelope["version"] == 1
        assert envelope["session_key"] == session.session_key
        assert envelope["guild_id"] == GUILD_ID
        assert envelope["players"] == [
            {
                "user_id": 12345,
                "user_name": "Alice",
                "character_id": mock_character.id,
                "is_dm": False,
            }
        ]
        world = envelope["world_state"]
        # begin_turn ran inside the turn: counter advanced, party synced,
        # and the pinned fact landed — the snapshot is POST-turn state.
        assert world["turn"] == 1
        assert world["players"]["Test Hero"]["hp"] == 44
        assert "The mayor is missing" in world["established_facts"]

    async def test_persist_failure_never_breaks_the_turn(
        self,
        manager,
        fake_session_repo,
        fake_npc_repo,
        fake_kg,
        chroma_calls,
        unique_channel_id,
        registry_cleanup,
        mock_character,
        monkeypatch,
    ):
        """The standing persist_failed policy: log, keep playing."""
        memory = FakeMemoryManager()

        async def _get_memory(campaign_id):
            return memory

        monkeypatch.setattr("dnd_bot.game.session.get_memory_manager", _get_memory)
        orch = FakeOrchestrator()
        monkeypatch.setattr("dnd_bot.game.session.get_orchestrator", lambda: orch)

        await self._session_with_player(
            manager, unique_channel_id, registry_cleanup, mock_character
        )
        fake_session_repo.save_snapshot_raises = RuntimeError("db locked")

        response = await manager.process_message(
            channel_id=unique_channel_id,
            user_id=12345,
            user_name="Alice",
            content="I search the room",
        )

        assert response is not None
        assert response.narrative == "You proceed carefully."
