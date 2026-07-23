"""DF-18 net: scene rescope on location change — registry half + turn seam.

WorldState-side rescoping (scene_items + scene NPC roster) is pinned in
test_world_state.py / test_world_state_sync.py. This file pins the
``SceneEntityRegistry.rescope_to_scene`` keep-rules and the
``GameSessionManager._rescope_scene_registry_after_move`` turn seam that
mirrors a move onto the registry (the two WorldState apply paths cannot
reach it).
"""

from types import SimpleNamespace

import pytest

from dnd_bot.game.scene.registry import SceneEntityRegistry
from dnd_bot.game.session import (
    GameSession,
    GameSessionManager,
    PlayerInfo,
    SessionState,
)
from dnd_bot.game.world_state import NPCState, WorldState
from dnd_bot.models.npc import Disposition, EntityType, SceneEntity


@pytest.fixture
def world() -> WorldState:
    return WorldState(current_location="Cellar")


@pytest.fixture
def registry() -> SceneEntityRegistry:
    return SceneEntityRegistry(campaign_id="camp", channel_id=0)


def _entity(
    registry: SceneEntityRegistry,
    name: str,
    entity_type: EntityType = EntityType.NPC,
    **fields,
) -> SceneEntity:
    return registry.register_entity(SceneEntity(
        name=name,
        entity_type=entity_type,
        disposition=Disposition.NEUTRAL,  # neutral skips the SRD auto-match
        **fields,
    ))


class TestRegistryRescope:
    def test_old_scene_npc_removed_new_scene_npc_kept(self, registry, world):
        guard = NPCState(name="Cellar Guard", location="Cellar")
        world.npcs[guard.id] = guard
        kept = _entity(registry, "Cellar Guard", npc_id=guard.id)
        _entity(registry, "Barkeep")  # old room; no world twin at Cellar

        removed = registry.rescope_to_scene(world)

        assert removed == 1
        assert registry.get_by_id(kept.id) is kept
        assert registry.get_by_name("Barkeep") is None

    def test_departed_npc_not_combat_targetable(self, registry, world):
        # Combat targeting iterates the full registry with no location
        # gate (orchestrator _initiate_combat_from_attack) — after the
        # rescope the old room's NPC is simply not there to target.
        _entity(registry, "Barkeep")
        registry.rescope_to_scene(world)
        assert registry.get_potential_targets() == []

    def test_npc_without_link_resolves_by_name(self, registry, world):
        guard = NPCState(name="Cellar Guard", location="Cellar")
        world.npcs[guard.id] = guard
        kept = _entity(registry, "Cellar Guard")  # no npc_id stamped
        assert registry.rescope_to_scene(world) == 0
        assert registry.get_by_id(kept.id) is kept

    def test_qualified_same_location_keeps_npc(self, registry, world):
        world.current_location = "the Cellar landing"
        guard = NPCState(name="Cellar Guard", location="Cellar landing")
        world.npcs[guard.id] = guard
        kept = _entity(registry, "Cellar Guard", npc_id=guard.id)

        assert registry.rescope_to_scene(world) == 0
        assert registry.get_by_id(kept.id) is kept

    def test_dead_entity_kept_for_db_persistence(self, registry, world):
        # An unpersisted death must survive until sync_to_npc_repo (DF-4);
        # rescoping it away would resurrect the NPC next session.
        dead = _entity(registry, "Barkeep", status="dead")
        assert registry.rescope_to_scene(world) == 0
        assert registry.get_by_id(dead.id) is dead

    def test_objects_follow_scene_items(self, registry, world):
        # The world clears scene_items on the same transition, so a
        # surviving key means "this object belongs to the NEW scene".
        world.scene_items["Wine Cask"] = "a dusty cask"
        kept = _entity(registry, "Wine Cask", entity_type=EntityType.OBJECT)
        _entity(registry, "Old Rope", entity_type=EntityType.OBJECT)

        removed = registry.rescope_to_scene(world)

        assert removed == 1
        assert registry.get_by_id(kept.id) is kept
        assert registry.get_by_name("Old Rope") is None


class TestTurnSeamRescope:
    def _session_with_world(self, location: str) -> GameSession:
        session = GameSession(
            id="rescope-session",
            channel_id=900901,
            guild_id=1,
            campaign_id="camp",
        )
        session.world_state = WorldState(current_location=location)
        return session

    async def test_move_rescopes_registry_and_reloads_db_roster(
        self, registry, monkeypatch
    ):
        manager = GameSessionManager()
        session = self._session_with_world("Cellar")
        _entity(registry, "Barkeep")  # stale old-room entity

        loads: list[str] = []

        async def _fake_load(location: str) -> int:
            loads.append(location)
            return 0

        monkeypatch.setattr(registry, "load_npcs_at_location", _fake_load)

        await manager._rescope_scene_registry_after_move(
            session, registry, "Tavern"
        )

        assert registry.get_by_name("Barkeep") is None
        # NPCs the DB records at the new location are loaded back in, so a
        # returning NPC keeps its canonical row id instead of re-minting.
        assert loads == ["Cellar"]

    async def test_no_move_leaves_registry_alone(self, registry, monkeypatch):
        manager = GameSessionManager()
        session = self._session_with_world("Tavern")
        stale = _entity(registry, "Barkeep")

        async def _fail_load(location: str) -> int:
            raise AssertionError("must not hit the DB when the party did not move")

        monkeypatch.setattr(registry, "load_npcs_at_location", _fail_load)

        await manager._rescope_scene_registry_after_move(
            session, registry, "Tavern"
        )
        assert registry.get_by_id(stale.id) is stale

    async def test_first_location_set_is_not_a_move(self, registry, monkeypatch):
        # Fresh sessions establish their first location; that must not
        # wipe the campaign preload (scene establishment, not a move).
        manager = GameSessionManager()
        session = self._session_with_world("Tavern")
        preloaded = _entity(registry, "Barkeep")

        async def _fail_load(location: str) -> int:
            raise AssertionError("scene establishment must not rescope")

        monkeypatch.setattr(registry, "load_npcs_at_location", _fail_load)

        await manager._rescope_scene_registry_after_move(session, registry, "")
        assert registry.get_by_id(preloaded.id) is preloaded

    async def test_db_load_failure_never_breaks_the_turn(
        self, registry, monkeypatch
    ):
        manager = GameSessionManager()
        session = self._session_with_world("Cellar")
        _entity(registry, "Barkeep")

        async def _boom(location: str) -> int:
            raise RuntimeError("db down")

        monkeypatch.setattr(registry, "load_npcs_at_location", _boom)

        # Must not raise (persist_failed policy); the rescope still landed.
        await manager._rescope_scene_registry_after_move(
            session, registry, "Tavern"
        )
        assert registry.get_by_name("Barkeep") is None


class _FakeMemory:
    """Just enough MemoryManager surface for process_message."""

    def __init__(self) -> None:
        self.buffer = SimpleNamespace(pinned_facts=[])

    def set_combat_state(self, in_combat: bool) -> None:
        pass

    async def add_player_message(self, content: str, author_name: str) -> None:
        pass

    async def add_dm_response(self, content: str, is_narration: bool) -> None:
        pass

    def update_scene(self, summary: str) -> None:
        pass


class _MovingOrchestrator:
    """process_action moves the party mid-turn, the way the narrator's
    change_location tool (or the extractor delta) does for real."""

    def __init__(self, new_location: str) -> None:
        self.new_location = new_location
        self.session = None

    def set_session(self, session) -> None:
        self.session = session

    def set_scene_registry(self, registry) -> None:
        pass

    async def process_action(self, action, player_name, context, **kwargs):
        self.session.world_state.current_location = self.new_location
        return SimpleNamespace(
            narrative="You push through the door into the tavern.",
            proposed_effects=[],
            mechanical_result=None,
            combat_triggered=False,
        )


class TestProcessMessageWiring:
    """Pins the process_message turn seam itself (not just the helper):
    the pre-turn location must be captured BEFORE process_action runs, and
    _rescope_scene_registry_after_move must be invoked after it. Reverting
    either half of the session.py wiring turns this red while every
    direct-call helper test above stays green."""

    async def test_process_message_rescopes_registry_after_a_move(
        self, registry, monkeypatch
    ):
        manager = GameSessionManager()
        session = GameSession(
            id="rescope-wiring",
            channel_id=900902,
            guild_id=1,
            campaign_id="camp",
        )
        session.world_state = WorldState(current_location="Cellar")
        session.state = SessionState.ACTIVE
        session.players[1] = PlayerInfo(
            user_id=1, user_name="eric",
            # Just enough Character surface for the store's begin_turn
            # party snapshot; everything else in the turn is faked out.
            character=SimpleNamespace(
                name="Tav",
                conditions=[],
                hp=SimpleNamespace(current=10, maximum=10),
                concentration_spell_id=None,
            ),
        )
        manager._sessions[f"discord:{session.channel_id}"] = session

        _entity(registry, "Barkeep")  # old-room entity, no twin in Tavern

        loads: list[str] = []

        async def _fake_load(location: str) -> int:
            loads.append(location)
            return 0

        monkeypatch.setattr(registry, "load_npcs_at_location", _fake_load)

        fake_orch = _MovingOrchestrator("Tavern")
        monkeypatch.setattr(
            "dnd_bot.game.session.get_orchestrator", lambda: fake_orch
        )

        async def _fake_memory(campaign_id: str) -> _FakeMemory:
            return _FakeMemory()

        monkeypatch.setattr(
            "dnd_bot.game.session.get_memory_manager", _fake_memory
        )
        monkeypatch.setattr(
            "dnd_bot.game.session.get_scene_registry",
            lambda campaign_id, session_key: registry,
        )
        async def _fake_build_context(*a, **k):
            return None

        monkeypatch.setattr(manager, "_build_context", _fake_build_context)

        async def _no_persist(sess) -> None:
            return None

        monkeypatch.setattr(manager, "_persist_world_snapshot", _no_persist)

        response = await manager.process_message(
            channel_id=session.channel_id, user_id=1,
            user_name="eric", content="I walk to the tavern",
        )

        assert response is not None
        # If process_message captured the location AFTER process_action (or
        # never called the seam), previous == current and nothing rescopes.
        assert registry.get_by_name("Barkeep") is None
        # The DB roster for the NEW location is loaded back in.
        assert loads == ["Tavern"]
