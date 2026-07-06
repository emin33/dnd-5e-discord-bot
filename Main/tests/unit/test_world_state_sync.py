"""REFACTOR_PLAN Step-4 net: pin the effect→WorldState sync chain.

Pinned FIRST against ``DMOrchestrator._sync_effect_to_world_state``
(commit 3b943f1), then flipped to the ``WorldStateStore.apply_effect``
seam when the chain moved verbatim into the single-writer store — the
only pin edits were the fixture and the call target; every branch diff
assertion is unchanged.

The chain's 11 branches had NO direct coverage before this net — Step 1
pinned only that every ``world_sync=True`` tool HAS a branch (source
inspection), not what each branch writes. The chain is synchronous and
pure over (world_state, effect) — no LLM, no DB, no registry.
"""

import pytest

from dnd_bot.game.session import GameSession
from dnd_bot.game.world_state import NPCState, WorldState
from dnd_bot.game.world_store import WorldStateStore
from dnd_bot.llm.effects import EffectType, ProposedEffect


@pytest.fixture
def world() -> WorldState:
    ws = WorldState(current_location="Tavern")
    ws.turn = 7
    return ws


@pytest.fixture
def store(world) -> WorldStateStore:
    return WorldStateStore(world)


def _npc(world: WorldState, name: str = "Merchant", **fields) -> NPCState:
    npc = NPCState(name=name, location="Tavern", **fields)
    world.npcs[npc.id] = npc
    return npc


class TestSceneAndTransferBranches:
    def test_spawn_object_records_item_and_transfer(self, store, world):
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.SPAWN_OBJECT,
            object_name="Jeweled Dagger",
            object_description="a dagger with a ruby pommel",
        ))
        assert world.scene_items == {"Jeweled Dagger": "a dagger with a ruby pommel"}
        assert world.recent_transfers == [
            "a dagger with a ruby pommel appeared in the scene"
        ]

    def test_spawn_object_falls_back_to_name(self, store, world):
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.SPAWN_OBJECT, object_name="Old Rope",
        ))
        assert world.scene_items == {"Old Rope": "Old Rope"}
        assert world.recent_transfers == ["Old Rope appeared in the scene"]

    def test_transfer_item_from_scene_removes_scene_item(self, store, world):
        world.spawn_item("Old Rope", "a coil of rope")
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.TRANSFER_ITEM,
            item_name="Old Rope", object_name="Old Rope",
            from_entity="scene", to_entity="player:Elara",
        ))
        assert world.scene_items == {}
        assert world.recent_transfers[-1] == (
            "Old Rope moved from scene to player:Elara"
        )

    def test_transfer_item_between_entities_keeps_scene(self, store, world):
        world.spawn_item("Old Rope", "a coil of rope")
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.TRANSFER_ITEM,
            item_name="Old Rope",
            from_entity="npc:merchant", to_entity="player:Elara",
        ))
        assert world.scene_items == {"Old Rope": "a coil of rope"}
        assert world.recent_transfers[-1] == (
            "Old Rope moved from npc:merchant to player:Elara"
        )

    def test_grant_currency_transfer_line(self, store, world):
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.GRANT_CURRENCY,
            gold=5, silver=3, source="Farmer", target="Elara",
        ))
        assert world.recent_transfers == ["Farmer gave 5gp, 3sp to Elara"]

    def test_consume_resource_transfer_line(self, store, world):
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.CONSUME_RESOURCE,
            resource_name="Arrow", quantity=2,
        ))
        assert world.recent_transfers == ["Consumed 2x Arrow"]

    def test_remove_entity_matches_slug_dialect(self, store, world):
        # The roster hands the narrator '[id: rusty-key]'-style slugs while
        # scene_items keys are display names — the branch compares slugified.
        world.spawn_item("Rusty Key", "an old iron key")
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.REMOVE_ENTITY, target="rusty-key",
        ))
        assert world.scene_items == {}

    def test_set_flag_writes_global_flags(self, store, world):
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.SET_FLAG,
            flag_name="bridge_destroyed", flag_value=True,
        ))
        assert world.global_flags == {"bridge_destroyed": True}


class TestLocationBranch:
    def test_change_location_moves_and_links_previous(self, store, world):
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.CHANGE_LOCATION,
            location_name="Cellar", location_description="A damp stone cellar",
        ))
        assert world.current_location == "Cellar"
        assert world.location_description == "A damp stone cellar"
        # The previous location becomes a connected exit (it's reachable).
        assert world.connected_locations == ["Tavern"]
        assert world.recent_transfers == ["party arrived at Cellar"]

    def test_change_location_to_same_place_still_logs(self, store, world):
        # Current behavior: a same-name change skips the connected-exit
        # append but still rewrites and logs the arrival.
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.CHANGE_LOCATION, location_name="Tavern",
        ))
        assert world.current_location == "Tavern"
        assert world.connected_locations == []
        assert world.recent_transfers == ["party arrived at Tavern"]


class TestNpcBranches:
    def test_add_npc_mints_npc_state_at_current_location(self, store, world):
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.ADD_NPC,
            npc_name="Grimjaw", npc_disposition="hostile",
            npc_description="A scarred dwarf",
        ))
        assert len(world.npcs) == 1
        npc = next(iter(world.npcs.values()))
        assert npc.name == "Grimjaw"
        assert npc.location == "Tavern"
        assert npc.disposition == "hostile"
        assert npc.description == "A scarred dwarf"
        assert npc.last_seen_turn == 7

    def test_add_npc_skips_existing_name(self, store, world):
        existing = _npc(world, "Grimjaw")
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.ADD_NPC, npc_name="Grimjaw",
        ))
        assert list(world.npcs) == [existing.id]  # no duplicate minted

    def test_ref_entity_bumps_recency_and_collects_alias(self, store, world):
        npc = _npc(world, "Merchant")
        npc.last_seen_turn = 2
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id=npc.id, ref_alias_used="the trader",
        ))
        assert npc.last_seen_turn == 7
        assert npc.aliases == ["the trader"]

    def test_update_entity_applies_every_field(self, store, world):
        npc = _npc(world, "Merchant", description="A trader", inventory=["ledger"])
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id=npc.id,
            update_disposition="Hostile",
            update_status="wounded",
            update_importance=True,
            update_description_addition="now scarred",
            update_add_items=["dagger"],
            update_remove_items=["Ledger"],  # case-insensitive removal
        ))
        assert npc.last_seen_turn == 7
        assert npc.disposition == "hostile"
        assert npc.alive is True
        assert npc.notes == "[wounded]"
        assert npc.important is True
        assert npc.description == "A trader now scarred"
        assert npc.inventory == ["dagger"]

    def test_update_entity_dead_status_flips_alive(self, store, world):
        npc = _npc(world, "Merchant")
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id=npc.id, update_status="dead",
        ))
        assert npc.alive is False
        assert npc.notes == ""  # dead is modeled, not noted


class TestUpdatePlayerBranch:
    def test_transfer_log_and_npc_inventory_mirroring(self, store, world):
        npc = _npc(world, "Innkeeper", inventory=["relic"])
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.UPDATE_PLAYER,
            player_item_grant=[{"name": "Relic", "source": f"npc:{npc.id}"}],
            player_item_remove=[
                {"name": "Coin Purse", "destination": f"npc:{npc.id}"}
            ],
            player_currency_delta={"gp": 5},
            player_hp_delta=-4,
            player_damage_type="fire",
            player_add_conditions=["poisoned"],
        ))
        # The grant sourced from the NPC strips it from their inventory;
        # the removal destined to the NPC lands in it ("I give the relic
        # to the innkeeper" sticks 20 turns later).
        assert npc.inventory == ["Coin Purse"]
        assert world.recent_transfers == [
            "player gained: Relic | player lost: Coin Purse | "
            "currency: {'gp': 5} | HP -4 (fire) | conditions+: ['poisoned']"
        ]

    def test_no_parts_records_nothing(self, store, world):
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.UPDATE_PLAYER,
        ))
        assert world.recent_transfers == []


class TestStoreDerivation:
    """The no-session/no-world guards moved to the call sites: the store is
    derived from the session (``GameSession.world_store``) and is None when
    there is no world state — callers skip the sync entirely."""

    def _session(self, channel_id: int) -> GameSession:
        return GameSession(
            id="ws-sync-session",
            channel_id=channel_id,
            guild_id=1,
            campaign_id="ws-sync-campaign",
        )

    def test_session_without_world_state_has_no_store(self, unique_channel_id):
        session = self._session(unique_channel_id)
        session.world_state = None
        assert session.world_store is None

    def test_session_store_wraps_the_live_world_state(
        self, world, unique_channel_id
    ):
        session = self._session(unique_channel_id)
        session.world_state = world
        store = session.world_store
        assert store is not None
        assert store.state is world
        store.apply_effect(ProposedEffect(
            effect_type=EffectType.SET_FLAG, flag_name="x", flag_value=True,
        ))
        assert world.global_flags == {"x": True}

    def test_apply_delta_delegates_to_the_state(self, store, world):
        from dnd_bot.game.world_state import StateDelta

        rejections = store.apply_delta(StateDelta(new_facts=["The bridge is out"]))
        assert rejections == []
        assert world.established_facts == ["The bridge is out"]
