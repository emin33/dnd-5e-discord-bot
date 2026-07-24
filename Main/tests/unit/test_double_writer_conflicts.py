"""Ordering/conflict tests for the turn's two state writers.

The pipeline deliberately commits the state extractor's delta first
(Step 3.6) and narrator tool effects last (Step 4), so tools win every
scalar field by construction. What that design leans on — and what these
tests pin — is the ACCUMULATOR fields (aliases, inventory, notes markers,
connected_locations), where "last writer wins" is meaningless and only
membership checks stand between a double-fire and duplicated state. Each
test replays a real both-writers-in-one-turn sequence in pipeline order:
``WorldState.apply_delta`` (extractor) then ``WorldStateStore.apply_effect``
(tool).
"""

import pytest

from dnd_bot.game.world_state import NPCState, NPCUpdate, StateDelta, WorldState
from dnd_bot.game.world_store import WorldStateStore
from dnd_bot.llm.effects import EffectType, ProposedEffect


def _store(world: WorldState) -> WorldStateStore:
    return WorldStateStore(world)


class TestAliasCaseFanOut:
    def test_ref_alias_case_variant_of_extractor_alias_not_duplicated(self):
        npc = NPCState(id="npc-1", name="Bram")
        world = WorldState(campaign_id="camp", npcs={"npc-1": npc})

        world.apply_delta(StateDelta(
            npc_updates=[NPCUpdate(id="npc-1", add_aliases=["Old Bram"])]
        ))
        _store(world).apply_effect(ProposedEffect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id="npc-1",
            ref_alias_used="old Bram",
        ))

        assert npc.aliases == ["Old Bram"]

    def test_extractor_alias_case_variant_of_name_not_added(self):
        npc = NPCState(id="npc-1", name="Old Bram")
        world = WorldState(campaign_id="camp", npcs={"npc-1": npc})

        world.apply_delta(StateDelta(
            npc_updates=[NPCUpdate(id="npc-1", add_aliases=["old bram"])]
        ))

        assert npc.aliases == []

    def test_case_correcting_rename_leaves_no_alias_residue(self):
        npc = NPCState(id="npc-1", name="old bram")
        world = WorldState(campaign_id="camp", npcs={"npc-1": npc})

        world.apply_delta(StateDelta(
            npc_updates=[NPCUpdate(id="npc-1", new_name="Old Bram")]
        ))

        # The rename applies (case correction is legitimate) but the old
        # spelling is not an identity worth keeping as an alias.
        assert npc.name == "Old Bram"
        assert npc.aliases == []


class TestInventoryCaseFanOut:
    def test_tool_add_case_variant_of_extractor_add_not_duplicated(self):
        npc = NPCState(id="npc-1", name="Innkeeper")
        world = WorldState(campaign_id="camp", npcs={"npc-1": npc})

        world.apply_delta(StateDelta(
            npc_updates=[NPCUpdate(id="npc-1", add_inventory=["brass key"])]
        ))
        _store(world).apply_effect(ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="npc-1",
            update_add_items=["Brass Key"],
        ))

        assert npc.inventory == ["brass key"]

    def test_player_gift_mirror_case_variant_not_duplicated(self):
        npc = NPCState(id="npc-1", name="Innkeeper", inventory=["brass key"])
        world = WorldState(campaign_id="camp", npcs={"npc-1": npc})

        _store(world).apply_effect(ProposedEffect(
            effect_type=EffectType.UPDATE_PLAYER,
            player_item_remove=[{"name": "Brass Key", "destination": "npc:npc-1"}],
        ))

        assert npc.inventory == ["brass key"]

    def test_extractor_add_case_variant_of_existing_not_duplicated(self):
        npc = NPCState(id="npc-1", name="Innkeeper", inventory=["Brass Key"])
        world = WorldState(campaign_id="camp", npcs={"npc-1": npc})

        world.apply_delta(StateDelta(
            npc_updates=[NPCUpdate(id="npc-1", add_inventory=["brass key"])]
        ))

        assert npc.inventory == ["Brass Key"]


class TestStatusMarkerPileUp:
    def test_restated_wounded_status_records_one_marker(self):
        npc = NPCState(id="npc-1", name="Guard")
        world = WorldState(campaign_id="camp", npcs={"npc-1": npc})
        wound = ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="npc-1",
            update_status="wounded",
        )

        for _ in range(3):
            _store(world).apply_effect(wound)

        assert npc.notes.count("[wounded]") == 1

    def test_distinct_statuses_both_recorded(self):
        npc = NPCState(id="npc-1", name="Guard")
        world = WorldState(campaign_id="camp", npcs={"npc-1": npc})

        for status in ("wounded", "captured"):
            _store(world).apply_effect(ProposedEffect(
                effect_type=EffectType.UPDATE_ENTITY,
                update_entity_id="npc-1",
                update_status=status,
            ))

        assert "[wounded]" in npc.notes
        assert "[captured]" in npc.notes


class TestLocationDoubleFire:
    def test_restated_move_records_origin_not_phantom_self_edge(self):
        # The soak shape: extractor applies location_change='the tavern',
        # then the narrator tool restates change_location('Tavern'). The
        # raw string compare used to append 'the tavern' — the place the
        # party is standing in — as a connection, and the true origin
        # ('Ash Gate') was lost because the tool read an already-moved
        # current_location.
        world = WorldState(campaign_id="camp", current_location="Ash Gate")

        world.apply_delta(StateDelta(location_change="the tavern"))
        _store(world).apply_effect(ProposedEffect(
            effect_type=EffectType.CHANGE_LOCATION,
            location_name="Tavern",
        ))

        assert world.current_location == "Tavern"
        assert "Ash Gate" in world.connected_locations
        assert not any(
            "tavern" in c.lower() for c in world.connected_locations
        )

    def test_extractor_only_move_records_origin_edge(self):
        world = WorldState(campaign_id="camp", current_location="Ash Gate")

        world.apply_delta(StateDelta(location_change="The Tavern"))

        assert world.connected_locations == ["Ash Gate"]

    def test_new_connection_naming_current_location_is_skipped(self):
        world = WorldState(campaign_id="camp", current_location="The Tavern")

        world.apply_delta(StateDelta(new_connections=["the tavern", "Cellar"]))

        assert world.connected_locations == ["Cellar"]

    def test_new_connection_spelling_variant_not_duplicated(self):
        world = WorldState(
            campaign_id="camp",
            current_location="Courtyard",
            connected_locations=["Ash Gate"],
        )

        world.apply_delta(StateDelta(new_connections=["the ash gate"]))

        assert world.connected_locations == ["Ash Gate"]


class TestDedupAliasNotGraftedBeforeValidation:
    @pytest.mark.asyncio
    async def test_deterministic_rewrite_defers_alias_to_apply(self):
        # dedup_effect rewrites add_npc('Old Bram') onto the roster's
        # 'Bram'. The alias must NOT be on the NPC yet — validation can
        # still reject the rewritten ref (e.g. the alias is another NPC's
        # canonical name), and a rejected effect must leave no residue.
        npc = NPCState(id="npc-1", name="Bram")
        world = WorldState(campaign_id="camp", npcs={"npc-1": npc})

        rewritten = await _store(world).dedup_effect(ProposedEffect(
            effect_type=EffectType.ADD_NPC,
            npc_name="Old Bram",
        ))

        assert rewritten.effect_type == EffectType.REF_ENTITY
        assert rewritten.ref_alias_used == "Old Bram"
        assert npc.aliases == []

        # The accepted path still accumulates it — through the one
        # post-validation writer.
        _store(world).apply_effect(rewritten)
        assert npc.aliases == ["Old Bram"]
