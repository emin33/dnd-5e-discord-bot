"""WorldStateStore snapshot round-trip (ROOT-3, DF-5).

The store owns the persistence FORMAT (to_snapshot/state_from_snapshot);
the session layer owns when/where it is written. These tests prove a
fully-populated live world survives dict -> JSON -> dict -> WorldState
with nothing dropped — the exact loss DF-5 documents ("current_location,
scene_items, established_facts, recent_events, and the live NPC roster
... gone" after restart).
"""

import json

import pytest
from pydantic import ValidationError

from dnd_bot.game.world_state import NPCState, QuestState, WorldState
from dnd_bot.game.world_store import WorldStateStore


def _populated_world() -> WorldState:
    """One WorldState with every serialized field family non-default."""
    ws = WorldState(
        turn=17,
        phase="dialogue",
        time_of_day="dusk",
        current_location="The Gilded Flagon",
        location_description="A smoky tavern with a low ceiling",
        connected_locations=["Market Square", "Back Alley"],
    )
    fred = NPCState(
        name="Fred",
        location="The Gilded Flagon",
        disposition="friendly",
        description="the fat innkeeper with a crimson scar",
        notes="[wounded]",
        important=True,
        inventory=["brass key", "ledger"],
        aliases=["the innkeeper", "the fat man"],
        last_seen_turn=16,
    )
    ghost = NPCState(name="Whispering Shade", alive=False)
    ws.npcs = {fred.id: fred, ghost.id: ghost}
    ws.quests = {
        "Find the relic": QuestState(
            name="Find the relic",
            giver="Fred",
            status="active",
            objectives=["Search the crypt", "Return to Fred"],
            location="Old Crypt",
        )
    }
    ws.sync_player("Test Hero", hp=31, max_hp=44, conditions=["poisoned"], concentration="bless")
    ws.scene_items = {"jeweled-dagger": "a dagger with a ruby pommel"}
    ws.recent_transfers = ["Fred gave 15gp to player"]
    ws.active_effects = ["bless (3 rounds)"]
    ws.recent_events = ["The party arrived at the tavern", "Fred mentioned the crypt"]
    ws.established_facts = ["The mayor is missing", "The crypt is sealed"]
    ws.global_flags = {"crypt_unsealed": False, "met_fred": True}
    return ws


class TestSnapshotRoundTrip:
    def test_every_field_family_survives(self):
        original = _populated_world()

        # The exact pipeline the session layer runs: dict -> JSON string
        # (session_snapshot.game_state) -> dict -> WorldState.
        payload = json.dumps(WorldStateStore(original).to_snapshot())
        restored = WorldStateStore.state_from_snapshot(json.loads(payload))

        assert restored.model_dump() == original.model_dump()

    def test_restored_state_is_functional(self):
        original = _populated_world()
        restored = WorldStateStore.state_from_snapshot(
            WorldStateStore(original).to_snapshot()
        )

        # NPC ids stayed canonical (the cross-layer identity anchor) and
        # alias resolution still works on the restored object.
        fred = restored._find_npc("the fat man")
        assert fred is not None
        assert fred.id in original.npcs
        assert restored.npcs[fred.id] is fred

        # The narrator-facing YAML renders from the restored state.
        yaml_out = restored.to_yaml()
        assert "The Gilded Flagon" in yaml_out
        assert "Fred" in yaml_out

        # And the store's write seams accept it.
        store = WorldStateStore(restored)
        store.add_established_fact("A new fact")
        assert "A new fact" in restored.established_facts

    def test_default_world_round_trips(self):
        original = WorldState()
        restored = WorldStateStore.state_from_snapshot(
            WorldStateStore(original).to_snapshot()
        )
        assert restored.model_dump() == original.model_dump()

    def test_invalid_payload_raises(self):
        with pytest.raises(ValidationError):
            WorldStateStore.state_from_snapshot({"turn": "not-an-int"})
