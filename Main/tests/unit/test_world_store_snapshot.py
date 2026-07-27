"""WorldStateStore snapshot round-trip (ROOT-3, DF-5).

The store owns the persistence FORMAT (to_snapshot/state_from_snapshot);
the session layer owns when/where it is written. These tests prove a
fully-populated live world survives dict -> JSON -> dict -> WorldState
with nothing dropped — the exact loss DF-5 documents ("current_location,
scene_items, established_facts, recent_events, and the live NPC roster
... gone" after restart).
"""

import pytest
import json

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
    # One authored, one written in play — so the round trip is exercised on a
    # ledger where the two are actually distinguishable.
    ws.canon_facts = ["The crypt is sealed"]
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

        # Fact provenance survives the restart. If it did not, every RESUMED
        # campaign would silently fall back to ranking its facts by recency —
        # the defect this marking exists to fix, reintroduced by the restore
        # path and invisible until a long campaign lost its canon.
        assert restored.canon_facts == ["The crypt is sealed"]
        store.add_established_fact("A new fact", canon=True)
        assert restored.canon_facts == ["The crypt is sealed", "A new fact"]

    def test_default_world_round_trips(self):
        original = WorldState()
        restored = WorldStateStore.state_from_snapshot(
            WorldStateStore(original).to_snapshot()
        )
        assert restored.model_dump() == original.model_dump()

    def test_invalid_payload_raises(self):
        with pytest.raises(ValidationError):
            WorldStateStore.state_from_snapshot({"turn": "not-an-int"})


def test_a_forced_reseed_leaves_no_trace_of_the_campaign_before_it():
    """`force` documents itself as "a fresh campaign reusing a session shell".

    Half a reset is incoherent rather than merely incomplete: the freshly
    seeded tavern used to arrive already carrying the PREVIOUS campaign's
    contradictory fact about it, at that campaign's turn count, with its
    exits and flags still in place.
    """
    from dnd_bot.game.world_state import QuestState, WorldState
    from dnd_bot.game.world_store import WorldStateStore

    state = WorldState(current_location="Old Cellar")
    store = WorldStateStore(state)
    state.turn = 40
    store.add_established_fact("Copper Finch burned to ashes.")
    store.add_established_fact("The old book said so.", canon=True)
    state.recent_events.append("The roof fell in.")
    state.recent_transfers.append("player gave: torch")
    state.active_effects.append("blessed")
    state.connected_locations.append("Old Sewer")
    state.quests["q"] = QuestState(id="q", name="Old Quest")
    state.global_flags["cellar_flooded"] = True
    state.phase = "combat"
    state.time_of_day = "midnight"
    state.location_description = "A flooded cellar reeking of mould."
    state.superseded_facts.append(
        {"fact": "The Ash Gate was closed from the inside.", "reason": "overturned"}
    )

    assert store.seed_opening_scene(
        location="Copper Finch", description="A rain-dark tavern.", force=True,
    )

    assert state.current_location == "Copper Finch"
    assert state.turn == 0
    assert state.established_facts == []
    assert state.canon_facts == []
    assert state.recent_events == []
    assert state.recent_transfers == []
    assert state.active_effects == []
    assert state.connected_locations == []
    assert state.quests == {}
    assert state.global_flags == {}
    assert state.phase == "exploration"
    assert state.time_of_day == "morning"
    assert state.location_description == "A rain-dark tavern."
    # The retirement list gates add_established_fact, so a sentence the
    # previous campaign overturned would silently refuse to install from the
    # new book -- dropped with no error and no trace.
    assert state.superseded_facts == []
    store.add_established_fact("The Ash Gate was closed from the inside.")
    assert state.established_facts == ["The Ash Gate was closed from the inside."]


def test_an_unforced_seed_does_not_reset_anything():
    """The reset is the FORCED path's job.

    An unforced seed runs only on a campaign that has not started, and the
    book's own facts are added after it returns — so it must not be reaching
    into campaign state at all.
    """
    from dnd_bot.game.world_state import WorldState
    from dnd_bot.game.world_store import WorldStateStore

    state = WorldState(current_location="")
    store = WorldStateStore(state)
    store.add_established_fact("Seeded before the scene was set.")
    state.global_flags["prologue_done"] = True

    assert store.seed_opening_scene(location="Copper Finch")

    assert state.established_facts == ["Seeded before the scene was set."]
    assert state.global_flags == {"prologue_done": True}


def test_a_forced_reseed_with_no_description_does_not_keep_the_old_scenery():
    """An EMPTY description still replaces the old one when forcing.

    Guarding on `if description:` left the new room wearing the previous
    campaign's scenery -- a book that describes its opening location only
    through its NPCs would inherit a flooded cellar.
    """
    from dnd_bot.game.world_state import WorldState
    from dnd_bot.game.world_store import WorldStateStore

    state = WorldState(current_location="Old Cellar")
    state.location_description = "A flooded cellar reeking of mould."
    store = WorldStateStore(state)
    state.turn = 12

    assert store.seed_opening_scene(location="Copper Finch", force=True)

    assert state.current_location == "Copper Finch"
    assert state.location_description == ""


@pytest.mark.asyncio
async def test_a_snapshot_reports_whether_it_actually_landed():
    """The bool the sourcebook install spends its retry marker on.

    `_persist_world_snapshot` logs and swallows -- correct per turn, where
    the next turn retries -- but a seeded install has no next turn, and
    "logged the failure and moved on" must not read as "durable".
    """
    from unittest.mock import patch
    from dnd_bot.game.session import GameSession, GameSessionManager
    from dnd_bot.game.world_state import WorldState

    manager = GameSessionManager.__new__(GameSessionManager)
    session = GameSession(id="s", channel_id=1, guild_id=1, campaign_id="c")
    session.world_state = WorldState(current_location="Copper Finch")

    class _OkRepo:
        async def save_world_snapshot(self, *_a, **_k): return None

    class _BrokenRepo:
        async def save_world_snapshot(self, *_a, **_k):
            raise RuntimeError("disk full")

    with patch("dnd_bot.game.session.get_session_repo", return_value=_OkRepo()):
        assert await manager._persist_world_snapshot(session) is True
    with patch("dnd_bot.game.session.get_session_repo", return_value=_BrokenRepo()):
        assert await manager._persist_world_snapshot(session) is False
