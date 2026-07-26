"""Installing a book into a campaign's live stores.

compile_sourcebook is pure; apply_sourcebook is the leg that actually
populates a campaign — graph first, then the opening scene through the
store, then that location's residents (which reads the residency edges the
graph leg just wrote, so the order is load-bearing).

Run against a real KnowledgeGraph and a real WorldStateStore. The seeding
path is exactly where a mistake is expensive: it writes the world the party
wakes up in, and a book applied onto a session already in play would
relocate them mid-scene.
"""

from __future__ import annotations

import json

import pytest
import yaml

from dnd_bot.game.knowledge.graph import KnowledgeGraph
from dnd_bot.game.knowledge.sourcebook_compiler import (
    apply_sourcebook, load_sourcebook,
)
from dnd_bot.game.world_state import NPCState, WorldState
from dnd_bot.game.world_store import WorldStateStore

from tests.integration.test_sourcebook_to_live_graph import _book
from tests.unit.test_scene_hydration import _MemoryRepo


async def _fresh():
    kg = KnowledgeGraph(campaign_id="camp", repository=_MemoryRepo())
    await kg.load()
    store = WorldStateStore(WorldState())
    return kg, store


@pytest.mark.asyncio
async def test_applying_a_book_populates_the_world_the_party_wakes_in():
    kg, store = await _fresh()

    compiled = await apply_sourcebook(
        _book(), campaign_id="camp", knowledge_graph=kg, world_store=store,
    )
    ws = store.state

    assert ws.current_location == "Copper Finch"
    assert ws.location_description.startswith("A rain-dark tavern")
    # Residents are on stage without anyone having narrated them into being.
    assert {n.name for n in ws.npcs.values()} == {"Mara Venn", "Toran Vex"}
    assert "everyone at the Copper Finch defers to" in " ".join(
        ws.established_facts
    )
    assert not compiled.warnings


@pytest.mark.asyncio
async def test_the_dead_are_not_placed_on_stage_by_applying_a_book():
    kg, store = await _fresh()

    await apply_sourcebook(
        _book(), campaign_id="camp", knowledge_graph=kg, world_store=store,
    )

    assert not any(n.name == "Old Bram" for n in store.state.npcs.values())


@pytest.mark.asyncio
async def test_secrets_stay_out_of_the_seeded_world():
    kg, store = await _fresh()

    compiled = await apply_sourcebook(
        _book(), campaign_id="camp", knowledge_graph=kg, world_store=store,
    )

    assert not any(
        "filed the lock" in fact for fact in store.state.established_facts
    )
    assert "filed the lock" not in store.state.to_yaml()
    assert any("filed the lock" in c.text for c in compiled.withheld)


@pytest.mark.asyncio
async def test_a_campaign_in_progress_is_never_overwritten():
    """Seeding onto a live session would relocate the party mid-scene."""
    kg, store = await _fresh()
    store.state.current_location = "Somewhere Else"
    store.state.turn = 12
    store.state.npcs["existing"] = NPCState(
        id="existing", name="Someone", location="Somewhere Else",
    )

    await apply_sourcebook(
        _book(), campaign_id="camp", knowledge_graph=kg, world_store=store,
    )

    assert store.state.current_location == "Somewhere Else"
    assert "existing" in store.state.npcs
    assert store.state.established_facts == []


@pytest.mark.asyncio
async def test_force_allows_a_deliberate_reseed():
    kg, store = await _fresh()
    store.state.current_location = "Somewhere Else"
    store.state.turn = 12

    await apply_sourcebook(
        _book(), campaign_id="camp", knowledge_graph=kg,
        world_store=store, force=True,
    )

    assert store.state.current_location == "Copper Finch"


@pytest.mark.asyncio
async def test_graph_rejections_surface_as_warnings_not_silence():
    """A partially-installed book must be visible, not quietly incomplete."""
    kg, store = await _fresh()

    class _RejectingGraph:
        def __init__(self, inner):
            self._inner = inner

        async def apply_operations(self, ops):
            await self._inner.apply_operations(ops)
            return ["add_edge: Source node not found: ghost"]

        def resolve_location_node(self, name):
            return self._inner.resolve_location_node(name)

        def residents_of(self, node_id):
            return self._inner.residents_of(node_id)

    compiled = await apply_sourcebook(
        _book(), campaign_id="camp",
        knowledge_graph=_RejectingGraph(kg), world_store=store,
    )

    assert any("graph rejected" in w for w in compiled.warnings)
    # The rest of the install still happened.
    assert store.state.current_location == "Copper Finch"


# ── Loading from disk ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_book_round_trips_through_yaml_and_json(tmp_path):
    payload = _book().model_dump(mode="json")
    yaml_path = tmp_path / "book.yaml"
    json_path = tmp_path / "book.json"
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    from_yaml = load_sourcebook(yaml_path)
    from_json = load_sourcebook(json_path)

    assert from_yaml.metadata.sourcebook_id == "ash-gate"
    assert from_yaml == from_json
    assert {n.id for n in from_yaml.npcs} == {
        "mara-venn", "toran-vex", "old-bram"
    }


def test_a_malformed_book_fails_at_load_not_half_way_through_install(tmp_path):
    """Validation is the schema's job, and it happens before anything lands."""
    payload = _book().model_dump(mode="json")
    payload["npcs"][0]["current_location_id"] = "no-such-place"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Exception) as excinfo:
        load_sourcebook(path)

    assert "no-such-place" in str(excinfo.value)
