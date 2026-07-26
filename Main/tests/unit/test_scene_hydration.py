"""Pin: arriving somewhere restores its known inhabitants.

``rescope_scene`` only DROPS. Leaving a tavern evicted its non-important
regulars from the scene roster and returning never brought them back, so the
narrator arrived to an empty room — and either ignored the barkeep who had
served the party for twenty turns, or invented a fresh one that the
returning-NPC re-anchoring had to reconcile after the fact. Verified before
the fix: Tavern -> Market -> Tavern left the roster holding only the
``important`` NPC.

Hydration reads the graph's durable residency record, and is bounded by two
invariants that outrank scene continuity: the dead stay dead, and the roster
is prompt budget rather than a census.
"""

from __future__ import annotations

import pytest

from dnd_bot.game.knowledge.graph import KnowledgeGraph
from dnd_bot.game.knowledge.models import (
    AddEdge, AddNode, Entity, EntityType, RelationType, Relationship, slugify,
)
from dnd_bot.game.session import GameSession
from dnd_bot.game.world_store import WorldStateStore
from dnd_bot.game.world_state import NPCState, WorldState
from dnd_bot.llm.orchestrator import DMOrchestrator


class _MemoryRepo:
    """KnowledgeGraph repository stub — write-through goes nowhere."""

    async def load_nodes(self, campaign_id):
        return []

    async def load_edges(self, campaign_id):
        return []

    async def upsert_node(self, entity):
        return None

    async def upsert_edge(self, rel):
        return None

    async def delete_node(self, campaign_id, node_id):
        return None

    async def delete_edge(self, *a, **k):
        return None

    async def delete_edges_by_source(self, *a, **k):
        return None


async def _graph_with(residents, location="Copper Finch", campaign="camp"):
    kg = KnowledgeGraph(campaign_id=campaign, repository=_MemoryRepo())
    await kg.load()
    ops = [AddNode(entity=Entity(
        node_id=slugify(location), entity_type=EntityType.LOCATION,
        name=location, campaign_id=campaign,
    ))]
    for node_id, name, props in residents:
        ops.append(AddNode(entity=Entity(
            node_id=node_id, entity_type=EntityType.NPC, name=name,
            campaign_id=campaign, properties=props,
        )))
    rejections = await kg.apply_operations(ops)
    assert not rejections, rejections
    edges = [AddEdge(relationship=Relationship(
        source_id=node_id, target_id=slugify(location),
        relation_type=RelationType.LOCATED_AT, campaign_id=campaign,
    )) for node_id, _n, _p in residents]
    assert not await kg.apply_operations(edges)
    return kg


def _orch(world_state, dead: dict | None = None):
    session = GameSession(id="s", channel_id=880, guild_id=1, campaign_id="camp")
    session.world_state = world_state
    if dead:
        session.campaign_dead_npcs.update(dead)
    orch = DMOrchestrator()
    orch.set_session(session)
    return orch


@pytest.mark.asyncio
async def test_returning_restores_the_locations_regulars():
    kg = await _graph_with([
        ("barkeep-id", "Barkeep", {"description": "A broad, quiet man.",
                                   "disposition": "friendly"}),
    ])
    ws = WorldState(current_location="Copper Finch")

    restored = _orch(ws)._hydrate_scene_from_knowledge(ws, kg, "Market")

    assert restored == ["Barkeep"]
    npc = ws.npcs["barkeep-id"]
    assert npc.location == "Copper Finch"
    assert npc.disposition == "friendly"
    assert npc.description == "A broad, quiet man."
    assert npc.alive is True


@pytest.mark.asyncio
async def test_staying_put_hydrates_nothing():
    """Only arrivals hydrate — not every turn spent in the same room."""
    kg = await _graph_with([("barkeep-id", "Barkeep", {})])
    ws = WorldState(current_location="Copper Finch")

    restored = _orch(ws)._hydrate_scene_from_knowledge(ws, kg, "Copper Finch")

    assert restored == []
    assert ws.npcs == {}


@pytest.mark.asyncio
async def test_equivalent_location_spelling_is_not_an_arrival():
    kg = await _graph_with([("barkeep-id", "Barkeep", {})])
    ws = WorldState(current_location="Copper Finch")

    restored = _orch(ws)._hydrate_scene_from_knowledge(ws, kg, "the Copper Finch")

    assert restored == []


@pytest.mark.asyncio
async def test_a_graph_node_flagged_dead_is_never_restored():
    kg = await _graph_with([
        ("ferryman-id", "Dead Ferryman", {"alive": "false"}),
        ("barkeep-id", "Barkeep", {}),
    ])
    ws = WorldState(current_location="Copper Finch")

    restored = _orch(ws)._hydrate_scene_from_knowledge(ws, kg, "Market")

    assert restored == ["Barkeep"]


@pytest.mark.asyncio
async def test_a_stale_residency_edge_cannot_resurrect_the_buried():
    """The dead-NPC invariant outranks scene continuity.

    The campaign's dead roster is keyed by the id it buried, which need not
    match the graph node still carrying a LOCATED_AT edge — so the guard has
    to match on identity, not just id.
    """
    kg = await _graph_with([
        ("bram-graph-id", "Old Bram", {}),   # graph still says alive, here
        ("barkeep-id", "Barkeep", {}),
    ])
    ws = WorldState(current_location="Copper Finch")
    dead = {"buried-bram": NPCState(id="buried-bram", name="Old Bram", alive=False)}

    restored = _orch(ws, dead=dead)._hydrate_scene_from_knowledge(ws, kg, "Market")

    assert restored == ["Barkeep"]
    assert not any(n.name == "Old Bram" for n in ws.npcs.values())


@pytest.mark.asyncio
async def test_already_present_npcs_are_not_duplicated():
    kg = await _graph_with([("barkeep-id", "Barkeep", {})])
    ws = WorldState(current_location="Copper Finch")
    ws.npcs["some-other-id"] = NPCState(
        id="some-other-id", name="Barkeep", location="Copper Finch",
    )

    restored = _orch(ws)._hydrate_scene_from_knowledge(ws, kg, "Market")

    assert restored == []
    assert len(ws.npcs) == 1


@pytest.mark.asyncio
async def test_a_crowded_location_is_capped_to_the_roster_budget():
    kg = await _graph_with([
        (f"npc-{i:02d}", f"Regular {i:02d}", {}) for i in range(12)
    ])
    ws = WorldState(current_location="Copper Finch")

    restored = _orch(ws)._hydrate_scene_from_knowledge(ws, kg, "Market")

    assert len(restored) == WorldStateStore.MAX_HYDRATED_RESIDENTS
    assert len(ws.npcs) == WorldStateStore.MAX_HYDRATED_RESIDENTS


@pytest.mark.asyncio
async def test_non_npc_residents_are_ignored():
    """Items recorded at a location are not scene NPCs."""
    kg = KnowledgeGraph(campaign_id="camp", repository=_MemoryRepo())
    await kg.load()
    assert not await kg.apply_operations([
        AddNode(entity=Entity(
            node_id="copper-finch", entity_type=EntityType.LOCATION,
            name="Copper Finch", campaign_id="camp",
        )),
        AddNode(entity=Entity(
            node_id="brass-compass", entity_type=EntityType.ITEM,
            name="brass compass", campaign_id="camp",
        )),
    ])
    assert not await kg.apply_operations([
        AddEdge(relationship=Relationship(
            source_id="brass-compass", target_id="copper-finch",
            relation_type=RelationType.LOCATED_AT, campaign_id="camp",
        )),
    ])
    ws = WorldState(current_location="Copper Finch")

    restored = _orch(ws)._hydrate_scene_from_knowledge(ws, kg, "Market")

    assert restored == []


@pytest.mark.asyncio
async def test_an_npc_who_fled_is_not_put_back_on_stage():
    """update_entity(status='fled') is the sanctioned off-stage channel and
    deliberately leaves the residency edge intact (remove_entity is forbidden
    for it), so residency alone cannot mean "still here"."""
    kg = await _graph_with([
        ("bandit-id", "Sable Quill", {"status": "fled", "disposition": "hostile"}),
        ("captive-id", "Tam Rook", {"status": "captured"}),
        ("barkeep-id", "Barkeep", {}),
    ])
    ws = WorldState(current_location="Copper Finch")

    restored = _orch(ws)._hydrate_scene_from_knowledge(ws, kg, "Market")

    assert restored == ["Barkeep"]


@pytest.mark.asyncio
async def test_a_surviving_status_marker_is_carried_into_the_roster():
    """An NPC who is still present but marked (e.g. wounded) keeps the note
    rather than silently shedding state the story established."""
    kg = await _graph_with([("guard-id", "Watch Sergeant", {"status": "wounded"})])
    ws = WorldState(current_location="Copper Finch")

    _orch(ws)._hydrate_scene_from_knowledge(ws, kg, "Market")

    assert ws.npcs["guard-id"].notes == "[wounded]"


@pytest.mark.asyncio
async def test_the_dead_guard_fails_closed_on_an_ambiguous_name():
    """Two buried 'Cultist's make resolve_unique_identity abstain; reading
    that abstention as "not dead" would resurrect one."""
    kg = await _graph_with([
        ("cultist-graph-id", "Cultist", {}),
        ("barkeep-id", "Barkeep", {}),
    ])
    ws = WorldState(current_location="Copper Finch")
    dead = {
        "cultist-a": NPCState(id="cultist-a", name="Cultist", alive=False),
        "cultist-b": NPCState(id="cultist-b", name="Cultist", alive=False),
    }

    restored = _orch(ws, dead=dead)._hydrate_scene_from_knowledge(ws, kg, "Market")

    assert restored == ["Barkeep"]
    assert not any(n.name == "Cultist" for n in ws.npcs.values())


@pytest.mark.asyncio
async def test_a_buried_npc_is_blocked_by_id_even_under_a_new_name():
    kg = await _graph_with([("bram-id", "The Ash Ferryman", {})])
    ws = WorldState(current_location="Copper Finch")
    dead = {"bram-id": NPCState(id="bram-id", name="Old Bram", alive=False)}

    assert _orch(ws, dead=dead)._hydrate_scene_from_knowledge(ws, kg, "Market") == []


@pytest.mark.asyncio
async def test_a_buried_npcs_alias_also_blocks_hydration():
    kg = await _graph_with([("ferryman-id", "The Ash Ferryman", {})])
    ws = WorldState(current_location="Copper Finch")
    dead = {
        "buried": NPCState(
            id="buried", name="Old Bram",
            aliases=["the ash ferryman"], alive=False,
        )
    }

    assert _orch(ws, dead=dead)._hydrate_scene_from_knowledge(ws, kg, "Market") == []


@pytest.mark.asyncio
async def test_a_location_spelling_variant_still_finds_its_residents():
    """The graph knows "Copper Finch"; the party arrived at "The Copper Finch"."""
    kg = await _graph_with([("barkeep-id", "Barkeep", {})], location="Copper Finch")
    ws = WorldState(current_location="The Copper Finch")

    restored = _orch(ws)._hydrate_scene_from_knowledge(ws, kg, "Market")

    assert restored == ["Barkeep"]


@pytest.mark.asyncio
async def test_a_paraphrased_destination_resolves_through_an_alias():
    """Narrators rename rooms. A live lore run walked the party back to "the
    tavern" and the Copper Finch came back empty, because no spelling-variant
    rule can bridge a descriptive paraphrase to a proper name."""
    kg = KnowledgeGraph(campaign_id="camp", repository=_MemoryRepo())
    await kg.load()
    assert not await kg.apply_operations([
        AddNode(entity=Entity(
            node_id="copper-finch", entity_type=EntityType.LOCATION,
            name="Copper Finch", aliases=["the tavern"], campaign_id="camp",
        )),
        AddNode(entity=Entity(
            node_id="barkeep-id", entity_type=EntityType.NPC,
            name="Barkeep", campaign_id="camp",
        )),
    ])
    assert not await kg.apply_operations([
        AddEdge(relationship=Relationship(
            source_id="barkeep-id", target_id="copper-finch",
            relation_type=RelationType.LOCATED_AT, campaign_id="camp",
        )),
    ])
    ws = WorldState(current_location="the tavern")

    restored = _orch(ws)._hydrate_scene_from_knowledge(ws, kg, "Ash Gate")

    assert restored == ["Barkeep"]
    # And the scene is re-anchored to canon's name, so every later comparison
    # (rescope, residency, fact relevance) lines up instead of missing.
    assert ws.current_location == "Copper Finch"
    assert ws.npcs["barkeep-id"].location == "Copper Finch"


@pytest.mark.asyncio
async def test_unknown_location_yields_nothing():
    kg = await _graph_with([("barkeep-id", "Barkeep", {})])
    ws = WorldState(current_location="Somewhere Unmapped")

    assert _orch(ws)._hydrate_scene_from_knowledge(ws, kg, "Market") == []
