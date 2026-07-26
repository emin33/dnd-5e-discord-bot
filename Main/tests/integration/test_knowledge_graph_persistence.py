"""Cross-store graph invariants against a real temporary SQLite database."""

from pathlib import Path

import pytest

from dnd_bot.data.database import Database
from dnd_bot.game.knowledge.graph import KnowledgeGraph
from dnd_bot.game.knowledge.models import (
    AddEdge,
    AddNode,
    Entity,
    EntityType,
    RelationType,
    Relationship,
    RemoveEdge,
)
from dnd_bot.game.knowledge.repository import KnowledgeGraphRepository


@pytest.fixture
async def graph_db(tmp_path: Path):
    db = Database(db_path=tmp_path / "graph.db")
    await db.connect()
    await db.execute(
        "INSERT INTO campaign (id, guild_id, name, dm_user_id) VALUES (?, ?, ?, ?)",
        ("campaign", 1, "Graph Test", 1),
    )
    await db.commit()
    yield db
    await db.disconnect()


def _node(
    node_id: str, entity_type: EntityType, campaign: str = "campaign"
) -> AddNode:
    return AddNode(entity=Entity(
        node_id=node_id,
        entity_type=entity_type,
        name=node_id,
        campaign_id=campaign,
    ))


def _located_at(target_id: str) -> AddEdge:
    return AddEdge(relationship=Relationship(
        source_id="npc",
        target_id=target_id,
        relation_type=RelationType.LOCATED_AT,
        campaign_id="campaign",
    ))


def _tie(
    source: str, target: str, relation: RelationType, campaign: str = "campaign"
) -> AddEdge:
    return AddEdge(relationship=Relationship(
        source_id=source,
        target_id=target,
        relation_type=relation,
        campaign_id=campaign,
    ))


def _outgoing(graph: KnowledgeGraph, source: str, target: str) -> set[str]:
    """Relation types the NARRATOR is shown for ``source`` → ``target``.

    Asserted through ``get_context_subgraph`` rather than the NetworkX
    object underneath, because the prompt is where a dropped relation does
    its damage — a pin on the internals could stay green while the context
    the narrator actually reads had lost an edge.
    """
    entry = next(
        e for e in graph.get_context_subgraph([source, target])
        if e["id"] == source
    )
    return {
        rel.removesuffix(f" {target}")
        for rel in entry.get("relationships", [])
        if rel.endswith(f" {target}")
    }


async def _pair_with_ties(
    repo: KnowledgeGraphRepository,
    campaign: str,
    relations: list[RelationType],
) -> KnowledgeGraph:
    """Two NPCs joined by ``relations``, in the order given."""
    graph = KnowledgeGraph(campaign, repo)
    await graph.load()
    rejections = await graph.apply_operations(
        [
            _node("mara", EntityType.NPC, campaign),
            _node("toran", EntityType.NPC, campaign),
        ]
        + [_tie("mara", "toran", relation, campaign) for relation in relations]
    )
    assert rejections == []
    return graph


@pytest.mark.asyncio
async def test_wildcard_edge_removal_matches_sqlite_after_reload(graph_db):
    repo = KnowledgeGraphRepository(graph_db)
    live = KnowledgeGraph("campaign", repo)
    await live.load()
    await live.apply_operations([
        _node("npc", EntityType.NPC),
        _node("old-location", EntityType.LOCATION),
        _node("other-old-location", EntityType.LOCATION),
        _located_at("old-location"),
        _located_at("other-old-location"),
    ])

    rejections = await live.apply_operations([RemoveEdge(
        source_id="npc",
        target_id="",
        relation_type=RelationType.LOCATED_AT,
    )])

    reloaded = KnowledgeGraph("campaign", repo)
    await reloaded.load()
    assert rejections == []
    assert live.node_count() == reloaded.node_count() == 3
    assert live.edge_count() == reloaded.edge_count() == 0


@pytest.mark.asyncio
async def test_distinct_relations_between_one_pair_both_survive_a_reload(graph_db):
    """Two relation types between the same pair are two edges, not a race.

    ``kg_edge``'s primary key is (campaign, source, target, relation_type),
    so SQLite always stored what the author wrote. The in-memory graph was a
    DiGraph, which holds at most ONE edge per (source, target) pair — so the
    second relation applied REPLACED the first, with zero rejections.
    ``apply_operations`` reported a clean run over a silent loss, and any
    receipt counting applied ops overstated what landed.

    The reload half is the one that matters most: unordered rows meant the
    winner was decided by row order, so the same campaign could present a
    different social graph after a restart with nothing in the logs.
    """
    repo = KnowledgeGraphRepository(graph_db)
    live = await _pair_with_ties(
        repo, "campaign", [RelationType.ALLIED_WITH, RelationType.HOSTILE_TO],
    )

    assert live.edge_count() == 2
    assert _outgoing(live, "mara", "toran") == {"allied_with", "hostile_to"}

    reloaded = KnowledgeGraph("campaign", repo)
    await reloaded.load()

    assert reloaded.edge_count() == 2
    assert _outgoing(reloaded, "mara", "toran") == {"allied_with", "hostile_to"}
    # Identical, not merely equinumerous — the whole narrator-visible
    # projection has to survive the round trip unchanged.
    assert (
        reloaded.get_context_subgraph(["mara", "toran"])
        == live.get_context_subgraph(["mara", "toran"])
    )


@pytest.mark.asyncio
async def test_removing_one_relation_leaves_its_sibling_standing(graph_db):
    """A targeted removal names a relation type; only that one goes.

    The persisted side always behaved this way (``DELETE ... AND
    relation_type = ?``). In memory the relation type was a post-hoc filter
    on the single edge the pair was allowed, so the two projections could
    disagree about what survived.
    """
    repo = KnowledgeGraphRepository(graph_db)
    live = await _pair_with_ties(
        repo, "campaign", [RelationType.ALLIED_WITH, RelationType.HOSTILE_TO],
    )

    rejections = await live.apply_operations([RemoveEdge(
        source_id="mara",
        target_id="toran",
        relation_type=RelationType.HOSTILE_TO,
    )])

    reloaded = KnowledgeGraph("campaign", repo)
    await reloaded.load()

    assert rejections == []
    assert live.edge_count() == reloaded.edge_count() == 1
    assert _outgoing(live, "mara", "toran") == {"allied_with"}
    assert _outgoing(reloaded, "mara", "toran") == {"allied_with"}


@pytest.mark.asyncio
async def test_apply_order_does_not_decide_the_social_graph(graph_db):
    """The same two ties, authored in either order, reload identically.

    This is the nondeterminism stated as an experiment rather than an
    assertion about scan order: build one campaign ALLIED-then-HOSTILE and
    another HOSTILE-then-ALLIED, reload both, compare. Under the DiGraph the
    last edge applied won, so the two campaigns disagreed about whether the
    pair were friends or enemies — from identical authored input.
    """
    await graph_db.execute(
        "INSERT INTO campaign (id, guild_id, name, dm_user_id) VALUES (?, ?, ?, ?)",
        ("mirror", 1, "Mirror", 1),
    )
    await graph_db.commit()
    repo = KnowledgeGraphRepository(graph_db)

    await _pair_with_ties(
        repo, "campaign", [RelationType.ALLIED_WITH, RelationType.HOSTILE_TO],
    )
    await _pair_with_ties(
        repo, "mirror", [RelationType.HOSTILE_TO, RelationType.ALLIED_WITH],
    )

    allied_first = KnowledgeGraph("campaign", repo)
    hostile_first = KnowledgeGraph("mirror", repo)
    await allied_first.load()
    await hostile_first.load()

    assert _outgoing(allied_first, "mara", "toran") == {"allied_with", "hostile_to"}
    assert (
        allied_first.get_context_subgraph(["mara", "toran"])
        == hostile_first.get_context_subgraph(["mara", "toran"])
    )
