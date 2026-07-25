"""Pin the sourcebook -> live-stores projection.

``models/sourcebook.py`` has been the authoring contract with no consumer but
its own schema test; this covers the compiler that finally reads it.

The assertions that matter most are about the VISIBILITY BOUNDARY. A campaign
book is mostly secrets, and a leak is invisible to every quality gate this
project has: self-consistency grading cannot flag unearned canon, because
leaked canon is perfectly consistent. So DM_ONLY and DISCOVERABLE claims must
never reach narrator-visible state, and the compiler keeps them in a separate
bucket that exists precisely so tests can assert their ABSENCE.
"""

from __future__ import annotations

import pytest

from dnd_bot.game.knowledge.models import AddEdge, AddNode, EntityType, RelationType
from dnd_bot.game.knowledge.sourcebook_compiler import compile_sourcebook
from dnd_bot.models.sourcebook import (
    CampaignSourcebook,
    CharacterStatus,
    KnowledgeClaim,
    LocationKind,
    LocationSpec,
    NPCSpec,
    ItemSpec,
    QuestObjective,
    QuestSpec,
    RelationshipKind,
    RelationshipSpec,
    RouteSpec,
    SourcebookMetadata,
    StartingState,
    Visibility,
)


def _book(**overrides) -> CampaignSourcebook:
    base = dict(
        metadata=SourcebookMetadata(
            sourcebook_id="ash-gate", title="The Ash Gate", pitch="A gate.",
        ),
        locations=[
            LocationSpec(
                id="copper-finch", name="Copper Finch",
                location_kind=LocationKind.BUILDING,
                description="A rain-dark tavern of copper lamps.",
            ),
            LocationSpec(
                id="ash-gate-arch", name="Ash Gate",
                location_kind=LocationKind.SITE,
                description="A cracked black arch.",
            ),
        ],
        routes=[RouteSpec(
            id="finch-to-gate", from_location_id="copper-finch",
            to_location_id="ash-gate-arch",
        )],
        npcs=[
            NPCSpec(
                id="mara-venn", name="Mara Venn", role="investigator",
                appearance="A sharp-eyed woman in a charcoal coat.",
                current_location_id="copper-finch",
            ),
            NPCSpec(
                id="old-bram", name="Old Bram", status=CharacterStatus.DEAD,
                summary="The ferryman who warned people away.",
                current_location_id="ash-gate-arch",
            ),
        ],
        items=[ItemSpec(
            id="sealed-reliquary", name="sealed reliquary",
            description="An iron box with a bronze pin.",
            default_location_id="copper-finch",
        )],
        relationships=[RelationshipSpec(
            id="mara-knew-bram", source_id="mara-venn", target_id="old-bram",
            kind=RelationshipKind.KNOWS,
        )],
        claims=[
            KnowledgeClaim(
                id="claim-gate-closed", subject_id="ash-gate-arch",
                text="The Ash Gate has been closed since Old Bram died.",
                visibility=Visibility.PUBLIC,
            ),
            KnowledgeClaim(
                id="claim-mara-lies", subject_id="mara-venn",
                text="Mara Venn filed the lock herself.",
                visibility=Visibility.DM_ONLY,
            ),
            KnowledgeClaim(
                id="claim-key-buried", subject_id="sealed-reliquary",
                text="The obsidian key is buried under the arch.",
                visibility=Visibility.DISCOVERABLE,
            ),
        ],
        starting_state=StartingState(
            location_id="copper-finch",
            opening_situation="Rain on the shutters; Mara is waiting.",
        ),
    )
    base.update(overrides)
    return CampaignSourcebook(**base)


def _nodes(compiled):
    return {op.entity.node_id: op.entity
            for op in compiled.graph_ops if isinstance(op, AddNode)}


def _edges(compiled):
    return {
        (op.relationship.source_id, op.relationship.target_id,
         op.relationship.relation_type)
        for op in compiled.graph_ops if isinstance(op, AddEdge)
    }


def test_entities_project_with_their_types_and_descriptions():
    compiled = compile_sourcebook(_book(), "camp")
    nodes = _nodes(compiled)

    assert nodes["copper-finch"].entity_type is EntityType.LOCATION
    assert nodes["mara-venn"].entity_type is EntityType.NPC
    assert nodes["sealed-reliquary"].entity_type is EntityType.ITEM
    assert nodes["mara-venn"].properties["description"].startswith("A sharp-eyed")
    assert not compiled.warnings


def test_authored_death_projects_as_not_alive():
    """The liveness signal hydration and continuity governance both read."""
    nodes = _nodes(compile_sourcebook(_book(), "camp"))

    assert nodes["old-bram"].properties["alive"] == "false"
    assert nodes["old-bram"].properties["status"] == "dead"
    assert nodes["mara-venn"].properties["alive"] == "true"


def test_residency_and_routes_become_traversable_edges():
    edges = _edges(compile_sourcebook(_book(), "camp"))

    assert ("mara-venn", "copper-finch", RelationType.LOCATED_AT) in edges
    assert ("sealed-reliquary", "copper-finch", RelationType.FOUND_AT) in edges
    # Bidirectional by default: both directions are walkable.
    assert ("copper-finch", "ash-gate-arch", RelationType.CONNECTED_TO) in edges
    assert ("ash-gate-arch", "copper-finch", RelationType.CONNECTED_TO) in edges


def test_one_way_routes_stay_one_way():
    book = _book(routes=[RouteSpec(
        id="drop", from_location_id="copper-finch",
        to_location_id="ash-gate-arch", bidirectional=False,
    )])
    edges = _edges(compile_sourcebook(book, "camp"))

    assert ("copper-finch", "ash-gate-arch", RelationType.CONNECTED_TO) in edges
    assert ("ash-gate-arch", "copper-finch", RelationType.CONNECTED_TO) not in edges


def test_richer_relationship_kinds_collapse_onto_retrieval_edges():
    """The book is the system of record; the graph only needs nearness."""
    book = _book(relationships=[
        RelationshipSpec(id="rel-one", source_id="mara-venn", target_id="old-bram",
                         kind=RelationshipKind.PARENT_OF),
        RelationshipSpec(id="rel-two", source_id="mara-venn", target_id="old-bram",
                         kind=RelationshipKind.RIVAL_OF),
        RelationshipSpec(id="rel-three", source_id="mara-venn",
                         target_id="sealed-reliquary",
                         kind=RelationshipKind.CARRIES),
    ])
    edges = _edges(compile_sourcebook(book, "camp"))

    assert ("mara-venn", "old-bram", RelationType.KNOWS) in edges
    assert ("mara-venn", "old-bram", RelationType.HOSTILE_TO) in edges
    assert ("mara-venn", "sealed-reliquary", RelationType.OWNS) in edges


def test_inactive_relationships_are_not_projected():
    """A lapsed alliance is history, not a live retrieval edge."""
    book = _book(relationships=[RelationshipSpec(
        id="rel-one", source_id="mara-venn", target_id="old-bram",
        kind=RelationshipKind.ALLIED_WITH, active=False,
    )])
    edges = _edges(compile_sourcebook(book, "camp"))

    assert ("mara-venn", "old-bram", RelationType.ALLIED_WITH) not in edges
    # Residency and routes are unaffected — only the relationship is dropped.
    assert ("mara-venn", "copper-finch", RelationType.LOCATED_AT) in edges


def test_undirected_relationships_project_both_ways():
    book = _book(relationships=[RelationshipSpec(
        id="rel-one", source_id="mara-venn", target_id="old-bram",
        kind=RelationshipKind.SIBLING_OF, directed=False,
    )])
    edges = _edges(compile_sourcebook(book, "camp"))

    assert ("mara-venn", "old-bram", RelationType.KNOWS) in edges
    assert ("old-bram", "mara-venn", RelationType.KNOWS) in edges


# ── The visibility boundary ─────────────────────────────────────────────────


def test_only_player_visible_claims_reach_world_state():
    compiled = compile_sourcebook(_book(), "camp")

    assert compiled.established_facts == [
        "The Ash Gate has been closed since Old Bram died."
    ]
    withheld = {c.id for c in compiled.withheld}
    assert withheld == {"claim-mara-lies", "claim-key-buried"}
    # The secret must appear nowhere the narrator can reach.
    assert not any(
        "filed the lock" in fact for fact in compiled.established_facts
    )


def test_starting_state_can_grant_a_specific_secret():
    """The party begins knowing one thing they would otherwise have to earn."""
    book = _book(starting_state=StartingState(
        location_id="copper-finch",
        opening_situation="Rain on the shutters.",
        player_known_claim_ids=["claim-key-buried"],
    ))
    compiled = compile_sourcebook(book, "camp")

    assert "The obsidian key is buried under the arch." in compiled.established_facts
    assert {c.id for c in compiled.withheld} == {"claim-mara-lies"}


def test_dm_only_claims_are_never_granted_implicitly():
    """Even a DISCOVERABLE grant must not drag DM_ONLY along with it."""
    book = _book(starting_state=StartingState(
        location_id="copper-finch", opening_situation="Rain.",
        player_known_claim_ids=["claim-key-buried"],
    ))
    compiled = compile_sourcebook(book, "camp")

    assert not any("filed the lock" in f for f in compiled.established_facts)


# ── Opening scene ───────────────────────────────────────────────────────────


def test_the_opening_scene_is_seeded_from_the_starting_location():
    compiled = compile_sourcebook(_book(), "camp")

    assert compiled.current_location == "Copper Finch"
    assert compiled.location_description.startswith("A rain-dark tavern")
    assert compiled.scene_items == {
        "sealed reliquary": "An iron box with a bronze pin."
    }
    assert compiled.opening_situation.startswith("Rain on the shutters")


def test_quest_edges_link_giver_and_objective():
    book = _book(quests=[QuestSpec(
        id="find-the-key", name="Find the Key",
        giver_ids=["mara-venn"],
        objectives=[QuestObjective(
            id="obj-search", description="Search beneath the arch.",
            location_ids=["ash-gate-arch"],
        )],
    )])
    compiled = compile_sourcebook(book, "camp")
    edges = _edges(compiled)

    assert _nodes(compiled)["find-the-key"].entity_type is EntityType.QUEST
    assert ("mara-venn", "find-the-key", RelationType.QUEST_GIVER) in edges
    assert ("find-the-key", "ash-gate-arch", RelationType.OBJECTIVE_AT) in edges


# ── Authoring mistakes degrade, they do not explode ─────────────────────────


def test_a_dangling_reference_warns_instead_of_raising():
    """Defense in depth, not the primary guard.

    CampaignSourcebook's own validator rejects dangling ids at construction
    ("npc drifter current location references missing id ..."), so a book
    built the normal way cannot reach this. It still matters for books built
    around validation — model_construct, a partially-applied edit, a future
    schema relaxation — because emitting an edge to a node that does not
    exist would be rejected by the graph anyway, and a warning beats a
    silent drop.
    """
    book = _book()
    book.npcs[0].current_location_id = "nowhere-at-all"  # post-validation
    compiled = compile_sourcebook(book, "camp")

    assert ("mara-venn", "nowhere-at-all", RelationType.LOCATED_AT) not in _edges(compiled)
    assert any("nowhere-at-all" in w for w in compiled.warnings)
    # The NPC itself still lands — one bad pointer must not lose the entity.
    assert "mara-venn" in _nodes(compiled)


def test_an_unknown_starting_location_warns_and_leaves_the_scene_empty():
    book = _book(starting_state=StartingState(
        location_id="copper-finch", opening_situation="Rain.",
    ))
    book.starting_state.location_id = "not-a-place"  # bypass authoring checks
    compiled = compile_sourcebook(book, "camp")

    assert compiled.current_location == ""
    assert any("not-a-place" in w for w in compiled.warnings)
