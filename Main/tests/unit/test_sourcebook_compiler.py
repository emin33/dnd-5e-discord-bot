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
    CanonStatus,
    CharacterStatus,
    InventoryEntry,
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
                id="ash-gate", name="Ash Gate",
                location_kind=LocationKind.SITE,
                description="A cracked black arch.",
            ),
        ],
        routes=[RouteSpec(
            id="finch-to-gate", from_location_id="copper-finch",
            to_location_id="ash-gate",
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
                current_location_id="ash-gate",
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
                id="claim-gate-closed", subject_id="ash-gate",
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


def _focused_book(**overrides) -> CampaignSourcebook:
    """A book with no cross-references, for single-channel assertions."""
    base = dict(
        metadata=SourcebookMetadata(
            sourcebook_id="focused", title="Focused", pitch="A test.",
        ),
        locations=[LocationSpec(
            id="copper-finch", name="Copper Finch",
            location_kind=LocationKind.BUILDING, description="A tavern.",
        )],
        starting_state=StartingState(
            location_id="copper-finch", opening_situation="Rain.",
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
    assert ("copper-finch", "ash-gate", RelationType.CONNECTED_TO) in edges
    assert ("ash-gate", "copper-finch", RelationType.CONNECTED_TO) in edges


def test_one_way_routes_stay_one_way():
    book = _book(routes=[RouteSpec(
        id="drop", from_location_id="copper-finch",
        to_location_id="ash-gate", bidirectional=False,
    )])
    edges = _edges(compile_sourcebook(book, "camp"))

    assert ("copper-finch", "ash-gate", RelationType.CONNECTED_TO) in edges
    assert ("ash-gate", "copper-finch", RelationType.CONNECTED_TO) not in edges


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


def test_a_public_but_false_claim_is_not_asserted_as_canon():
    """Public != true. A rumour everyone repeats is still not a fact."""
    book = _book(claims=[
        KnowledgeClaim(id="claim-rumour", subject_id="mara-venn",
                       text="Mara Venn is secretly the gate's warden.",
                       visibility=Visibility.PUBLIC,
                       canon_status=CanonStatus.FALSE),
        KnowledgeClaim(id="claim-legend", subject_id="ash-gate",
                       text="The arch was raised in a single night.",
                       visibility=Visibility.PUBLIC,
                       canon_status=CanonStatus.LEGEND),
    ])
    compiled = compile_sourcebook(book, "camp")

    assert compiled.established_facts == []
    assert {c.id for c in compiled.withheld} == {"claim-rumour", "claim-legend"}
    assert any("canon_status=false" in n for n in compiled.withheld_notes)


def test_only_the_player_facing_quest_hook_is_projected():
    """`summary` is where the author writes the answer to the mystery."""
    book = _book(
        quests=[QuestSpec(
            id="find-the-key", name="Find the Key",
            hook="Someone has been filing the gate's lock.",
            summary="Mara Venn filed it herself to sell passage.",
            giver_ids=["mara-venn"],
        )],
        starting_state=StartingState(
            location_id="copper-finch", opening_situation="Rain.",
            active_quest_ids=["find-the-key"],
        ),
    )
    compiled = compile_sourcebook(book, "camp")
    node = _nodes(compiled)["find-the-key"]

    assert node.properties["description"] == "Someone has been filing the gate's lock."
    assert "filed it herself" not in str(node.properties)


def test_quests_the_party_has_not_reached_are_not_projected_at_all():
    """Every quest as a permanent BFS-reachable node hands over the endgame."""
    book = _book(quests=[
        QuestSpec(id="find-the-key", name="Find the Key", hook="A filed lock.",
                  giver_ids=["mara-venn"]),
        QuestSpec(id="endgame-quest", name="Close the Gate",
                  hook="The gate must be sealed forever.",
                  summary="Only Bram's bones can seal it.",
                  giver_ids=["old-bram"],
                  objectives=[QuestObjective(
                      id="obj-seal", description="Seal the arch.",
                      location_ids=["ash-gate"],
                  )]),
    ], starting_state=StartingState(
        location_id="copper-finch", opening_situation="Rain.",
        active_quest_ids=["find-the-key"],
    ))
    compiled = compile_sourcebook(book, "camp")
    edges = _edges(compiled)

    assert "find-the-key" in _nodes(compiled)
    assert "endgame-quest" not in _nodes(compiled)
    assert any("endgame-quest" in n for n in compiled.withheld_notes)
    # Its giver and objective edges go too — they point at a node that was
    # deliberately not projected.
    assert ("old-bram", "endgame-quest", RelationType.QUEST_GIVER) not in edges
    assert ("endgame-quest", "ash-gate", RelationType.OBJECTIVE_AT) not in edges
    # And withholding it is NOT an authoring mistake. `warnings` is what an
    # author reads to find real defects (genuine dangling refs, ids that are
    # not slugify(name), same-named NPCs the graph would merge); a correctly
    # authored book that keeps a quest inactive must not land in there.
    assert not compiled.warnings


def test_a_privately_described_relationship_becomes_no_edge():
    """The schema's way of saying nobody knows this tie.

    The lossy collapse makes publishing it worse than merely imprecise: a
    covert chain of command (SERVES) would surface as a plain alliance.
    """
    book = _book(relationships=[RelationshipSpec(
        id="kale-serves-vex", source_id="mara-venn", target_id="old-bram",
        kind=RelationshipKind.SERVES,
        private_description="Nobody knows Mara answers to Bram's killer.",
    )])
    compiled = compile_sourcebook(book, "camp")
    edges = _edges(compiled)

    assert ("mara-venn", "old-bram", RelationType.ALLIED_WITH) not in edges
    assert any("kale-serves-vex" in n for n in compiled.withheld_notes)


def test_a_publicly_described_relationship_is_still_projected():
    book = _book(relationships=[RelationshipSpec(
        id="known-tie", source_id="mara-venn", target_id="old-bram",
        kind=RelationshipKind.SERVES,
        public_description="Everyone knows Mara worked for the ferryman.",
        private_description="And she resented every hour of it.",
    )])

    assert ("mara-venn", "old-bram", RelationType.ALLIED_WITH) in _edges(
        compile_sourcebook(book, "camp")
    )


def test_a_hidden_possession_is_neither_owned_nor_described():
    """The ownership edge alone answers 'who has the forged deed'."""
    book = _focused_book(
        items=[ItemSpec(id="forged-deed", name="forged deed",
                        description="Proof that Mara sold the passage.")],
        npcs=[NPCSpec(id="mara-venn", name="Mara Venn",
                      appearance="A sharp-eyed woman.",
                      current_location_id="copper-finch",
                      inventory=[InventoryEntry(item_id="forged-deed",
                                                hidden=True)])],
    )
    compiled = compile_sourcebook(book, "camp")

    assert ("mara-venn", "forged-deed", RelationType.OWNS) not in _edges(compiled)
    # And the item itself is not described into the graph, since a concealed
    # entry is its only presence in the book.
    assert "forged-deed" not in _nodes(compiled)
    assert any("forged-deed" in n for n in compiled.withheld_notes)


def test_an_openly_carried_item_is_still_projected():
    book = _focused_book(
        items=[ItemSpec(id="brass-compass", name="brass compass",
                        description="A palm-sized compass.")],
        npcs=[NPCSpec(id="mara-venn", name="Mara Venn",
                      appearance="A sharp-eyed woman.",
                      current_location_id="copper-finch",
                      inventory=[InventoryEntry(item_id="brass-compass")])],
    )
    compiled = compile_sourcebook(book, "camp")

    assert "brass-compass" in _nodes(compiled)
    assert ("mara-venn", "brass-compass", RelationType.OWNS) in _edges(compiled)


def test_a_missing_character_is_not_placed_anywhere():
    book = _focused_book(npcs=[NPCSpec(
        id="lost-clerk", name="Lost Clerk", status=CharacterStatus.MISSING,
        current_location_id="copper-finch",
    )])
    compiled = compile_sourcebook(book, "camp")

    assert "lost-clerk" in _nodes(compiled)   # they exist
    assert not [e for e in _edges(compiled) if e[0] == "lost-clerk"]
    assert any("lost-clerk" in n for n in compiled.withheld_notes)


def test_same_named_npcs_warn_because_the_graph_merges_them():
    """Two 'Cultist's: one is destroyed on apply. Say so at compile time."""
    book = _focused_book(npcs=[
        NPCSpec(id="cultist-one", name="Cultist", appearance="A hooded figure."),
        NPCSpec(id="cultist-two", name="Cultist", appearance="Another."),
    ])
    compiled = compile_sourcebook(book, "camp")

    assert any("cultist-two" in w and "MERGES" in w for w in compiled.warnings)


def test_a_location_id_that_is_not_its_slug_warns_about_forking():
    book = _book(locations=[LocationSpec(
        id="the-finch", name="Copper Finch",
        location_kind=LocationKind.BUILDING, description="A tavern.",
    )], routes=[], npcs=[], items=[], relationships=[], claims=[],
        starting_state=StartingState(
            location_id="the-finch", opening_situation="Rain."))
    compiled = compile_sourcebook(book, "camp")

    assert any("the-finch" in w and "fork" in w for w in compiled.warnings)


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
    book = _book(
        quests=[QuestSpec(
            id="find-the-key", name="Find the Key",
            hook="Something is wrong with the gate's lock.",
            giver_ids=["mara-venn"],
            objectives=[QuestObjective(
                id="obj-search", description="Search beneath the arch.",
                location_ids=["ash-gate"],
            )],
        )],
        starting_state=StartingState(
            location_id="copper-finch", opening_situation="Rain.",
            active_quest_ids=["find-the-key"],
        ),
    )
    compiled = compile_sourcebook(book, "camp")
    edges = _edges(compiled)

    assert _nodes(compiled)["find-the-key"].entity_type is EntityType.QUEST
    assert ("mara-venn", "find-the-key", RelationType.QUEST_GIVER) in edges
    assert ("find-the-key", "ash-gate", RelationType.OBJECTIVE_AT) in edges


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
