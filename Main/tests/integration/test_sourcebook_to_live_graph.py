"""A compiled sourcebook must survive contact with the real graph.

The unit pins check the projection's shape; this checks that the ops the
compiler emits are ones the live KnowledgeGraph actually accepts, and that
the result answers the questions the game asks of it — who is here, what is
near this, and (the one nothing else can check) what the party must NOT be
told yet.

The last assertion closes the loop with scene hydration: a book-authored NPC
should be restorable on arrival exactly like one established in play, since
both end up as the same durable residency record.
"""

from __future__ import annotations

import pytest

from dnd_bot.game.knowledge.graph import KnowledgeGraph
from dnd_bot.game.knowledge.sourcebook_compiler import compile_sourcebook
from dnd_bot.game.world_state import WorldState
from dnd_bot.game.world_store import WorldStateStore
from dnd_bot.models.sourcebook import (
    CampaignSourcebook, CharacterStatus, KnowledgeClaim, LocationKind,
    LocationSpec, NPCSpec, RouteSpec, SourcebookMetadata, StartingState,
    Visibility,
)

from tests.unit.test_scene_hydration import _MemoryRepo


def _book() -> CampaignSourcebook:
    return CampaignSourcebook(
        metadata=SourcebookMetadata(
            sourcebook_id="ash-gate", title="The Ash Gate", pitch="A gate.",
        ),
        locations=[
            LocationSpec(id="copper-finch", name="Copper Finch",
                         location_kind=LocationKind.BUILDING,
                         description="A rain-dark tavern."),
            LocationSpec(id="ash-gate", name="Ash Gate",
                         location_kind=LocationKind.SITE,
                         description="A cracked black arch."),
        ],
        routes=[RouteSpec(id="finch-to-gate", from_location_id="copper-finch",
                          to_location_id="ash-gate")],
        npcs=[
            NPCSpec(id="mara-venn", name="Mara Venn",
                    appearance="A sharp-eyed woman in a charcoal coat.",
                    current_location_id="copper-finch"),
            NPCSpec(id="toran-vex", name="Toran Vex",
                    appearance="A nervous clerk.",
                    current_location_id="copper-finch"),
            NPCSpec(id="old-bram", name="Old Bram", status=CharacterStatus.DEAD,
                    summary="The dead ferryman.",
                    current_location_id="ash-gate"),
        ],
        claims=[
            # Both claims name Mara Venn, who is on stage — so scene-relevance
            # would surface EITHER. Only visibility separates them.
            KnowledgeClaim(id="claim-public", subject_id="mara-venn",
                           text="Mara Venn is the investigator everyone at "
                                "the Copper Finch defers to.",
                           visibility=Visibility.PUBLIC),
            KnowledgeClaim(id="claim-secret", subject_id="mara-venn",
                           text="Mara Venn filed the lock herself.",
                           visibility=Visibility.DM_ONLY),
        ],
        starting_state=StartingState(
            location_id="copper-finch",
            opening_situation="Rain on the shutters.",
        ),
    )


async def _loaded_graph(book, campaign="camp"):
    kg = KnowledgeGraph(campaign_id=campaign, repository=_MemoryRepo())
    await kg.load()
    compiled = compile_sourcebook(book, campaign)
    rejections = await kg.apply_operations(compiled.graph_ops)
    return kg, compiled, rejections


@pytest.mark.asyncio
async def test_the_whole_book_applies_without_rejections():
    kg, compiled, rejections = await _loaded_graph(_book())

    assert rejections == []
    assert kg.node_count() == 5      # 2 locations + 3 NPCs
    assert kg.edge_count() == compiled.edge_count
    assert not compiled.warnings


@pytest.mark.asyncio
async def test_the_graph_can_answer_who_is_here():
    kg, _compiled, _ = await _loaded_graph(_book())

    node = kg.resolve_location_node("Copper Finch")
    residents = {e.name for e in kg.residents_of(node)}

    assert residents == {"Mara Venn", "Toran Vex"}


@pytest.mark.asyncio
async def test_book_authored_npcs_hydrate_on_arrival():
    """Closes the loop: authored residents restore like played-in ones."""
    kg, compiled, _ = await _loaded_graph(_book())
    ws = WorldState(current_location=compiled.current_location)
    store = WorldStateStore(ws)

    node = kg.resolve_location_node(ws.current_location)
    restored = store.hydrate_residents(kg.residents_of(node))

    assert sorted(restored) == ["Mara Venn", "Toran Vex"]
    assert ws.npcs["mara-venn"].description.startswith("A sharp-eyed")


@pytest.mark.asyncio
async def test_an_authored_death_is_not_hydrated_at_its_location():
    """Old Bram is recorded AT the Ash Gate and must stay dead there."""
    kg, _compiled, _ = await _loaded_graph(_book())
    ws = WorldState(current_location="Ash Gate")
    store = WorldStateStore(ws)

    node = kg.resolve_location_node("Ash Gate")
    assert {e.name for e in kg.residents_of(node)} == {"Old Bram"}
    assert store.hydrate_residents(kg.residents_of(node)) == []


@pytest.mark.asyncio
async def test_secrets_never_reach_narrator_visible_state():
    """The assertion no other gate in this project can make.

    Leaked canon is perfectly self-consistent, so consistency grading cannot
    see it. Ground truth from the book is the only way to catch it.
    """
    kg, compiled, _ = await _loaded_graph(_book())
    ws = WorldState(current_location=compiled.current_location)
    ws.established_facts.extend(compiled.established_facts)
    store = WorldStateStore(ws)
    node = kg.resolve_location_node(ws.current_location)
    store.hydrate_residents(kg.residents_of(node))

    secret = "filed the lock"
    surface = ws.to_yaml() + "\n" + kg.to_context_yaml(
        kg.get_context_subgraph([e.node_id for e in kg.residents_of(node)])
    )

    assert secret not in surface
    # Positive control: a PUBLIC claim about the same on-stage NPC does
    # surface, so the secret's absence is visibility doing the work — not
    # scene-relevance filtering that would have hidden it anyway.
    assert "everyone at the Copper Finch defers to" in ws.to_yaml()
    # And the book still holds the secret, for the DM layer and assertions.
    assert any(secret in c.text for c in compiled.withheld)
