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
from dnd_bot.game.knowledge.matcher import action_entity_names
from dnd_bot.game.knowledge.sourcebook_compiler import (
    apply_sourcebook, load_sourcebook,
)
from dnd_bot.game.world_state import NPCState, WorldState
from dnd_bot.game.world_store import WorldStateStore
from dnd_bot.models.sourcebook import KnowledgeClaim, Visibility

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


# ── Reaching authored canon that is not in the room ─────────────────────────


def _book_with_offstage_canon():
    """The seeded book plus a public claim about a place elsewhere.

    The shared fixture anchors both its claims to an on-stage NPC, so it
    cannot express the case that matters here: canon whose subject the
    party can ask after but is not standing in front of.
    """
    book = _book()
    book.claims.append(KnowledgeClaim(
        id="claim-gate-closed", subject_id="ash-gate",
        text=(
            "The Ash Gate has been shut to travellers since Old Bram the "
            "ferryman died; his boat the Grey Hind still rots at the landing."
        ),
        visibility=Visibility.PUBLIC,
    ))
    return book


@pytest.mark.asyncio
async def test_asking_after_an_offstage_subject_reaches_its_authored_canon():
    """The whole seam: graph resolution -> fact anchor -> narrator prompt.

    Seeded lore read as amnesiac because facts were anchored to where the
    party STOOD while graph entities were seeded from what the player SAID.
    A question that names the Ash Gate by name could not reach the Ash
    Gate's canon from the tavern, however directly it was asked.
    """
    kg, store = await _fresh()
    await apply_sourcebook(
        _book_with_offstage_canon(), campaign_id="camp",
        knowledge_graph=kg, world_store=store,
    )
    action = (
        "I ask Mara Venn what she knows about the Ash Gate "
        "and why it is closed."
    )

    projected = store.state.to_yaml(
        action_text=action, action_entities=action_entity_names(kg, action),
    )

    assert "Grey Hind" in projected
    # ... and the on-stage NPC's own canon is still there beside it.
    assert "everyone at the Copper Finch defers to" in projected


@pytest.mark.asyncio
async def test_offstage_canon_stays_out_when_nobody_raised_it():
    """The anchor is what the player raised, not everything that exists."""
    kg, store = await _fresh()
    await apply_sourcebook(
        _book_with_offstage_canon(), campaign_id="camp",
        knowledge_graph=kg, world_store=store,
    )
    action = "I order a bowl of stew and warm my hands by the fire."

    projected = store.state.to_yaml(
        action_text=action, action_entities=action_entity_names(kg, action),
    )

    assert "Grey Hind" not in projected
    assert "everyone at the Copper Finch defers to" in projected


@pytest.mark.asyncio
async def test_reaching_offstage_canon_never_reaches_a_secret():
    """Widening retrieval must not widen the visibility boundary."""
    kg, store = await _fresh()
    await apply_sourcebook(
        _book_with_offstage_canon(), campaign_id="camp",
        knowledge_graph=kg, world_store=store,
    )
    action = "I ask about the Ash Gate, Old Bram, and who filed the lock."

    projected = store.state.to_yaml(
        action_text=action, action_entities=action_entity_names(kg, action),
    )

    assert "filed the lock" not in projected


@pytest.mark.asyncio
async def test_a_word_that_merely_contains_a_name_reaches_no_canon():
    """The shipped path, not just the WorldState-local half.

    Graph tier-1 resolution is bare substring — it answers "Ash Gate" to
    "Ash Gateway" — and its output is what `session._build_context` feeds
    the fact projection. This is the assertion that would have caught it.
    """
    kg, store = await _fresh()
    await apply_sourcebook(
        _book_with_offstage_canon(), campaign_id="camp",
        knowledge_graph=kg, world_store=store,
    )
    action = "I study the Ash Gateway Ledger, a bound volume of tariffs."

    projected = store.state.to_yaml(
        action_text=action, action_entities=action_entity_names(kg, action),
    )

    assert "Grey Hind" not in projected


@pytest.mark.asyncio
async def test_asking_after_the_dead_does_not_put_them_on_stage():
    """Fact anchoring widens facts only — the roster stays scene-scoped."""
    kg, store = await _fresh()
    await apply_sourcebook(
        _book_with_offstage_canon(), campaign_id="camp",
        knowledge_graph=kg, world_store=store,
    )
    action = "I call out for Old Bram and listen for an answer."

    data = yaml.safe_load(store.state.to_yaml(
        action_text=action, action_entities=action_entity_names(kg, action),
    ))

    assert {n["name"] for n in data["npcs_here"]} == {"Mara Venn", "Toran Vex"}


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


# ── The shipping seam ───────────────────────────────────────────────────────
#
# Everything above composes `to_yaml(action_text=..., action_entities=...)` by
# hand. That is the arrangement the feature NEEDS, not proof that production
# makes it: `GameSessionManager._build_context` is the only path that feeds
# the narrator, and a mutation test showed the whole feature could be deleted
# from it with all 1557 tests still green. These two pin the caller.


async def _seeded_session():
    """A real applied book behind a real GameSession, ready for _build_context."""
    from unittest.mock import MagicMock

    from dnd_bot.game.session import GameSession, GameSessionManager

    kg, store = await _fresh()
    await apply_sourcebook(
        _book_with_offstage_canon(), campaign_id="camp",
        knowledge_graph=kg, world_store=store,
    )
    session = GameSession(id="s", channel_id=1, guild_id=1, campaign_id="camp")
    session.world_state = store.state
    session.knowledge_graph = kg
    manager = GameSessionManager.__new__(GameSessionManager)
    return manager, session, MagicMock()


@pytest.mark.asyncio
async def test_the_narrator_context_reaches_offstage_canon_the_player_asked_for():
    """End to end through the ONLY path that builds the narrator's context.

    The Ash Gate is not the room the party is standing in and Old Bram is not
    on stage, so nothing about the scene can reach this fact — the action text
    is the only thing that can, and `_build_context` is what has to pass it.
    """
    manager, session, memory = await _seeded_session()

    context = await manager._build_context(
        session, memory, "I ask Mara Venn what she knows about the Ash Gate.",
    )

    assert "Grey Hind" in context.world_state_yaml
    # Positive control: the scene projection still happens at all.
    assert "Copper Finch" in context.world_state_yaml


@pytest.mark.asyncio
async def test_the_narrator_context_needs_the_graph_to_reach_that_canon():
    """The graph half is load-bearing, not decoration.

    After a book is applied the party's WorldState knows two on-stage NPCs and
    no connected locations, so its LOCAL vocabulary cannot resolve "Ash Gate"
    at all. If `_build_context` stopped passing `action_entities`, the feature
    would silently degrade to scene-only on exactly the seeded campaigns it
    was built for.
    """
    manager, session, memory = await _seeded_session()
    action = "I ask Mara Venn what she knows about the Ash Gate."

    with_graph = await manager._build_context(session, memory, action)
    session.knowledge_graph = None
    without_graph = await manager._build_context(session, memory, action)

    assert "Grey Hind" in with_graph.world_state_yaml
    assert "Grey Hind" not in without_graph.world_state_yaml


@pytest.mark.asyncio
async def test_naming_one_npc_does_not_reach_a_different_npcs_canon():
    """Token boundaries are not entity boundaries.

    A one-word name is a token of every longer name containing it, so asking
    after Mara Venn used to anchor an unrelated NPC called Mara — and put her
    canon in the prompt. This codebase makes that ordinary rather than exotic:
    naming-promotion leaves bare first names in aliases.
    """
    manager, session, memory = await _seeded_session()
    session.world_state.npcs["mara-of-thornwood"] = NPCState(
        id="mara-of-thornwood", name="Mara", location="Thornwood",
    )
    session.world_state.established_facts.append(
        "Mara of Thornwood poisoned the well and fled north."
    )

    context = await manager._build_context(
        session, memory, "I ask Mara Venn what she saw at the gate.",
    )

    assert "poisoned the well" not in context.world_state_yaml
    # Positive control: naming her outright DOES reach her canon.
    named = await manager._build_context(session, memory, "I ask Mara about the well.")
    assert "poisoned the well" in named.world_state_yaml


@pytest.mark.asyncio
async def test_applying_a_book_marks_its_facts_as_authored():
    """The install is where provenance is recorded; nothing else knows it.

    `get_scene_relevant_facts` ranks on this mark, so if the install stopped
    setting it the ranking would silently revert to pure recency and every
    long-campaign guarantee below would go with it.
    """
    kg, store = await _fresh()

    compiled = await apply_sourcebook(
        _book(), campaign_id="camp", knowledge_graph=kg, world_store=store,
    )

    assert compiled.established_facts
    assert store.state.canon_facts == store.state.established_facts
    # Withheld canon is not in the ledger, so it is not marked either.
    assert not any("filed the lock" in f for f in store.state.canon_facts)


@pytest.mark.asyncio
async def test_seeded_canon_still_reaches_the_prompt_late_in_a_campaign():
    """The generalisation the shipped evidence did not cover.

    The 0-of-3 -> 3-of-3 result behind this feature was measured entirely at
    turn 1 on a fresh ledger, where "most recently written" and "true" agree.
    They stop agreeing at six play facts about the same subject. This drives
    the real `_build_context` at a ledger depth a long campaign reaches, and
    it fails on a build that ranks the action budget by recency alone.
    """
    manager, session, memory = await _seeded_session()
    action = "I ask Mara Venn what she knows about the Ash Gate."

    turn_one = await manager._build_context(session, memory, action)
    # Forty turns of narrator chatter about the same subject.
    for n in range(40):
        session.world_state.established_facts.append(
            f"Old Bram's name came up at the Ash Gate again, note {n}."
        )
    much_later = await manager._build_context(session, memory, action)

    assert "Grey Hind" in turn_one.world_state_yaml
    assert "Grey Hind" in much_later.world_state_yaml
    # Positive control: the play facts crowding it are genuinely in scope —
    # without this, an assertion above could pass on a ledger that never grew.
    assert "note 39" in much_later.world_state_yaml
