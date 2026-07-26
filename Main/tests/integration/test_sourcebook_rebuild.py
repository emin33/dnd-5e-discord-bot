"""Rebuilding the disposable indexes from canonical rows.

``SOURCEBOOK_COMPILER_DESIGN.md``: "If graph or vector projection fails, the
import remains recoverable: rebuild projections from canonical SQLite records
instead of asking a model to regenerate lore." That sentence is only true if
canon is *sufficient* — and sufficiency is not something you can assert about
a schema, only about a rebuild.

So these run against a real migrated database, a real ``KnowledgeGraph``
persisting to that same database, and a real ``WorldStateStore``. The central
test destroys the graph outright and puts it back from a 64-character key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnd_bot.data.repositories.sourcebook_repo import SourcebookRepository
from dnd_bot.game.knowledge.graph import KnowledgeGraph
from dnd_bot.game.knowledge.models import AddEdge, AddNode, Entity, EntityType
from dnd_bot.game.knowledge.repository import KnowledgeGraphRepository
from dnd_bot.game.knowledge.sourcebook_compiler import (
    compile_from_canon, compile_sourcebook, install_sourcebook,
    projection_fingerprint, rebuild_indexes,
)
from dnd_bot.game.world_state import WorldState
from dnd_bot.game.world_store import WorldStateStore

from tests.integration.test_sourcebook_canon_repo import make_db, rich_book

# Text that exists ONLY in withheld canon. If any of it reaches an index, the
# secret is retrievable by semantic similarity alone — with no scene, entity
# or keyword to make the leak traceable afterwards.
SECRETS = (
    "filed the lock herself",          # DM_ONLY claim
    "paid to look away",               # private_history
    "Ink too new",                     # item present only as hidden inventory
    "the deed proves the motive",      # quest summary (hook is public, this is not)
    "reports to them nightly",         # private-only tie, to a faction
    "bought his silence",              # private-only tie between two NPCs
    "under the floor",                 # note on a CONCEALED inventory entry
    "cargo nobody signed for",         # hook of a quest that is not active
)
# Deliberately NOT a secret: the Warden's Ledger is placed at the Ash Gate, so
# its description is public and SHOULD be indexed. What is concealed is that
# Toran has one — an ownership EDGE, not text, which is why
# test_concealment_and_private_ties_survive_the_canon_round_trip checks edges.


def _assert_secrets_are_really_in_the_book(book) -> None:
    """The positive control the leak assertions are worthless without.

    Every string in SECRETS must exist in the fixture. Otherwise a reworded
    fixture — or a typo here — turns every "not in" assertion below into a
    tautology, and the suite goes on reporting that secrets are contained
    while checking nothing at all.
    """
    corpus = book.model_dump_json()
    missing = [secret for secret in SECRETS if secret not in corpus]
    assert not missing, f"SECRETS no longer present in the book: {missing}"


class _RecordingVectorStore:
    """Records exactly the text handed to the index, and nothing else.

    A fake here is safe in the one way that matters: it cannot flatter the
    implementation, because the assertion is about what was PASSED to it.
    """

    def __init__(self, fail_on: set[str] | None = None) -> None:
        # A LIST per node, not a dict slot: keying by node_id would let a
        # clean upsert overwrite a transient leak, and the leak would vanish
        # from `corpus` before any assertion saw it.
        self.writes: list[tuple[str, str]] = []
        self._fail_on = fail_on or set()

    def add_entity_description(
        self, *, campaign_id, node_id, entity_type, name, description,
        aliases=None,
    ) -> bool:
        self.writes.append((node_id, "\n".join(
            [f"{entity_type}: {name}", description, *(aliases or [])]
        )))
        return node_id not in self._fail_on

    @property
    def node_ids(self) -> set[str]:
        return {node_id for node_id, _ in self.writes}

    @property
    def corpus(self) -> str:
        return "\n".join(document for _, document in self.writes)


def _node_ids(compiled) -> set[str]:
    return {
        op.entity.node_id for op in compiled.graph_ops if isinstance(op, AddNode)
    }


def _edges(compiled) -> set[tuple[str, str, str]]:
    return {
        (op.relationship.source_id, op.relationship.target_id,
         op.relationship.relation_type.value)
        for op in compiled.graph_ops if isinstance(op, AddEdge)
    }


@pytest.fixture
async def rig(tmp_path: Path):
    db = await make_db(tmp_path, "rebuild.db")
    kg_repo = KnowledgeGraphRepository(db=db)
    graph = KnowledgeGraph(campaign_id="camp", repository=kg_repo)
    await graph.load()
    store = WorldStateStore(WorldState())
    try:
        yield db, SourcebookRepository(db=db), graph, store
    finally:
        await db.disconnect()


async def _install(rig, vector_store=None):
    _db, repo, graph, store = rig
    return await install_sourcebook(
        rich_book(), campaign_id="camp", repository=repo,
        knowledge_graph=graph, world_store=store, vector_store=vector_store,
    )


# ── Sufficiency ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_projection_from_rows_is_the_projection_from_the_file(rig):
    """The equivalence the whole design rests on.

    Not "roughly the same set of nodes" — the same ops, in the same order,
    with the same withheld list and the same warnings. Anything weaker and
    "canonical" would mean "canonical except for the parts nobody checked".
    """
    _db, repo, _graph, _store = rig
    book = rich_book()
    receipt = await repo.import_book(book)

    from_file = compile_sourcebook(book, "camp")
    from_rows = await compile_from_canon(repo, receipt.sourcebook_key, "camp")

    assert projection_fingerprint(from_rows) == projection_fingerprint(from_file)
    # Not a vacuous comparison of two empty projections: the fixture has to be
    # projecting nodes, edges, facts AND withholding things for the equality
    # above to mean anything.
    assert from_file.node_count and from_file.edge_count
    assert from_file.established_facts and from_file.withheld
    assert from_file.withheld_notes


@pytest.mark.asyncio
async def test_the_graph_can_be_destroyed_and_rebuilt_from_canon_alone(rig):
    """Delete every node the campaign has, then put the world back from a key.

    This is the whole point of the canonical layer. Before it, a compiled
    book existed only in the graph — a rebuildable index with nothing to
    rebuild *from*.
    """
    db, repo, graph, _store = rig
    installed = await _install(rig)
    before = (graph.node_count(), graph.edge_count())
    residents_before = {
        e.name for e in graph.residents_of(graph.resolve_location_node("Copper Finch"))
    }
    assert before[0] and before[1]

    # The index is gone. Only the canonical tables and the key remain.
    await db.execute("DELETE FROM kg_node")
    await db.commit()
    rebuilt_graph = KnowledgeGraph(
        campaign_id="camp", repository=KnowledgeGraphRepository(db=db)
    )
    await rebuilt_graph.load()
    assert rebuilt_graph.node_count() == 0

    receipt = await rebuild_indexes(
        repository=repo,
        sourcebook_key=installed.sourcebook_key,
        campaign_id="camp",
        knowledge_graph=rebuilt_graph,
    )

    assert (rebuilt_graph.node_count(), rebuilt_graph.edge_count()) == before
    assert receipt.graph_rejections == []
    assert (receipt.nodes_added, receipt.edges_added) == before
    assert receipt.preserved_nodes == []
    assert {
        e.name for e in
        rebuilt_graph.residents_of(rebuilt_graph.resolve_location_node("Copper Finch"))
    } == residents_before
    # And it survives a reload, so the rebuild reached the database, not just
    # the in-memory NetworkX graph.
    reloaded = KnowledgeGraph(
        campaign_id="camp", repository=KnowledgeGraphRepository(db=db)
    )
    await reloaded.load()
    assert (reloaded.node_count(), reloaded.edge_count()) == before


@pytest.mark.asyncio
async def test_rebuilding_a_healthy_index_repairs_rather_than_duplicates(rig):
    _db, repo, graph, _store = rig
    installed = await _install(rig)
    before = (graph.node_count(), graph.edge_count())

    for _ in range(2):
        await rebuild_indexes(
            repository=repo, sourcebook_key=installed.sourcebook_key,
            campaign_id="camp", knowledge_graph=graph,
        )

    assert (graph.node_count(), graph.edge_count()) == before


@pytest.mark.asyncio
async def test_rebuilding_an_unimported_book_fails_loudly(rig):
    _db, repo, graph, _store = rig

    with pytest.raises(LookupError, match="deadbeef"):
        await rebuild_indexes(
            repository=repo, sourcebook_key="deadbeef", campaign_id="camp",
            knowledge_graph=graph,
        )


# ── Installing ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_installing_a_book_populates_canon_indexes_and_the_scene(rig):
    """One call, all four truth layers, in the order the design doc gives."""
    _db, repo, graph, store = rig

    installed = await _install(rig)
    world = store.state

    assert installed.imported.total_rows > 0
    assert installed.scene_seeded
    assert installed.rebuilt.graph_rejections == []
    assert installed.rebuilt.nodes_added == installed.rebuilt.projected_nodes
    # Against DISTINCT edges, not the op count: two objectives of the same
    # quest sit in the Copper Finch, so the projection emits that OBJECTIVE_AT
    # edge twice and the graph keeps one. Exactly the gap between "what canon
    # asked for" and "what the graph took" that the receipt now measures.
    assert installed.rebuilt.edges_added == len(_edges(installed.compiled))
    assert installed.rebuilt.edges_added < installed.rebuilt.projected_edges
    # Canon
    assert await repo.sourcebook_keys_for_campaign("camp") == [
        installed.sourcebook_key
    ]
    # Overlay: what the book says they already know, and where they woke up.
    assert [c.claim_id for c in await repo.discovery_log("camp")] == [
        "claim-known-start"
    ]
    finch = await repo.region_contents(
        installed.sourcebook_key, "copper-finch", campaign_id="camp"
    )
    assert not finch.is_untouched
    # Index + scene
    assert graph.node_count() == installed.rebuilt.projected_nodes
    assert world.current_location == "Copper Finch"
    assert {n.name for n in world.npcs.values()} == {"Mara Venn", "Toran Vex"}
    # The opening location is marked visited at turn 0 — otherwise "authored
    # in a region the party has not touched" counts the starting tavern.
    row = await _db.fetch_one(
        """
        SELECT first_visited_turn FROM campaign_location_state
        WHERE campaign_id = 'camp' AND location_id = 'copper-finch'
        """
    )
    assert row is not None and row[0] == 0


@pytest.mark.asyncio
async def test_a_lossy_round_trip_stops_the_install(rig):
    """The install COMPARES the two projections; it does not merely hope.

    Once the graph is built from canon rather than from the book, every
    secret depends on canon's fidelity too — so "the rows reproduce the book"
    has to be checked, not assumed.
    """
    _db, repo, graph, _store = rig
    real_load = repo.load_book

    async def _amnesiac(key: str):
        book = await real_load(key)
        book.npcs = [n for n in book.npcs if n.id != "sable-quill"]
        return book

    repo.load_book = _amnesiac  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="do not reproduce the book"):
        await _install(rig)

    assert not graph.has_node("sable-quill")


@pytest.mark.asyncio
async def test_a_lossy_round_trip_that_would_publish_a_secret_stops_the_install(rig):
    """The case that motivates the check.

    Losing one bool — `InventoryEntry.hidden` — gives a concealed item an
    ownership edge and its text to the vector index, with no warning, no
    rejection and nothing in `withheld_notes` to show it happened.
    """
    _db, repo, graph, _store = rig
    real_load = repo.load_book

    async def _forgetful(key: str):
        book = await real_load(key)
        for npc in book.npcs:
            for entry in npc.inventory:
                entry.hidden = False
        return book

    repo.load_book = _forgetful  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="do not reproduce the book"):
        await _install(rig)

    assert not graph.has_node("forged-deed")


@pytest.mark.asyncio
async def test_a_faithful_round_trip_installs(rig):
    """Positive control for the two above: the real round trip passes the
    check, so those failures are the injected loss and not the check itself."""
    _db, _repo, graph, store = rig

    installed = await _install(rig)

    assert installed.scene_seeded
    assert graph.has_node("sable-quill")
    assert store.state.current_location == "Copper Finch"


# ── Rebuilding onto a campaign in play ──────────────────────────────────────


@pytest.mark.asyncio
async def test_a_rebuild_does_not_resurrect_the_dead(rig):
    """The invariant that outranks index freshness.

    The graph merges by node id with `properties.update()`, and canon always
    says `alive: "true"` for a character the BOOK considers living. So a
    rebuild that overwrote existing nodes would revert a death: the tool path
    leaves `alive: true` beside `status: dead` — and the narrator is only ever
    shown `alive` — while the delta path writes no `status` at all, clearing
    both of `hydrate_residents`' gates and walking the corpse back on stage.
    """
    _db, repo, graph, store = rig
    installed = await _install(rig)

    # Play kills Toran, by each of the two paths that write a death.
    toran = graph.get_entity("toran-vex")
    toran.properties.update({"alive": "false", "status": "dead"})
    mara = graph.get_entity("mara-venn")
    mara.properties["alive"] = "false"          # delta path: no `status`
    mara.properties["description"] = "Face down in the taproom."

    receipt = await rebuild_indexes(
        repository=repo, sourcebook_key=installed.sourcebook_key,
        campaign_id="camp", knowledge_graph=graph,
    )

    assert graph.get_entity("toran-vex").properties["alive"] == "false"
    assert graph.get_entity("mara-venn").properties["alive"] == "false"
    assert graph.get_entity("mara-venn").properties["description"] == (
        "Face down in the taproom."
    )
    assert {"toran-vex", "mara-venn"} <= set(receipt.preserved_nodes)
    # And nobody is hydrated back onto the stage.
    node = graph.resolve_location_node("Copper Finch")
    assert store.hydrate_residents(graph.residents_of(node)) == []


@pytest.mark.asyncio
async def test_a_node_the_graph_silently_refused_is_not_indexed(rig):
    """The graph can decline a node WITHOUT rejecting it.

    A new NPC whose proper name a durable node already carries merges into
    that node (one collision) or abstains (several) — and either way
    ``apply_operations`` returns no rejection, so a receipt built from the
    projection's own counts would claim a node that does not exist. Indexing
    that entity would then leave an orphan document in the vector store
    pointing at a node id nothing can resolve.
    """
    db, repo, graph, _store = rig
    installed = await _install(rig)
    kg_repo = KnowledgeGraphRepository(db=db)

    # Play lost Mara's node and minted a differently-identified one under the
    # same proper name — the exact shape the naming-promotion work guards.
    await kg_repo.delete_node("camp", "mara-venn")
    await kg_repo.upsert_node(Entity(
        node_id="the-woman-in-grey", entity_type=EntityType.NPC,
        name="Mara Venn", campaign_id="camp",
        properties={"description": "Play wrote this one."},
    ))
    reloaded = KnowledgeGraph(campaign_id="camp", repository=kg_repo)
    await reloaded.load()
    assert not reloaded.has_node("mara-venn")
    vector = _RecordingVectorStore()

    receipt = await rebuild_indexes(
        repository=repo, sourcebook_key=installed.sourcebook_key,
        campaign_id="camp", knowledge_graph=reloaded, vector_store=vector,
    )

    # The graph merged rather than created, and said nothing about the NODE —
    # only its orphaned edges complain, which names the wrong culprit.
    assert not reloaded.has_node("mara-venn")
    assert all(r.startswith("add_edge") for r in receipt.graph_rejections)
    assert not any("add_node" in r for r in receipt.graph_rejections)
    # Counting the projection would have reported a node that isn't there.
    assert receipt.nodes_added < receipt.projected_nodes
    # So the index must not carry a document for a node that isn't there.
    assert "mara-venn" not in vector.node_ids
    assert "the-woman-in-grey" not in vector.node_ids
    # Positive control: everything the graph DID hold was indexed.
    assert "toran-vex" in vector.node_ids
    assert receipt.embedded == len(vector.node_ids)


@pytest.mark.asyncio
async def test_a_deliberate_reseed_can_overwrite(rig):
    """`overwrite=True` is the escape hatch — and it must actually differ."""
    _db, repo, graph, _store = rig
    installed = await _install(rig)
    graph.get_entity("toran-vex").properties["description"] = "Scratched out."

    await rebuild_indexes(
        repository=repo, sourcebook_key=installed.sourcebook_key,
        campaign_id="camp", knowledge_graph=graph, overwrite=True,
    )

    assert graph.get_entity("toran-vex").properties["description"] == (
        "A nervous clerk."
    )


@pytest.mark.asyncio
async def test_a_rebuild_restores_only_what_the_graph_lost(rig):
    """The realistic damage: rows gone, not rows stale."""
    db, repo, graph, _store = rig
    installed = await _install(rig)
    graph.get_entity("mara-venn").properties["description"] = "Play wrote this."

    await db.execute("DELETE FROM kg_node WHERE node_id = 'toran-vex'")
    await db.commit()
    reloaded = KnowledgeGraph(
        campaign_id="camp", repository=KnowledgeGraphRepository(db=db)
    )
    await reloaded.load()
    reloaded.get_entity("mara-venn").properties["description"] = "Play wrote this."
    assert not reloaded.has_node("toran-vex")

    receipt = await rebuild_indexes(
        repository=repo, sourcebook_key=installed.sourcebook_key,
        campaign_id="camp", knowledge_graph=reloaded,
    )

    assert reloaded.has_node("toran-vex")
    assert receipt.nodes_added == 1
    assert "toran-vex" not in receipt.preserved_nodes
    assert reloaded.get_entity("mara-venn").properties["description"] == (
        "Play wrote this."
    )


@pytest.mark.asyncio
async def test_installing_onto_a_campaign_in_progress_leaves_the_scene_alone(rig):
    _db, repo, graph, store = rig
    store.state.current_location = "Somewhere Else"
    store.state.turn = 12

    installed = await _install(rig)

    assert not installed.scene_seeded
    assert store.state.current_location == "Somewhere Else"
    assert store.state.established_facts == []
    # Canon and the graph still landed — only the live scene was protected.
    assert installed.imported.total_rows > 0
    assert graph.node_count() > 0


@pytest.mark.asyncio
async def test_reinstalling_the_same_book_is_a_no_op_for_canon(rig):
    _db, repo, graph, _store = rig
    first = await _install(rig)
    second = await _install(rig)

    assert second.imported.already_imported
    assert second.sourcebook_key == first.sourcebook_key
    assert await repo.sourcebook_keys_for_campaign("camp") == [
        first.sourcebook_key
    ]


# ── The visibility boundary, on the rebuilt indexes ─────────────────────────


@pytest.mark.asyncio
async def test_secrets_do_not_reach_the_rebuilt_vector_index(rig):
    """The one leak no consistency grader can catch.

    Leaked canon is perfectly self-consistent, and an embedding leak is worse
    than a prompt leak: it resurfaces on similarity, from any turn, with
    nothing in the trace to explain where it came from.
    """
    _db, _repo, _graph, _store = rig
    _assert_secrets_are_really_in_the_book(rich_book())
    vector = _RecordingVectorStore()

    installed = await _install(rig, vector_store=vector)

    assert installed.rebuilt.embedded == installed.rebuilt.projected_nodes
    assert not installed.rebuilt.vector_skipped
    assert installed.rebuilt.vector_complete
    for secret in SECRETS:
        assert secret not in vector.corpus, f"leaked to the vector index: {secret}"
    # Positive controls: the index is not simply empty, and the channels the
    # secrets would have travelled on are live. Aliases especially — with no
    # aliased entity in the book, an alias-borne leak would be untestable.
    assert "A sharp-eyed woman" in vector.corpus
    assert "Someone is paying to keep the arch shut" in vector.corpus
    assert "the investigator" in vector.corpus
    assert "the black gate" in vector.corpus
    # An item whose only presence is a concealed one is not indexed at all,
    # but one that is ALSO placed at a location is — minus the concealment.
    assert "forged-deed" not in vector.node_ids
    assert "warden-ledger" in vector.node_ids


@pytest.mark.asyncio
async def test_a_withheld_quest_reaches_neither_index(rig):
    """`quest-salt-run` is not in `active_quest_ids`, so it does not exist yet
    as far as play is concerned — including semantically."""
    _db, _repo, graph, _store = rig
    vector = _RecordingVectorStore()

    installed = await _install(rig, vector_store=vector)

    assert not graph.has_node("quest-salt-run")
    assert "quest-salt-run" not in vector.node_ids
    assert "cargo nobody signed for" not in vector.corpus
    assert graph.has_node("quest-ash-gate")          # positive control
    assert any("quest-salt-run" in note
               for note in installed.compiled.withheld_notes)


@pytest.mark.asyncio
async def test_concealment_and_private_ties_survive_the_canon_round_trip(rig):
    """Both guards, exercised where the guard is the ONLY thing stopping them.

    `warden-ledger` earns a node on its own (it is placed at the Ash Gate), so
    a broken `hidden` check publishes "Toran carries it" rather than merely
    losing a node. `rel-secret-debt` runs between two NPCs that both have
    nodes, so a broken visibility check publishes the tie rather than
    reporting a dangling reference.
    """
    _db, repo, _graph, _store = rig
    installed = await _install(rig)
    edges = _edges(installed.compiled)

    assert "warden-ledger" in _node_ids(installed.compiled)
    assert ("toran-vex", "warden-ledger", "owns") not in edges
    assert ("mara-venn", "toran-vex", "knows") not in edges
    assert ("mara-venn", "toran-vex", "allied_with") not in edges
    # Positive control: a NON-hidden inventory entry does produce ownership.
    assert ("sable-quill", "warden-seal", "owns") in edges
    notes = installed.compiled.withheld_notes
    assert any("toran-vex->warden-ledger: hidden" in note for note in notes)
    assert any("rel-secret-debt" in note for note in notes)


@pytest.mark.asyncio
async def test_a_vector_index_that_refused_every_write_is_not_success(rig):
    """`add_entity_description` swallows its own errors and returns False, so
    counting only successes would let a rebuild that indexed NOTHING pass."""
    _db, repo, graph, _store = rig
    installed = await _install(rig)
    refusing = _RecordingVectorStore(fail_on=set(_node_ids(installed.compiled)))

    receipt = await rebuild_indexes(
        repository=repo, sourcebook_key=installed.sourcebook_key,
        campaign_id="camp", knowledge_graph=graph, vector_store=refusing,
    )

    assert receipt.embedded == 0
    assert receipt.embed_failures == receipt.projected_nodes
    assert not receipt.vector_complete


@pytest.mark.asyncio
async def test_a_vector_store_that_raises_does_not_abort_the_install(rig):
    """The most disposable layer must not strand the install with canon
    written, the campaign bound and no scene seeded."""
    _db, _repo, graph, store = rig

    class _Exploding:
        def add_entity_description(self, **_kwargs):
            raise RuntimeError("chroma is on fire")

    installed = await _install(rig, vector_store=_Exploding())

    assert installed.scene_seeded
    assert store.state.current_location == "Copper Finch"
    assert graph.node_count() > 0
    assert not installed.rebuilt.vector_complete
    assert any("chroma is on fire" in w for w in installed.rebuilt.warnings)


@pytest.mark.asyncio
async def test_secrets_do_not_reach_the_seeded_world_state(rig):
    _db, _repo, _graph, store = rig
    _assert_secrets_are_really_in_the_book(rich_book())

    installed = await _install(rig)
    # `established_facts`, NOT to_yaml(): to_yaml emits only facts relevant to
    # the CURRENT SCENE, so a fact about somewhere else is filtered out
    # whether or not the compiler leaked it. Asserting against the rendered
    # view made every check here unfalsifiable.
    facts = "\n".join(store.state.established_facts)

    for secret in SECRETS:
        assert secret not in facts, f"leaked into established facts: {secret}"
    assert "everyone at the Copper Finch defers to" in facts
    # A public claim that is not settled truth is withheld too: asserting a
    # legend as fact is the same failure wearing a different hat. Old Bram is
    # at the Ash Gate, so only the unfiltered list can see this.
    assert "walks the quay" not in facts
    assert "sealed by the flood" not in facts          # PUBLIC but DISPUTED
    assert "flooded three winters running" in facts    # PLAYER_KNOWN, canon
    assert any("filed the lock" in c.text for c in installed.compiled.withheld)
    assert "filed the lock" not in store.state.to_yaml()


@pytest.mark.asyncio
async def test_the_vector_index_is_skipped_when_no_store_is_given(rig):
    _db, _repo, _graph, _store = rig

    installed = await _install(rig)

    assert installed.rebuilt.vector_skipped
    assert installed.rebuilt.embedded == 0
    assert installed.rebuilt.vector_complete
