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
from dnd_bot.game.knowledge.models import AddNode
from dnd_bot.game.knowledge.repository import KnowledgeGraphRepository
from dnd_bot.game.knowledge.sourcebook_compiler import (
    compile_from_canon, compile_sourcebook, install_sourcebook, rebuild_indexes,
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
    "reports to them nightly",         # private-only relationship
)


class _RecordingVectorStore:
    """Records exactly the text handed to the index, and nothing else.

    A fake here is safe in the one way that matters: it cannot flatter the
    implementation, because the assertion is about what was PASSED to it.
    """

    def __init__(self) -> None:
        self.documents: dict[str, str] = {}

    def add_entity_description(
        self, *, campaign_id, node_id, entity_type, name, description,
        aliases=None,
    ) -> bool:
        self.documents[node_id] = "\n".join(
            [f"{entity_type}: {name}", description, *(aliases or [])]
        )
        return True

    @property
    def corpus(self) -> str:
        return "\n".join(self.documents.values())


def _shape(ops):
    """Graph ops minus their wall-clock stamps.

    ``Entity``/``Relationship`` default ``created_at`` to ``utcnow()``, so two
    identical projections built a millisecond apart never compare equal.
    Everything else — ids, order, properties, aliases, relation types — is
    left in, which is the part that has to match.
    """
    shapes = []
    for op in ops:
        data = op.model_dump(mode="json")
        for holder in ("entity", "relationship"):
            if isinstance(data.get(holder), dict):
                data[holder].pop("created_at", None)
                data[holder].pop("updated_at", None)
        shapes.append(data)
    return shapes


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

    assert _shape(from_rows.graph_ops) == _shape(from_file.graph_ops)
    assert from_rows.established_facts == from_file.established_facts
    assert from_rows.withheld == from_file.withheld
    assert from_rows.withheld_notes == from_file.withheld_notes
    assert from_rows.warnings == from_file.warnings
    assert from_rows.current_location == from_file.current_location
    assert from_rows.location_description == from_file.location_description
    assert from_rows.scene_items == from_file.scene_items
    assert from_rows.opening_situation == from_file.opening_situation
    # Not a vacuous comparison of two empty projections.
    assert from_file.node_count and from_file.edge_count and from_file.withheld


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
    assert graph.node_count() == installed.rebuilt.nodes
    assert world.current_location == "Copper Finch"
    assert {n.name for n in world.npcs.values()} == {"Mara Venn", "Toran Vex"}


@pytest.mark.asyncio
async def test_the_install_reads_canon_back_rather_than_trusting_memory(rig):
    """A lossy import must break the install, not survive as a healthy graph.

    Proven by making the round trip lie: if ``install_sourcebook`` compiled
    the in-memory book, the graph would be fully populated regardless of what
    canon actually holds.
    """
    _db, repo, graph, store = rig
    real_load = repo.load_book

    async def _amnesiac(key: str):
        book = await real_load(key)
        book.npcs = [n for n in book.npcs if n.id != "sable-quill"]
        return book

    repo.load_book = _amnesiac  # type: ignore[method-assign]
    installed = await _install(rig)

    projected = {
        op.entity.node_id for op in installed.compiled.graph_ops
        if isinstance(op, AddNode)
    }
    assert "sable-quill" not in projected
    assert not graph.has_node("sable-quill")
    # Positive control: the rest of the book still installed, so the absence
    # above is canon being read back, not the install having failed outright.
    assert "mara-venn" in projected
    assert graph.has_node("mara-venn")


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
    vector = _RecordingVectorStore()

    installed = await _install(rig, vector_store=vector)

    assert installed.rebuilt.embedded == installed.rebuilt.nodes
    assert not installed.rebuilt.vector_skipped
    for secret in SECRETS:
        assert secret not in vector.corpus, f"leaked to the vector index: {secret}"
    # Positive control: the index is not simply empty.
    assert "A sharp-eyed woman" in vector.corpus
    assert "Someone is paying to keep the arch shut" in vector.corpus
    assert "forged-deed" not in vector.documents


@pytest.mark.asyncio
async def test_secrets_do_not_reach_the_seeded_world_state(rig):
    _db, _repo, _graph, store = rig

    installed = await _install(rig)
    surface = store.state.to_yaml()

    for secret in SECRETS:
        assert secret not in surface
    assert "everyone at the Copper Finch defers to" in surface
    # A public claim that is not settled truth is withheld too: asserting a
    # legend as fact is the same failure wearing a different hat.
    assert "walks the quay" not in surface
    assert any("filed the lock" in c.text for c in installed.compiled.withheld)


@pytest.mark.asyncio
async def test_the_vector_index_is_skipped_when_no_store_is_given(rig):
    _db, _repo, _graph, _store = rig

    installed = await _install(rig)

    assert installed.rebuilt.vector_skipped
    assert installed.rebuilt.embedded == 0
