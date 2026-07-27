"""The shelf: how an authored book becomes something a DM can pick.

`load_sourcebook` read a file and `import_book` wrote canon, but nothing
joined them — so no book could ever BECOME available and the wizard would
have offered an empty list forever. These pin that middle step against a
real migrated database and the real repository.

The shipped example book is exercised here too, because a worked example
that has drifted out of validity is worse than none: it is the first thing
an author copies.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnd_bot.data.repositories.sourcebook_repo import SourcebookRepository
from dnd_bot.game.knowledge.sourcebook_compiler import (
    compile_sourcebook, load_sourcebook,
)
from dnd_bot.game.knowledge.sourcebook_library import (
    available_books, scan_library,
)

from tests.integration.test_sourcebook_canon_repo import make_db, rich_book

SHIPPED = Path(__file__).resolve().parents[2] / "data" / "sourcebooks"


@pytest.fixture
async def repo(tmp_path: Path):
    db = await make_db(tmp_path, "library.db")
    try:
        yield SourcebookRepository(db=db)
    finally:
        await db.disconnect()


def _write(directory: Path, name: str, book) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(
        yaml.safe_dump(book.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
async def test_a_book_dropped_in_the_folder_becomes_selectable(repo, tmp_path):
    shelf = tmp_path / "books"
    _write(shelf, "ash.yaml", rich_book())

    scan = await scan_library(repo, shelf)

    assert scan.ok
    assert [h.sourcebook_id for h in scan.imported] == ["ash-gate"]
    assert [h.title for h in await available_books(repo)] == ["The Ash Gate"]


@pytest.mark.asyncio
async def test_a_malformed_book_is_named_not_silently_dropped(repo, tmp_path):
    """The DM has to be told WHICH file and WHY.

    Silently offering a shorter list means the first symptom is a campaign
    starting with no world in it, which is a much harder thing to diagnose
    than a filename.
    """
    shelf = tmp_path / "books"
    _write(shelf, "good.yaml", rich_book())
    (shelf / "broken.yaml").write_text("metadata: {sourcebook_id: x", encoding="utf-8")
    bad = rich_book().model_dump(mode="json")
    bad["npcs"][0]["current_location_id"] = "no-such-place"
    (shelf / "dangling.yaml").write_text(yaml.safe_dump(bad), encoding="utf-8")

    scan = await scan_library(repo, shelf)

    assert not scan.ok
    assert {name for name, _ in scan.rejected} == {"broken.yaml", "dangling.yaml"}
    assert any("no-such-place" in reason for _, reason in scan.rejected)
    # The readable book still landed — one bad file does not empty the shelf.
    assert [h.sourcebook_id for h in scan.imported] == ["ash-gate"]


@pytest.mark.asyncio
async def test_rescanning_an_unchanged_shelf_writes_nothing(repo, tmp_path):
    shelf = tmp_path / "books"
    _write(shelf, "ash.yaml", rich_book())

    first = await scan_library(repo, shelf)
    second = await scan_library(repo, shelf)

    assert [h.sourcebook_key for h in second.imported] == [
        h.sourcebook_key for h in first.imported
    ]
    assert len(await repo.list_books()) == 1


@pytest.mark.asyncio
async def test_editing_a_book_offers_the_new_version_not_both(repo, tmp_path):
    """A revision is a new version in canon, but one entry in the menu.

    Offering every version would ask a DM to choose between two
    identical-looking titles.
    """
    shelf = tmp_path / "books"
    path = _write(shelf, "ash.yaml", rich_book())
    await scan_library(repo, shelf)

    edited = rich_book()
    edited.metadata.pitch = "A gate, and a ledger nobody signed."
    _write(shelf, path.name, edited)
    await scan_library(repo, shelf)

    assert len(await repo.list_books()) == 2       # canon keeps both
    offered = await available_books(repo)
    assert len(offered) == 1                        # the menu shows one
    assert offered[0].pitch == "A gate, and a ledger nobody signed."


@pytest.mark.asyncio
async def test_no_shelf_is_not_an_error(repo, tmp_path):
    """A deployment with no authored books is a normal deployment."""
    scan = await scan_library(repo, tmp_path / "nope")

    assert scan.ok and scan.imported == []
    assert await available_books(repo) == []


@pytest.mark.asyncio
async def test_non_book_files_are_ignored(repo, tmp_path):
    shelf = tmp_path / "books"
    _write(shelf, "ash.yaml", rich_book())
    (shelf / "README.md").write_text("not a book", encoding="utf-8")
    (shelf / "notes.txt").write_text("also not a book", encoding="utf-8")

    scan = await scan_library(repo, shelf)

    assert scan.ok
    assert len(scan.imported) == 1


# ── The shipped example ─────────────────────────────────────────────────────


def test_the_shipped_example_book_is_valid():
    """A worked example that has drifted is worse than none.

    It is the first thing an author copies, so it must load, compile, and
    demonstrate the two rules its own header teaches.
    """
    path = SHIPPED / "ash_gate_primer.yaml"
    assert path.is_file(), "the shipped example book is missing"

    book = load_sourcebook(path)
    compiled = compile_sourcebook(book, "camp")

    assert book.metadata.sourcebook_id == "ash-gate-primer"
    assert compiled.current_location and compiled.opening_situation
    # It teaches visibility: some claims are withheld, some reach play.
    assert compiled.established_facts and compiled.withheld
    # And it teaches appearance-vs-summary: no NPC relies on the fallback,
    # so the book compiles without the "bare name" warning.
    assert not compiled.warnings


@pytest.mark.asyncio
async def test_the_shipped_example_imports_and_is_offered(repo):
    scan = await scan_library(repo, SHIPPED)

    assert scan.ok, f"shipped shelf has unreadable books: {scan.rejected}"
    assert "The Ash Gate Primer" in {h.title for h in await available_books(repo)}


# ── The wizard seam ─────────────────────────────────────────────────────────
#
# `install_for_campaign` is the ONLY production path that puts an authored
# world into a campaign. Everything above proves the parts; these prove the
# thing the cog actually calls.


async def _session_for(campaign_id: str):
    from dnd_bot.game.knowledge.graph import KnowledgeGraph
    from dnd_bot.game.session import GameSession
    from dnd_bot.game.world_state import WorldState
    from tests.unit.test_scene_hydration import _MemoryRepo

    session = GameSession(id="s", channel_id=1, guild_id=1, campaign_id=campaign_id)
    session.world_state = WorldState()
    session.knowledge_graph = KnowledgeGraph(
        campaign_id=campaign_id, repository=_MemoryRepo(),
    )
    await session.knowledge_graph.load()
    return session


@pytest.mark.asyncio
async def test_the_wizard_path_installs_an_authored_world(repo):
    """Book -> canon -> graph -> the room the party wakes up in."""
    from dnd_bot.game.knowledge.sourcebook_library import install_for_campaign

    key = (await repo.import_book(rich_book())).sourcebook_key
    session = await _session_for("camp")

    outcome = await install_for_campaign(repo, session, "camp", key)

    assert outcome.installed and not outcome.error
    assert outcome.title == "The Ash Gate"
    assert "Copper Finch" in outcome.scene
    assert "Rain on the shutters." in outcome.scene
    assert session.world_state.current_location == "Copper Finch"
    assert {n.name for n in session.world_state.npcs.values()} == {
        "Mara Venn", "Toran Vex",
    }
    assert session.knowledge_graph.node_count() > 0
    assert await repo.sourcebook_keys_for_campaign("camp") == [key]


@pytest.mark.asyncio
async def test_the_opening_scene_carries_no_secrets(repo):
    """`compiled` also holds `withheld`. Handing that to the narrator's
    opening would publish the book's secrets in its first paragraph."""
    from dnd_bot.game.knowledge.sourcebook_library import install_for_campaign

    key = (await repo.import_book(rich_book())).sourcebook_key
    session = await _session_for("camp")

    outcome = await install_for_campaign(repo, session, "camp", key)

    assert "filed the lock" not in outcome.scene       # DM_ONLY claim
    assert "paid to look away" not in outcome.scene    # private_history
    assert "deed proves the motive" not in outcome.scene  # quest summary
    # Positive control: the public material IS there.
    assert "rain-dark tavern" in outcome.scene.casefold()


@pytest.mark.asyncio
async def test_a_missing_book_reports_instead_of_raising(repo):
    """The party is already in a started session; it must stay playable."""
    from dnd_bot.game.knowledge.sourcebook_library import install_for_campaign

    session = await _session_for("camp")

    outcome = await install_for_campaign(repo, session, "camp", "not-imported")

    assert not outcome.installed
    assert "not imported" in outcome.error
    assert session.world_state.current_location in ("", None)


@pytest.mark.asyncio
async def test_a_session_already_in_play_is_not_relocated(repo):
    """seed_opening_scene refuses a live scene, and the caller must respect
    that rather than narrate the book's opening over the party's room."""
    from dnd_bot.game.knowledge.sourcebook_library import install_for_campaign

    key = (await repo.import_book(rich_book())).sourcebook_key
    session = await _session_for("camp")
    session.world_state.current_location = "Somewhere Else"
    session.world_state.turn = 12

    outcome = await install_for_campaign(repo, session, "camp", key)

    assert not outcome.installed
    assert "already had a scene" in outcome.error
    assert session.world_state.current_location == "Somewhere Else"


class _CollectingRegistry:
    """Just enough registry: `_execute_add_npc` only calls register_entity.

    The real one reaches for Chroma and the voice stack, which this test has
    no opinion about -- what is under test is whether the executor resolves
    the NPC through the world store instead of minting a twin.
    """

    def __init__(self) -> None:
        self.registered: list = []

    def register_entity(self, entity) -> None:
        self.registered.append(entity)


@pytest.mark.asyncio
async def test_an_opening_npc_resolves_to_the_book_s_cast_not_a_twin(repo):
    """The executor must carry the SESSION, or the roster splits.

    `_execute_add_npc` routes through `world_store.ensure_npc` only when it
    has a session -- and that is the single place a tool-path NPCState is
    minted. Sessionless, every opening NPC became a SceneEntity with no
    `npc_id`, unlinked from world state. After a sourcebook install that is
    worse than untidy: the book has already put its cast on the roster, so
    an add_npc for one of them mints a second Mara Venn.
    """
    from dnd_bot.game.session import GameSession
    from dnd_bot.llm.effects import EffectExecutor, EffectType, ProposedEffect
    from dnd_bot.game.knowledge.sourcebook_library import install_for_campaign

    key = (await repo.import_book(rich_book())).sourcebook_key
    session = await _session_for("camp")
    await install_for_campaign(repo, session, "camp", key)
    seeded_ids = {npc.id for npc in session.world_state.npcs.values()}
    assert len(seeded_ids) == 2

    executor = EffectExecutor(
        scene_registry=_CollectingRegistry(),
        session=session,
    )
    await executor.execute(ProposedEffect(
        effect_type=EffectType.ADD_NPC,
        npc_name="Mara Venn",
        npc_description="A sharp-eyed woman in a charcoal coat.",
    ))

    # The DISCRIMINATING fact: the SceneEntity carries the book's NPC id.
    # Sessionless, `ensure_npc` is never reached and npc_id stays None --
    # and world_state would be untouched either way, so asserting only that
    # the roster is unchanged would pass whether or not the fix is present.
    mara_id = next(
        npc.id for npc in session.world_state.npcs.values()
        if npc.name == "Mara Venn"
    )
    assert [e.npc_id for e in executor.scene_registry.registered] == [mara_id]
    # And no twin: the roster is unchanged and still the book's cast.
    assert {npc.id for npc in session.world_state.npcs.values()} == seeded_ids
    assert sorted(n.name for n in session.world_state.npcs.values()) == [
        "Mara Venn", "Toran Vex",
    ]
