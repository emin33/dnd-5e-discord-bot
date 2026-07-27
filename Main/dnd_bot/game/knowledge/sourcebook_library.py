"""The shelf of authored sourcebooks a DM can start a campaign from.

`load_sourcebook` reads one book from a path and `import_book` writes it to
canon. Nothing sat between them, so a book had no way to BECOME available:
the wizard would have offered an empty list forever.

This is that middle step, and it is deliberately thin. Scan a directory,
validate each file, import the ones that parse, and report the ones that do
not — by name and reason, never by silently offering a shorter list. A DM
who drops a malformed book in the folder must be told, because the failure
they would otherwise see is their campaign starting with no world in it.

Importing is content-addressed and idempotent, so re-scanning costs nothing
and a book edited between scans becomes a NEW version rather than mutating
the one a campaign is already playing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import structlog

from ...models.sourcebook_canon import SourcebookHeader
from .sourcebook_compiler import load_sourcebook

logger = structlog.get_logger()

# Both authoring formats `load_sourcebook` accepts.
BOOK_SUFFIXES = (".yaml", ".yml", ".json")


@dataclass
class LibraryScan:
    """What a scan found, and what it could not read.

    ``rejected`` is not an error list to be swallowed — it is the answer to
    "why is my book not in the menu", and the caller is expected to show it.
    """

    imported: list[SourcebookHeader] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejected


async def scan_library(repository, directory: Path | str | None = None) -> LibraryScan:
    """Import every readable book in *directory* and return the shelf.

    Validation is the schema's job: `CampaignSourcebook` rejects dangling
    references, containment cycles and duplicate ids on construction, so a
    broken book fails HERE — at the shelf, with a filename attached — rather
    than half-installing into someone's campaign later.
    """
    from ...config import get_settings

    root = Path(directory) if directory is not None else get_settings().sourcebook_dir
    scan = LibraryScan()
    if not root.is_dir():
        # Not an error: a deployment with no authored books is a normal
        # deployment. The wizard simply offers nothing.
        logger.info("sourcebook_library_absent", path=str(root))
        return scan

    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in BOOK_SUFFIXES:
            continue
        try:
            book = load_sourcebook(path)
        except Exception as exc:
            # Deliberately broad: a hand-authored file can fail as YAML, as
            # JSON, or as schema validation, and the DM needs the same
            # treatment for all three — named, not dropped.
            scan.rejected.append((path.name, str(exc)))
            logger.warning(
                "sourcebook_rejected", file=path.name, error=str(exc)
            )
            continue
        receipt = await repository.import_book(book)
        header = await repository.get_header(receipt.sourcebook_key)
        if header is not None:
            scan.imported.append(header)

    logger.info(
        "sourcebook_library_scanned",
        path=str(root),
        imported=len(scan.imported),
        rejected=len(scan.rejected),
    )
    return scan


@dataclass
class InstallOutcome:
    """What the campaign got, and what to tell the DM if it did not.

    ``error`` is a message for a human, not an exception to re-raise: by the
    time this runs the party is in a started session, and failing the session
    to punish a bad book would cost them the session too.
    """

    scene: str = ""
    title: str = ""
    error: str = ""

    @property
    def installed(self) -> bool:
        return bool(self.scene)


async def install_for_campaign(
    repository, session, campaign_id: str, sourcebook_key: str
) -> InstallOutcome:
    """Put an authored world into a live session, and describe the result.

    Deliberately outside the Discord layer. This is the only production path
    that installs a book, so it is the one thing here that most needs a test
    — and a cog cannot be imported without py-cord, which the test
    environment does not have. The cog keeps the parts that are genuinely
    Discord: reading the option, and showing ``error``.

    Call AFTER the session exists (the graph and world store come from it)
    and BEFORE the opening narration, which needs ``scene`` to describe the
    seeded room instead of inventing a competing one.
    """
    from ...game.world_store import WorldStateStore
    from .sourcebook_compiler import install_sourcebook

    # PREFLIGHT, before anything is written. `install_sourcebook` binds the
    # campaign and rebuilds the graph BEFORE it seeds the scene, so a scene
    # the store then refuses left canon bound and a fully populated graph
    # behind an "improvised world" message -- a campaign that reads as
    # bookless while carrying the book. The refusal is knowable up front:
    # `seed_opening_scene` declines exactly when the session is already in
    # play, and this runs at session start where that is a real possibility
    # only if something else got there first.
    world_state = getattr(session, "world_state", None)
    if world_state is not None and (
        world_state.turn > 0 or bool(world_state.npcs)
    ):
        logger.warning(
            "sourcebook_install_preflight_refused",
            campaign_id=campaign_id,
            sourcebook_key=sourcebook_key[:12],
            turn=world_state.turn,
            npcs=len(world_state.npcs),
        )
        return InstallOutcome(
            error="the session already had a scene in progress",
        )

    try:
        book = await repository.load_book(sourcebook_key)
        if book is None:
            raise LookupError(f"sourcebook {sourcebook_key[:12]} is not imported")
        installed = await install_sourcebook(
            book,
            campaign_id=campaign_id,
            repository=repository,
            knowledge_graph=session.knowledge_graph,
            world_store=WorldStateStore(session.world_state),
        )
    except Exception as exc:
        logger.error(
            "sourcebook_install_failed",
            campaign_id=campaign_id,
            sourcebook_key=sourcebook_key[:12],
            error=str(exc),
            exc_info=True,
        )
        return InstallOutcome(error=str(exc))

    if not installed.scene_seeded:
        # seed_opening_scene refuses a session already in play. Reaching here
        # means something started the world first, and narrating the book's
        # opening over it would relocate the party mid-scene.
        logger.warning(
            "sourcebook_scene_not_seeded",
            campaign_id=campaign_id, sourcebook_key=sourcebook_key[:12],
        )
        return InstallOutcome(
            title=book.metadata.title,
            error="the session already had a scene in progress",
        )

    compiled = installed.compiled
    present = ", ".join(npc.name for npc in session.world_state.npcs.values())
    # Only what the compiler already cleared for play. `compiled` also carries
    # `withheld`, and handing that to the narrator's opening would publish the
    # book's secrets in its first paragraph.
    scene = "\n".join(part for part in (
        f"Location: {compiled.current_location}",
        compiled.location_description,
        compiled.opening_situation,
        f"Present: {present}" if present else "",
    ) if part)
    logger.info(
        "sourcebook_installed_for_campaign",
        campaign_id=campaign_id,
        sourcebook=book.metadata.sourcebook_id,
        location=compiled.current_location,
        npcs_on_stage=len(session.world_state.npcs),
    )
    return InstallOutcome(scene=scene, title=book.metadata.title)


async def available_books(repository) -> list[SourcebookHeader]:
    """Every imported version, newest first, one entry per book.

    A book edited on disk imports as a NEW version beside the old one, so the
    raw table grows a row per revision. Offering all of them would ask a DM
    to choose between two identical-looking titles, so only the most recent
    version of each ``sourcebook_id`` is shown.
    """
    headers = await repository.list_books()
    newest: dict[str, SourcebookHeader] = {}
    for header in headers:
        # list_books is ordered by imported_at, so the last write wins.
        newest[header.sourcebook_id] = header
    return sorted(newest.values(), key=lambda h: h.title.casefold())
