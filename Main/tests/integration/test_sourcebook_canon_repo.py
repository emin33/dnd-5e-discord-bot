"""The canonical sourcebook layer, against a REAL migrated SQLite database.

Fakes are not an option here and the reason is on the record: the
single-writer work shipped a HIGH-severity silent no-write because the unit
fakes had no ``equipped`` semantics and no ``get_item_by_index``, so the
suite could not see a write that neither happened nor was receipted. This
layer is denser in exactly the same way — composite primary keys, cascade
paths, ``COALESCE`` overlay resolution, a recursive CTE — and every one of
those is a place a fake would agree with a wrong implementation.

So: real ``Database``, real migrations, real ``SourcebookRepository``, on a
tmp file per test.

The book below is deliberately awkward. It has an NPC who is a faction
member by one authoring route and a leader by another, an undirected
rivalry, an INACTIVE hostility, a claim that is public and false, a region
nobody visits, and lists whose order disagrees with their ids. Each of those
is a place the implementation could be wrong while still looking right.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnd_bot.data.database import Database
from dnd_bot.data.repositories.sourcebook_repo import (
    ROLE_LEADER, ROLE_MEMBER, ROLE_NOTABLE, SourcebookRepository,
)
from dnd_bot.models.sourcebook import (
    BehaviorProfile, CampaignSourcebook, CanonStatus, CharacterStatus,
    CreatureSpec, EncounterSpec, FactionSpec, HistoricalEvent, InventoryEntry,
    ItemSpec, KnowledgeClaim, LocationKind, LocationSpec, LoreDomainKind,
    LoreDomainSpec, NPCSpec, Provenance, QuestObjective, QuestSpec,
    RelationshipBeat, RelationshipKind, RelationshipSpec, RouteSpec,
    SourcebookMetadata, StartingState, StatBlock, StoryArcSpec, StoryBeat,
    Visibility,
)


def rich_book() -> CampaignSourcebook:
    """A book that touches every canonical table and several traps."""
    return CampaignSourcebook(
        metadata=SourcebookMetadata(
            sourcebook_id="ash-gate",
            title="The Ash Gate",
            pitch="Something was sealed, and someone lied about why.",
            tone=["rain-soaked", "close"],
            themes=["debt", "silence"],
            safety_boundaries=["no harm to children"],
            authoring_notes=["Bram's death is the hinge; do not reveal early."],
        ),
        locations=[
            LocationSpec(id="ashvale", name="Ashvale",
                         location_kind=LocationKind.REGION,
                         summary="A drowned valley."),
            LocationSpec(id="emberhold", name="Emberhold",
                         location_kind=LocationKind.SETTLEMENT,
                         parent_location_id="ashvale",
                         description="A town of wet stone."),
            LocationSpec(id="copper-finch", name="Copper Finch",
                         location_kind=LocationKind.BUILDING,
                         parent_location_id="emberhold",
                         description="A rain-dark tavern.",
                         atmosphere=["low talk"],
                         sensory_details=["woodsmoke", "wet wool"],
                         notable_features=["a scorched hearth"],
                         hazards=["a loose stair"],
                         access_rules=["closed at third bell"],
                         map_coordinates=(12.5, -3.25)),
            LocationSpec(id="ash-gate", name="Ash Gate",
                         aliases=["the arch", "the black gate"],
                         location_kind=LocationKind.SITE,
                         parent_location_id="emberhold",
                         description="A cracked black arch."),
            # A whole region the party has no reason to reach.
            LocationSpec(id="far-shore", name="Far Shore",
                         location_kind=LocationKind.REGION),
            LocationSpec(id="salt-quay", name="Salt Quay",
                         location_kind=LocationKind.SETTLEMENT,
                         parent_location_id="far-shore"),
        ],
        routes=[
            RouteSpec(id="finch-to-gate", from_location_id="copper-finch",
                      to_location_id="ash-gate", travel_time="a quarter hour",
                      hazards=["flooded lane"]),
            RouteSpec(id="gate-to-quay", from_location_id="ash-gate",
                      to_location_id="salt-quay", bidirectional=False,
                      access_requirements=["a warden's seal"]),
        ],
        factions=[
            FactionSpec(id="kestrel-guild", name="Kestrel Guild",
                        aliases=["the Guild", "kestrels"],
                        headquarters_id="emberhold",
                        territory_location_ids=["emberhold", "ash-gate"],
                        # mara-venn is ALSO a plain member via her own
                        # faction_ids: the two authoring routes must not
                        # collapse into one another. Two notable members, so
                        # list ORDER inside a membership role is exercised.
                        leader_ids=["mara-venn"],
                        notable_member_ids=["toran-vex", "old-bram"],
                        ideology=["debts are sacred"],
                        goals=["own the gate"],
                        ranks=["fledgling", "kestrel"]),
            FactionSpec(id="ash-wardens", name="Ash Wardens",
                        headquarters_id="ash-gate",
                        territory_location_ids=["ash-gate"],
                        leader_ids=["sable-quill"]),
        ],
        items=[
            ItemSpec(id="forged-deed", name="Forged Deed", category="document",
                     description="Ink too new for the date it claims.",
                     history=["drawn up the night of the sealing"],
                     significance="Proof of who profited.",
                     mechanics=["DC 15 Investigation to spot"]),
            ItemSpec(id="brass-key", name="Brass Key", category="key",
                     description="Warm, always.", charges=3,
                     default_location_id="copper-finch"),
            ItemSpec(id="warden-seal", name="Warden's Seal", category="badge",
                     aliases=["the seal"],
                     description="A char-black disc.", attunement="none"),
            # Not unique, so it can be BOTH placed at a location and carried.
            # That combination is what makes the `hidden` guard testable: the
            # item earns a graph node on its own, so the ONLY thing standing
            # between play and "Toran has it" is the concealment check.
            ItemSpec(id="warden-ledger", name="Warden's Ledger",
                     category="document", unique=False,
                     description="Columns of names, one of them scratched out.",
                     default_location_id="ash-gate"),
        ],
        # Order deliberately not id order — sort_order must preserve THIS.
        npcs=[
            NPCSpec(id="toran-vex", name="Toran Vex",
                    aliases=["the clerk"],
                    appearance="A nervous clerk.", role="ledger-keeper",
                    faction_ids=["kestrel-guild"],
                    current_location_id="copper-finch",
                    # Concealed, and the item is placed at the Ash Gate on its
                    # own — so a broken hidden-check publishes "Toran carries
                    # the ledger" rather than merely losing a node.
                    inventory=[InventoryEntry(item_id="warden-ledger",
                                              hidden=True, quantity=2,
                                              notes="under the floor")]),
            NPCSpec(id="mara-venn", name="Mara Venn",
                    aliases=["the investigator", "Venn"],
                    appearance="A sharp-eyed woman in a charcoal coat.",
                    role="investigator", pronouns="she/her", ancestry="human",
                    age="forties",
                    public_history=["Came upriver after the flood."],
                    private_history=["She was paid to look away."],
                    behavior=BehaviorProfile(
                        voice="clipped", values=["a debt paid"],
                        goals=["keep the deed buried"], fears=["the ledger"],
                        decision_rules=["never answer twice the same way"],
                    ),
                    faction_ids=["kestrel-guild"],
                    current_location_id="copper-finch",
                    inventory=[InventoryEntry(item_id="forged-deed",
                                              hidden=True, notes="sewn in")],
                    stat_block=StatBlock(challenge_rating="2",
                                         armor_class=13, hit_points=32,
                                         abilities={"dex": 14}),
                    provenance=Provenance(source_type="model",
                                          source_label="draft-3", revision=2)),
            NPCSpec(id="old-bram", name="Old Bram",
                    status=CharacterStatus.DEAD,
                    appearance="A river-worn coat, still damp.",
                    summary="The dead ferryman.",
                    current_location_id="ash-gate"),
            NPCSpec(id="sable-quill", name="Sable Quill",
                    appearance="Ash on her cuffs.",
                    # Two factions: membership list ORDER has to survive.
                    faction_ids=["ash-wardens", "kestrel-guild"],
                    current_location_id="ash-gate",
                    inventory=[InventoryEntry(item_id="warden-seal",
                                              equipped=True)]),
            NPCSpec(id="wren-ashlow", name="Wren Ashlow",
                    status=CharacterStatus.MISSING,
                    home_location_id="salt-quay"),
        ],
        creatures=[
            CreatureSpec(id="gate-hound", name="Gate Hound",
                         ecology="Nests in the arch's shadow.",
                         common_location_ids=["ash-gate"],
                         stat_block=StatBlock(challenge_rating="1/2",
                                              hit_points=11)),
        ],
        lore_domains=[
            LoreDomainSpec(id="saltwrit-law", name="Saltwrit",
                           domain_kind=LoreDomainKind.LAW,
                           tenets=["a sealed door is a settled debt"],
                           associated_entity_ids=["kestrel-guild", "emberhold"]),
        ],
        relationships=[
            RelationshipSpec(id="rel-sable-hates-mara", source_id="sable-quill",
                             target_id="mara-venn",
                             kind=RelationshipKind.HOSTILE_TO, valence=-70,
                             public_description="They do not speak."),
            # Undirected: hostility must be found from BOTH ends.
            RelationshipSpec(id="rel-toran-rivals-sable", source_id="toran-vex",
                             target_id="sable-quill",
                             kind=RelationshipKind.RIVAL_OF, directed=False,
                             public_description="An old ledger dispute.",
                             history=[RelationshipBeat(
                                 event_id="event-gate-sealed",
                                 description="Both signed the wrong page.",
                                 valence_after=-30)]),
            # Fear is not enmity, however the graph flattens it.
            RelationshipSpec(id="rel-mara-fears-bram", source_id="mara-venn",
                             target_id="old-bram", kind=RelationshipKind.FEARS,
                             public_description="She will not name him."),
            # Secret chain of command: no public description at all.
            RelationshipSpec(id="rel-secret-service", source_id="toran-vex",
                             target_id="ash-wardens",
                             kind=RelationshipKind.SERVES,
                             private_description="He reports to them nightly."),
            # The same guard, but between two NPCs that BOTH have graph nodes.
            # rel-secret-service targets a faction, which has no node at all,
            # so it would be withheld even with the check removed — this one
            # is the case where the visibility check is doing the work alone.
            RelationshipSpec(id="rel-secret-debt", source_id="mara-venn",
                             target_id="toran-vex", kind=RelationshipKind.OWES,
                             private_description="She bought his silence for "
                                                 "sixty crowns."),
            # Over and done with — must not answer "who is hostile now".
            RelationshipSpec(id="rel-old-grudge", source_id="old-bram",
                             target_id="mara-venn",
                             kind=RelationshipKind.HOSTILE_TO, active=False,
                             public_description="Long settled."),
        ],
        claims=[
            KnowledgeClaim(id="claim-public-mara", subject_id="mara-venn",
                           text="Mara Venn is the investigator everyone at "
                                "the Copper Finch defers to.",
                           visibility=Visibility.PUBLIC),
            KnowledgeClaim(id="claim-secret-deed", subject_id="mara-venn",
                           text="Mara Venn filed the lock herself.",
                           visibility=Visibility.DM_ONLY,
                           contradiction_group="who-sealed-the-gate"),
            KnowledgeClaim(id="claim-rumor-bram", subject_id="old-bram",
                           text="Old Bram walks the quay on wet nights.",
                           canon_status=CanonStatus.LEGEND,
                           visibility=Visibility.PUBLIC),
            KnowledgeClaim(id="claim-find-key", subject_id="brass-key",
                           text="The brass key is warm because the gate is not "
                                "sealed, only shut.",
                           visibility=Visibility.DISCOVERABLE),
            KnowledgeClaim(id="claim-find-gate", subject_id="ash-gate",
                           text="The Ash Gate was closed from the inside.",
                           visibility=Visibility.DISCOVERABLE,
                           valid_from_event_id="event-gate-sealed"),
            KnowledgeClaim(id="claim-find-warden", subject_id="sable-quill",
                           text="Sable Quill kept the second seal.",
                           visibility=Visibility.DISCOVERABLE,
                           evidence_claim_ids=["claim-find-key"],
                           known_by_ids=["mara-venn"]),
            KnowledgeClaim(id="claim-known-start", subject_id="emberhold",
                           text="Emberhold has flooded three winters running.",
                           visibility=Visibility.PLAYER_KNOWN),
            KnowledgeClaim(id="claim-old-story", subject_id="ash-gate",
                           text="The gate was sealed by the flood.",
                           canon_status=CanonStatus.DISPUTED,
                           visibility=Visibility.PUBLIC,
                           contradiction_group="who-sealed-the-gate",
                           invalidated_by_event_id="event-bram-drowned"),
        ],
        # Listed with the LATER event first, so authored list order and
        # chronological sort_order disagree. The schema keeps both on purpose;
        # storing only one would silently reorder somebody's timeline.
        timeline=[
            HistoricalEvent(id="event-bram-drowned", title="Bram Goes Under",
                            date_label="That same winter", sort_order=2,
                            summary="The ferryman did not come back.",
                            participant_ids=["old-bram"],
                            location_ids=["ash-gate"],
                            cause_event_ids=["event-gate-sealed"],
                            consequence_ids=["claim-secret-deed"]),
            HistoricalEvent(id="event-gate-sealed", title="The Sealing",
                            date_label="Three winters back", sort_order=1,
                            summary="The arch was shut in one night.",
                            location_ids=["ash-gate"],
                            visibility=Visibility.PUBLIC),
        ],
        quests=[
            QuestSpec(id="quest-ash-gate", name="What the Gate Kept",
                      summary="Mara sealed it; the deed proves the motive.",
                      hook="Someone is paying to keep the arch shut.",
                      stakes=["the town's water rights"],
                      giver_ids=["mara-venn"],
                      objectives=[
                          QuestObjective(id="obj-find-key",
                                         description="Find the brass key.",
                                         location_ids=["copper-finch"],
                                         completion_conditions=["key in hand"],
                                         involved_entity_ids=["brass-key"]),
                          QuestObjective(id="obj-open-gate",
                                         description="Open the arch.",
                                         prerequisite_objective_ids=["obj-find-key"],
                                         failure_conditions=["the gate is rewelded"],
                                         location_ids=["ash-gate", "copper-finch"]),
                      ],
                      reveal_claim_ids=["claim-find-gate"],
                      reward_item_ids=["brass-key"],
                      success_consequences=["the guild loses the gate"]),
            # Not active at start: the compiler withholds it.
            QuestSpec(id="quest-salt-run", name="The Salt Run",
                      hook="A cargo nobody signed for.",
                      objectives=[QuestObjective(id="obj-reach-quay",
                                                 description="Reach the quay.",
                                                 location_ids=["salt-quay"])]),
        ],
        story_arcs=[
            StoryArcSpec(id="arc-the-seal", name="The Seal",
                         premise="A debt was paid in stone.",
                         central_question="Who profits from a shut door?",
                         themes=["debt"],
                         involved_entity_ids=["mara-venn", "ash-gate"],
                         beats=[StoryBeat(id="beat-first-clue",
                                          title="A Warm Key",
                                          purpose="Seed the contradiction.",
                                          trigger_conditions=["the party asks about the gate"],
                                          location_ids=["copper-finch"],
                                          reveal_claim_ids=["claim-find-key"])],
                         escalation_clocks={"gate": 3}),
        ],
        encounters=[
            EncounterSpec(id="enc-gate-hounds", name="Hounds at the Arch",
                          location_ids=["ash-gate"],
                          participant_ids=["sable-quill"],
                          trigger_conditions=["approaching after dark"],
                          noncombat_solutions=["show the warden's seal"]),
        ],
        starting_state=StartingState(
            location_id="copper-finch",
            opening_situation="Rain on the shutters.",
            active_quest_ids=["quest-ash-gate"],
            active_story_arc_ids=["arc-the-seal"],
            player_known_claim_ids=["claim-known-start"],
            initial_clocks={"gate": 0},
        ),
    )


async def make_db(tmp_path: Path, name: str = "canon.db") -> Database:
    db = Database(db_path=tmp_path / name)
    await db.connect()
    await db.execute(
        "INSERT INTO campaign (id, guild_id, name, dm_user_id) VALUES (?,?,?,?)",
        ("camp", 1, "Camp", 1),
    )
    await db.execute(
        "INSERT INTO campaign (id, guild_id, name, dm_user_id) VALUES (?,?,?,?)",
        ("other", 1, "Other", 1),
    )
    await db.commit()
    return db


@pytest.fixture
async def rig(tmp_path: Path):
    db = await make_db(tmp_path)
    try:
        yield db, SourcebookRepository(db=db)
    finally:
        await db.disconnect()


async def _imported(repo: SourcebookRepository, book=None) -> str:
    receipt = await repo.import_book(book or rich_book())
    return receipt.sourcebook_key


async def _bound(repo: SourcebookRepository, campaign_id: str = "camp") -> str:
    key = await _imported(repo)
    await repo.bind_campaign(campaign_id, key)
    return key


# ── Import fidelity ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_book_round_trips_through_canonical_rows_exactly(rig):
    """The assertion the whole rebuild path stands on.

    If canon loses ANYTHING — a hidden inventory flag, a behaviour profile, a
    story beat, the order of a list — then the projection rebuilt from rows
    is not the projection compiled from the file, and calling the graph
    "disposable" is a lie.
    """
    _db, repo = rig
    book = rich_book()

    key = await repo.import_book(book)
    restored = await repo.load_book(key.sourcebook_key)

    assert restored == book


@pytest.mark.asyncio
async def test_authored_list_order_survives_and_is_not_id_order(rig):
    """``sort_order`` earns its column here.

    The book lists NPCs starting with Toran; sorted by id they would start
    with Mara. Reading rows back in PK order would silently reorder every
    list in the book, and the compiled projection with it.
    """
    _db, repo = rig
    book = rich_book()
    authored = [str(n.id) for n in book.npcs]
    assert authored != sorted(authored), "fixture no longer tests ordering"

    key = await repo.import_book(book)
    restored = await repo.load_book(key.sourcebook_key)

    assert [str(n.id) for n in restored.npcs] == authored


@pytest.mark.asyncio
async def test_both_authoring_routes_into_a_faction_are_kept_apart(rig):
    """Mara is a member by her own list and a leader by the faction's.

    Storing membership once would either lose her leadership or promote
    every member to leader. The round trip is the only place that shows it.
    """
    _db, repo = rig
    key = await _imported(repo)

    restored = await repo.load_book(key)
    mara = next(n for n in restored.npcs if n.id == "mara-venn")
    guild = next(f for f in restored.factions if f.id == "kestrel-guild")

    assert [str(f) for f in mara.faction_ids] == ["kestrel-guild"]
    assert [str(i) for i in guild.leader_ids] == ["mara-venn"]
    assert [str(i) for i in guild.notable_member_ids] == ["toran-vex", "old-bram"]
    # Order WITHIN a role, and an NPC in two factions, both survive.
    sable = next(n for n in restored.npcs if n.id == "sable-quill")
    assert [str(f) for f in sable.faction_ids] == ["ash-wardens", "kestrel-guild"]


@pytest.mark.asyncio
async def test_reimporting_identical_bytes_writes_nothing(rig):
    db, repo = rig
    book = rich_book()

    first = await repo.import_book(book)
    before = await db.fetch_one("SELECT COUNT(*) FROM sourcebook_npc")
    second = await repo.import_book(book)
    after = await db.fetch_one("SELECT COUNT(*) FROM sourcebook_npc")

    assert second.sourcebook_key == first.sourcebook_key
    assert second.already_imported is True
    assert second.row_counts == {}
    assert after[0] == before[0]


@pytest.mark.asyncio
async def test_an_edited_book_is_a_new_version_beside_the_old_one(rig):
    """Identity is content, so editing cannot overwrite what a party is playing."""
    _db, repo = rig
    original = rich_book()
    edited = rich_book()
    edited.claims[0].text = "Mara Venn is barely tolerated at the Copper Finch."

    first = await repo.import_book(original)
    second = await repo.import_book(edited)

    assert first.sourcebook_key != second.sourcebook_key
    assert {h.sourcebook_key for h in await repo.list_books()} == {
        first.sourcebook_key, second.sourcebook_key,
    }
    still = await repo.load_book(first.sourcebook_key)
    assert still.claims[0].text == original.claims[0].text


@pytest.mark.asyncio
async def test_identity_covers_the_whole_book_including_where_it_starts(rig):
    """Two books differing only in `starting_state` are DIFFERENT books.

    If the hash skipped it, the second import would silently no-op and the
    campaign would wake up in the wrong tavern knowing the wrong things.
    """
    _db, repo = rig
    base = rich_book()
    moved = rich_book()
    moved.starting_state.location_id = "ash-gate"
    reknown = rich_book()
    reknown.starting_state.player_known_claim_ids = ["claim-public-mara"]

    keys = {
        (await repo.import_book(b)).sourcebook_key
        for b in (base, moved, reknown)
    }

    assert len(keys) == 3


@pytest.mark.asyncio
async def test_books_that_compare_equal_get_the_same_key(rig):
    """Identity must not fork on an encoding artefact.

    `-0.0 == 0.0` is True in Python, so these two books ARE equal — but they
    JSON-encode differently, and a naive hash would mint a second version of a
    book a campaign is already playing.
    """
    _db, repo = rig
    positive = rich_book()
    negative = rich_book()
    positive.locations[2].map_coordinates = (0.0, 12.0)
    negative.locations[2].map_coordinates = (-0.0, 12.0)
    assert positive == negative

    first = await repo.import_book(positive)
    second = await repo.import_book(negative)

    assert second.sourcebook_key == first.sourcebook_key
    assert second.already_imported
    assert len(await repo.list_books()) == 1


@pytest.mark.asyncio
async def test_an_imported_book_reports_what_it_actually_wrote(rig):
    """Per-table counts, not just a total: a table skipped entirely by the
    importer would still leave `total_rows > 0` looking healthy."""
    _db, repo = rig

    counts = (await repo.import_book(rich_book())).row_counts

    assert counts["sourcebook"] == 1
    assert counts["sourcebook_location"] == 6
    assert counts["sourcebook_route"] == 2
    assert counts["sourcebook_npc"] == 5
    assert counts["sourcebook_item"] == 4
    assert counts["sourcebook_faction"] == 2
    assert counts["sourcebook_claim"] == 8
    assert counts["sourcebook_event"] == 2
    assert counts["sourcebook_relationship"] == 6
    assert counts["sourcebook_quest"] == 2
    assert counts["sourcebook_quest_objective"] == 3
    assert counts["sourcebook_quest_objective_location"] == 4
    assert counts["sourcebook_npc_inventory"] == 3
    assert counts["sourcebook_faction_territory"] == 3
    # 4 from npc.faction_ids (Sable names two) + 2 leaders + 2 notables
    assert counts["sourcebook_npc_faction"] == 8
    # creatures + lore domains + story arcs + encounters
    assert counts["sourcebook_aux_record"] == 4


@pytest.mark.asyncio
async def test_a_header_describes_the_version_it_names(rig):
    _db, repo = rig
    key = await _imported(repo)

    header = await repo.get_header(key)

    assert header.sourcebook_key == key
    assert header.sourcebook_id == "ash-gate"
    assert header.title == "The Ash Gate"
    assert header.pitch.startswith("Something was sealed")
    assert header.ruleset == "dnd5e"


@pytest.mark.asyncio
async def test_a_timeline_keeps_authored_order_and_chronology_apart(rig):
    """The fixture lists the later event first. Both facts are data."""
    _db, repo = rig
    key = await _imported(repo)

    restored = await repo.load_book(key)

    assert [str(e.id) for e in restored.timeline] == [
        "event-bram-drowned", "event-gate-sealed",
    ]
    assert [e.sort_order for e in restored.timeline] == [2, 1]


@pytest.mark.asyncio
async def test_replacing_an_unbound_version_rewrites_its_rows(rig):
    """The happy path of `replace`, which the refusal test cannot reach."""
    db, repo = rig
    key = await _imported(repo)
    await db.execute(
        "UPDATE sourcebook_npc SET name = 'Tampered' WHERE id = 'mara-venn'"
    )
    await db.commit()

    receipt = await repo.import_book(rich_book(), replace=True)

    assert receipt.sourcebook_key == key
    assert not receipt.already_imported
    restored = await repo.load_book(key)
    assert next(n for n in restored.npcs if n.id == "mara-venn").name == "Mara Venn"
    row = await db.fetch_one("SELECT COUNT(*) FROM sourcebook_npc")
    assert row[0] == 5


@pytest.mark.asyncio
async def test_a_failed_import_leaves_no_partial_rows(tmp_path):
    """A half-imported world is worse than none: the missing half is invisible.

    The failure is injected mid-write rather than simulated, and the
    assertion sweeps EVERY ``sourcebook*`` table from sqlite_master — so a
    table added later without transactional coverage fails this test instead
    of quietly leaking rows.
    """
    book = rich_book()

    class _FailingDatabase:
        """The real Database, with one write that dies like a disk error."""

        def __init__(self, inner: Database, fail_on: int) -> None:
            self._inner = inner
            self._fail_on = fail_on
            self.calls = 0

        def __getattr__(self, name):
            return getattr(self._inner, name)

        async def execute(self, sql, parameters=()):
            self.calls += 1
            if self.calls == self._fail_on:
                raise RuntimeError("disk went away mid-import")
            return await self._inner.execute(sql, parameters)

    db = await make_db(tmp_path)
    # Closed in `finally`: a non-transactional import leaves an open write
    # transaction, and an unclosed aiosqlite connection hangs interpreter
    # exit — which would turn this assertion's failure into a timeout.
    try:
        # Count the writes a clean import makes, then break it in the middle.
        counter = _FailingDatabase(db, fail_on=0)
        await SourcebookRepository(db=counter).import_book(book)
        midpoint = max(2, counter.calls // 2)
        await db.execute("DELETE FROM sourcebook")
        await db.commit()

        breaking = _FailingDatabase(db, fail_on=midpoint)
        with pytest.raises(RuntimeError, match="disk went away"):
            await SourcebookRepository(db=breaking).import_book(book)

        tables = await db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name LIKE 'sourcebook%'"
        )
        assert tables, "no canonical tables found — the sweep would pass vacuously"
        leaked = {}
        for (name,) in tables:
            row = await db.fetch_one(f"SELECT COUNT(*) FROM {name}")
            if row[0]:
                leaked[name] = row[0]
        assert leaked == {}
    finally:
        await db.rollback()
        await db.disconnect()


@pytest.mark.asyncio
async def test_replacing_a_version_a_campaign_is_playing_is_refused(rig):
    """Replace cascades through the party's discovery log. Refuse, loudly."""
    _db, repo = rig
    key = await _bound(repo)
    await repo.record_discovery("camp", key, "claim-find-key", turn=4, via="test")

    with pytest.raises(ValueError, match="camp"):
        await repo.import_book(rich_book(), replace=True)

    log = await repo.discovery_log("camp")
    assert [c.claim_id for c in log] == ["claim-find-key"]


# ── Binding: one version of a book per campaign ─────────────────────────────


@pytest.mark.asyncio
async def test_fixing_a_typo_does_not_leave_a_campaign_playing_two_versions(rig):
    """The defect this section exists for.

    The key is a content hash, so editing one line of prose mints a NEW key.
    Binding it beside the old one made every overlay query answer for two
    versions at once — the pre-edit and post-edit text of the same claim id
    both returned as settled canon, and since results order by a sha256, which
    one a caller saw when collapsing by claim id was effectively random.
    """
    _db, repo = rig
    first = await _bound(repo)
    await repo.record_discovery("camp", first, "claim-find-key", turn=7, via="dug")

    edited = rich_book()
    edited.claims[0].text = "Mara Venn is barely tolerated at the Copper Finch."
    second = (await repo.import_book(edited)).sourcebook_key
    receipt = await repo.bind_campaign("camp", second)

    assert second != first
    assert await repo.sourcebook_keys_for_campaign("camp") == [second]
    assert receipt.superseded_keys == [first]
    # One row per claim id, not two.
    effective = await repo.effective_claims("camp")
    ids = [c.claim_id for c in effective]
    assert len(ids) == len(set(ids)) == 8
    assert {c.text for c in effective if c.claim_id == "claim-public-mara"} == {
        "Mara Venn is barely tolerated at the Copper Finch."
    }


@pytest.mark.asyncio
async def test_advancing_a_version_carries_the_party_s_discoveries_across(rig):
    """A prose fix must not un-earn what the party earned.

    Discovery is keyed on the sourcebook_key, so without carry-forward every
    claim reads back undiscovered under the new key and gets re-offered.
    """
    _db, repo = rig
    first = await _bound(repo)
    await repo.record_discovery("camp", first, "claim-find-key", turn=7, via="dug")
    await repo.supersede_claim("camp", first, "claim-old-story", "claim-find-gate",
                               canon_status=CanonStatus.FALSE, note="disproved")
    await repo.record_visit("camp", first, "ash-gate", turn=4)

    edited = rich_book()
    edited.metadata.pitch = "A gate, and a ledger nobody signed."
    second = (await repo.import_book(edited)).sourcebook_key
    receipt = await repo.bind_campaign("camp", second)

    log = await repo.discovery_log("camp")
    assert [(c.claim_id, c.discovered_at_turn, c.discovered_via) for c in log] == [
        ("claim-find-key", 7, "dug"),
    ]
    assert receipt.claims_carried >= 2 and receipt.visits_carried == 1
    overturned = next(
        c for c in await repo.effective_claims("camp")
        if c.claim_id == "claim-old-story"
    )
    assert overturned.superseded_by_claim_id == "claim-find-gate"
    assert overturned.effective_canon_status is CanonStatus.FALSE
    assert overturned.note == "disproved"
    gate = await repo.region_contents(key := second, "ash-gate", campaign_id="camp")
    assert gate.unvisited_location_ids == [] and key
    assert "claim-find-key" not in {
        c.claim_id for c in await repo.undiscovered_claims("camp")
    }


@pytest.mark.asyncio
async def test_a_claim_the_new_version_dropped_is_left_behind(rig):
    """Carry-forward is by stable id, not a blind copy: a claim the author
    deleted must not be resurrected as a dangling overlay row."""
    _db, repo = rig
    first = await _bound(repo)
    await repo.record_discovery("camp", first, "claim-find-warden", turn=3, via="x")
    await repo.record_discovery("camp", first, "claim-find-key", turn=4, via="y")

    trimmed = rich_book()
    trimmed.claims = [c for c in trimmed.claims if c.id != "claim-find-warden"]
    trimmed.quests[0].reveal_claim_ids = ["claim-find-gate"]
    second = (await repo.import_book(trimmed)).sourcebook_key
    await repo.bind_campaign("camp", second)

    assert [c.claim_id for c in await repo.discovery_log("camp")] == [
        "claim-find-key"
    ]


@pytest.mark.asyncio
async def test_a_supplement_binds_beside_a_base_module(rig):
    """Only versions of the SAME sourcebook_id collide. A different book is a
    different book, and campaigns are meant to be able to play both."""
    _db, repo = rig
    base = await _bound(repo)
    supplement = rich_book()
    supplement.metadata.sourcebook_id = "salt-quay-supplement"
    other = (await repo.import_book(supplement)).sourcebook_key

    receipt = await repo.bind_campaign("camp", other)

    assert receipt.superseded_keys == []
    assert sorted(await repo.sourcebook_keys_for_campaign("camp")) == sorted(
        [base, other]
    )


@pytest.mark.asyncio
async def test_binding_an_unimported_key_fails_loudly(rig):
    _db, repo = rig
    with pytest.raises(LookupError, match="deadbeef"):
        await repo.bind_campaign("camp", "deadbeef")


@pytest.mark.asyncio
async def test_unbinding_hides_a_book_without_destroying_the_overlay(rig):
    _db, repo = rig
    key = await _bound(repo)
    await repo.record_discovery("camp", key, "claim-find-key", turn=2, via="z")

    assert await repo.unbind_campaign("camp", key)

    assert await repo.sourcebook_keys_for_campaign("camp") == []
    assert await repo.effective_claims("camp") == []
    # Rebinding restores it — the rows were hidden, not dropped.
    await repo.bind_campaign("camp", key)
    assert [c.claim_id for c in await repo.discovery_log("camp")] == [
        "claim-find-key"
    ]


@pytest.mark.asyncio
async def test_one_campaign_s_binding_is_not_anothers(rig):
    _db, repo = rig
    key = await _bound(repo, "camp")

    assert await repo.sourcebook_keys_for_campaign("other") == []
    assert await repo.sourcebook_keys_for_campaign("camp") == [key]


# ── Claims: the queries this layer exists for ───────────────────────────────


@pytest.mark.asyncio
async def test_undiscovered_discoverable_claims_is_a_query(rig):
    """The design doc's first named question.

    PUBLIC claims are not "earned" — they are true out loud. DM_ONLY ones are
    not on offer. Only DISCOVERABLE canon is a thing a party can go and get.
    """
    _db, repo = rig
    key = await _bound(repo)

    pending = await repo.undiscovered_claims("camp")

    assert [c.claim_id for c in pending] == [
        "claim-find-key", "claim-find-gate", "claim-find-warden",
    ]
    assert all(c.visibility is Visibility.DISCOVERABLE for c in pending)

    await repo.record_discovery("camp", key, "claim-find-gate", turn=7,
                                via="asked Mara")
    after = await repo.undiscovered_claims("camp")

    assert [c.claim_id for c in after] == ["claim-find-key", "claim-find-warden"]


@pytest.mark.asyncio
async def test_an_unbound_book_answers_nothing_for_a_campaign(rig):
    """Importing a module is not the same as playing it."""
    _db, repo = rig
    await _imported(repo)  # imported, never bound

    assert await repo.undiscovered_claims("camp") == []
    assert await repo.effective_claims("camp") == []


@pytest.mark.asyncio
async def test_one_partys_discoveries_do_not_leak_into_anothers(rig):
    """Two campaigns, one book: the overlay is per-campaign or it is nothing."""
    _db, repo = rig
    key = await _bound(repo, "camp")
    await repo.bind_campaign("other", key)

    await repo.record_discovery("camp", key, "claim-find-key", turn=3, via="dug")

    assert [c.claim_id for c in await repo.undiscovered_claims("camp")] == [
        "claim-find-gate", "claim-find-warden",
    ]
    assert [c.claim_id for c in await repo.undiscovered_claims("other")] == [
        "claim-find-key", "claim-find-gate", "claim-find-warden",
    ]


@pytest.mark.asyncio
async def test_when_they_learned_it_is_recorded_once(rig):
    """"What did the party learn, and when" is only answerable if the first
    answer stands. A re-discovery must not rewrite the turn."""
    _db, repo = rig
    key = await _bound(repo)

    await repo.record_discovery("camp", key, "claim-find-key", turn=3,
                                via="found it")
    await repo.record_discovery("camp", key, "claim-find-key", turn=19,
                                via="found it again")

    learned = await repo.discovery_log("camp")
    assert [(c.claim_id, c.discovered_at_turn, c.discovered_via) for c in learned] == [
        ("claim-find-key", 3, "found it"),
    ]


@pytest.mark.asyncio
async def test_the_party_starts_knowing_what_the_book_says_they_know(rig):
    _db, repo = rig
    key = await _bound(repo)

    granted = await repo.seed_starting_knowledge("camp", key)

    log = await repo.discovery_log("camp")
    assert granted == 1
    assert [(c.claim_id, c.discovered_at_turn, c.discovered_via) for c in log] == [
        ("claim-known-start", 0, "starting_state"),
    ]


@pytest.mark.asyncio
async def test_the_discovery_log_is_ordered_by_when_not_by_id(rig):
    _db, repo = rig
    key = await _bound(repo)

    await repo.record_discovery("camp", key, "claim-find-warden", turn=2, via="a")
    await repo.record_discovery("camp", key, "claim-find-key", turn=11, via="b")
    await repo.record_discovery("camp", key, "claim-find-gate", turn=5, via="c")

    assert [c.claim_id for c in await repo.discovery_log("camp")] == [
        "claim-find-warden", "claim-find-gate", "claim-find-key",
    ]


@pytest.mark.asyncio
async def test_play_supersedes_canon_without_editing_the_book(rig):
    """The second named query, and the invariant underneath it.

    A campaign that overturns authored canon writes the OVERLAY. The book is
    immutable — another campaign playing the same module must still find the
    original truth, or "canonical" means nothing.
    """
    _db, repo = rig
    key = await _bound(repo, "camp")
    await repo.bind_campaign("other", key)

    await repo.supersede_claim(
        "camp", key, "claim-old-story", "claim-find-gate",
        canon_status=CanonStatus.FALSE,
        note="the party proved the arch was shut from inside",
    )

    ours = {c.claim_id: c for c in await repo.effective_claims("camp")}
    theirs = {c.claim_id: c for c in await repo.effective_claims("other")}
    book = await repo.load_book(key)
    canon = next(c for c in book.claims if c.id == "claim-old-story")

    assert ours["claim-old-story"].superseded_by_claim_id == "claim-find-gate"
    assert ours["claim-old-story"].effective_canon_status is CanonStatus.FALSE
    assert ours["claim-old-story"].is_superseded
    # Untouched for everyone else, and untouched in canon.
    assert theirs["claim-old-story"].superseded_by_claim_id is None
    assert theirs["claim-old-story"].effective_canon_status is CanonStatus.DISPUTED
    assert canon.canon_status is CanonStatus.DISPUTED


@pytest.mark.asyncio
async def test_a_typo_in_a_superseding_id_is_refused(rig):
    """There is no foreign key here, and the failure would be silent AND total.

    A superseded claim drops out of `include_superseded=False`, so a bad id
    deletes a fact from retrieval with nothing put in its place.
    """
    _db, repo = rig
    key = await _bound(repo)

    with pytest.raises(ValueError, match="claim-typo-does-not-exist"):
        await repo.supersede_claim(
            "camp", key, "claim-public-mara", "claim-typo-does-not-exist"
        )
    with pytest.raises(ValueError, match="itself"):
        await repo.supersede_claim(
            "camp", key, "claim-public-mara", "claim-public-mara"
        )
    with pytest.raises(ValueError, match="no-such-claim"):
        await repo.supersede_claim("camp", key, "no-such-claim", "claim-find-key")

    standing = {c.claim_id for c in
                await repo.effective_claims("camp", include_superseded=False)}
    assert "claim-public-mara" in standing


@pytest.mark.asyncio
async def test_why_canon_was_overturned_survives_a_later_correction(rig):
    """`note` is preserved on omission, exactly like canon_status. The reason
    is the part most worth keeping."""
    _db, repo = rig
    key = await _bound(repo)
    await repo.supersede_claim("camp", key, "claim-old-story", "claim-find-gate",
                               canon_status=CanonStatus.FALSE,
                               note="the party proved it was shut from inside")

    await repo.supersede_claim("camp", key, "claim-old-story", "claim-find-key")

    overturned = next(c for c in await repo.effective_claims("camp")
                      if c.claim_id == "claim-old-story")
    assert overturned.superseded_by_claim_id == "claim-find-key"
    assert overturned.effective_canon_status is CanonStatus.FALSE
    assert overturned.note == "the party proved it was shut from inside"


@pytest.mark.asyncio
async def test_an_unearned_claim_reports_the_campaign_s_verdict_not_the_book_s(rig):
    """A party can disprove a claim before ever finding it.

    Advertising that at the book's CANON confidence would hand retrieval a
    fact this campaign has already ruled false.
    """
    _db, repo = rig
    key = await _bound(repo)
    await repo.supersede_claim("camp", key, "claim-find-warden", "claim-find-key",
                               canon_status=CanonStatus.FALSE, note="a forgery")

    pending = {c.claim_id: c for c in await repo.undiscovered_claims("camp")}

    unearned = pending["claim-find-warden"]
    assert unearned.effective_canon_status is CanonStatus.FALSE
    assert unearned.superseded_by_claim_id == "claim-find-key"
    assert unearned.is_superseded
    assert unearned.note == "a forgery"
    # Still unearned, though — that is a question about discovery, not truth.
    assert not unearned.discovered
    # And a claim play has NOT touched still reads as the book wrote it.
    assert pending["claim-find-gate"].effective_canon_status is CanonStatus.CANON
    assert not pending["claim-find-gate"].is_superseded


@pytest.mark.asyncio
async def test_discovering_a_claim_play_had_already_overturned(rig):
    """supersede_claim creates the row first, with discovered = 0. The later
    discovery has to flip it without the overlay row's existence hiding it."""
    _db, repo = rig
    key = await _bound(repo)
    await repo.supersede_claim("camp", key, "claim-find-key", "claim-find-gate")

    assert "claim-find-key" in {
        c.claim_id for c in await repo.undiscovered_claims("camp")
    }
    assert await repo.record_discovery("camp", key, "claim-find-key",
                                       turn=6, via="found it anyway")

    log = {c.claim_id: c for c in await repo.discovery_log("camp")}
    assert log["claim-find-key"].discovered_at_turn == 6
    assert log["claim-find-key"].discovered_via == "found it anyway"
    assert "claim-find-key" not in {
        c.claim_id for c in await repo.undiscovered_claims("camp")
    }


@pytest.mark.asyncio
async def test_a_discovery_reports_whether_it_actually_granted_anything(rig):
    """An UPSERT's rowcount is 1 either way, so returning it would report
    grants that never happened — and seed_starting_knowledge sums these."""
    _db, repo = rig
    key = await _bound(repo)

    assert await repo.record_discovery("camp", key, "claim-find-key", turn=1, via="a")
    assert not await repo.record_discovery("camp", key, "claim-find-key", turn=2, via="b")
    assert await repo.seed_starting_knowledge("camp", key) == 1
    assert await repo.seed_starting_knowledge("camp", key) == 0


@pytest.mark.asyncio
async def test_a_claim_query_can_be_narrowed_to_one_subject(rig):
    _db, repo = rig
    await _bound(repo)

    about_gate = await repo.undiscovered_claims("camp", subject_id="ash-gate")

    assert [c.claim_id for c in about_gate] == ["claim-find-gate"]
    assert [c.claim_id for c in
            await repo.undiscovered_claims("camp", subject_id="no-one")] == []


@pytest.mark.asyncio
async def test_asking_for_nothing_returns_nothing(rig):
    """The empty-iterable short circuits: an empty IN () is a SQL error, and
    'no filter' must never silently mean 'no filtering'."""
    _db, repo = rig
    key = await _bound(repo)

    assert await repo.effective_claims("camp", visibilities=[]) == []
    assert await repo.ties_to(key, "mara-venn", kinds=[]) == []
    # A generator is consumed exactly once, and still filters.
    generated = await repo.effective_claims(
        "camp", visibilities=(v for v in [Visibility.PUBLIC])
    )
    assert {c.claim_id for c in generated} == {
        "claim-public-mara", "claim-rumor-bram", "claim-old-story",
    }


@pytest.mark.asyncio
async def test_superseded_claims_can_be_filtered_out_of_retrieval(rig):
    _db, repo = rig
    key = await _bound(repo)
    await repo.supersede_claim("camp", key, "claim-old-story", "claim-find-gate")

    standing = await repo.effective_claims("camp", include_superseded=False)

    assert "claim-old-story" not in {c.claim_id for c in standing}
    assert "claim-public-mara" in {c.claim_id for c in standing}


@pytest.mark.asyncio
async def test_effective_claims_can_be_narrowed_to_what_play_may_see(rig):
    _db, repo = rig
    await _bound(repo)

    visible = await repo.effective_claims(
        "camp", visibilities=[Visibility.PUBLIC, Visibility.PLAYER_KNOWN]
    )

    ids = {c.claim_id for c in visible}
    assert "claim-secret-deed" not in ids
    assert "claim-find-key" not in ids
    assert {"claim-public-mara", "claim-rumor-bram", "claim-known-start"} <= ids


# ── Entity queries ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_faction_roster_is_one_query(rig):
    """Default is a row per AUTHORING ROUTE — both facts about Mara are real."""
    _db, repo = rig
    key = await _imported(repo)

    roster = await repo.faction_members(key, "kestrel-guild")

    assert {(m.npc_id, m.membership_role) for m in roster} == {
        ("mara-venn", ROLE_LEADER),
        ("mara-venn", ROLE_MEMBER),
        ("toran-vex", ROLE_MEMBER),
        ("toran-vex", ROLE_NOTABLE),
        ("old-bram", ROLE_NOTABLE),
        ("sable-quill", ROLE_MEMBER),
    }
    # Names come from the NPC join, not from the membership row.
    assert dict((m.npc_id, m.npc_name) for m in roster)["old-bram"] == "Old Bram"
    # And status, so a roster can tell you a notable member is dead.
    assert {m.status for m in roster if m.npc_id == "old-bram"} == {"dead"}


@pytest.mark.asyncio
async def test_a_roster_can_be_asked_for_people_instead_of_routes(rig):
    """`distinct` collapses to one row per NPC, keeping the strongest role.

    Without it, anything that counts or narrates the roster doubles anyone
    the book named twice.
    """
    _db, repo = rig
    key = await _imported(repo)

    people = await repo.faction_members(key, "kestrel-guild", distinct=True)

    assert [(m.npc_id, m.membership_role) for m in people] == [
        ("mara-venn", ROLE_LEADER),
        ("old-bram", ROLE_NOTABLE),
        ("toran-vex", ROLE_NOTABLE),
        ("sable-quill", ROLE_MEMBER),
    ]


@pytest.mark.asyncio
async def test_hostility_is_found_from_both_ends_of_an_undirected_tie(rig):
    """The trap: an undirected tie is hostility in BOTH directions.

    Matching only ``target_id`` would silently halve the answer — and a test
    that just asserted "Sable is hostile to Mara" would pass anyway.
    """
    _db, repo = rig
    key = await _imported(repo)

    at_mara = await repo.hostile_to(key, "mara-venn")
    at_sable = await repo.hostile_to(key, "sable-quill")
    at_toran = await repo.hostile_to(key, "toran-vex")

    assert [t.relationship_id for t in at_mara] == ["rel-sable-hates-mara"]
    assert [t.relationship_id for t in at_sable] == ["rel-toran-rivals-sable"]
    # Same undirected edge, reached from the source side.
    assert [t.relationship_id for t in at_toran] == ["rel-toran-rivals-sable"]


@pytest.mark.asyncio
async def test_fear_is_not_enmity_and_a_settled_grudge_is_not_current(rig):
    """Two distinctions the graph cannot make, which is why canon is queried.

    ``FEARS`` and ``HOSTILE_TO`` both collapse onto one ``hostile_to`` edge
    there, and the graph has no notion of an inactive tie at all.
    """
    _db, repo = rig
    key = await _imported(repo)

    assert await repo.hostile_to(key, "old-bram") == []
    assert [t.relationship_id for t in
            await repo.ties_to(key, "old-bram", kinds=[RelationshipKind.FEARS])
            ] == ["rel-mara-fears-bram"]
    # rel-old-grudge targets Mara but is inactive.
    assert "rel-old-grudge" not in {
        t.relationship_id for t in await repo.hostile_to(key, "mara-venn")
    }
    assert "rel-old-grudge" in {
        t.relationship_id
        for t in await repo.ties_to(key, "mara-venn", active_only=False)
    }


@pytest.mark.asyncio
async def test_a_tie_the_author_kept_private_is_still_in_canon(rig):
    """The compiler drops it from the graph. Canon must still hold it —
    that is the difference between a system of record and an index."""
    _db, repo = rig
    key = await _imported(repo)

    secret = await repo.ties_to(key, "ash-wardens",
                                kinds=[RelationshipKind.SERVES])

    assert [t.relationship_id for t in secret] == ["rel-secret-service"]
    assert secret[0].public_description == ""
    assert "reports to them nightly" in secret[0].private_description


@pytest.mark.asyncio
async def test_everything_authored_in_a_region_is_recursive(rig):
    """Asking about a region answers for every room beneath it.

    Compared as LISTS, not sets. quest-ash-gate has two objectives inside this
    subtree, so a dropped DISTINCT returns it twice — and a set() would have
    swallowed the duplicate silently.
    """
    _db, repo = rig
    key = await _imported(repo)

    region = await repo.region_contents(key, "ashvale")

    assert region.location_ids == [
        "ash-gate", "ashvale", "copper-finch", "emberhold",
    ]
    assert region.npc_ids == [
        "mara-venn", "old-bram", "sable-quill", "toran-vex",
    ]
    assert region.item_ids == ["brass-key", "warden-ledger"]
    assert region.quest_ids == ["quest-ash-gate"]
    assert region.faction_ids == ["ash-wardens", "kestrel-guild"]


@pytest.mark.asyncio
async def test_a_faction_is_found_by_territory_and_by_headquarters(rig):
    """Two independent arms of one query; neither may carry the other."""
    _db, repo = rig
    key = await _imported(repo)

    # copper-finch: no faction is headquartered here and none claims it as
    # territory, so neither arm fires.
    assert (await repo.region_contents(key, "copper-finch")).faction_ids == []
    # emberhold: Kestrel HQ *and* Kestrel territory.
    assert (await repo.region_contents(key, "emberhold")).faction_ids == [
        "ash-wardens", "kestrel-guild",
    ]


@pytest.mark.asyncio
async def test_an_npc_is_authored_into_a_region_by_home_as_well_as_by_presence(rig):
    """Wren is MISSING and stands nowhere — only `home_location_id` finds her."""
    _db, repo = rig
    key = await _imported(repo)

    far = await repo.region_contents(key, "far-shore")

    assert far.npc_ids == ["wren-ashlow"]


@pytest.mark.asyncio
async def test_a_region_the_party_has_not_touched(rig):
    _db, repo = rig
    key = await _bound(repo)

    far = await repo.region_contents(key, "far-shore", campaign_id="camp")
    assert far.is_untouched
    assert set(far.location_ids) == {"far-shore", "salt-quay"}
    assert set(far.quest_ids) == {"quest-salt-run"}

    await repo.record_visit("camp", key, "salt-quay", turn=12)
    after = await repo.region_contents(key, "far-shore", campaign_id="camp")

    assert not after.is_untouched
    assert after.unvisited_location_ids == ["far-shore"]


@pytest.mark.asyncio
async def test_an_unknown_region_is_empty_not_untouched(rig):
    """"Nothing authored here" and "authored and unvisited" are different
    answers; a typo'd id must not read as a whole untouched world."""
    _db, repo = rig
    key = await _bound(repo)

    nowhere = await repo.region_contents(key, "no-such-place",
                                         campaign_id="camp")

    assert nowhere.location_ids == []
    assert nowhere.npc_ids == [] and nowhere.item_ids == []
    assert nowhere.quest_ids == [] and nowhere.faction_ids == []
    assert not nowhere.is_untouched


@pytest.mark.asyncio
async def test_a_region_is_never_called_untouched_without_checking(rig):
    """Omitting `campaign_id` asks a different question, and the answer must
    not be "untouched" — that is the direction that dumps unearned lore."""
    _db, repo = rig
    key = await _bound(repo)
    await repo.record_visit("camp", key, "salt-quay", turn=3)

    unasked = await repo.region_contents(key, "far-shore")
    asked = await repo.region_contents(key, "far-shore", campaign_id="camp")

    assert unasked.location_ids == asked.location_ids
    assert unasked.unvisited_location_ids == []
    assert not unasked.is_untouched
    assert asked.unvisited_location_ids == ["far-shore"]


@pytest.mark.asyncio
async def test_visits_remember_the_first_arrival(rig):
    _db, repo = rig
    key = await _bound(repo)

    await repo.record_visit("camp", key, "ash-gate", turn=4)
    await repo.record_visit("camp", key, "ash-gate", turn=30)

    row = await _db_row(rig, "ash-gate")
    assert (row[0], row[1]) == (4, 30)


async def _db_row(rig, location_id: str):
    db, _repo = rig
    return await db.fetch_one(
        """
        SELECT first_visited_turn, last_visited_turn
        FROM campaign_location_state
        WHERE campaign_id = 'camp' AND location_id = ?
        """,
        (location_id,),
    )


# ── Lifecycle ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deleting_a_version_takes_its_overlay_with_it(rig):
    """No orphan discovery rows pointing at a book that no longer exists."""
    db, repo = rig
    key = await _bound(repo)
    await repo.record_discovery("camp", key, "claim-find-key", turn=1, via="x")
    await repo.record_visit("camp", key, "ash-gate", turn=1)

    assert await repo.delete_book(key)

    for table in ("sourcebook", "sourcebook_claim", "sourcebook_npc",
                  "campaign_sourcebook", "campaign_claim_state",
                  "campaign_location_state"):
        row = await db.fetch_one(f"SELECT COUNT(*) FROM {table}")
        assert row[0] == 0, f"{table} kept rows after the book was deleted"


@pytest.mark.asyncio
async def test_a_missing_book_reads_as_missing_not_as_empty(rig):
    _db, repo = rig
    assert await repo.load_book("not-a-key") is None
    assert await repo.get_header("not-a-key") is None
