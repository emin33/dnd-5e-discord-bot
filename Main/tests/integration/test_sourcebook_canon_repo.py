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
                        headquarters_id="emberhold",
                        territory_location_ids=["emberhold", "ash-gate"],
                        # mara-venn is ALSO a plain member via her own
                        # faction_ids: the two authoring routes must not
                        # collapse into one another.
                        leader_ids=["mara-venn"],
                        notable_member_ids=["toran-vex"],
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
                     description="A char-black disc.", attunement="none"),
        ],
        # Order deliberately not id order — sort_order must preserve THIS.
        npcs=[
            NPCSpec(id="toran-vex", name="Toran Vex",
                    appearance="A nervous clerk.", role="ledger-keeper",
                    faction_ids=["kestrel-guild"],
                    current_location_id="copper-finch"),
            NPCSpec(id="mara-venn", name="Mara Venn",
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
                    status=CharacterStatus.DEAD, summary="The dead ferryman.",
                    current_location_id="ash-gate"),
            NPCSpec(id="sable-quill", name="Sable Quill",
                    appearance="Ash on her cuffs.",
                    faction_ids=["ash-wardens"],
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
        timeline=[
            HistoricalEvent(id="event-gate-sealed", title="The Sealing",
                            date_label="Three winters back", sort_order=1,
                            summary="The arch was shut in one night.",
                            location_ids=["ash-gate"],
                            visibility=Visibility.PUBLIC),
            HistoricalEvent(id="event-bram-drowned", title="Bram Goes Under",
                            date_label="That same winter", sort_order=2,
                            summary="The ferryman did not come back.",
                            participant_ids=["old-bram"],
                            location_ids=["ash-gate"],
                            cause_event_ids=["event-gate-sealed"],
                            consequence_ids=["claim-secret-deed"]),
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
                                         location_ids=["ash-gate"]),
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
    assert [str(i) for i in guild.notable_member_ids] == ["toran-vex"]


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
    _db, repo = rig
    key = await _imported(repo)

    roster = await repo.faction_members(key, "kestrel-guild")

    assert {(m.npc_id, m.membership_role) for m in roster} == {
        ("mara-venn", ROLE_LEADER),
        ("mara-venn", ROLE_MEMBER),
        ("toran-vex", ROLE_MEMBER),
        ("toran-vex", ROLE_NOTABLE),
    }
    assert {m.npc_name for m in roster} == {"Mara Venn", "Toran Vex"}


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
    """Asking about a region answers for every room beneath it."""
    _db, repo = rig
    key = await _imported(repo)

    region = await repo.region_contents(key, "ashvale")

    assert set(region.location_ids) == {
        "ashvale", "emberhold", "copper-finch", "ash-gate",
    }
    assert set(region.npc_ids) == {
        "mara-venn", "toran-vex", "old-bram", "sable-quill",
    }
    assert set(region.item_ids) == {"brass-key"}
    assert set(region.quest_ids) == {"quest-ash-gate"}
    assert set(region.faction_ids) == {"kestrel-guild", "ash-wardens"}


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
    assert not nowhere.is_untouched


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
