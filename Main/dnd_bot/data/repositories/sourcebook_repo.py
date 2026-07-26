"""Canonical storage for authored sourcebooks, and the queries play makes.

``SOURCEBOOK_COMPILER_DESIGN.md``: SQLite is canonical; the knowledge graph
and the vector index are rebuildable projections OF it. Before migration 007
that was aspirational — a compiled book existed only in the graph, so there
was nothing to rebuild from and no way to answer a question the graph's nine
relation types cannot express.

Two responsibilities, and they pull in opposite directions on purpose:

**Import is exact.** :meth:`import_book` writes a validated
:class:`CampaignSourcebook` and :meth:`load_book` reads back a book that
compares equal to it — including list ORDER, which is why every entity table
carries ``sort_order``. That equality is what makes the graph disposable: the
projection rebuilt from rows is the projection compiled from the file, not an
approximation of it.

**Query is overlay-aware.** The runtime's questions are joins, not traversals
— "which discoverable claims has this party not yet earned", "what did they
learn and when", "which claim supersedes which now that play overturned
canon". Those resolve the campaign overlay OVER the immutable book, so canon
is never edited by play.

Nothing here writes ``WorldState``. Seeding the opening scene stays with
``WorldStateStore``, the single writer.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Optional, Sequence

import structlog

from ...models.sourcebook import (
    CampaignSourcebook,
    CanonStatus,
    KnowledgeClaim,
    Provenance,
    RelationshipKind,
    Visibility,
)
from ...models.sourcebook_canon import (
    AuthoredTie,
    CampaignClaim,
    FactionMember,
    ImportReceipt,
    RegionContents,
    SourcebookHeader,
)
from ..database import Database, get_database

logger = structlog.get_logger()

# Which membership list a sourcebook_npc_faction row came from. Kept apart so
# the round trip puts each id back where the author wrote it instead of
# inventing leadership out of plain membership.
ROLE_MEMBER = "member"
ROLE_LEADER = "leader"
ROLE_NOTABLE = "notable"

# Authored kinds that mean "wants X to come to harm". The graph flattens
# HOSTILE_TO, RIVAL_OF *and* FEARS onto one ``hostile_to`` edge; FEARS is
# excluded here because being afraid of someone is not being their enemy,
# and the whole point of querying canon instead of the index is that the
# distinction survives. A caller wanting the graph's broader reading passes
# ``kinds`` explicitly.
HOSTILE_KINDS: frozenset[RelationshipKind] = frozenset({
    RelationshipKind.HOSTILE_TO,
    RelationshipKind.RIVAL_OF,
})

_AUX_KINDS = ("creature", "lore_domain", "story_arc", "encounter")


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _opt_text(value: object) -> str | None:
    return None if value is None else str(value)


def _flag(value: object) -> bool:
    return bool(value)


def _opt_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return int(str(value))


def _load_dict(raw: object) -> dict[str, Any]:
    if not raw:
        return {}
    parsed = json.loads(str(raw))
    return parsed if isinstance(parsed, dict) else {}


def _load_list(raw: object) -> list[Any]:
    if not raw:
        return []
    parsed = json.loads(str(raw))
    return parsed if isinstance(parsed, list) else []


def _dump(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _named_fields(spec: Any) -> dict[str, Any]:
    """The NamedEntity columns every authored entity shares."""
    return {
        "id": str(spec.id),
        "name": spec.name,
        "summary": spec.summary,
        "aliases_json": _dump(list(spec.aliases)),
        "tags_json": _dump(list(spec.tags)),
        "provenance_json": _dump(spec.provenance.model_dump(mode="json")),
    }


def _named_payload(row: Sequence[Any]) -> dict[str, Any]:
    """Inverse of :func:`_named_fields`.

    Every entity SELECT leads with the same six columns in the same order,
    so this stays the single place that knows their positions.
    """
    entity_id, name, summary, aliases, tags, provenance = tuple(row)[:6]
    return {
        "id": _text(entity_id),
        "name": _text(name),
        "summary": _text(summary),
        "aliases": _load_list(aliases),
        "tags": _load_list(tags),
        "provenance": _load_dict(provenance),
    }


class SourcebookRepository:
    """Canonical sourcebook rows plus the per-campaign overlay on them."""

    def __init__(self, db: Optional[Database] = None) -> None:
        self._db = db

    async def _get_db(self) -> Database:
        if self._db:
            return self._db
        return await get_database()

    # ==================== Identity ====================

    @staticmethod
    def content_hash(book: CampaignSourcebook) -> str:
        """A version's identity is its content.

        sha256 over canonical JSON, so re-importing the same bytes is a
        detectable no-op and an edited book is a NEW version rather than a
        silent overwrite of the one a campaign is already playing.
        """
        payload = _dump(book.model_dump(mode="json"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ==================== Import ====================

    async def import_book(
        self, book: CampaignSourcebook, *, replace: bool = False
    ) -> ImportReceipt:
        """Write a validated book to the canonical tables, atomically.

        The whole import runs inside one savepoint: a book either lands
        completely or not at all, because a half-imported world is worse
        than no world — the missing half is invisible at read time.

        Already-imported is a no-op, not an error: the key is derived from
        content, so an identical key means identical bytes. ``replace``
        re-writes those rows, and REFUSES when a campaign is bound to the
        version, because the delete cascades through the campaign's
        discovery log.
        """
        key = self.content_hash(book)
        db = await self._get_db()

        existing = await db.fetch_one(
            "SELECT sourcebook_key FROM sourcebook WHERE sourcebook_key = ?",
            (key,),
        )
        if existing and not replace:
            return ImportReceipt(
                sourcebook_key=key,
                sourcebook_id=str(book.metadata.sourcebook_id),
                already_imported=True,
            )
        if existing and replace:
            bound = await db.fetch_all(
                "SELECT campaign_id FROM campaign_sourcebook WHERE sourcebook_key = ?",
                (key,),
            )
            if bound:
                names = ", ".join(sorted(_text(row[0]) for row in bound))
                raise ValueError(
                    f"refusing to replace sourcebook {key[:12]}: campaigns "
                    f"[{names}] are bound to it, and replacing cascades "
                    "through their discovery and visit overlays. Unbind "
                    "them first if the loss is intended."
                )

        counts: dict[str, int] = {}

        def bump(table: str, n: int = 1) -> None:
            if n:
                counts[table] = counts.get(table, 0) + n

        async with await db.transaction():
            if existing:
                await db.execute(
                    "DELETE FROM sourcebook WHERE sourcebook_key = ?", (key,)
                )

            meta = book.metadata
            await db.execute(
                """
                INSERT INTO sourcebook
                    (sourcebook_key, sourcebook_id, schema_version, title,
                     pitch, ruleset, metadata_json, starting_state_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    str(meta.sourcebook_id),
                    meta.schema_version,
                    meta.title,
                    meta.pitch,
                    meta.ruleset,
                    _dump({
                        "tone": list(meta.tone),
                        "themes": list(meta.themes),
                        "safety_boundaries": list(meta.safety_boundaries),
                        "authoring_notes": list(meta.authoring_notes),
                    }),
                    _dump(book.starting_state.model_dump(mode="json")),
                ),
            )
            bump("sourcebook")

            # Order is load-bearing: SQLite enforces foreign keys
            # immediately, so every parent lands before the join rows that
            # reference it.
            for order, location in enumerate(book.locations):
                named = _named_fields(location)
                await db.execute(
                    """
                    INSERT INTO sourcebook_location
                        (sourcebook_key, id, sort_order, name, summary,
                         location_kind, parent_location_id, description,
                         aliases_json, tags_json, detail_json, provenance_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key, named["id"], order, named["name"], named["summary"],
                        location.location_kind.value,
                        _opt_text(location.parent_location_id),
                        location.description,
                        named["aliases_json"], named["tags_json"],
                        _dump({
                            "atmosphere": list(location.atmosphere),
                            "sensory_details": list(location.sensory_details),
                            "notable_features": list(location.notable_features),
                            "hazards": list(location.hazards),
                            "access_rules": list(location.access_rules),
                            "map_coordinates": (
                                list(location.map_coordinates)
                                if location.map_coordinates else None
                            ),
                        }),
                        named["provenance_json"],
                    ),
                )
            bump("sourcebook_location", len(book.locations))

            for order, route in enumerate(book.routes):
                await db.execute(
                    """
                    INSERT INTO sourcebook_route
                        (sourcebook_key, id, sort_order, from_location_id,
                         to_location_id, bidirectional, travel_time, distance,
                         description, detail_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key, str(route.id), order, str(route.from_location_id),
                        str(route.to_location_id), int(route.bidirectional),
                        route.travel_time, route.distance, route.description,
                        _dump({
                            "access_requirements": list(route.access_requirements),
                            "hazards": list(route.hazards),
                        }),
                    ),
                )
            bump("sourcebook_route", len(book.routes))

            for order, faction in enumerate(book.factions):
                named = _named_fields(faction)
                await db.execute(
                    """
                    INSERT INTO sourcebook_faction
                        (sourcebook_key, id, sort_order, name, summary,
                         headquarters_id, aliases_json, tags_json,
                         profile_json, provenance_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key, named["id"], order, named["name"], named["summary"],
                        _opt_text(faction.headquarters_id),
                        named["aliases_json"], named["tags_json"],
                        _dump({
                            "ideology": list(faction.ideology),
                            "goals": list(faction.goals),
                            "methods": list(faction.methods),
                            "resources": list(faction.resources),
                            "ranks": list(faction.ranks),
                        }),
                        named["provenance_json"],
                    ),
                )
            bump("sourcebook_faction", len(book.factions))

            for order, item in enumerate(book.items):
                named = _named_fields(item)
                await db.execute(
                    """
                    INSERT INTO sourcebook_item
                        (sourcebook_key, id, sort_order, name, summary,
                         category, description, significance, attunement,
                         charges, is_unique, default_location_id,
                         aliases_json, tags_json, detail_json, provenance_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key, named["id"], order, named["name"], named["summary"],
                        item.category, item.description, item.significance,
                        item.attunement, item.charges, int(item.unique),
                        _opt_text(item.default_location_id),
                        named["aliases_json"], named["tags_json"],
                        _dump({
                            "history": list(item.history),
                            "mechanics": list(item.mechanics),
                        }),
                        named["provenance_json"],
                    ),
                )
            bump("sourcebook_item", len(book.items))

            for order, npc in enumerate(book.npcs):
                named = _named_fields(npc)
                await db.execute(
                    """
                    INSERT INTO sourcebook_npc
                        (sourcebook_key, id, sort_order, name, summary, status,
                         role, appearance, pronouns, ancestry, age,
                         current_location_id, home_location_id, aliases_json,
                         tags_json, behavior_json, public_history_json,
                         private_history_json, stat_block_json, provenance_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key, named["id"], order, named["name"], named["summary"],
                        npc.status.value, npc.role, npc.appearance,
                        npc.pronouns, npc.ancestry, npc.age,
                        _opt_text(npc.current_location_id),
                        _opt_text(npc.home_location_id),
                        named["aliases_json"], named["tags_json"],
                        _dump(npc.behavior.model_dump(mode="json")),
                        _dump(list(npc.public_history)),
                        _dump(list(npc.private_history)),
                        (
                            _dump(npc.stat_block.model_dump(mode="json"))
                            if npc.stat_block else None
                        ),
                        named["provenance_json"],
                    ),
                )
            bump("sourcebook_npc", len(book.npcs))

            membership = 0
            for npc in book.npcs:
                for order, faction_id in enumerate(npc.faction_ids):
                    await db.execute(
                        """
                        INSERT INTO sourcebook_npc_faction
                            (sourcebook_key, npc_id, faction_id,
                             membership_role, sort_order)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (key, str(npc.id), str(faction_id), ROLE_MEMBER, order),
                    )
                    membership += 1
            for faction in book.factions:
                for role, ids in (
                    (ROLE_LEADER, faction.leader_ids),
                    (ROLE_NOTABLE, faction.notable_member_ids),
                ):
                    for order, npc_id in enumerate(ids):
                        await db.execute(
                            """
                            INSERT INTO sourcebook_npc_faction
                                (sourcebook_key, npc_id, faction_id,
                                 membership_role, sort_order)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (key, str(npc_id), str(faction.id), role, order),
                        )
                        membership += 1
            bump("sourcebook_npc_faction", membership)

            inventory = 0
            for npc in book.npcs:
                for order, entry in enumerate(npc.inventory):
                    await db.execute(
                        """
                        INSERT INTO sourcebook_npc_inventory
                            (sourcebook_key, npc_id, item_id, quantity,
                             equipped, hidden, notes, sort_order)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            key, str(npc.id), str(entry.item_id), entry.quantity,
                            int(entry.equipped), int(entry.hidden), entry.notes,
                            order,
                        ),
                    )
                    inventory += 1
            bump("sourcebook_npc_inventory", inventory)

            territory = 0
            for faction in book.factions:
                for order, location_id in enumerate(faction.territory_location_ids):
                    await db.execute(
                        """
                        INSERT INTO sourcebook_faction_territory
                            (sourcebook_key, faction_id, location_id, sort_order)
                        VALUES (?, ?, ?, ?)
                        """,
                        (key, str(faction.id), str(location_id), order),
                    )
                    territory += 1
            bump("sourcebook_faction_territory", territory)

            objectives = 0
            objective_locations = 0
            for order, quest in enumerate(book.quests):
                named = _named_fields(quest)
                await db.execute(
                    """
                    INSERT INTO sourcebook_quest
                        (sourcebook_key, id, sort_order, name, summary, hook,
                         expiry_trigger, aliases_json, tags_json, detail_json,
                         giver_ids_json, reveal_claim_ids_json,
                         reward_item_ids_json, provenance_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key, named["id"], order, named["name"], named["summary"],
                        quest.hook, _opt_text(quest.expiry_trigger),
                        named["aliases_json"], named["tags_json"],
                        _dump({
                            "stakes": list(quest.stakes),
                            "success_consequences": list(quest.success_consequences),
                            "failure_consequences": list(quest.failure_consequences),
                        }),
                        _dump([str(i) for i in quest.giver_ids]),
                        _dump([str(i) for i in quest.reveal_claim_ids]),
                        _dump([str(i) for i in quest.reward_item_ids]),
                        named["provenance_json"],
                    ),
                )
                for obj_order, objective in enumerate(quest.objectives):
                    await db.execute(
                        """
                        INSERT INTO sourcebook_quest_objective
                            (sourcebook_key, id, quest_id, sort_order,
                             description, detail_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            key, str(objective.id), str(quest.id), obj_order,
                            objective.description,
                            _dump({
                                "prerequisite_objective_ids": [
                                    str(i) for i in objective.prerequisite_objective_ids
                                ],
                                "completion_conditions": list(
                                    objective.completion_conditions
                                ),
                                "failure_conditions": list(objective.failure_conditions),
                                "involved_entity_ids": [
                                    str(i) for i in objective.involved_entity_ids
                                ],
                            }),
                        ),
                    )
                    objectives += 1
                    for loc_order, location_id in enumerate(objective.location_ids):
                        await db.execute(
                            """
                            INSERT INTO sourcebook_quest_objective_location
                                (sourcebook_key, objective_id, location_id,
                                 sort_order)
                            VALUES (?, ?, ?, ?)
                            """,
                            (key, str(objective.id), str(location_id), loc_order),
                        )
                        objective_locations += 1
            bump("sourcebook_quest", len(book.quests))
            bump("sourcebook_quest_objective", objectives)
            bump("sourcebook_quest_objective_location", objective_locations)

            for order, relationship in enumerate(book.relationships):
                await db.execute(
                    """
                    INSERT INTO sourcebook_relationship
                        (sourcebook_key, id, sort_order, source_id, target_id,
                         kind, custom_kind, directed, valence, active,
                         public_description, private_description, history_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key, str(relationship.id), order,
                        str(relationship.source_id), str(relationship.target_id),
                        relationship.kind.value, relationship.custom_kind,
                        int(relationship.directed), relationship.valence,
                        int(relationship.active),
                        relationship.public_description,
                        relationship.private_description,
                        _dump([
                            beat.model_dump(mode="json")
                            for beat in relationship.history
                        ]),
                    ),
                )
            bump("sourcebook_relationship", len(book.relationships))

            for order, event in enumerate(book.timeline):
                await db.execute(
                    """
                    INSERT INTO sourcebook_event
                        (sourcebook_key, id, title, date_label, sort_order,
                         list_order, summary, visibility, detail_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key, str(event.id), event.title, event.date_label,
                        event.sort_order, order, event.summary,
                        event.visibility.value,
                        _dump({
                            "participant_ids": [str(i) for i in event.participant_ids],
                            "location_ids": [str(i) for i in event.location_ids],
                            "cause_event_ids": [str(i) for i in event.cause_event_ids],
                            "consequence_ids": [str(i) for i in event.consequence_ids],
                        }),
                    ),
                )
            bump("sourcebook_event", len(book.timeline))

            for order, claim in enumerate(book.claims):
                await db.execute(
                    """
                    INSERT INTO sourcebook_claim
                        (sourcebook_key, id, sort_order, subject_id, claim_text,
                         canon_status, visibility, superseded_by_claim_id,
                         contradiction_group, valid_from_event_id,
                         invalidated_by_event_id, known_by_json, evidence_json,
                         provenance_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key, str(claim.id), order, str(claim.subject_id),
                        claim.text, claim.canon_status.value,
                        claim.visibility.value,
                        # NULL by construction: the v1 contract has no
                        # authored-supersession field. See migration 007.
                        None,
                        claim.contradiction_group,
                        _opt_text(claim.valid_from_event_id),
                        _opt_text(claim.invalidated_by_event_id),
                        _dump([str(i) for i in claim.known_by_ids]),
                        _dump([str(i) for i in claim.evidence_claim_ids]),
                        _dump(claim.provenance.model_dump(mode="json")),
                    ),
                )
            bump("sourcebook_claim", len(book.claims))

            aux = 0
            for kind, records in (
                ("creature", book.creatures),
                ("lore_domain", book.lore_domains),
                ("story_arc", book.story_arcs),
                ("encounter", book.encounters),
            ):
                for order, record in enumerate(records):
                    await db.execute(
                        """
                        INSERT INTO sourcebook_aux_record
                            (sourcebook_key, record_kind, id, name, sort_order,
                             payload_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            key, kind, str(record.id), record.name, order,
                            _dump(record.model_dump(mode="json")),
                        ),
                    )
                    aux += 1
            bump("sourcebook_aux_record", aux)

        receipt = ImportReceipt(
            sourcebook_key=key,
            sourcebook_id=str(meta.sourcebook_id),
            already_imported=False,
            row_counts=counts,
        )
        logger.info(
            "sourcebook_imported",
            sourcebook=str(meta.sourcebook_id),
            key=key[:12],
            rows=receipt.total_rows,
            replaced=bool(existing),
        )
        return receipt

    async def delete_book(self, sourcebook_key: str) -> bool:
        """Drop a version and everything that cascades from it."""
        db = await self._get_db()
        cursor = await db.execute(
            "DELETE FROM sourcebook WHERE sourcebook_key = ?", (sourcebook_key,)
        )
        await db.commit()
        return bool(getattr(cursor, "rowcount", 0))

    # ==================== Headers ====================

    async def get_header(self, sourcebook_key: str) -> SourcebookHeader | None:
        db = await self._get_db()
        row = await db.fetch_one(
            """
            SELECT sourcebook_key, sourcebook_id, schema_version, title,
                   pitch, ruleset, imported_at
            FROM sourcebook WHERE sourcebook_key = ?
            """,
            (sourcebook_key,),
        )
        return self._row_to_header(row) if row else None

    async def list_books(self) -> list[SourcebookHeader]:
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT sourcebook_key, sourcebook_id, schema_version, title,
                   pitch, ruleset, imported_at
            FROM sourcebook ORDER BY imported_at, sourcebook_id
            """
        )
        return [self._row_to_header(row) for row in rows]

    def _row_to_header(self, row: Sequence[Any]) -> SourcebookHeader:
        return SourcebookHeader(
            sourcebook_key=_text(row[0]),
            sourcebook_id=_text(row[1]),
            schema_version=_text(row[2]) or "1.0",
            title=_text(row[3]),
            pitch=_text(row[4]),
            ruleset=_text(row[5]),
            imported_at=_text(row[6]),
        )

    # ==================== Rebuild source ====================

    async def load_book(self, sourcebook_key: str) -> CampaignSourcebook | None:
        """Reconstruct the authored book from canonical rows.

        The rebuild path's foundation, and the assertion that makes the graph
        genuinely disposable: the returned book compares EQUAL to the one
        imported, so recompiling it yields the same projection as compiling
        the original file. Validation runs again on the way out — a round
        trip that lost a reference fails here rather than silently shipping a
        thinner world.
        """
        db = await self._get_db()
        header = await db.fetch_one(
            """
            SELECT sourcebook_id, schema_version, title, pitch, ruleset,
                   metadata_json, starting_state_json
            FROM sourcebook WHERE sourcebook_key = ?
            """,
            (sourcebook_key,),
        )
        if not header:
            return None

        metadata: dict[str, Any] = {
            "sourcebook_id": _text(header[0]),
            "schema_version": _text(header[1]) or "1.0",
            "title": _text(header[2]),
            "pitch": _text(header[3]),
            "ruleset": _text(header[4]),
            **_load_dict(header[5]),
        }

        payload: dict[str, Any] = {
            "metadata": metadata,
            "starting_state": _load_dict(header[6]),
            "locations": await self._load_locations(sourcebook_key),
            "routes": await self._load_routes(sourcebook_key),
            "npcs": await self._load_npcs(sourcebook_key),
            "factions": await self._load_factions(sourcebook_key),
            "items": await self._load_items(sourcebook_key),
            "relationships": await self._load_relationships(sourcebook_key),
            "claims": [c.model_dump(mode="json") for c in
                       await self._load_claims(sourcebook_key)],
            "timeline": await self._load_timeline(sourcebook_key),
            "quests": await self._load_quests(sourcebook_key),
        }
        for kind in _AUX_KINDS:
            payload[f"{kind}s"] = await self._load_aux(sourcebook_key, kind)
        return CampaignSourcebook.model_validate(payload)

    async def _load_locations(self, key: str) -> list[dict[str, Any]]:
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT id, name, summary, aliases_json, tags_json, provenance_json,
                   location_kind, parent_location_id, description, detail_json
            FROM sourcebook_location WHERE sourcebook_key = ? ORDER BY sort_order
            """,
            (key,),
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = _named_payload(row)
            payload.update({
                "location_kind": _text(row[6]),
                "parent_location_id": _opt_text(row[7]),
                "description": _text(row[8]),
            })
            payload.update(_load_dict(row[9]))
            out.append(payload)
        return out

    async def _load_routes(self, key: str) -> list[dict[str, Any]]:
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT id, from_location_id, to_location_id, bidirectional,
                   travel_time, distance, description, detail_json
            FROM sourcebook_route WHERE sourcebook_key = ? ORDER BY sort_order
            """,
            (key,),
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            payload: dict[str, Any] = {
                "id": _text(row[0]),
                "from_location_id": _text(row[1]),
                "to_location_id": _text(row[2]),
                "bidirectional": _flag(row[3]),
                "travel_time": _text(row[4]),
                "distance": _text(row[5]),
                "description": _text(row[6]),
            }
            payload.update(_load_dict(row[7]))
            out.append(payload)
        return out

    async def _load_npcs(self, key: str) -> list[dict[str, Any]]:
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT id, name, summary, aliases_json, tags_json, provenance_json,
                   status, role, appearance, pronouns, ancestry, age,
                   current_location_id, home_location_id, behavior_json,
                   public_history_json, private_history_json, stat_block_json
            FROM sourcebook_npc WHERE sourcebook_key = ? ORDER BY sort_order
            """,
            (key,),
        )
        factions = await self._membership_by_npc(key, ROLE_MEMBER)
        inventory = await self._inventory_by_npc(key)
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = _named_payload(row)
            npc_id = payload["id"]
            payload.update({
                "status": _text(row[6]),
                "role": _text(row[7]),
                "appearance": _text(row[8]),
                "pronouns": _text(row[9]),
                "ancestry": _text(row[10]),
                "age": _text(row[11]),
                "current_location_id": _opt_text(row[12]),
                "home_location_id": _opt_text(row[13]),
                "behavior": _load_dict(row[14]),
                "public_history": _load_list(row[15]),
                "private_history": _load_list(row[16]),
                "stat_block": _load_dict(row[17]) if row[17] else None,
                "faction_ids": factions.get(npc_id, []),
                "inventory": inventory.get(npc_id, []),
            })
            out.append(payload)
        return out

    async def _membership_by_npc(
        self, key: str, role: str
    ) -> dict[str, list[str]]:
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT npc_id, faction_id FROM sourcebook_npc_faction
            WHERE sourcebook_key = ? AND membership_role = ?
            ORDER BY npc_id, sort_order
            """,
            (key, role),
        )
        out: dict[str, list[str]] = {}
        for row in rows:
            out.setdefault(_text(row[0]), []).append(_text(row[1]))
        return out

    async def _membership_by_faction(
        self, key: str, role: str
    ) -> dict[str, list[str]]:
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT faction_id, npc_id FROM sourcebook_npc_faction
            WHERE sourcebook_key = ? AND membership_role = ?
            ORDER BY faction_id, sort_order
            """,
            (key, role),
        )
        out: dict[str, list[str]] = {}
        for row in rows:
            out.setdefault(_text(row[0]), []).append(_text(row[1]))
        return out

    async def _inventory_by_npc(self, key: str) -> dict[str, list[dict[str, Any]]]:
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT npc_id, item_id, quantity, equipped, hidden, notes
            FROM sourcebook_npc_inventory WHERE sourcebook_key = ?
            ORDER BY npc_id, sort_order
            """,
            (key,),
        )
        out: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            out.setdefault(_text(row[0]), []).append({
                "item_id": _text(row[1]),
                "quantity": int(row[2]),
                "equipped": _flag(row[3]),
                "hidden": _flag(row[4]),
                "notes": _text(row[5]),
            })
        return out

    async def _load_factions(self, key: str) -> list[dict[str, Any]]:
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT id, name, summary, aliases_json, tags_json, provenance_json,
                   headquarters_id, profile_json
            FROM sourcebook_faction WHERE sourcebook_key = ? ORDER BY sort_order
            """,
            (key,),
        )
        leaders = await self._membership_by_faction(key, ROLE_LEADER)
        notables = await self._membership_by_faction(key, ROLE_NOTABLE)
        territory = await self._territory_by_faction(key)
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = _named_payload(row)
            faction_id = payload["id"]
            payload.update({
                "headquarters_id": _opt_text(row[6]),
                "leader_ids": leaders.get(faction_id, []),
                "notable_member_ids": notables.get(faction_id, []),
                "territory_location_ids": territory.get(faction_id, []),
            })
            payload.update(_load_dict(row[7]))
            out.append(payload)
        return out

    async def _territory_by_faction(self, key: str) -> dict[str, list[str]]:
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT faction_id, location_id FROM sourcebook_faction_territory
            WHERE sourcebook_key = ? ORDER BY faction_id, sort_order
            """,
            (key,),
        )
        out: dict[str, list[str]] = {}
        for row in rows:
            out.setdefault(_text(row[0]), []).append(_text(row[1]))
        return out

    async def _load_items(self, key: str) -> list[dict[str, Any]]:
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT id, name, summary, aliases_json, tags_json, provenance_json,
                   category, description, significance, attunement, charges,
                   is_unique, default_location_id, detail_json
            FROM sourcebook_item WHERE sourcebook_key = ? ORDER BY sort_order
            """,
            (key,),
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = _named_payload(row)
            payload.update({
                "category": _text(row[6]),
                "description": _text(row[7]),
                "significance": _text(row[8]),
                "attunement": _text(row[9]),
                "charges": _opt_int(row[10]),
                "unique": _flag(row[11]),
                "default_location_id": _opt_text(row[12]),
            })
            payload.update(_load_dict(row[13]))
            out.append(payload)
        return out

    async def _load_relationships(self, key: str) -> list[dict[str, Any]]:
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT id, source_id, target_id, kind, custom_kind, directed,
                   valence, active, public_description, private_description,
                   history_json
            FROM sourcebook_relationship WHERE sourcebook_key = ?
            ORDER BY sort_order
            """,
            (key,),
        )
        return [
            {
                "id": _text(row[0]),
                "source_id": _text(row[1]),
                "target_id": _text(row[2]),
                "kind": _text(row[3]),
                "custom_kind": _opt_text(row[4]),
                "directed": _flag(row[5]),
                "valence": _opt_int(row[6]),
                "active": _flag(row[7]),
                "public_description": _text(row[8]),
                "private_description": _text(row[9]),
                "history": _load_list(row[10]),
            }
            for row in rows
        ]

    async def _load_timeline(self, key: str) -> list[dict[str, Any]]:
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT id, title, date_label, sort_order, summary, visibility,
                   detail_json
            FROM sourcebook_event WHERE sourcebook_key = ? ORDER BY list_order
            """,
            (key,),
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            payload: dict[str, Any] = {
                "id": _text(row[0]),
                "title": _text(row[1]),
                "date_label": _text(row[2]),
                "sort_order": int(row[3]),
                "summary": _text(row[4]),
                "visibility": _text(row[5]),
            }
            payload.update(_load_dict(row[6]))
            out.append(payload)
        return out

    async def _load_quests(self, key: str) -> list[dict[str, Any]]:
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT id, name, summary, aliases_json, tags_json, provenance_json,
                   hook, expiry_trigger, detail_json, giver_ids_json,
                   reveal_claim_ids_json, reward_item_ids_json
            FROM sourcebook_quest WHERE sourcebook_key = ? ORDER BY sort_order
            """,
            (key,),
        )
        objectives = await self._objectives_by_quest(key)
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = _named_payload(row)
            payload.update({
                "hook": _text(row[6]),
                "expiry_trigger": _opt_text(row[7]),
                "giver_ids": _load_list(row[9]),
                "reveal_claim_ids": _load_list(row[10]),
                "reward_item_ids": _load_list(row[11]),
                "objectives": objectives.get(payload["id"], []),
            })
            payload.update(_load_dict(row[8]))
            out.append(payload)
        return out

    async def _objectives_by_quest(
        self, key: str
    ) -> dict[str, list[dict[str, Any]]]:
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT quest_id, id, description, detail_json
            FROM sourcebook_quest_objective WHERE sourcebook_key = ?
            ORDER BY quest_id, sort_order
            """,
            (key,),
        )
        locations = await self._locations_by_objective(key)
        out: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            payload: dict[str, Any] = {
                "id": _text(row[1]),
                "description": _text(row[2]),
                "location_ids": locations.get(_text(row[1]), []),
            }
            payload.update(_load_dict(row[3]))
            out.setdefault(_text(row[0]), []).append(payload)
        return out

    async def _locations_by_objective(self, key: str) -> dict[str, list[str]]:
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT objective_id, location_id
            FROM sourcebook_quest_objective_location WHERE sourcebook_key = ?
            ORDER BY objective_id, sort_order
            """,
            (key,),
        )
        out: dict[str, list[str]] = {}
        for row in rows:
            out.setdefault(_text(row[0]), []).append(_text(row[1]))
        return out

    async def _load_aux(self, key: str, record_kind: str) -> list[dict[str, Any]]:
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT payload_json FROM sourcebook_aux_record
            WHERE sourcebook_key = ? AND record_kind = ? ORDER BY sort_order
            """,
            (key, record_kind),
        )
        return [_load_dict(row[0]) for row in rows]

    # ==================== Campaign binding ====================

    async def bind_campaign(self, campaign_id: str, sourcebook_key: str) -> None:
        """Point a campaign's overlay at a book version."""
        db = await self._get_db()
        await db.execute(
            """
            INSERT INTO campaign_sourcebook (campaign_id, sourcebook_key)
            VALUES (?, ?)
            ON CONFLICT(campaign_id, sourcebook_key) DO NOTHING
            """,
            (campaign_id, sourcebook_key),
        )
        await db.commit()

    async def unbind_campaign(self, campaign_id: str, sourcebook_key: str) -> bool:
        db = await self._get_db()
        cursor = await db.execute(
            """
            DELETE FROM campaign_sourcebook
            WHERE campaign_id = ? AND sourcebook_key = ?
            """,
            (campaign_id, sourcebook_key),
        )
        await db.commit()
        return bool(getattr(cursor, "rowcount", 0))

    async def sourcebook_keys_for_campaign(self, campaign_id: str) -> list[str]:
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT sourcebook_key FROM campaign_sourcebook
            WHERE campaign_id = ? ORDER BY applied_at, sourcebook_key
            """,
            (campaign_id,),
        )
        return [_text(row[0]) for row in rows]

    # ==================== Overlay writes ====================

    async def record_discovery(
        self,
        campaign_id: str,
        sourcebook_key: str,
        claim_id: str,
        *,
        turn: int = 0,
        via: str = "",
    ) -> bool:
        """Mark a claim as earned by this party, at this turn.

        First discovery wins: ``discovered_at_turn`` answers "when did they
        learn it", so a later re-discovery must not rewrite the answer.
        """
        db = await self._get_db()
        cursor = await db.execute(
            """
            INSERT INTO campaign_claim_state
                (campaign_id, sourcebook_key, claim_id, discovered,
                 discovered_at_turn, discovered_via)
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(campaign_id, sourcebook_key, claim_id) DO UPDATE SET
                discovered = 1,
                discovered_at_turn = COALESCE(
                    campaign_claim_state.discovered_at_turn, excluded.discovered_at_turn
                ),
                discovered_via = CASE
                    WHEN campaign_claim_state.discovered = 1
                        THEN campaign_claim_state.discovered_via
                    ELSE excluded.discovered_via
                END,
                updated_at = CURRENT_TIMESTAMP
            """,
            (campaign_id, sourcebook_key, claim_id, turn, via),
        )
        await db.commit()
        return bool(getattr(cursor, "rowcount", 0))

    async def seed_starting_knowledge(
        self, campaign_id: str, sourcebook_key: str
    ) -> int:
        """Grant the claims ``starting_state`` says the party already knows."""
        db = await self._get_db()
        row = await db.fetch_one(
            "SELECT starting_state_json FROM sourcebook WHERE sourcebook_key = ?",
            (sourcebook_key,),
        )
        if not row:
            return 0
        claim_ids = _load_dict(row[0]).get("player_known_claim_ids") or []
        granted = 0
        for claim_id in claim_ids:
            if await self.record_discovery(
                campaign_id, sourcebook_key, str(claim_id),
                turn=0, via="starting_state",
            ):
                granted += 1
        return granted

    async def supersede_claim(
        self,
        campaign_id: str,
        sourcebook_key: str,
        claim_id: str,
        superseded_by_claim_id: str,
        *,
        canon_status: CanonStatus | None = None,
        note: str = "",
    ) -> bool:
        """Record that play overturned a claim.

        Writes the OVERLAY, never the book: canon stays immutable, and the
        campaign carries the correction. ``canon_status`` optionally demotes
        the old claim (typically to ``FALSE`` or ``DISPUTED``) so retrieval
        stops asserting it as settled truth.
        """
        db = await self._get_db()
        cursor = await db.execute(
            """
            INSERT INTO campaign_claim_state
                (campaign_id, sourcebook_key, claim_id, superseded_by_claim_id,
                 canon_status, note)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(campaign_id, sourcebook_key, claim_id) DO UPDATE SET
                superseded_by_claim_id = excluded.superseded_by_claim_id,
                canon_status = COALESCE(
                    excluded.canon_status, campaign_claim_state.canon_status
                ),
                note = excluded.note,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                campaign_id, sourcebook_key, claim_id, superseded_by_claim_id,
                canon_status.value if canon_status else None, note,
            ),
        )
        await db.commit()
        return bool(getattr(cursor, "rowcount", 0))

    async def record_visit(
        self,
        campaign_id: str,
        sourcebook_key: str,
        location_id: str,
        *,
        turn: int = 0,
    ) -> bool:
        """Mark an authored location as touched by this party."""
        db = await self._get_db()
        cursor = await db.execute(
            """
            INSERT INTO campaign_location_state
                (campaign_id, sourcebook_key, location_id, visited,
                 first_visited_turn, last_visited_turn)
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(campaign_id, sourcebook_key, location_id) DO UPDATE SET
                visited = 1,
                first_visited_turn = COALESCE(
                    campaign_location_state.first_visited_turn,
                    excluded.first_visited_turn
                ),
                last_visited_turn = excluded.last_visited_turn,
                updated_at = CURRENT_TIMESTAMP
            """,
            (campaign_id, sourcebook_key, location_id, turn, turn),
        )
        await db.commit()
        return bool(getattr(cursor, "rowcount", 0))

    # ==================== Claim queries ====================

    _CLAIM_COLUMNS = """
        c.sourcebook_key, c.id, c.subject_id, c.claim_text, c.canon_status,
        c.visibility, c.superseded_by_claim_id, c.contradiction_group,
        c.valid_from_event_id, c.invalidated_by_event_id, c.known_by_json,
        c.evidence_json, c.provenance_json
    """

    def _row_to_claim(self, row: Sequence[Any]) -> KnowledgeClaim:
        """Map the ``_CLAIM_COLUMNS`` run, which every claim query selects first.

        Unpacked by name rather than indexed: the overlay queries append
        their own columns after this run, and off-by-one there would produce
        a claim whose text is somebody else's.
        """
        (
            _key, claim_id, subject_id, text, canon_status, visibility,
            _superseded, contradiction_group, valid_from, invalidated_by,
            known_by, evidence, provenance,
        ) = tuple(row)[:13]
        return KnowledgeClaim(
            id=_text(claim_id),
            subject_id=_text(subject_id),
            text=_text(text),
            canon_status=CanonStatus(_text(canon_status)),
            visibility=Visibility(_text(visibility)),
            known_by_ids=_load_list(known_by),
            evidence_claim_ids=_load_list(evidence),
            valid_from_event_id=_opt_text(valid_from),
            invalidated_by_event_id=_opt_text(invalidated_by),
            contradiction_group=_opt_text(contradiction_group),
            provenance=Provenance.model_validate(_load_dict(provenance)),
        )

    async def _load_claims(self, key: str) -> list[KnowledgeClaim]:
        db = await self._get_db()
        rows = await db.fetch_all(
            f"SELECT {self._CLAIM_COLUMNS} FROM sourcebook_claim c "
            "WHERE c.sourcebook_key = ? ORDER BY c.sort_order",
            (key,),
        )
        return [self._row_to_claim(row) for row in rows]

    async def undiscovered_claims(
        self,
        campaign_id: str,
        *,
        visibility: Visibility = Visibility.DISCOVERABLE,
        subject_id: str | None = None,
    ) -> list[CampaignClaim]:
        """Which claims of this visibility has the party not yet earned.

        The query the design doc names first, and the one a document blob
        cannot answer without loading and scanning the whole book every turn.
        Defaults to DISCOVERABLE — PUBLIC claims are not "earned", they are
        simply true out loud, and DM_ONLY ones are not on offer at all.
        """
        db = await self._get_db()
        params: list[Any] = [campaign_id, visibility.value]
        subject_clause = ""
        if subject_id:
            subject_clause = "AND c.subject_id = ?"
            params.append(subject_id)
        rows = await db.fetch_all(
            f"""
            SELECT {self._CLAIM_COLUMNS}
            FROM sourcebook_claim c
            JOIN campaign_sourcebook cs
              ON cs.sourcebook_key = c.sourcebook_key
            LEFT JOIN campaign_claim_state s
              ON s.campaign_id = cs.campaign_id
             AND s.sourcebook_key = c.sourcebook_key
             AND s.claim_id = c.id
            WHERE cs.campaign_id = ?
              AND c.visibility = ?
              {subject_clause}
              AND COALESCE(s.discovered, 0) = 0
            ORDER BY c.sourcebook_key, c.sort_order
            """,
            tuple(params),
        )
        return [
            CampaignClaim(
                sourcebook_key=_text(row[0]),
                claim=self._row_to_claim(row),
                effective_canon_status=CanonStatus(_text(row[4])),
            )
            for row in rows
        ]

    async def effective_claims(
        self,
        campaign_id: str,
        *,
        visibilities: Iterable[Visibility] | None = None,
        include_superseded: bool = True,
        discovered_only: bool = False,
    ) -> list[CampaignClaim]:
        """Canon as this campaign now stands, overlay resolved over book.

        ``effective_canon_status`` is the campaign's override when it set
        one, otherwise the book's — so a claim the party disproved reads as
        FALSE here without the immutable book having changed.
        """
        db = await self._get_db()
        params: list[Any] = [campaign_id]
        clauses: list[str] = []
        if visibilities is not None:
            values = [v.value for v in visibilities]
            if not values:
                return []
            clauses.append(
                f"AND c.visibility IN ({','.join('?' for _ in values)})"
            )
            params.extend(values)
        if discovered_only:
            clauses.append("AND COALESCE(s.discovered, 0) = 1")
        if not include_superseded:
            clauses.append(
                "AND COALESCE(s.superseded_by_claim_id, c.superseded_by_claim_id) "
                "IS NULL"
            )
        rows = await db.fetch_all(
            f"""
            SELECT {self._CLAIM_COLUMNS},
                   COALESCE(s.discovered, 0),
                   s.discovered_at_turn,
                   COALESCE(s.discovered_via, ''),
                   COALESCE(s.superseded_by_claim_id, c.superseded_by_claim_id),
                   COALESCE(s.canon_status, c.canon_status),
                   COALESCE(s.note, '')
            FROM sourcebook_claim c
            JOIN campaign_sourcebook cs
              ON cs.sourcebook_key = c.sourcebook_key
            LEFT JOIN campaign_claim_state s
              ON s.campaign_id = cs.campaign_id
             AND s.sourcebook_key = c.sourcebook_key
             AND s.claim_id = c.id
            WHERE cs.campaign_id = ?
              {' '.join(clauses)}
            ORDER BY c.sourcebook_key, c.sort_order
            """,
            tuple(params),
        )
        return [
            CampaignClaim(
                sourcebook_key=_text(row[0]),
                claim=self._row_to_claim(row),
                discovered=_flag(row[13]),
                discovered_at_turn=_opt_int(row[14]),
                discovered_via=_text(row[15]),
                superseded_by_claim_id=_opt_text(row[16]),
                effective_canon_status=CanonStatus(_text(row[17])),
                note=_text(row[18]),
            )
            for row in rows
        ]

    async def discovery_log(self, campaign_id: str) -> list[CampaignClaim]:
        """What did the party learn, and when — canon joined to the overlay."""
        discovered = await self.effective_claims(
            campaign_id, discovered_only=True
        )
        return sorted(
            discovered,
            key=lambda c: (
                c.discovered_at_turn if c.discovered_at_turn is not None else 0,
                c.claim_id,
            ),
        )

    # ==================== Entity queries ====================

    async def faction_members(
        self, sourcebook_key: str, faction_id: str
    ) -> list[FactionMember]:
        """Every NPC the book places in a faction, however it said so."""
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT f.faction_id, f.npc_id, n.name, f.membership_role, n.status
            FROM sourcebook_npc_faction f
            JOIN sourcebook_npc n
              ON n.sourcebook_key = f.sourcebook_key AND n.id = f.npc_id
            WHERE f.sourcebook_key = ? AND f.faction_id = ?
            ORDER BY f.membership_role, f.sort_order, f.npc_id
            """,
            (sourcebook_key, faction_id),
        )
        return [
            FactionMember(
                faction_id=_text(row[0]),
                npc_id=_text(row[1]),
                npc_name=_text(row[2]),
                membership_role=_text(row[3]),
                status=_text(row[4]),
            )
            for row in rows
        ]

    async def ties_to(
        self,
        sourcebook_key: str,
        entity_id: str,
        *,
        kinds: Iterable[RelationshipKind] | None = None,
        active_only: bool = True,
    ) -> list[AuthoredTie]:
        """Ties pointing AT an entity, in the book's own vocabulary.

        An undirected tie counts in both directions — the schema's way of
        saying "these two are mutually X". Missing that would silently halve
        the answer to "who is hostile to me", which is exactly the kind of
        absence no assertion about the result's contents would catch.
        """
        db = await self._get_db()
        params: list[Any] = [sourcebook_key, entity_id, entity_id]
        clauses = ["r.sourcebook_key = ?",
                   "(r.target_id = ? OR (r.directed = 0 AND r.source_id = ?))"]
        if active_only:
            clauses.append("r.active = 1")
        if kinds is not None:
            values = [k.value for k in kinds]
            if not values:
                return []
            clauses.append(f"r.kind IN ({','.join('?' for _ in values)})")
            params.extend(values)
        rows = await db.fetch_all(
            f"""
            SELECT r.id, r.source_id, r.target_id, r.kind, r.custom_kind,
                   r.directed, r.valence, r.public_description,
                   r.private_description,
                   COALESCE(sn.name, ''), COALESCE(tn.name, '')
            FROM sourcebook_relationship r
            LEFT JOIN sourcebook_npc sn
              ON sn.sourcebook_key = r.sourcebook_key AND sn.id = r.source_id
            LEFT JOIN sourcebook_npc tn
              ON tn.sourcebook_key = r.sourcebook_key AND tn.id = r.target_id
            WHERE {' AND '.join(clauses)}
            ORDER BY r.sort_order
            """,
            tuple(params),
        )
        return [
            AuthoredTie(
                relationship_id=_text(row[0]),
                source_id=_text(row[1]),
                source_name=_text(row[9]),
                target_id=_text(row[2]),
                target_name=_text(row[10]),
                kind=RelationshipKind(_text(row[3])),
                custom_kind=_opt_text(row[4]),
                directed=_flag(row[5]),
                valence=_opt_int(row[6]),
                public_description=_text(row[7]),
                private_description=_text(row[8]),
            )
            for row in rows
        ]

    async def hostile_to(
        self, sourcebook_key: str, entity_id: str
    ) -> list[AuthoredTie]:
        """Every authored antagonist of an entity."""
        return await self.ties_to(
            sourcebook_key, entity_id, kinds=HOSTILE_KINDS
        )

    async def region_contents(
        self,
        sourcebook_key: str,
        region_id: str,
        *,
        campaign_id: str | None = None,
    ) -> RegionContents:
        """Everything authored inside a location subtree.

        Recursive over containment, so asking about a region answers for
        every district, building and room beneath it. With ``campaign_id``
        the result also carries which of those the party has never touched —
        the "untouched region" half of the query.
        """
        db = await self._get_db()
        subtree = """
            WITH RECURSIVE region(id) AS (
                SELECT id FROM sourcebook_location
                 WHERE sourcebook_key = :key AND id = :region
                UNION
                SELECT l.id FROM sourcebook_location l
                  JOIN region ON l.parent_location_id = region.id
                 WHERE l.sourcebook_key = :key
            )
        """
        binds = {"key": sourcebook_key, "region": region_id}

        async def scalars(sql: str) -> list[str]:
            cursor = await db.connection.execute(subtree + sql, binds)
            rows = await cursor.fetchall()
            return [_text(row[0]) for row in rows]

        location_ids = await scalars("SELECT id FROM region ORDER BY id")
        if not location_ids:
            return RegionContents(region_id=region_id)

        npc_ids = await scalars(
            """
            SELECT DISTINCT n.id FROM sourcebook_npc n
            WHERE n.sourcebook_key = :key
              AND (n.current_location_id IN (SELECT id FROM region)
                   OR n.home_location_id IN (SELECT id FROM region))
            ORDER BY n.id
            """
        )
        item_ids = await scalars(
            """
            SELECT DISTINCT i.id FROM sourcebook_item i
            WHERE i.sourcebook_key = :key
              AND i.default_location_id IN (SELECT id FROM region)
            ORDER BY i.id
            """
        )
        quest_ids = await scalars(
            """
            SELECT DISTINCT o.quest_id FROM sourcebook_quest_objective o
            JOIN sourcebook_quest_objective_location ol
              ON ol.sourcebook_key = o.sourcebook_key AND ol.objective_id = o.id
            WHERE o.sourcebook_key = :key
              AND ol.location_id IN (SELECT id FROM region)
            ORDER BY o.quest_id
            """
        )
        faction_ids = await scalars(
            """
            SELECT DISTINCT f.id FROM sourcebook_faction f
            WHERE f.sourcebook_key = :key
              AND (f.headquarters_id IN (SELECT id FROM region)
                   OR EXISTS (
                       SELECT 1 FROM sourcebook_faction_territory t
                       WHERE t.sourcebook_key = f.sourcebook_key
                         AND t.faction_id = f.id
                         AND t.location_id IN (SELECT id FROM region)
                   ))
            ORDER BY f.id
            """
        )

        unvisited = list(location_ids)
        if campaign_id:
            visited_rows = await db.fetch_all(
                """
                SELECT location_id FROM campaign_location_state
                WHERE campaign_id = ? AND sourcebook_key = ? AND visited = 1
                """,
                (campaign_id, sourcebook_key),
            )
            visited = {_text(row[0]) for row in visited_rows}
            unvisited = [lid for lid in location_ids if lid not in visited]

        return RegionContents(
            region_id=region_id,
            location_ids=location_ids,
            npc_ids=npc_ids,
            item_ids=item_ids,
            quest_ids=quest_ids,
            faction_ids=faction_ids,
            unvisited_location_ids=unvisited,
        )


_repo: Optional[SourcebookRepository] = None


async def get_sourcebook_repo() -> SourcebookRepository:
    """Get the global sourcebook repository."""
    global _repo
    if _repo is None:
        _repo = SourcebookRepository()
    return _repo
