"""Project an authored sourcebook into the live campaign stores.

``dnd_bot/models/sourcebook.py`` is the authoring contract: immutable
canonical input, and the system of record. This module is the compiler its
docstring promised — it turns a :class:`CampaignSourcebook` into knowledge
graph operations plus an opening world-state seed, so a campaign can start
with a populated world instead of one the narrator invents turn by turn.

Two properties matter more than completeness:

**The projection is deliberately lossy, in one direction.** The sourcebook
carries far richer structure than the graph's nine retrieval-shaped relation
types (``parent_of``, ``owes``, ``fears`` and friends all collapse toward
``knows``). That is fine because the book remains the system of record; the
graph exists to answer "what is near what" for context assembly. Nothing
reads the graph expecting authored fidelity.

**Visibility is enforced on EVERY channel, not just claims.** A campaign book
is mostly secrets, and leaked canon is invisible to self-consistency grading
precisely because it is perfectly consistent — so the boundary has to hold
everywhere authored text can reach the narrator, which is more places than it
first appears:

- claims: only PUBLIC/PLAYER_KNOWN, and only ``CanonStatus.CANON`` — a claim
  can be public and *false*, and asserting a legend as fact is a different
  bug with the same shape;
- quests: the player-facing ``hook``, never the ``summary`` where an author
  writes the twist, and only quests ``starting_state`` says are active;
- relationships: a tie authored with only a ``private_description`` is the
  schema's way of saying nobody knows it, so it becomes no edge;
- inventory: ``hidden`` entries yield no ownership edge, and an item whose
  only presence in the book is a concealed one is not projected at all;
- NPCs: ``private_history`` is never read, and a MISSING character is not
  placed anywhere.

Everything excluded lands in ``withheld`` / ``withheld_notes`` — a bucket
that exists so tests can assert an ABSENCE, which is the only way this class
of failure gets caught at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from ...models.sourcebook import (
    CampaignSourcebook,
    CanonStatus,
    CharacterStatus,
    KnowledgeClaim,
    RelationshipKind,
    Visibility,
)
from .models import (
    AddEdge,
    AddNode,
    Entity,
    EntityType,
    GraphOperation,
    RelationType,
    Relationship,
    slugify,
)

logger = structlog.get_logger()


# The book's relationship vocabulary is richer than the graph's. Map each
# authored kind onto the closest retrieval edge; unmapped social ties become
# KNOWS so the entities stay connected for BFS rather than dropping out.
_RELATION_MAP: dict[RelationshipKind, RelationType] = {
    RelationshipKind.HOSTILE_TO: RelationType.HOSTILE_TO,
    RelationshipKind.RIVAL_OF: RelationType.HOSTILE_TO,
    RelationshipKind.FEARS: RelationType.HOSTILE_TO,
    RelationshipKind.ALLIED_WITH: RelationType.ALLIED_WITH,
    RelationshipKind.FRIEND_OF: RelationType.ALLIED_WITH,
    RelationshipKind.SERVES: RelationType.ALLIED_WITH,
    RelationshipKind.MEMBER_OF: RelationType.ALLIED_WITH,
    RelationshipKind.LEADS: RelationType.ALLIED_WITH,
    RelationshipKind.OWNS: RelationType.OWNS,
    RelationshipKind.CARRIES: RelationType.OWNS,
    RelationshipKind.CONTROLS: RelationType.OWNS,
    RelationshipKind.CREATED: RelationType.OWNS,
    RelationshipKind.LOCATED_AT: RelationType.LOCATED_AT,
    RelationshipKind.CONNECTED_TO: RelationType.CONNECTED_TO,
    RelationshipKind.QUEST_GIVER: RelationType.QUEST_GIVER,
}

# Claims the party may be told. Everything else is the DM's.
_VISIBLE_TO_PLAY = frozenset({Visibility.PUBLIC, Visibility.PLAYER_KNOWN})

# Only settled truth becomes an established fact. A claim can be PUBLIC and
# FALSE (a rumour everyone repeats) or LEGEND (told, not verified); projecting
# either as an established fact asserts it as canon in the narrator's context.
_ASSERTABLE = frozenset({CanonStatus.CANON})

# Sourcebook statuses that mean "not walking around alive".
_NOT_ALIVE = frozenset({CharacterStatus.DEAD, CharacterStatus.UNDEAD})

# Statuses that mean "not standing anywhere the party can find". Placing one
# of these would assert a location the book never claims.
_UNPLACED = frozenset({CharacterStatus.MISSING, CharacterStatus.UNKNOWN})


@dataclass
class CompiledSourcebook:
    """The projection: what the game gets, and what it must not get."""

    graph_ops: list[GraphOperation] = field(default_factory=list)
    current_location: str = ""
    location_description: str = ""
    established_facts: list[str] = field(default_factory=list)
    scene_items: dict[str, str] = field(default_factory=dict)
    opening_situation: str = ""
    # Claims withheld from play — ground truth for tests and for the DM
    # layer, never projected into narrator-visible state.
    withheld: list[KnowledgeClaim] = field(default_factory=list)
    # Everything else the boundary held back (secret ties, concealed items,
    # inactive quests), as "<channel> <id>: <reason>". Assertable absence.
    withheld_notes: list[str] = field(default_factory=list)
    # Non-fatal authoring problems (dangling references the schema's own
    # validators do not cover). Compilation continues around them.
    warnings: list[str] = field(default_factory=list)

    @property
    def node_count(self) -> int:
        return sum(1 for op in self.graph_ops if isinstance(op, AddNode))

    @property
    def edge_count(self) -> int:
        return sum(1 for op in self.graph_ops if isinstance(op, AddEdge))


def _npc_properties(npc) -> dict[str, str]:
    # Player-facing fields first. `summary` is the author's shelf for "what
    # this character IS", twist included, so it is the last resort rather
    # than the first choice. private_history is never read at all.
    description = npc.appearance or npc.role or npc.summary
    properties = {
        "description": description,
        "alive": "false" if npc.status in _NOT_ALIVE else "true",
    }
    if npc.status is not CharacterStatus.ALIVE:
        properties["status"] = npc.status.value
    if npc.role:
        properties["role"] = npc.role
    # No disposition: the book models behavior as values/goals/fears rather
    # than a single hostility dial, and inventing one here would hand the
    # narrator an authored-looking fact nobody wrote.
    return {k: v for k, v in properties.items() if v}


def compile_sourcebook(
    book: CampaignSourcebook,
    campaign_id: str,
) -> CompiledSourcebook:
    """Compile an authored book into graph ops and an opening world seed.

    Pure: performs no I/O and mutates nothing. Callers apply the result.
    """
    out = CompiledSourcebook()
    known_ids: set[str] = set()

    def _add_node(spec, entity_type: EntityType, properties: dict[str, str]) -> None:
        known_ids.add(spec.id)
        out.graph_ops.append(AddNode(entity=Entity(
            node_id=spec.id,
            entity_type=entity_type,
            name=spec.name,
            campaign_id=campaign_id,
            aliases=list(spec.aliases),
            properties={k: v for k, v in properties.items() if v},
        )))

    # An item whose ONLY presence in the book is a concealed inventory entry
    # is not something the world can show yet; projecting a described node
    # for it hands the narrator the secret without the ownership edge.
    concealed_only: set[str] = set()
    for item in book.items:
        entries = [
            entry
            for npc in book.npcs
            for entry in npc.inventory
            if entry.item_id == item.id
        ]
        if entries and all(e.hidden for e in entries) and not item.default_location_id:
            concealed_only.add(item.id)

    active_quests = set(book.starting_state.active_quest_ids)

    for location in book.locations:
        _add_node(location, EntityType.LOCATION, {
            "description": location.description or location.summary,
            "kind": location.location_kind.value,
        })
        if location.id != slugify(location.name):
            out.warnings.append(
                f"location {location.id!r} id is not slugify(name) "
                f"({slugify(location.name)!r}) — code that resolves a location "
                "by slugified name will fork a second node for it"
            )

    seen_npc_names: dict[str, str] = {}
    for npc in book.npcs:
        _add_node(npc, EntityType.NPC, _npc_properties(npc))
        key = npc.name.strip().casefold()
        if key in seen_npc_names:
            out.warnings.append(
                f"npc {npc.id!r} shares the name {npc.name!r} with "
                f"{seen_npc_names[key]!r} — the graph MERGES same-named NPCs, "
                "so one of them will be destroyed on apply; give them "
                "distinguishing names"
            )
        else:
            seen_npc_names[key] = npc.id

    for item in book.items:
        if item.id in concealed_only:
            out.withheld_notes.append(
                f"item {item.id}: only ever held as a hidden inventory entry"
            )
            continue
        _add_node(item, EntityType.ITEM, {
            "description": item.description or item.summary,
            "category": item.category,
        })

    for quest in book.quests:
        if quest.id not in active_quests:
            out.withheld_notes.append(
                f"quest {quest.id}: not listed in starting_state.active_quest_ids"
            )
            continue
        # `hook` is what the party could have heard. `summary` is where the
        # author writes the answer, and it is NOT projected.
        _add_node(quest, EntityType.QUEST, {"description": quest.hook})
    # Factions and lore domains have no graph entity type. They stay in the
    # book; membership still projects as ALLIED_WITH edges between NPCs and
    # whatever faction node a future schema version introduces.

    edges: list[tuple[str, str, RelationType]] = []

    def _edge(source: str, target: str, relation: RelationType, why: str) -> None:
        if source not in known_ids or target not in known_ids:
            out.warnings.append(
                f"{why}: dangling reference {source!r} -> {target!r}"
            )
            return
        edges.append((source, target, relation))

    for npc in book.npcs:
        if npc.status in _UNPLACED:
            out.withheld_notes.append(
                f"npc {npc.id}: status {npc.status.value} — not placed anywhere"
            )
        elif npc.current_location_id:
            _edge(npc.id, npc.current_location_id, RelationType.LOCATED_AT,
                  f"npc {npc.id} current_location")
        elif npc.home_location_id:
            _edge(npc.id, npc.home_location_id, RelationType.LOCATED_AT,
                  f"npc {npc.id} home_location")
        for entry in npc.inventory:
            item_id = getattr(entry, "item_id", "") or ""
            if not item_id:
                continue
            if entry.hidden:
                # A concealed possession must not be published as ownership:
                # the edge alone answers "who has the forged deed".
                out.withheld_notes.append(
                    f"inventory {npc.id}->{item_id}: hidden"
                )
                continue
            if item_id in concealed_only:
                continue
            _edge(npc.id, item_id, RelationType.OWNS,
                  f"npc {npc.id} inventory")

    for item in book.items:
        if item.default_location_id:
            _edge(item.id, item.default_location_id, RelationType.FOUND_AT,
                  f"item {item.id} default_location")

    for location in book.locations:
        if location.parent_location_id:
            _edge(location.id, location.parent_location_id,
                  RelationType.CONNECTED_TO, f"location {location.id} parent")

    for route in book.routes:
        _edge(route.from_location_id, route.to_location_id,
              RelationType.CONNECTED_TO, f"route {route.id}")
        if route.bidirectional:
            _edge(route.to_location_id, route.from_location_id,
                  RelationType.CONNECTED_TO, f"route {route.id} (reverse)")

    for relationship in book.relationships:
        if not relationship.active:
            continue
        # A tie the author described ONLY privately is a secret tie. The
        # collapse onto retrieval types makes publishing it worse, not
        # merely lossy: a covert chain of command (SERVES) would surface as
        # a plain alliance, disclosing the conspiracy in the first retrieval.
        if relationship.private_description and not relationship.public_description:
            out.withheld_notes.append(
                f"relationship {relationship.id}: private_description only"
            )
            continue
        relation = _RELATION_MAP.get(relationship.kind, RelationType.KNOWS)
        _edge(relationship.source_id, relationship.target_id, relation,
              f"relationship {relationship.id}")
        if not relationship.directed:
            _edge(relationship.target_id, relationship.source_id, relation,
                  f"relationship {relationship.id} (reverse)")

    for quest in book.quests:
        for giver_id in quest.giver_ids:
            _edge(giver_id, quest.id, RelationType.QUEST_GIVER,
                  f"quest {quest.id} giver")
        for objective in quest.objectives:
            for location_id in objective.location_ids:
                _edge(quest.id, location_id, RelationType.OBJECTIVE_AT,
                      f"quest {quest.id} objective {objective.id}")

    for source, target, relation in edges:
        out.graph_ops.append(AddEdge(relationship=Relationship(
            source_id=source, target_id=target,
            relation_type=relation, campaign_id=campaign_id,
        )))

    # ── Opening scene ────────────────────────────────────────────────────
    start = book.starting_state
    out.opening_situation = start.opening_situation
    origin = next((l for l in book.locations if l.id == start.location_id), None)
    if origin is None:
        out.warnings.append(
            f"starting_state.location_id {start.location_id!r} is not a location"
        )
    else:
        out.current_location = origin.name
        out.location_description = origin.description or origin.summary
        for item in book.items:
            if item.default_location_id == origin.id:
                out.scene_items[item.name] = item.description or item.summary

    # ── Claims: the visibility boundary ──────────────────────────────────
    granted = set(start.player_known_claim_ids)
    for claim in book.claims:
        visible = claim.visibility in _VISIBLE_TO_PLAY or claim.id in granted
        if not visible:
            out.withheld.append(claim)
        elif claim.canon_status not in _ASSERTABLE:
            # Public but not settled truth. Stating a rumour or a known
            # falsehood as an established fact makes it canon in the
            # narrator's context, which is the same failure wearing a
            # different hat.
            out.withheld.append(claim)
            out.withheld_notes.append(
                f"claim {claim.id}: visible but canon_status="
                f"{claim.canon_status.value}"
            )
        else:
            out.established_facts.append(claim.text)

    logger.info(
        "sourcebook_compiled",
        sourcebook=book.metadata.sourcebook_id,
        nodes=out.node_count,
        edges=out.edge_count,
        facts=len(out.established_facts),
        withheld_claims=len(out.withheld),
        withheld_other=len(out.withheld_notes),
        warnings=len(out.warnings),
    )
    return out
