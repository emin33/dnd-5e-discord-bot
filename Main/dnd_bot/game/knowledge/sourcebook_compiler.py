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

**Visibility is enforced at the boundary.** Only PUBLIC and PLAYER_KNOWN
claims — plus whatever ``starting_state`` explicitly grants — reach world
state. DM_ONLY and DISCOVERABLE claims are compiled into a separate
``withheld`` bucket that never touches the narrator's context. A campaign
book is full of secrets the party has to earn, and leaked canon is invisible
to self-consistency grading precisely because it is perfectly consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from ...models.sourcebook import (
    CampaignSourcebook,
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

# Sourcebook statuses that mean "not walking around alive".
_NOT_ALIVE = frozenset({CharacterStatus.DEAD, CharacterStatus.UNDEAD})


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
    description = npc.appearance or npc.summary or npc.role
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

    for location in book.locations:
        _add_node(location, EntityType.LOCATION, {
            "description": location.description or location.summary,
            "kind": location.location_kind.value,
        })
    for npc in book.npcs:
        _add_node(npc, EntityType.NPC, _npc_properties(npc))
    for item in book.items:
        _add_node(item, EntityType.ITEM, {
            "description": item.description or item.summary,
            "category": item.category,
        })
    for quest in book.quests:
        _add_node(quest, EntityType.QUEST, {"description": quest.summary})
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
        if npc.current_location_id:
            _edge(npc.id, npc.current_location_id, RelationType.LOCATED_AT,
                  f"npc {npc.id} current_location")
        elif npc.home_location_id:
            _edge(npc.id, npc.home_location_id, RelationType.LOCATED_AT,
                  f"npc {npc.id} home_location")
        for entry in npc.inventory:
            item_id = getattr(entry, "item_id", "") or ""
            if item_id:
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
        if claim.visibility in _VISIBLE_TO_PLAY or claim.id in granted:
            out.established_facts.append(claim.text)
        else:
            out.withheld.append(claim)

    logger.info(
        "sourcebook_compiled",
        sourcebook=book.metadata.sourcebook_id,
        nodes=out.node_count,
        edges=out.edge_count,
        facts=len(out.established_facts),
        withheld=len(out.withheld),
        warnings=len(out.warnings),
    )
    return out
