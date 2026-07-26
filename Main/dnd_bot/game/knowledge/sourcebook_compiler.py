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

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

from ...models.sourcebook import (
    CampaignSourcebook,
    CanonStatus,
    CharacterStatus,
    KnowledgeClaim,
    RelationshipKind,
    Visibility,
)
from ...models.sourcebook_canon import ImportReceipt, RebuildReceipt
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
        # A withheld quest has no node, so every edge touching it would report
        # itself as a dangling reference — an authoring defect the author did
        # not commit. Its absence is already in withheld_notes; `warnings` is
        # reserved for mistakes worth acting on.
        if quest.id not in active_quests:
            continue
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

    _log_compiled(book, out)
    return out


def load_sourcebook(path: Path | str) -> CampaignSourcebook:
    """Read an authored book from YAML or JSON.

    Validation is the schema's job — CampaignSourcebook rejects dangling
    references, containment cycles and malformed ids on construction, so a
    bad book fails here rather than half-compiling into a live campaign.
    """
    import json

    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return CampaignSourcebook.model_validate(data)


async def apply_sourcebook(
    book: CampaignSourcebook,
    *,
    campaign_id: str,
    knowledge_graph,
    world_store,
    force: bool = False,
) -> CompiledSourcebook:
    """Compile a book and install it into a campaign's live stores.

    Graph first, then the opening scene, then its residents — the order
    matters because hydration reads the residency edges the graph leg just
    wrote. World-state writes go through the store, never around it.

    Collaborators are passed explicitly rather than reaching through a
    session so this is testable against a real graph and a real store
    without standing up a session.
    """
    compiled = compile_sourcebook(book, campaign_id)
    label = str(book.metadata.sourcebook_id)

    await _install_graph(compiled, campaign_id, knowledge_graph, label)
    _seed_opening_scene(compiled, knowledge_graph, world_store, label, force)
    return compiled


async def _install_graph(
    compiled: CompiledSourcebook,
    campaign_id: str,
    knowledge_graph,
    label: str,
    ops: list[GraphOperation] | None = None,
) -> list[str]:
    """Apply the projection's ops; report what the graph refused.

    ``ops`` narrows what is applied (the rebuild path filters out nodes the
    graph already has); rejections still land on ``compiled.warnings``.
    """
    rejections = await knowledge_graph.apply_operations(
        compiled.graph_ops if ops is None else ops
    )
    if rejections:
        # The graph rejects rather than raises; surface them as warnings so a
        # partially-installed book is visible instead of quietly incomplete.
        compiled.warnings.extend(f"graph rejected: {r}" for r in rejections)
        # `sourcebook_ref`, not `sourcebook`: this fires from the book-object
        # path (which has the authored id) and the canon path (which has only
        # a key), and a field that means two different things across callers
        # is worse than one that promises less.
        logger.warning(
            "sourcebook_graph_rejections",
            sourcebook_ref=label,
            campaign=campaign_id,
            count=len(rejections),
        )
    return list(rejections)


def _seed_opening_scene(
    compiled: CompiledSourcebook,
    knowledge_graph,
    world_store,
    label: str,
    force: bool,
) -> bool:
    """The world-state leg. Every write goes through the store."""
    seeded = world_store.seed_opening_scene(
        location=compiled.current_location,
        description=compiled.location_description,
        scene_items=compiled.scene_items,
        force=force,
    )
    if not seeded:
        logger.warning(
            "sourcebook_scene_not_seeded",
            sourcebook=label,
            location=compiled.current_location,
        )
        return False

    for fact in compiled.established_facts:
        world_store.add_established_fact(fact)

    location_node = knowledge_graph.resolve_location_node(
        compiled.current_location
    )
    if location_node:
        world_store.hydrate_residents(
            knowledge_graph.residents_of(location_node)
        )

    logger.info(
        "sourcebook_applied",
        sourcebook=label,
        location=compiled.current_location,
        npcs_on_stage=len(world_store.state.npcs),
    )
    return True


# ── Rebuilding the indexes from canonical rows ──────────────────────────────
#
# The design doc's contract: "If graph or vector projection fails, the import
# remains recoverable: rebuild projections from canonical SQLite records
# instead of asking a model to regenerate lore." Everything below is that
# sentence made executable. `apply_sourcebook` needs the book OBJECT; these
# need only a `sourcebook_key`, which is what makes the graph and the
# embeddings genuinely disposable.


@dataclass
class InstalledSourcebook:
    """What a production install wrote, across all four layers."""

    imported: "ImportReceipt"
    rebuilt: "RebuildReceipt"
    compiled: CompiledSourcebook
    scene_seeded: bool = False

    @property
    def sourcebook_key(self) -> str:
        return self.imported.sourcebook_key


def projection_fingerprint(compiled: CompiledSourcebook) -> dict[str, object]:
    """A comparable shape for a projection.

    Graph ops minus their wall-clock stamps — ``Entity``/``Relationship``
    default ``created_at`` to ``utcnow()``, so two identical projections built
    a millisecond apart never compare equal. Everything the game can see (ids,
    ORDER, properties, aliases, relation types, the withheld buckets, the
    opening scene) is kept.
    """
    ops: list[dict[str, Any]] = []
    for op in compiled.graph_ops:
        data = op.model_dump(mode="json")
        for holder in ("entity", "relationship"):
            if isinstance(data.get(holder), dict):
                data[holder].pop("created_at", None)
                data[holder].pop("updated_at", None)
        ops.append(data)
    return {
        "graph_ops": ops,
        "established_facts": list(compiled.established_facts),
        "withheld": [claim.model_dump(mode="json") for claim in compiled.withheld],
        "withheld_notes": list(compiled.withheld_notes),
        "warnings": list(compiled.warnings),
        "current_location": compiled.current_location,
        "location_description": compiled.location_description,
        "scene_items": dict(compiled.scene_items),
        "opening_situation": compiled.opening_situation,
    }


def _assert_canon_reproduces(
    from_book: CompiledSourcebook,
    from_canon: CompiledSourcebook,
    sourcebook_key: str,
) -> None:
    """Refuse to install when the rows cannot reproduce the book."""
    expected = projection_fingerprint(from_book)
    actual = projection_fingerprint(from_canon)
    if expected == actual:
        return
    differing = sorted(k for k in expected if expected[k] != actual[k])
    logger.error(
        "sourcebook_canon_projection_mismatch",
        key=sourcebook_key[:12],
        fields=differing,
    )
    raise ValueError(
        f"canonical rows for sourcebook {sourcebook_key[:12]} do not "
        f"reproduce the book: {differing} differ. Refusing to install — the "
        "visibility boundary is enforced on the projection, so a lossy round "
        "trip can publish a secret with no warning to show for it."
    )


async def compile_from_canon(
    repository, sourcebook_key: str, campaign_id: str
) -> CompiledSourcebook:
    """Compile straight from canonical rows — no book file involved.

    Equivalent to ``compile_sourcebook(load_sourcebook(path), campaign_id)``
    for any book that was imported, which is the property the round-trip
    gate pins: if canon lost anything, these two projections diverge.
    """
    book = await repository.load_book(sourcebook_key)
    if book is None:
        raise LookupError(f"no imported sourcebook with key {sourcebook_key!r}")
    return compile_sourcebook(book, campaign_id)


async def rebuild_indexes(
    *,
    repository,
    sourcebook_key: str,
    campaign_id: str,
    knowledge_graph,
    vector_store=None,
    overwrite: bool = False,
) -> "RebuildReceipt":
    """Regenerate the graph (and optionally the vector index) from canon.

    **Nodes the graph already has are left completely alone**, which is what
    makes this safe to run on a campaign in play. The graph merges by node id
    via ``properties.update()``, and canon always says ``alive: "true"`` for a
    character the BOOK considers living — so overwriting would revert
    everything play wrote about that entity. Concretely, before this rule: an
    NPC the party killed came back. The tool path left the node saying
    ``alive: true`` beside ``status: dead`` and the narrator only ever sees
    ``alive``; the delta path writes no ``status`` at all, so both of
    ``hydrate_residents``' gates cleared and the corpse walked back on stage.
    "The dead stay dead" outranks index freshness.

    Skipping existing nodes still repairs the failure this exists for. The
    design doc's case is a projection that FAILED — rows lost, never written,
    a half-applied import — where the nodes are missing, not stale. Pass
    ``overwrite=True`` to re-assert canon over a live graph deliberately.

    Edges are always applied: they are keyed by (source, target, relation) and
    carry no play-authored state, so re-adding one is idempotent.

    Vector content is derived from the compiler's OWN node ops rather than
    re-read from the canonical tables. That is deliberate: the visibility
    boundary is enforced in exactly one place, and an embedding is the worst
    possible second place to re-derive it — a leaked secret in the vector
    index resurfaces on semantic similarity alone, with no scene, entity or
    keyword to make the leak traceable.
    """
    compiled = await compile_from_canon(repository, sourcebook_key, campaign_id)
    return await _rebuild_from_compiled(
        compiled,
        sourcebook_key=sourcebook_key,
        campaign_id=campaign_id,
        knowledge_graph=knowledge_graph,
        vector_store=vector_store,
        overwrite=overwrite,
    )


async def _rebuild_from_compiled(
    compiled: CompiledSourcebook,
    *,
    sourcebook_key: str,
    campaign_id: str,
    knowledge_graph,
    vector_store=None,
    overwrite: bool = False,
) -> "RebuildReceipt":
    receipt = RebuildReceipt(
        sourcebook_key=sourcebook_key,
        campaign_id=campaign_id,
        projected_nodes=compiled.node_count,
        projected_edges=compiled.edge_count,
    )

    ops = list(compiled.graph_ops)
    if not overwrite:
        preserved = [
            op.entity.node_id
            for op in ops
            if isinstance(op, AddNode) and knowledge_graph.has_node(op.entity.node_id)
        ]
        receipt.preserved_nodes = sorted(preserved)
        skip = set(preserved)
        ops = [
            op for op in ops
            if not (isinstance(op, AddNode) and op.entity.node_id in skip)
        ]

    before_nodes = knowledge_graph.node_count()
    before_edges = knowledge_graph.edge_count()
    receipt.graph_rejections = await _install_graph(
        compiled, campaign_id, knowledge_graph, sourcebook_key[:12], ops=ops,
    )
    # Measured, not assumed. The graph MERGES or ABSTAINS on a proper-name
    # collision and returns without raising, so apply_operations reports
    # nothing and the projection's own count would overstate what landed.
    receipt.nodes_added = knowledge_graph.node_count() - before_nodes
    receipt.edges_added = knowledge_graph.edge_count() - before_edges
    receipt.warnings = list(compiled.warnings)

    if vector_store is not None:
        # Snapshot to plain values BEFORE going off-loop: the graph mutates
        # Entity objects in place (aliases are appended to on merge), and
        # indexing the op's entity rather than the node the graph actually
        # holds would drift index and graph apart exactly where a merge
        # touched. Entities the graph does not hold are skipped, so a rejected
        # or abstained node cannot leave an orphan document behind pointing at
        # a node id that does not exist.
        indexable = [
            (
                entity.node_id, entity.entity_type.value, entity.name,
                entity.properties.get("description", ""), list(entity.aliases),
            )
            for entity in (
                knowledge_graph.get_entity(op.entity.node_id)
                for op in compiled.graph_ops if isinstance(op, AddNode)
            )
            if entity is not None
        ]

        def _embed() -> tuple[int, int]:
            done = failed = 0
            for node_id, entity_type, name, description, aliases in indexable:
                if vector_store.add_entity_description(
                    campaign_id=campaign_id,
                    node_id=node_id,
                    entity_type=entity_type,
                    name=name,
                    description=description,
                    aliases=aliases,
                ):
                    done += 1
                else:
                    failed += 1
            return done, failed

        # Chroma client init and per-entity embedding are blocking work; a
        # whole-book rebuild would otherwise freeze the event loop (AQ-ASYNC-03).
        try:
            receipt.embedded, receipt.embed_failures = await asyncio.to_thread(_embed)
        except Exception as exc:
            # The vector index is the most disposable layer there is — it can
            # be rebuilt from these same rows at any time. Letting it abort an
            # install would leave canon written and the campaign bound with no
            # scene and no receipt for the caller to retry from.
            receipt.embed_failures = len(indexable)
            receipt.warnings.append(f"vector rebuild failed: {exc}")
            logger.warning(
                "sourcebook_vector_rebuild_failed",
                key=sourcebook_key[:12], campaign=campaign_id,
                error=str(exc), exc_info=True,
            )
        receipt.vector_skipped = False
        if receipt.embed_failures:
            # add_entity_description swallows its own errors and returns False,
            # so counting only successes would let a rebuild that indexed
            # NOTHING report success.
            logger.warning(
                "sourcebook_vector_rebuild_incomplete",
                key=sourcebook_key[:12],
                campaign=campaign_id,
                embedded=receipt.embedded,
                failed=receipt.embed_failures,
            )

    logger.info(
        "sourcebook_indexes_rebuilt",
        key=sourcebook_key[:12],
        campaign=campaign_id,
        nodes_added=receipt.nodes_added,
        edges_added=receipt.edges_added,
        preserved=len(receipt.preserved_nodes),
        embedded=receipt.embedded,
        rejections=len(receipt.graph_rejections),
    )
    return receipt


async def install_sourcebook(
    book: CampaignSourcebook,
    *,
    campaign_id: str,
    repository,
    knowledge_graph,
    world_store,
    vector_store=None,
    force: bool = False,
) -> InstalledSourcebook:
    """The production import path: canon first, indexes derived from it.

    The indexes are built by reading the rows BACK rather than from the book
    still in memory, and the two projections are then COMPARED. A lossy import
    raises here instead of surviving as a healthy-looking graph beside
    canonical tables that quietly cannot reproduce it. That comparison is not
    ceremony: the compiler enforces the visibility boundary in exactly one
    place, so once the graph is built from canon rather than from the book,
    every secret depends on canon's fidelity too. Losing a single `hidden`
    flag would give a concealed item a node, an ownership edge, and its secret
    text in the vector index — with no warning, no rejection, and nothing in
    ``withheld_notes`` to show it ever happened.

    **Not atomic, but idempotent.** The stages commit independently, so a
    failure part-way through leaves canon written and the campaign bound. Just
    re-run it: the import no-ops on an identical key, binding is a no-op,
    starting knowledge is first-wins, and the rebuild preserves nodes the
    graph already has.
    """
    imported = await repository.import_book(book)
    key = imported.sourcebook_key

    await repository.bind_campaign(campaign_id, key)
    await repository.seed_starting_knowledge(campaign_id, key)

    # Read the rows back, and prove they reproduce the book.
    compiled = await compile_from_canon(repository, key, campaign_id)
    _assert_canon_reproduces(compile_sourcebook(book, campaign_id), compiled, key)

    rebuilt = await _rebuild_from_compiled(
        compiled,
        sourcebook_key=key,
        campaign_id=campaign_id,
        knowledge_graph=knowledge_graph,
        vector_store=vector_store,
    )

    seeded = _seed_opening_scene(
        compiled, knowledge_graph, world_store,
        str(book.metadata.sourcebook_id), force,
    )
    if seeded:
        # The party wakes up here, so the world has been touched here. Without
        # this, "authored in a region the party has not touched" would count
        # the opening tavern as untouched forever.
        await repository.record_visit(
            campaign_id, key, str(book.starting_state.location_id), turn=0,
        )

    return InstalledSourcebook(
        imported=imported, rebuilt=rebuilt, compiled=compiled,
        scene_seeded=seeded,
    )


def _log_compiled(book: CampaignSourcebook, out: CompiledSourcebook) -> None:
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
