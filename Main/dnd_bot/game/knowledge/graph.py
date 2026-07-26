"""KnowledgeGraph — NetworkX MultiDiGraph with SQLite write-through persistence."""

from datetime import datetime
from typing import Any, Optional

import networkx as nx
import structlog
import yaml

from .models import (
    AddEdge,
    AddNode,
    Entity,
    EntityType,
    GraphOperation,
    RelationType,
    RemoveEdge,
    RemoveNode,
    UpdateNode,
    slugify,
)
from .repository import KnowledgeGraphRepository
from ..identity import (
    entity_identity_keys,
    identity_keys,
    is_generic_npc_label,
    name_is_fragment_of,
)

logger = structlog.get_logger()


class KnowledgeGraph:
    """In-memory graph with write-through SQLite persistence.

    All read queries hit the NetworkX MultiDiGraph (microsecond latency).
    All mutations write to both NetworkX and SQLite via the repository.

    **Edge identity is (source, target, relation_type)** — the same key
    ``kg_edge``'s primary key uses, which is why the in-memory shape is a
    MultiDiGraph keyed by relation type rather than a DiGraph. A DiGraph
    holds at most ONE edge per (source, target) pair, so a second relation
    between the same two entities silently REPLACED the first: an author
    who wrote both ALLIED_WITH and HOSTILE_TO between two NPCs got one of
    them, with no rejection to show for it. ``apply_operations`` returned
    clean and any receipt counting applied ops overstated what landed.

    Worse than the loss was its instability: ``load_edges`` decided the
    winner by row order, so the same campaign could present a different
    social graph after a reload. Keying on the relation type removes both
    — the projection stays lossy where it is *designed* to be (24 authored
    RelationshipKinds collapse onto 9 RelationTypes in
    ``sourcebook_compiler``), and stops being lossy where nothing intended
    it.

    Every edge also carries its relation type in the ``relationship`` data
    attribute, duplicating the key. Readers filter on the attribute, and
    keeping it means the key change did not have to reach all of them.
    """

    def __init__(self, campaign_id: str, repository: KnowledgeGraphRepository):
        self._campaign_id = campaign_id
        self._repo = repository
        self._graph = nx.MultiDiGraph()
        self._entities: dict[str, Entity] = {}  # node_id → Entity model
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Load full graph from SQLite. Called once at session start."""
        nodes = await self._repo.load_nodes(self._campaign_id)
        edges = await self._repo.load_edges(self._campaign_id)

        for entity in nodes:
            self._graph.add_node(entity.node_id, entity_type=entity.entity_type.value)
            self._entities[entity.node_id] = entity

        for rel in edges:
            if rel.source_id in self._graph and rel.target_id in self._graph:
                self._graph.add_edge(
                    rel.source_id,
                    rel.target_id,
                    key=rel.relation_type.value,
                    relationship=rel.relation_type.value,
                    weight=rel.weight,
                )

        self._loaded = True
        logger.info(
            "knowledge_graph_loaded",
            campaign_id=self._campaign_id,
            nodes=self.node_count(),
            edges=self.edge_count(),
        )

    # ------------------------------------------------------------------
    # Mutations (write-through)
    # ------------------------------------------------------------------

    async def apply_operations(self, ops: list[GraphOperation]) -> list[str]:
        """Apply a batch of graph operations. Returns rejection messages.

        Node creation is hoisted so an edge never references an endpoint that
        does not exist yet. Everything else runs in the order its producer
        wrote, because within a batch the sequence IS the intent.

        A blanket type-priority sort (add_edge ranked ahead of remove_edge)
        silently inverted the one batch that depends on order: the bridge
        relocates an NPC by removing every LOCATED_AT edge from them and then
        adding the new one. Inverted, the wildcard removal deleted the
        residency edge that had just been written, leaving the NPC with NO
        residency at all — so ``residents_of`` no longer listed them and
        scene hydration had nothing to restore. That is how an authored
        tavern regular vanished from the roster on return (lore_recall
        20260726_011328 lost Mara Venn, _011459 lost Toran Vex; both deltas
        carried an ``npc_updates[].location`` for exactly the NPC that went
        missing, and the six runs without one all kept both residents).
        """
        rejections: list[str] = []

        # Stable sort: AddNode first, everything else in authored order.
        sorted_ops = sorted(ops, key=lambda o: 0 if o.op == "add_node" else 1)

        for op in sorted_ops:
            try:
                if isinstance(op, AddNode):
                    await self._apply_add_node(op)
                elif isinstance(op, AddEdge):
                    await self._apply_add_edge(op)
                elif isinstance(op, UpdateNode):
                    await self._apply_update_node(op)
                elif isinstance(op, RemoveEdge):
                    await self._apply_remove_edge(op)
                elif isinstance(op, RemoveNode):
                    await self._apply_remove_node(op)
            except Exception as e:
                rejections.append(f"{op.op}: {e}")

        if ops:
            logger.debug(
                "kg_operations_applied",
                count=len(ops),
                rejections=len(rejections),
            )

        return rejections

    def _npc_proper_name_collisions(self, node_id: str, name: str) -> list[Entity]:
        """NPC nodes other than ``node_id`` already holding proper name ``name``.

        The canonical-npc-identity invariant: at most one durable NPC node
        per proper name. Generic role labels ("the guard") abstain — they
        are archetypes, not identities. Matching uses the codebase's exact
        identity-key bar (leading-title stripping only, no fuzz).
        """
        if is_generic_npc_label(name):
            return []
        keys = identity_keys(name)
        if not keys:
            return []
        return [
            entity
            for nid, entity in self._entities.items()
            if nid != node_id
            and entity.entity_type == EntityType.NPC
            and keys & entity_identity_keys(entity)
        ]

    async def _apply_add_node(self, op: AddNode) -> None:
        entity = op.entity
        if entity.node_id in self._entities:
            # Merge: update existing instead of duplicate
            existing = self._entities[entity.node_id]
            existing.properties.update(entity.properties)
            existing.aliases = list(set(existing.aliases + entity.aliases))
            existing.updated_at = datetime.utcnow()
            self._entities[entity.node_id] = existing
            await self._repo.upsert_node(existing)
            return

        # Proper-name uniqueness at the write seam: a new NPC node whose
        # proper name a durable NPC node already carries must merge into it
        # (unique holder) or abstain (ambiguous) — never create a parallel
        # identity that later collides when a generic node's name is
        # promoted (the cross-store naming-promotion defect).
        if entity.entity_type == EntityType.NPC:
            collisions = self._npc_proper_name_collisions(entity.node_id, entity.name)
            if len(collisions) == 1:
                existing = collisions[0]
                existing.properties.update(entity.properties)
                held = {a.casefold() for a in existing.aliases}
                held.add(existing.name.casefold())
                for alias in (entity.name, *entity.aliases):
                    if alias and alias.casefold() not in held:
                        existing.aliases.append(alias)
                        held.add(alias.casefold())
                existing.updated_at = datetime.utcnow()
                await self._repo.upsert_node(existing)
                logger.info(
                    "kg_add_node_merged_into_same_named_npc",
                    proposed_node_id=entity.node_id,
                    target_node_id=existing.node_id,
                    name=entity.name,
                )
                return
            if collisions:
                logger.warning(
                    "kg_add_node_abstained_proper_name_collision",
                    proposed_node_id=entity.node_id,
                    name=entity.name,
                    colliding=[e.node_id for e in collisions],
                )
                return

        self._graph.add_node(entity.node_id, entity_type=entity.entity_type.value)
        self._entities[entity.node_id] = entity
        await self._repo.upsert_node(entity)

    async def _apply_add_edge(self, op: AddEdge) -> None:
        rel = op.relationship
        if rel.source_id not in self._graph:
            raise ValueError(f"Source node not found: {rel.source_id}")
        if rel.target_id not in self._graph:
            raise ValueError(f"Target node not found: {rel.target_id}")

        # Keyed by relation type, so re-adding the SAME relation updates it in
        # place (matching the repository's ON CONFLICT ... DO UPDATE) while a
        # DIFFERENT relation between the same pair lands beside it.
        self._graph.add_edge(
            rel.source_id,
            rel.target_id,
            key=rel.relation_type.value,
            relationship=rel.relation_type.value,
            weight=rel.weight,
        )
        await self._repo.upsert_edge(rel)

    async def _apply_update_node(self, op: UpdateNode) -> None:
        if op.node_id not in self._entities:
            raise ValueError(f"Node not found: {op.node_id}")

        entity = self._entities[op.node_id]
        entity.properties.update(op.properties)
        if op.aliases is not None:
            entity.aliases = list(set(entity.aliases + op.aliases))
        entity.updated_at = datetime.utcnow()
        await self._repo.upsert_node(entity)

    async def _apply_remove_edge(self, op: RemoveEdge) -> None:
        if op.target_id:
            # The relation type is the edge KEY, so this removes exactly the
            # relation named and leaves any other relation between the same
            # pair standing — which is what the repository's DELETE ... AND
            # relation_type = ? has always done on the persisted side.
            if self._graph.has_edge(op.source_id, op.target_id, op.relation_type.value):
                self._graph.remove_edge(
                    op.source_id, op.target_id, key=op.relation_type.value
                )
            await self._repo.delete_edge(
                self._campaign_id,
                op.source_id,
                op.target_id,
                op.relation_type.value,
            )
            return

        # Empty target is the bridge's explicit "remove every relationship of
        # this type from this source" operation. Apply identical wildcard
        # semantics to the in-memory and persisted projections.
        matching = [
            (source, target, key)
            for source, target, key in self._graph.out_edges(op.source_id, keys=True)
            if key == op.relation_type.value
        ]
        self._graph.remove_edges_from(matching)
        await self._repo.delete_edges_by_source(
            self._campaign_id,
            op.source_id,
            op.relation_type.value,
        )

    async def _apply_remove_node(self, op: RemoveNode) -> None:
        if op.node_id in self._graph:
            self._graph.remove_node(op.node_id)
        self._entities.pop(op.node_id, None)
        await self._repo.delete_node(self._campaign_id, op.node_id)

    # ------------------------------------------------------------------
    # Read queries
    # ------------------------------------------------------------------

    def get_entity(self, node_id: str) -> Optional[Entity]:
        return self._entities.get(node_id)

    def resolve_entity_reference(self, reference: str) -> Optional[Entity]:
        """Resolve a unique graph entity by ID, display name, or alias.

        Narrator tools naturally emit human-readable slugs such as
        ``tomas-kell`` even when a state extractor assigned the canonical graph
        node a UUID.  Resolution remains conservative: ambiguous names or
        aliases abstain instead of guessing into durable state.
        """
        from ..identity import resolve_unique_identity

        return resolve_unique_identity(reference, self._entities.values())

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        """Number of distinct (source, target, relation_type) edges.

        Counts parallel edges individually, so this now agrees row-for-row
        with what ``kg_edge`` holds. Under the old DiGraph it counted
        (source, target) PAIRS, which under-reported every campaign whose
        author wrote two relations between the same two entities — and
        ``RebuildReceipt.edges_added``, measured as a delta of this number,
        under-reported with it.
        """
        return self._graph.number_of_edges()

    def has_node(self, node_id: str) -> bool:
        return node_id in self._graph

    def get_all_names(self) -> dict[str, str]:
        """Return {lowercase_name_or_alias: node_id} for entity matching."""
        names: dict[str, str] = {}
        for node_id, entity in self._entities.items():
            names[entity.name.lower()] = node_id
            for alias in entity.aliases:
                names[alias.lower()] = node_id
        return names

    def get_entities_for_indexing(self) -> list[Entity]:
        """Return entities that have descriptions worth indexing in ChromaDB."""
        return [
            e for e in self._entities.values()
            if e.properties.get("description")
            and e.properties.get("placeholder") != "true"
        ]

    def dead_npcs(self) -> list[Entity]:
        """NPC nodes canon records as NOT alive — authored or played-in.

        The graph is the one campaign-wide store that carries both kinds:
        ``sourcebook_compiler`` writes ``alive=false`` for a character the
        book authored as DEAD/UNDEAD, and ``bridge`` writes it when someone
        dies in play. Scene hydration already treats this property as
        authoritative — ``WorldStateStore.hydrate_residents`` refuses to
        restore anyone carrying it — so exposing it here lets the continuity
        layer guard prose about exactly the same set, rather than the
        strictly smaller one that happens to have a session or a DB row.

        Only an EXPLICIT ``alive=false`` counts. An absent property means no
        writer has spoken, which is not a death.

        Returns entities in stable id order; callers apply their own policy
        (a live revival outranking a stale node, name matching, caps).
        """
        return sorted(
            (
                entity
                for entity in self._entities.values()
                if entity.entity_type == EntityType.NPC
                and str(entity.properties.get("alive", "")).strip().lower()
                == "false"
            ),
            key=lambda entity: entity.node_id,
        )

    async def promote_entity_name(self, node_id: str, new_name: str) -> bool:
        """Rename an entity, moving the old name to aliases.

        Used when an unnamed NPC ('the hooded stranger') gets a proper
        name from the narrator.

        Enforces the proper-name uniqueness invariant at the write seam:
        when another durable NPC node already carries ``new_name``, the
        promotion abstains (returns False) rather than minting a second
        node with the same canonical name. The split identity stays split
        — recoverable evidence — instead of becoming an irreversible
        name collision.
        """
        entity = self._entities.get(node_id)
        if not entity:
            return False

        # A "new name" whose every word already sits in the current label is
        # an excerpt, not a naming event ('Choir' offered as an alias for
        # 'a Choir acolyte' is the faction's name, not the person's) —
        # promoting it hijacks the excerpted word's identity. Abstain.
        if name_is_fragment_of(new_name, entity.name):
            logger.info(
                "entity_name_promotion_fragment_abstained",
                node_id=node_id,
                current_name=entity.name,
                new_name=new_name,
            )
            return False

        if entity.entity_type == EntityType.NPC:
            collisions = self._npc_proper_name_collisions(node_id, new_name)
            if collisions:
                logger.warning(
                    "entity_name_promotion_collision_abstained",
                    node_id=node_id,
                    new_name=new_name,
                    colliding=[e.node_id for e in collisions],
                )
                return False

        old_name = entity.name
        entity.name = new_name
        if old_name and old_name.lower() != new_name.lower():
            if old_name not in entity.aliases:
                entity.aliases.append(old_name)
        entity.properties["named"] = "true"
        entity.updated_at = datetime.utcnow()

        await self._repo.upsert_node(entity)

        logger.info(
            "entity_name_promoted",
            node_id=node_id,
            old_name=old_name,
            new_name=new_name,
        )
        return True

    # ------------------------------------------------------------------
    # BFS subgraph retrieval
    # ------------------------------------------------------------------

    def resolve_location_node(self, location_name: str) -> Optional[str]:
        """Find a location node by name, tolerating spelling variance.

        The rest of the codebase compares locations with
        ``locations_equivalent`` ("the Copper Finch" == "Copper Finch"), but
        a raw ``slugify`` lookup is exact — so a narrator-supplied variant
        silently matched nothing.
        """
        from ..identity import locations_equivalent

        name = (location_name or "").strip()
        if not name:
            return None
        slug = slugify(name)
        if slug in self._graph:
            return slug
        for node_id, entity in self._entities.items():
            if entity.entity_type != EntityType.LOCATION:
                continue
            if locations_equivalent(entity.name, name):
                return node_id
            # Aliases are how canon declares what a place may be CALLED.
            # Narrators paraphrase constantly — a live run returned the party
            # to "the tavern", which no spelling-variant rule can bridge to
            # "Copper Finch", so the room came back empty.
            if any(locations_equivalent(alias, name) for alias in entity.aliases):
                return node_id
        return None

    def residents_of(
        self,
        location_id: str,
        relation_types: tuple[RelationType, ...] = (RelationType.LOCATED_AT,),
    ) -> list[Entity]:
        """Entities recorded as being AT a location, by incoming edge.

        The graph is otherwise queried only by BFS from seeds, which answers
        "what is related to what the narrator just mentioned" — it cannot
        answer "who lives here", so arriving somewhere could not repopulate
        the scene with its known inhabitants.

        Returns entities in stable id order; callers apply their own policy
        (alive-only, caps, dead-roster exclusion). Unknown locations and
        unloaded graphs yield an empty list rather than raising.

        Deduplicated by node: one entity can now hold SEVERAL edges into the
        same location (edges are keyed by relation type), so a caller asking
        for more than one ``relation_types`` would otherwise get whoever
        satisfies two of them listed twice — and a duplicated resident is a
        duplicated NPC on stage after hydration.
        """
        if not location_id or location_id not in self._graph:
            return []
        wanted = {rel.value for rel in relation_types}
        residents: list[Entity] = []
        seen: set[str] = set()
        for source_id, _target, key in self._graph.in_edges(location_id, keys=True):
            if key not in wanted or source_id in seen:
                continue
            entity = self._entities.get(source_id)
            if entity is not None:
                seen.add(source_id)
                residents.append(entity)
        return sorted(residents, key=lambda e: e.node_id)

    def get_context_subgraph(
        self,
        seed_ids: list[str],
        radius: float = 2.0,
        max_entities: int = 15,
        no_expand_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve entities within BFS radius of seed nodes.

        Returns a list of entity dicts with their relationships, ready
        for YAML serialization and narrator injection.

        ``no_expand_ids`` marks seeds that are included verbatim but do NOT
        expand a BFS neighborhood. Speculative retrieval (vector similarity)
        earns the matched node only; hopping to its neighbors is how an
        off-screen NPC leaked into an unrelated prompt through a
        semantically similar item (soak 20260722_230128, turn 55).
        """
        if not seed_ids:
            return []

        frozen = set(no_expand_ids or [])

        # Union BFS neighborhoods from all seeds
        combined_nodes: set[str] = set()
        for sid in seed_ids:
            if sid not in self._graph:
                continue
            if sid in frozen:
                combined_nodes.add(sid)
                continue
            try:
                sub = nx.ego_graph(self._graph, sid, radius=radius, distance="weight", undirected=True)
                combined_nodes.update(sub.nodes())
            except nx.NetworkXError:
                continue

        if not combined_nodes:
            return []

        # Preserve caller priority. ``seed_ids`` is deliberately ordered as
        # explicit text matches first, then current-scene entities. Turning it
        # into a set and sorting lexicographically allowed ambient UUIDs to
        # displace an explicitly named entity under ``max_entities``.
        ordered_seeds: list[str] = []
        seen_seeds: set[str] = set()
        for seed_id in seed_ids:
            if seed_id in combined_nodes and seed_id not in seen_seeds:
                ordered_seeds.append(seed_id)
                seen_seeds.add(seed_id)
        ordered = ordered_seeds + sorted(combined_nodes - seen_seeds)

        # Cap at max_entities
        ordered = ordered[:max_entities]

        # Build output
        result = []
        for node_id in ordered:
            entity = self._entities.get(node_id)
            if not entity:
                continue

            # Collect outgoing relationships within the subgraph. Parallel
            # edges are distinct relations between the same pair, so an NPC
            # who is both allied with and hostile to another now reports
            # BOTH — under the old DiGraph one of the two never existed to
            # be reported.
            #
            # Sorted within each direction because insertion order is not a
            # stable fact: a graph built by play and the same graph reloaded
            # from SQLite would otherwise order these differently, changing
            # the narrator's prompt for no reason anyone authored.
            outgoing = []
            for _, target, data in self._graph.edges(node_id, data=True):
                if target in combined_nodes:
                    target_entity = self._entities.get(target)
                    target_name = target_entity.name if target_entity else target
                    rel_type = data.get("relationship", "related_to")
                    outgoing.append(f"{rel_type} {target_name}")

            # Collect incoming relationships within the subgraph
            incoming = []
            for source, _, data in self._graph.in_edges(node_id, data=True):
                if source in combined_nodes and source != node_id:
                    source_entity = self._entities.get(source)
                    source_name = source_entity.name if source_entity else source
                    rel_type = data.get("relationship", "related_to")
                    incoming.append(f"{source_name} {rel_type} this")

            relationships = sorted(outgoing) + sorted(incoming)

            entry: dict[str, Any] = {
                "id": entity.node_id,
                "name": entity.name,
                "type": entity.entity_type.value,
            }

            # Include key properties (skip placeholder/internal markers)
            for key in ("description", "disposition", "alive", "location"):
                if key in entity.properties and entity.properties[key]:
                    entry[key] = entity.properties[key]

            if relationships:
                entry["relationships"] = relationships

            result.append(entry)

        return result

    def to_context_yaml(
        self,
        seed_ids: list[str],
        radius: float = 2.0,
        max_entities: int = 15,
        no_expand_ids: set[str] | None = None,
    ) -> str:
        """Serialize relevant subgraph as YAML for narrator context injection."""
        subgraph = self.get_context_subgraph(
            seed_ids, radius, max_entities, no_expand_ids=no_expand_ids
        )
        if not subgraph:
            return ""
        return yaml.dump(
            {"known_entities": subgraph},
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
