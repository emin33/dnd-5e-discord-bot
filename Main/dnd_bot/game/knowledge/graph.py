"""KnowledgeGraph — NetworkX DiGraph with SQLite write-through persistence."""

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
    RemoveEdge,
    RemoveNode,
    UpdateNode,
)
from .repository import KnowledgeGraphRepository
from ..identity import entity_identity_keys, identity_keys, is_generic_npc_label

logger = structlog.get_logger()


class KnowledgeGraph:
    """In-memory graph with write-through SQLite persistence.

    All read queries hit the NetworkX DiGraph (microsecond latency).
    All mutations write to both NetworkX and SQLite via the repository.
    """

    def __init__(self, campaign_id: str, repository: KnowledgeGraphRepository):
        self._campaign_id = campaign_id
        self._repo = repository
        self._graph = nx.DiGraph()
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
        """Apply a batch of graph operations. Returns rejection messages."""
        rejections: list[str] = []

        # Sort by operation priority: AddNode → AddEdge → UpdateNode → RemoveEdge → RemoveNode
        priority = {"add_node": 0, "add_edge": 1, "update_node": 2, "remove_edge": 3, "remove_node": 4}
        sorted_ops = sorted(ops, key=lambda o: priority.get(o.op, 5))

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

        self._graph.add_edge(
            rel.source_id,
            rel.target_id,
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
            if self._graph.has_edge(op.source_id, op.target_id):
                edge = self._graph.get_edge_data(op.source_id, op.target_id) or {}
                if edge.get("relationship") == op.relation_type.value:
                    self._graph.remove_edge(op.source_id, op.target_id)
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
        matching_targets = [
            target
            for _, target, data in self._graph.out_edges(op.source_id, data=True)
            if data.get("relationship") == op.relation_type.value
        ]
        self._graph.remove_edges_from(
            (op.source_id, target) for target in matching_targets
        )
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

            # Collect outgoing relationships within the subgraph
            relationships = []
            for _, target, data in self._graph.edges(node_id, data=True):
                if target in combined_nodes:
                    target_entity = self._entities.get(target)
                    target_name = target_entity.name if target_entity else target
                    rel_type = data.get("relationship", "related_to")
                    relationships.append(f"{rel_type} {target_name}")

            # Collect incoming relationships within the subgraph
            for source, _, data in self._graph.in_edges(node_id, data=True):
                if source in combined_nodes and source != node_id:
                    source_entity = self._entities.get(source)
                    source_name = source_entity.name if source_entity else source
                    rel_type = data.get("relationship", "related_to")
                    relationships.append(f"{source_name} {rel_type} this")

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
