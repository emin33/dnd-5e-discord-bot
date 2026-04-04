"""EntityNameMatcher — multi-tier entity resolution against the knowledge graph.

Tier 1: Substring matching (fast, exact)
Tier 2: Scene seeding (always-on, guarantees context for current scene)
Tier 3: Vector similarity (fallback for fuzzy/descriptive references)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from .models import slugify

if TYPE_CHECKING:
    from .graph import KnowledgeGraph
    from ...memory.vector_store import VectorStore

logger = structlog.get_logger()


class EntityNameMatcher:
    """Multi-tier entity resolution against the knowledge graph."""

    def __init__(self, graph: "KnowledgeGraph"):
        self._graph = graph
        self._index: list[tuple[str, str]] | None = None  # (lowercase_name, node_id)

    def _build_index(self) -> list[tuple[str, str]]:
        """Build sorted lookup index from current graph state."""
        names = self._graph.get_all_names()  # {lowercase_name: node_id}
        # Sort by name length descending so longer matches take priority
        return sorted(names.items(), key=lambda pair: len(pair[0]), reverse=True)

    # ------------------------------------------------------------------
    # Tier 1: Substring matching (existing)
    # ------------------------------------------------------------------

    def match(self, text: str) -> list[str]:
        """Return node_ids for entities whose names appear in the text."""
        if not text:
            return []

        if self._index is None:
            self._index = self._build_index()

        text_lower = text.lower()
        matched_ids: list[str] = []
        seen: set[str] = set()

        for name, node_id in self._index:
            if node_id in seen:
                continue
            # Skip very short names (1-2 chars) to avoid false positives
            if len(name) <= 2:
                continue
            if name in text_lower:
                matched_ids.append(node_id)
                seen.add(node_id)

        return matched_ids

    # ------------------------------------------------------------------
    # Tier 2: Scene seeding (always-on)
    # ------------------------------------------------------------------

    def scene_seeds(self, world_state) -> list[str]:
        """Return node_ids for the current scene: location + present NPCs.

        Guarantees graph context injection every turn regardless of
        whether the player names entities verbatim.
        """
        seeds: list[str] = []
        if not world_state:
            return seeds

        # Current location
        if world_state.current_location:
            loc_id = slugify(world_state.current_location)
            if self._graph.has_node(loc_id):
                seeds.append(loc_id)

        # NPCs at current location
        for npc_name, npc_state in world_state.npcs.items():
            if npc_state.location == world_state.current_location and npc_state.alive:
                npc_id = slugify(npc_name)
                if self._graph.has_node(npc_id):
                    seeds.append(npc_id)

        return seeds

    # ------------------------------------------------------------------
    # Tier 3: Vector similarity fallback
    # ------------------------------------------------------------------

    def vector_match(
        self,
        text: str,
        campaign_id: str,
        vector_store: "VectorStore",
    ) -> list[str]:
        """Semantic fallback: search entity descriptions by vector similarity.

        Only called when substring match returns empty. Searches ChromaDB
        for entity descriptions that are semantically close to the player's text.
        """
        if not text:
            return []

        try:
            results = vector_store.search_entities(
                campaign_id=campaign_id,
                query=text,
                n_results=3,
            )
            # Only return node_ids that actually exist in the graph
            return [
                r["node_id"] for r in results
                if self._graph.has_node(r["node_id"])
            ]
        except Exception as e:
            logger.warning("vector_match_failed", error=str(e))
            return []

    def rebuild_index(self) -> None:
        """Force rebuild of the lookup index after graph mutations."""
        self._index = None
