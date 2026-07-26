"""EntityNameMatcher — multi-tier entity resolution against the knowledge graph.

Tier 1: Substring matching (fast, exact)
Tier 2: Scene seeding (always-on, guarantees context for current scene)
Tier 3: Vector similarity (fallback for fuzzy/descriptive references)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

from ..identity import names_addressed_in_text
from .models import slugify

if TYPE_CHECKING:
    from .graph import KnowledgeGraph
    from ...memory.vector_store import VectorStore

logger = structlog.get_logger()


# Semantic entity retrieval is a high-recall fallback, but a bare embedding
# match is not enough authority to put an off-screen campaign entity into the
# narrator prompt.  Require one distinctive lexical anchor from the entity's
# name, aliases, or description.  This still supports references such as
# "the scarred dwarf" while preventing generic phrases such as "the man" from
# recalling every male NPC with a vaguely similar description.
_GENERIC_ENTITY_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "he", "her", "hers", "him", "his", "i", "in", "into",
    "is", "it", "its", "item", "location", "male", "female", "man", "my",
    "named", "npc", "object", "of", "old", "on", "or", "our", "person",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "place", "she", "someone", "that", "the", "their", "them", "these",
    "they", "this", "those", "to", "unknown", "was", "we", "were",
    "with", "woman", "you", "young", "your",
}


def _distinctive_words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", (text or "").casefold())
        if len(word) >= 3 and word not in _GENERIC_ENTITY_WORDS
    }


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
        """Return node_ids for the full scene context.

        WorldState is a durable campaign record, not a scene-membership list:
        it can retain important off-screen NPCs and historical connections.
        Treating that catalog as presence caused old entities to enter every
        narrator prompt and contaminate future narrative memories.

        Active quests and known map connections remain available through
        structured state and explicit mention retrieval; they are not ambient
        graph seeds. Scene items are cleared when the party moves, so they are
        safe to treat as current presence.
        """
        seeds: list[str] = []
        seen: set[str] = set()
        if not world_state:
            return seeds

        def _try_add(node_id: str) -> None:
            if node_id and node_id not in seen and self._graph.has_node(node_id):
                seeds.append(node_id)
                seen.add(node_id)

        # Current location
        if world_state.current_location:
            _try_add(slugify(world_state.current_location))

        # NPCState.id is the KG identity anchor. Only NPCs whose canonical
        # location equals the party's current location are scene members.
        for npc_state in world_state.get_npcs_at_location():
            _try_add(npc_state.id)

        # Scene items (objects present in current location)
        for item_id in world_state.scene_items:
            _try_add(slugify(item_id))

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
            query_words = _distinctive_words(text)
            grounded: list[str] = []
            for result in results:
                node_id = result["node_id"]
                entity = self._graph.get_entity(node_id)
                if entity is None or not self._graph.has_node(node_id):
                    continue
                # A name/alias token is a strong identity anchor. Descriptive
                # prose is much noisier, so require two independent words
                # there. A single accidental overlap (the production failure
                # was the word "one") must not recall an off-screen entity.
                identity_words = _distinctive_words(
                    " ".join([entity.name, *entity.aliases])
                )
                description_words = _distinctive_words(
                    entity.properties.get("description", "")
                )
                if (
                    query_words.intersection(identity_words)
                    or len(query_words.intersection(description_words)) >= 2
                ):
                    grounded.append(node_id)
            return grounded
        except Exception as e:
            logger.warning("vector_match_failed", error=str(e), exc_info=True)
            return []

    def rebuild_index(self) -> None:
        """Force rebuild of the lookup index after graph mutations."""
        self._index = None


def action_entity_names(graph: "KnowledgeGraph | None", text: str) -> list[str]:
    """Every name borne by the entities the player's text names outright.

    Tier-1 candidates come from the same :meth:`match` that seeds graph
    context, then :func:`entity_named_in_text` decides. That second gate is
    not redundant: ``match`` is deliberately loose — bare substring, any
    name over two characters — because a speculative graph seed only costs
    an entity card in the prompt. Anchoring durable campaign facts is a
    stronger claim than that evidence supports, so it re-tests on token
    boundaries and ignores placeholder names. "I push through the brambles"
    seeds Bram's card; it must not open Bram's file.

    Names, not node ids: callers anchor prose against these. The player's
    own wording is NOT the anchor — resolving "the black arch" to Ash Gate
    is the whole point.

    Never raises. This runs on the turn's hot path purely to widen
    retrieval; a graph problem must degrade to the scene-only projection,
    not cost the player their turn.
    """
    if graph is None or not text:
        return []
    try:
        matcher = EntityNameMatcher(graph)
        entities = [
            entity for entity in (
                graph.get_entity(node_id) for node_id in matcher.match(text)
            ) if entity is not None
        ]
        # One resolution across all candidates, so the longest name claims its
        # tokens. Per-entity calls made "I ask Mara Venn ..." also name an
        # unrelated Mara, and put her canon in the prompt.
        names: list[str] = []
        for matched in names_addressed_in_text(
            text, [[entity.name, *entity.aliases] for entity in entities]
        ):
            names.extend(matched)
        return names
    except Exception as e:
        logger.warning("action_entity_names_failed", error=str(e), exc_info=True)
        return []
