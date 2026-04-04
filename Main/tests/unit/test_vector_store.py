"""Integration tests for VectorStore — entity indexing, search, and narrative recall.

Uses a real ChromaDB instance (temp directory) to catch API behavior changes
like the 1.4.0 silent-update bug that broke entity indexing.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from dnd_bot.memory.vector_store import VectorStore


@pytest.fixture
def vector_store(tmp_path):
    """Create a VectorStore with a temp ChromaDB directory."""
    vs = VectorStore(persist_directory=str(tmp_path / "chroma"))
    yield vs


CAMPAIGN_ID = "test-campaign-001"


# ======================================================================
# Entity description indexing
# ======================================================================


class TestEntityDescriptionIndexing:
    """Tests for add_entity_description + search_entities."""

    def test_add_entity_description_persists(self, vector_store):
        """Entity descriptions must actually be stored (not silently dropped)."""
        result = vector_store.add_entity_description(
            campaign_id=CAMPAIGN_ID,
            node_id="grimjaw",
            entity_type="npc",
            name="Grimjaw",
            description="A scarred dwarf blacksmith with a missing left eye",
        )
        assert result is True

        # Verify it's actually in the collection
        col = vector_store._get_collection(CAMPAIGN_ID)
        stored = col.get(ids=["entity_grimjaw"])
        assert stored["ids"] == ["entity_grimjaw"]
        assert "scarred dwarf" in stored["documents"][0]

    def test_add_entity_description_upsert(self, vector_store):
        """Updating an existing entity description replaces it."""
        vector_store.add_entity_description(
            campaign_id=CAMPAIGN_ID,
            node_id="grimjaw",
            entity_type="npc",
            name="Grimjaw",
            description="A scarred dwarf",
        )
        # Update with richer description
        vector_store.add_entity_description(
            campaign_id=CAMPAIGN_ID,
            node_id="grimjaw",
            entity_type="npc",
            name="Grimjaw",
            description="A grizzled dwarf blacksmith, heavily scarred, missing his left eye",
        )

        col = vector_store._get_collection(CAMPAIGN_ID)
        stored = col.get(ids=["entity_grimjaw"])
        assert "grizzled dwarf blacksmith" in stored["documents"][0]

    def test_add_multiple_entities(self, vector_store):
        """Multiple entities can be indexed independently."""
        entities = [
            ("grimjaw", "npc", "Grimjaw", "A scarred dwarf blacksmith"),
            ("ironforge-tavern", "location", "Ironforge Tavern", "A smoky tavern with low ceilings and ale-stained tables"),
            ("find-the-sword", "quest", "Find the Sword", "Retrieve the ancient blade from the goblin caves"),
        ]
        for node_id, etype, name, desc in entities:
            vector_store.add_entity_description(
                campaign_id=CAMPAIGN_ID,
                node_id=node_id,
                entity_type=etype,
                name=name,
                description=desc,
            )

        col = vector_store._get_collection(CAMPAIGN_ID)
        stored = col.get(ids=[f"entity_{e[0]}" for e in entities])
        assert len(stored["ids"]) == 3

    def test_aliases_included_in_content(self, vector_store):
        """Aliases are appended to the indexed content for broader matching."""
        vector_store.add_entity_description(
            campaign_id=CAMPAIGN_ID,
            node_id="grimjaw",
            entity_type="npc",
            name="Grimjaw",
            description="A dwarf blacksmith",
            aliases=["the scarred dwarf", "old one-eye"],
        )

        col = vector_store._get_collection(CAMPAIGN_ID)
        stored = col.get(ids=["entity_grimjaw"])
        doc = stored["documents"][0]
        assert "the scarred dwarf" in doc
        assert "old one-eye" in doc


# ======================================================================
# Entity vector search
# ======================================================================


class TestEntityVectorSearch:
    """Tests for search_entities — the 'describe it, don't name it' flow."""

    def _seed_entities(self, vs):
        """Seed a small world of entities for search tests."""
        entities = [
            ("grimjaw", "npc", "Grimjaw",
             "A scarred dwarf blacksmith with a missing left eye and soot-covered arms"),
            ("elara", "npc", "Elara",
             "A tall elven woman with silver hair, carries a staff topped with a glowing crystal"),
            ("ironforge-tavern", "location", "Ironforge Tavern",
             "A smoky underground tavern with low stone ceilings, ale-stained oak tables, and a roaring hearth"),
            ("goblin-caves", "location", "Goblin Caves",
             "Dark winding tunnels beneath the mountain, filled with the stench of rot and goblins"),
            ("find-the-blade", "quest", "Find the Ancient Blade",
             "Retrieve the legendary sword Frostbite from deep within the goblin caves"),
        ]
        for node_id, etype, name, desc in entities:
            vs.add_entity_description(
                campaign_id=CAMPAIGN_ID,
                node_id=node_id,
                entity_type=etype,
                name=name,
                description=desc,
            )

    def test_search_by_description(self, vector_store):
        """Player describes entity without naming it — vector search finds it."""
        self._seed_entities(vector_store)

        results = vector_store.search_entities(
            campaign_id=CAMPAIGN_ID,
            query="the one-eyed dwarf at the forge",
        )
        assert len(results) > 0
        assert results[0]["node_id"] == "grimjaw"

    def test_search_by_appearance(self, vector_store):
        """Search by physical appearance finds the right NPC."""
        self._seed_entities(vector_store)

        results = vector_store.search_entities(
            campaign_id=CAMPAIGN_ID,
            query="the elf woman with the glowing staff",
        )
        assert len(results) > 0
        assert results[0]["node_id"] == "elara"

    def test_search_location_by_description(self, vector_store):
        """Describing a location finds the right place."""
        self._seed_entities(vector_store)

        results = vector_store.search_entities(
            campaign_id=CAMPAIGN_ID,
            query="that underground bar with the stone ceiling",
        )
        assert len(results) > 0
        assert results[0]["node_id"] == "ironforge-tavern"

    def test_search_respects_max_distance(self, vector_store):
        """Results beyond max_distance are filtered out."""
        self._seed_entities(vector_store)

        # Very strict threshold — may return nothing for vague queries
        results = vector_store.search_entities(
            campaign_id=CAMPAIGN_ID,
            query="something completely unrelated like a spaceship",
            max_distance=0.3,
        )
        # Either empty or very few results, none should be a strong match
        for r in results:
            assert r["distance"] <= 0.3

    def test_search_empty_collection(self, vector_store):
        """Searching an empty collection returns empty results, not an error."""
        results = vector_store.search_entities(
            campaign_id=CAMPAIGN_ID,
            query="anything",
        )
        assert results == []

    def test_search_returns_metadata(self, vector_store):
        """Search results include node_id, name, entity_type, and distance."""
        self._seed_entities(vector_store)

        results = vector_store.search_entities(
            campaign_id=CAMPAIGN_ID,
            query="scarred blacksmith",
        )
        assert len(results) > 0
        r = results[0]
        assert "node_id" in r
        assert "name" in r
        assert "distance" in r


# ======================================================================
# Narrative chunk storage and recall
# ======================================================================


class TestNarrativeChunkRecall:
    """Tests for add_narrative_chunk + recall_narratives_for_entities."""

    def test_narrative_chunk_stored_and_recalled(self, vector_store):
        """Narrative chunks tagged with entity IDs can be recalled."""
        vector_store.add_narrative_chunk(
            campaign_id=CAMPAIGN_ID,
            chunk_id="turn-5",
            narrative_text="Grimjaw slams his hammer down on the anvil, sparks flying. "
                          "'The blade you seek,' he growls, 'was taken into the caves.'",
            entity_ids=["grimjaw", "find-the-blade"],
            turn=5,
            location="ironforge-tavern",
        )

        # current_turn must exceed turn + min_turn_age (default 12)
        # so turn-5 chunks are old enough to recall at turn 25+
        results = vector_store.recall_narratives_for_entities(
            campaign_id=CAMPAIGN_ID,
            entity_ids=["grimjaw"],
            query_text="what did the dwarf say about the blade",
            max_results=3,
            current_turn=25,
        )
        assert len(results) >= 1
        assert "Grimjaw" in results[0]["content"]

    def test_narrative_recall_empty_when_no_chunks(self, vector_store):
        """Recall returns empty when no narrative chunks exist."""
        results = vector_store.recall_narratives_for_entities(
            campaign_id=CAMPAIGN_ID,
            entity_ids=["grimjaw"],
            query_text="anything",
            max_results=3,
            current_turn=25,
        )
        assert results == []

    def test_multiple_chunks_ranked_by_relevance(self, vector_store):
        """Multiple chunks about the same entity are ranked by relevance."""
        vector_store.add_narrative_chunk(
            campaign_id=CAMPAIGN_ID,
            chunk_id="turn-3",
            narrative_text="Grimjaw grunts a greeting and returns to hammering a horseshoe.",
            entity_ids=["grimjaw"],
            turn=3,
            location="ironforge-tavern",
        )
        vector_store.add_narrative_chunk(
            campaign_id=CAMPAIGN_ID,
            chunk_id="turn-7",
            narrative_text="Grimjaw leans in close. 'The caves are guarded by a troll,' he whispers. "
                          "'Take the left passage at the fork.'",
            entity_ids=["grimjaw", "goblin-caves"],
            turn=7,
            location="ironforge-tavern",
        )

        results = vector_store.recall_narratives_for_entities(
            campaign_id=CAMPAIGN_ID,
            entity_ids=["grimjaw"],
            query_text="what did grimjaw say about the caves and the troll",
            max_results=2,
            current_turn=30,
        )
        assert len(results) >= 1
        # The troll/caves chunk should rank higher for this query
        assert "troll" in results[0]["content"]


# ======================================================================
# Entity + narrative integration (the full web)
# ======================================================================


class TestEntityNarrativeIntegration:
    """Tests the full 'describe it, find it, recall context' pipeline."""

    def test_describe_npc_find_entity_recall_narration(self, vector_store):
        """
        Full pipeline:
        1. Entity indexed with description
        2. Player describes entity without naming it
        3. Vector search finds the entity
        4. Narrative recall pulls past prose about that entity
        """
        # 1. Index entity
        vector_store.add_entity_description(
            campaign_id=CAMPAIGN_ID,
            node_id="hagrid",
            entity_type="npc",
            name="Hagrid",
            description="An enormous bearded man, nearly eight feet tall, with wild tangled hair",
            aliases=["the bearded man", "the giant"],
        )

        # 2. Store a narrative about this entity
        vector_store.add_narrative_chunk(
            campaign_id=CAMPAIGN_ID,
            chunk_id="turn-12",
            narrative_text="The enormous man ducks through the doorway, his wild beard "
                          "trailing snowflakes. 'Name's Hagrid,' he rumbles.",
            entity_ids=["hagrid"],
            turn=12,
            location="the-tavern",
        )

        # 3. Player describes without naming
        search_results = vector_store.search_entities(
            campaign_id=CAMPAIGN_ID,
            query="that huge bearded fellow who came in from the snow",
        )
        assert len(search_results) > 0
        assert search_results[0]["node_id"] == "hagrid"

        # 4. Use found entity to recall past narration
        found_id = search_results[0]["node_id"]
        recalled = vector_store.recall_narratives_for_entities(
            campaign_id=CAMPAIGN_ID,
            entity_ids=[found_id],
            query_text="what happened when the big man arrived",
            max_results=2,
            current_turn=30,
        )
        assert len(recalled) >= 1
        assert "Hagrid" in recalled[0]["content"]
