"""Tests for the knowledge graph system — models, graph, bridge, matcher."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from dnd_bot.game.knowledge.models import (
    AddEdge,
    AddNode,
    DEFAULT_WEIGHTS,
    Entity,
    EntityType,
    GraphOperation,
    Relationship,
    RelationType,
    RemoveEdge,
    RemoveNode,
    UpdateNode,
    slugify,
)
from dnd_bot.game.knowledge.graph import KnowledgeGraph
from dnd_bot.game.knowledge.bridge import DeltaBridge
from dnd_bot.game.knowledge.matcher import EntityNameMatcher
from dnd_bot.game.world_state import StateDelta, NPCState, NPCUpdate, QuestState, WorldState


# ======================================================================
# slugify
# ======================================================================

class TestSlugify:
    def test_basic(self):
        assert slugify("Grimjaw") == "grimjaw"

    def test_spaces_to_hyphens(self):
        assert slugify("Ironforge Tavern") == "ironforge-tavern"

    def test_special_characters_stripped(self):
        assert slugify("The Dragon's Lair!") == "the-dragons-lair"

    def test_multiple_spaces(self):
        assert slugify("  Market   Square  ") == "market-square"

    def test_already_slugified(self):
        assert slugify("some-slug") == "some-slug"

    def test_empty_string(self):
        assert slugify("") == ""

    def test_unicode_stripped(self):
        assert slugify("Café del Sol") == "caf-del-sol"

    def test_numbers_preserved(self):
        assert slugify("Level 3 Dungeon") == "level-3-dungeon"


# ======================================================================
# Entity model
# ======================================================================

class TestEntityModel:
    def test_create_npc(self):
        e = Entity(
            node_id="grimjaw",
            entity_type=EntityType.NPC,
            name="Grimjaw",
            campaign_id="camp-1",
            properties={"disposition": "hostile", "alive": "true"},
        )
        assert e.entity_type == EntityType.NPC
        assert e.properties["disposition"] == "hostile"

    def test_create_location(self):
        e = Entity(
            node_id="tavern",
            entity_type=EntityType.LOCATION,
            name="Tavern",
            campaign_id="camp-1",
        )
        assert e.entity_type == EntityType.LOCATION
        assert e.aliases == []
        assert e.properties == {}

    def test_entity_with_aliases(self):
        e = Entity(
            node_id="barkeep-thom",
            entity_type=EntityType.NPC,
            name="Barkeep Thom",
            campaign_id="camp-1",
            aliases=["the barkeep", "old thom"],
        )
        assert len(e.aliases) == 2


# ======================================================================
# Relationship model
# ======================================================================

class TestRelationshipModel:
    def test_create_with_default_weight(self):
        r = Relationship(
            source_id="grimjaw",
            target_id="tavern",
            relation_type=RelationType.LOCATED_AT,
            weight=DEFAULT_WEIGHTS[RelationType.LOCATED_AT],
            campaign_id="camp-1",
        )
        assert r.weight == 0.3

    def test_default_weights_all_present(self):
        for rt in RelationType:
            assert rt in DEFAULT_WEIGHTS


# ======================================================================
# Graph operations
# ======================================================================

class TestGraphOperations:
    def test_add_node_op(self):
        entity = Entity(
            node_id="test", entity_type=EntityType.NPC,
            name="Test", campaign_id="c",
        )
        op = AddNode(entity=entity)
        assert op.op == "add_node"

    def test_update_node_op(self):
        op = UpdateNode(node_id="test", properties={"alive": "false"})
        assert op.op == "update_node"
        assert op.aliases is None

    def test_remove_edge_op(self):
        op = RemoveEdge(
            source_id="a", target_id="b",
            relation_type=RelationType.LOCATED_AT,
        )
        assert op.op == "remove_edge"


# ======================================================================
# KnowledgeGraph
# ======================================================================

def _make_mock_repo():
    """Create a mock repository that does nothing on persistence calls."""
    repo = AsyncMock()
    repo.load_nodes = AsyncMock(return_value=[])
    repo.load_edges = AsyncMock(return_value=[])
    repo.upsert_node = AsyncMock()
    repo.upsert_edge = AsyncMock()
    repo.delete_node = AsyncMock()
    repo.delete_edge = AsyncMock()
    repo.delete_edges_by_source = AsyncMock()
    return repo


def _make_entity(node_id, entity_type=EntityType.NPC, name=None, **props):
    return Entity(
        node_id=node_id,
        entity_type=entity_type,
        name=name or node_id.replace("-", " ").title(),
        campaign_id="test-campaign",
        properties=props,
    )


def _make_relationship(source_id, target_id, rel_type, weight=None):
    return Relationship(
        source_id=source_id,
        target_id=target_id,
        relation_type=rel_type,
        weight=weight or DEFAULT_WEIGHTS.get(rel_type, 1.0),
        campaign_id="test-campaign",
    )


class TestKnowledgeGraph:

    @pytest.fixture
    def kg(self):
        repo = _make_mock_repo()
        return KnowledgeGraph("test-campaign", repo)

    async def test_load_empty(self, kg):
        await kg.load()
        assert kg.node_count() == 0
        assert kg.edge_count() == 0

    async def test_add_node_and_retrieve(self, kg):
        await kg.load()
        entity = _make_entity("grimjaw", name="Grimjaw")
        await kg.apply_operations([AddNode(entity=entity)])

        assert kg.node_count() == 1
        assert kg.get_entity("grimjaw").name == "Grimjaw"

    async def test_add_edge_between_nodes(self, kg):
        await kg.load()
        npc = _make_entity("grimjaw", name="Grimjaw")
        loc = _make_entity("tavern", EntityType.LOCATION, "Tavern")
        rel = _make_relationship("grimjaw", "tavern", RelationType.LOCATED_AT)

        await kg.apply_operations([
            AddNode(entity=npc),
            AddNode(entity=loc),
            AddEdge(relationship=rel),
        ])

        assert kg.edge_count() == 1

    async def test_add_edge_missing_target_rejected(self, kg):
        await kg.load()
        npc = _make_entity("grimjaw", name="Grimjaw")
        rel = _make_relationship("grimjaw", "nonexistent", RelationType.LOCATED_AT)

        rejections = await kg.apply_operations([
            AddNode(entity=npc),
            AddEdge(relationship=rel),
        ])

        assert len(rejections) == 1
        assert "not found" in rejections[0].lower()

    async def test_update_node_properties(self, kg):
        await kg.load()
        entity = _make_entity("grimjaw", disposition="hostile")
        await kg.apply_operations([AddNode(entity=entity)])

        await kg.apply_operations([
            UpdateNode(node_id="grimjaw", properties={"disposition": "neutral"}),
        ])

        updated = kg.get_entity("grimjaw")
        assert updated.properties["disposition"] == "neutral"

    async def test_update_nonexistent_node_rejected(self, kg):
        await kg.load()
        rejections = await kg.apply_operations([
            UpdateNode(node_id="nonexistent", properties={"foo": "bar"}),
        ])
        assert len(rejections) == 1

    async def test_remove_node_cascades_edges(self, kg):
        await kg.load()
        npc = _make_entity("grimjaw")
        loc = _make_entity("tavern", EntityType.LOCATION, "Tavern")
        rel = _make_relationship("grimjaw", "tavern", RelationType.LOCATED_AT)

        await kg.apply_operations([
            AddNode(entity=npc),
            AddNode(entity=loc),
            AddEdge(relationship=rel),
        ])
        assert kg.edge_count() == 1

        await kg.apply_operations([RemoveNode(node_id="grimjaw")])
        assert kg.node_count() == 1  # tavern remains
        assert kg.edge_count() == 0  # edge removed
        assert kg.get_entity("grimjaw") is None

    async def test_add_duplicate_node_merges(self, kg):
        await kg.load()
        e1 = _make_entity("grimjaw", disposition="hostile")
        e2 = _make_entity("grimjaw", description="A gruff dwarf")

        await kg.apply_operations([AddNode(entity=e1)])
        await kg.apply_operations([AddNode(entity=e2)])

        merged = kg.get_entity("grimjaw")
        assert merged.properties["disposition"] == "hostile"
        assert merged.properties["description"] == "A gruff dwarf"

    async def test_get_all_names(self, kg):
        await kg.load()
        entity = _make_entity("grimjaw", name="Grimjaw")
        entity.aliases = ["the dwarf", "old grim"]
        await kg.apply_operations([AddNode(entity=entity)])

        names = kg.get_all_names()
        assert "grimjaw" in names
        assert "the dwarf" in names
        assert "old grim" in names
        assert names["grimjaw"] == "grimjaw"

    async def test_has_node(self, kg):
        await kg.load()
        assert not kg.has_node("grimjaw")
        await kg.apply_operations([AddNode(entity=_make_entity("grimjaw"))])
        assert kg.has_node("grimjaw")


# ======================================================================
# BFS Subgraph Retrieval
# ======================================================================

class TestSubgraphRetrieval:

    @pytest.fixture
    async def populated_kg(self):
        """Build a small graph: NPC → Location → NPC chain."""
        repo = _make_mock_repo()
        kg = KnowledgeGraph("test-campaign", repo)
        await kg.load()

        npc1 = _make_entity("grimjaw", name="Grimjaw", disposition="hostile")
        npc2 = _make_entity("barkeep", name="Barkeep Thom", disposition="friendly")
        loc = _make_entity("tavern", EntityType.LOCATION, "The Tavern", description="A cozy tavern")
        item = _make_entity("sword", EntityType.ITEM, "Rusted Sword")

        await kg.apply_operations([
            AddNode(entity=npc1),
            AddNode(entity=npc2),
            AddNode(entity=loc),
            AddNode(entity=item),
            AddEdge(relationship=_make_relationship("grimjaw", "tavern", RelationType.LOCATED_AT)),
            AddEdge(relationship=_make_relationship("barkeep", "tavern", RelationType.LOCATED_AT)),
            AddEdge(relationship=_make_relationship("grimjaw", "sword", RelationType.OWNS)),
        ])
        return kg

    async def test_seed_retrieves_neighbors(self, populated_kg):
        result = populated_kg.get_context_subgraph(["grimjaw"])
        names = {e["name"] for e in result}
        assert "Grimjaw" in names
        assert "The Tavern" in names  # 1 hop via LOCATED_AT

    async def test_two_hop_retrieval(self, populated_kg):
        result = populated_kg.get_context_subgraph(["grimjaw"], radius=2.0)
        names = {e["name"] for e in result}
        # Grimjaw → Tavern → Barkeep (2 hops: 0.3 + 0.3 = 0.6, within radius)
        assert "Barkeep Thom" in names

    async def test_empty_seeds_returns_empty(self, populated_kg):
        result = populated_kg.get_context_subgraph([])
        assert result == []

    async def test_nonexistent_seed_returns_empty(self, populated_kg):
        result = populated_kg.get_context_subgraph(["nonexistent"])
        assert result == []

    async def test_max_entities_cap(self, populated_kg):
        result = populated_kg.get_context_subgraph(["grimjaw"], max_entities=2)
        assert len(result) <= 2

    async def test_yaml_output(self, populated_kg):
        yaml_str = populated_kg.to_context_yaml(["grimjaw"])
        assert yaml_str != ""
        assert "Grimjaw" in yaml_str
        assert "known_entities" in yaml_str

    async def test_yaml_empty_on_no_match(self, populated_kg):
        assert populated_kg.to_context_yaml(["nonexistent"]) == ""

    async def test_relationships_in_output(self, populated_kg):
        result = populated_kg.get_context_subgraph(["grimjaw"])
        grimjaw = next(e for e in result if e["name"] == "Grimjaw")
        assert "relationships" in grimjaw
        rel_strs = grimjaw["relationships"]
        assert any("located_at" in r for r in rel_strs)

    async def test_properties_in_output(self, populated_kg):
        result = populated_kg.get_context_subgraph(["grimjaw"])
        grimjaw = next(e for e in result if e["name"] == "Grimjaw")
        assert grimjaw.get("disposition") == "hostile"


# ======================================================================
# DeltaBridge
# ======================================================================

class TestDeltaBridge:

    @pytest.fixture
    def bridge(self):
        return DeltaBridge("test-campaign")

    @pytest.fixture
    def world_state(self):
        ws = WorldState()
        ws.current_location = "Market Square"
        return ws

    def test_empty_delta_no_ops(self, bridge, world_state):
        delta = StateDelta()
        ops = bridge.convert(delta, world_state)
        assert ops == []

    def test_new_npc_creates_node_and_edge(self, bridge, world_state):
        delta = StateDelta(
            new_npcs=[NPCState(
                name="Grimjaw",
                disposition="hostile",
                description="A gruff dwarf",
                location="Market Square",
            )]
        )
        ops = bridge.convert(delta, world_state)

        add_nodes = [o for o in ops if isinstance(o, AddNode)]
        add_edges = [o for o in ops if isinstance(o, AddEdge)]

        # Should create NPC node + location node (placeholder if needed) + LOCATED_AT edge
        npc_nodes = [n for n in add_nodes if n.entity.entity_type == EntityType.NPC]
        assert len(npc_nodes) == 1
        assert npc_nodes[0].entity.name == "Grimjaw"
        assert npc_nodes[0].entity.properties["disposition"] == "hostile"

        located_at = [e for e in add_edges if e.relationship.relation_type == RelationType.LOCATED_AT]
        assert len(located_at) == 1

    def test_location_change_creates_location_node(self, bridge, world_state):
        delta = StateDelta(
            location_change="The Dark Forest",
            location_description="A dense, foreboding forest",
        )
        ops = bridge.convert(delta, world_state)

        add_nodes = [o for o in ops if isinstance(o, AddNode)]
        loc_nodes = [n for n in add_nodes if n.entity.entity_type == EntityType.LOCATION]
        assert len(loc_nodes) >= 1
        assert any(n.entity.name == "The Dark Forest" for n in loc_nodes)

    def test_location_change_creates_connected_to_edge(self, bridge, world_state):
        delta = StateDelta(
            location_change="The Dark Forest",
            location_description="A dense, foreboding forest",
        )
        ops = bridge.convert(delta, world_state, previous_location="Market Square")

        add_edges = [o for o in ops if isinstance(o, AddEdge)]
        connected = [e for e in add_edges if e.relationship.relation_type == RelationType.CONNECTED_TO]
        # Bidirectional: Market Square → Dark Forest AND Dark Forest → Market Square
        assert len(connected) == 2

    def test_location_change_no_edge_without_previous(self, bridge, world_state):
        delta = StateDelta(
            location_change="The Dark Forest",
            location_description="A dense, foreboding forest",
        )
        ops = bridge.convert(delta, world_state, previous_location="")

        add_edges = [o for o in ops if isinstance(o, AddEdge)]
        connected = [e for e in add_edges if e.relationship.relation_type == RelationType.CONNECTED_TO]
        assert len(connected) == 0

    def test_new_connections_bidirectional(self, bridge, world_state):
        delta = StateDelta(new_connections=["North Gate", "Eastern Road"])
        # Current location is "Market Square"
        ops = bridge.convert(delta, world_state)

        add_edges = [o for o in ops if isinstance(o, AddEdge)]
        connected_to = [e for e in add_edges if e.relationship.relation_type == RelationType.CONNECTED_TO]
        # Each connection should produce 2 edges (bidirectional)
        assert len(connected_to) == 4  # 2 connections × 2 directions

    def test_npc_update_generates_update_node(self, bridge, world_state):
        delta = StateDelta(
            npc_updates=[NPCUpdate(name="Grimjaw", disposition="neutral")]
        )
        ops = bridge.convert(delta, world_state, existing_node_ids={"grimjaw"})

        update_nodes = [o for o in ops if isinstance(o, UpdateNode)]
        assert len(update_nodes) == 1
        assert update_nodes[0].properties["disposition"] == "neutral"

    def test_removed_npc_clears_location_not_node(self, bridge, world_state):
        delta = StateDelta(removed_npcs=["Grimjaw"])
        ops = bridge.convert(delta, world_state)

        # Should update node (clear location) + remove edge, but NOT remove node
        remove_nodes = [o for o in ops if isinstance(o, RemoveNode)]
        assert len(remove_nodes) == 0

        update_nodes = [o for o in ops if isinstance(o, UpdateNode)]
        assert len(update_nodes) == 1
        assert update_nodes[0].properties["location"] == ""

        remove_edges = [o for o in ops if isinstance(o, RemoveEdge)]
        assert len(remove_edges) == 1
        assert remove_edges[0].relation_type == RelationType.LOCATED_AT

    def test_npc_death_keeps_node(self, bridge, world_state):
        delta = StateDelta(
            npc_updates=[NPCUpdate(name="Grimjaw", alive=False)]
        )
        ops = bridge.convert(delta, world_state, existing_node_ids={"grimjaw"})

        remove_nodes = [o for o in ops if isinstance(o, RemoveNode)]
        assert len(remove_nodes) == 0

        update_nodes = [o for o in ops if isinstance(o, UpdateNode)]
        assert len(update_nodes) == 1
        assert update_nodes[0].properties["alive"] == "false"

    def test_placeholder_location_created_for_unknown_npc_location(self, bridge, world_state):
        delta = StateDelta(
            new_npcs=[NPCState(name="Spy", location="Hidden Base")]
        )
        ops = bridge.convert(delta, world_state)

        add_nodes = [o for o in ops if isinstance(o, AddNode)]
        loc_nodes = [n for n in add_nodes if n.entity.entity_type == EntityType.LOCATION]
        # "Hidden Base" should be created as a placeholder
        placeholders = [n for n in loc_nodes if n.entity.properties.get("placeholder") == "true"]
        assert len(placeholders) >= 1

    def test_npc_location_change_updates_edge(self, bridge, world_state):
        delta = StateDelta(
            npc_updates=[NPCUpdate(name="Grimjaw", location="Castle")]
        )
        ops = bridge.convert(delta, world_state, existing_node_ids={"grimjaw"})

        remove_edges = [o for o in ops if isinstance(o, RemoveEdge)]
        add_edges = [o for o in ops if isinstance(o, AddEdge)]

        assert len(remove_edges) == 1  # old LOCATED_AT removed
        located_at_adds = [e for e in add_edges if e.relationship.relation_type == RelationType.LOCATED_AT]
        assert len(located_at_adds) == 1  # new LOCATED_AT added


# ======================================================================
# EntityNameMatcher
# ======================================================================

class TestEntityNameMatcher:

    @pytest.fixture
    async def matcher_kg(self):
        repo = _make_mock_repo()
        kg = KnowledgeGraph("test-campaign", repo)
        await kg.load()

        npc = _make_entity("grimjaw", name="Grimjaw")
        npc.aliases = ["the dwarf", "old grim"]
        loc = _make_entity("ironforge-tavern", EntityType.LOCATION, "Ironforge Tavern")
        item = _make_entity("iron-dagger", EntityType.ITEM, "Iron Dagger")

        await kg.apply_operations([
            AddNode(entity=npc),
            AddNode(entity=loc),
            AddNode(entity=item),
        ])
        return kg

    async def test_exact_name_match(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        result = matcher.match("I talk to Grimjaw")
        assert "grimjaw" in result

    async def test_case_insensitive(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        result = matcher.match("i talk to grimjaw")
        assert "grimjaw" in result

    async def test_alias_match(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        result = matcher.match("the dwarf looks angry")
        assert "grimjaw" in result

    async def test_longer_name_preferred(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        # "Ironforge Tavern" should match before "Iron Dagger" would partially match
        result = matcher.match("I go to the Ironforge Tavern")
        assert "ironforge-tavern" in result

    async def test_no_match_returns_empty(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        result = matcher.match("I look around the empty room")
        assert result == []

    async def test_multiple_matches(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        result = matcher.match("Grimjaw sits in the Ironforge Tavern")
        assert "grimjaw" in result
        assert "ironforge-tavern" in result

    async def test_empty_text(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        assert matcher.match("") == []

    async def test_substring_in_sentence(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        result = matcher.match("What does old grim want from us?")
        assert "grimjaw" in result

    async def test_rebuild_index(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        matcher.match("test")  # builds initial index
        matcher.rebuild_index()
        assert matcher._index is None
        # Next match rebuilds
        result = matcher.match("Grimjaw")
        assert "grimjaw" in result

    # --- Scene seeds ---

    async def test_scene_seeds_current_location(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        ws = WorldState()
        ws.current_location = "Ironforge Tavern"
        seeds = matcher.scene_seeds(ws)
        assert "ironforge-tavern" in seeds

    async def test_scene_seeds_npc_at_location(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        ws = WorldState()
        ws.current_location = "Ironforge Tavern"
        ws.npcs["Grimjaw"] = NPCState(name="Grimjaw", location="Ironforge Tavern")
        seeds = matcher.scene_seeds(ws)
        assert "grimjaw" in seeds
        assert "ironforge-tavern" in seeds

    async def test_scene_seeds_empty_world_state(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        assert matcher.scene_seeds(None) == []

    async def test_scene_seeds_unknown_location(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        ws = WorldState()
        ws.current_location = "Unknown Place"
        seeds = matcher.scene_seeds(ws)
        assert seeds == []

    # --- Vector match ---

    async def test_vector_match_with_mock_store(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        mock_store = MagicMock()
        mock_store.search_entities.return_value = [
            {"node_id": "grimjaw", "name": "Grimjaw", "distance": 0.8}
        ]
        result = matcher.vector_match("the scarred dwarf", "test-campaign", mock_store)
        assert "grimjaw" in result
        mock_store.search_entities.assert_called_once()

    async def test_vector_match_filters_missing_nodes(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        mock_store = MagicMock()
        mock_store.search_entities.return_value = [
            {"node_id": "nonexistent", "name": "Ghost", "distance": 0.5}
        ]
        result = matcher.vector_match("some text", "test-campaign", mock_store)
        assert result == []

    async def test_vector_match_empty_text(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        mock_store = MagicMock()
        assert matcher.vector_match("", "test-campaign", mock_store) == []


# ======================================================================
# Unnamed NPC Detection
# ======================================================================

class TestUnnamedNPCDetection:

    @pytest.fixture
    def bridge(self):
        return DeltaBridge("test-campaign")

    @pytest.fixture
    def world_state(self):
        ws = WorldState()
        ws.current_location = "Market Square"
        return ws

    def test_unnamed_npc_flagged(self, bridge, world_state):
        delta = StateDelta(
            new_npcs=[NPCState(name="the hooded stranger", description="A mysterious figure")]
        )
        ops = bridge.convert(delta, world_state)
        npc_nodes = [o for o in ops if isinstance(o, AddNode) and o.entity.entity_type == EntityType.NPC]
        assert len(npc_nodes) == 1
        assert npc_nodes[0].entity.properties.get("named") == "false"

    def test_named_npc_no_flag(self, bridge, world_state):
        delta = StateDelta(
            new_npcs=[NPCState(name="Grimjaw", description="A gruff dwarf")]
        )
        ops = bridge.convert(delta, world_state)
        npc_nodes = [o for o in ops if isinstance(o, AddNode) and o.entity.entity_type == EntityType.NPC]
        assert len(npc_nodes) == 1
        assert "named" not in npc_nodes[0].entity.properties

    def test_article_prefix_detected(self, bridge, world_state):
        for name in ["a burly dwarf", "an old wizard", "the guard captain"]:
            delta = StateDelta(new_npcs=[NPCState(name=name)])
            ops = bridge.convert(delta, world_state)
            npc_nodes = [o for o in ops if isinstance(o, AddNode) and o.entity.entity_type == EntityType.NPC]
            assert npc_nodes[0].entity.properties.get("named") == "false", f"Expected unnamed for: {name}"


# ======================================================================
# Quest Extraction
# ======================================================================

class TestQuestBridge:

    @pytest.fixture
    def bridge(self):
        return DeltaBridge("test-campaign")

    @pytest.fixture
    def world_state(self):
        ws = WorldState()
        ws.current_location = "Village"
        return ws

    def test_new_quest_creates_node(self, bridge, world_state):
        delta = StateDelta(new_quests=[QuestState(
            name="Find the Amulet",
            giver="Marrowind",
            objectives=["Retrieve the amulet from the ruins"],
            location="Shadow Ruins",
        )])
        ops = bridge.convert(delta, world_state, existing_node_ids={"marrowind"})

        quest_nodes = [o for o in ops if isinstance(o, AddNode) and o.entity.entity_type == EntityType.QUEST]
        assert len(quest_nodes) == 1
        assert quest_nodes[0].entity.name == "Find the Amulet"
        assert quest_nodes[0].entity.properties["giver"] == "Marrowind"

    def test_quest_links_to_giver(self, bridge, world_state):
        delta = StateDelta(new_quests=[QuestState(
            name="Find the Amulet", giver="Marrowind",
        )])
        ops = bridge.convert(delta, world_state, existing_node_ids={"marrowind"})

        edges = [o for o in ops if isinstance(o, AddEdge)]
        giver_edges = [e for e in edges if e.relationship.relation_type == RelationType.QUEST_GIVER]
        assert len(giver_edges) == 1
        assert giver_edges[0].relationship.source_id == "marrowind"
        assert giver_edges[0].relationship.target_id == "find-the-amulet"

    def test_quest_links_to_location(self, bridge, world_state):
        delta = StateDelta(new_quests=[QuestState(
            name="Find the Amulet", location="Shadow Ruins",
        )])
        ops = bridge.convert(delta, world_state)

        edges = [o for o in ops if isinstance(o, AddEdge)]
        loc_edges = [e for e in edges if e.relationship.relation_type == RelationType.OBJECTIVE_AT]
        assert len(loc_edges) == 1
        assert loc_edges[0].relationship.target_id == "shadow-ruins"
