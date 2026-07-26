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
from dnd_bot.game.knowledge.bridge import DeltaBridge, NamePromotion, _is_proper_name
from dnd_bot.game.knowledge.matcher import EntityNameMatcher, action_entity_names
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

    def test_scene_registry_uses_canonical_slugify(self):
        """Scene entity refs and KG node IDs must share one slugify.

        A one-sided tweak would silently decouple scene refs from KG
        nodes (slug-vs-UUID identity was a dataflow-audit root cause).
        """
        from dnd_bot.game.scene import registry as scene_registry

        assert scene_registry.slugify is slugify


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

    async def test_remove_specific_edge_keeps_other_targets(self, kg):
        await kg.load()
        await kg.apply_operations([
            AddNode(entity=_make_entity("npc")),
            AddNode(entity=_make_entity("loc-a", EntityType.LOCATION)),
            AddNode(entity=_make_entity("loc-b", EntityType.LOCATION)),
            AddEdge(relationship=_make_relationship(
                "npc", "loc-a", RelationType.LOCATED_AT,
            )),
            AddEdge(relationship=_make_relationship(
                "npc", "loc-b", RelationType.LOCATED_AT,
            )),
        ])

        await kg.apply_operations([RemoveEdge(
            source_id="npc",
            target_id="loc-a",
            relation_type=RelationType.LOCATED_AT,
        )])

        assert kg.edge_count() == 1
        kg._repo.delete_edge.assert_awaited_once_with(
            "test-campaign", "npc", "loc-a", "located_at",
        )
        kg._repo.delete_edges_by_source.assert_not_awaited()

    async def test_remove_edge_empty_target_is_consistent_wildcard(self, kg):
        await kg.load()
        await kg.apply_operations([
            AddNode(entity=_make_entity("npc")),
            AddNode(entity=_make_entity("loc-a", EntityType.LOCATION)),
            AddNode(entity=_make_entity("loc-b", EntityType.LOCATION)),
            AddEdge(relationship=_make_relationship(
                "npc", "loc-a", RelationType.LOCATED_AT,
            )),
            AddEdge(relationship=_make_relationship(
                "npc", "loc-b", RelationType.LOCATED_AT,
            )),
        ])

        await kg.apply_operations([RemoveEdge(
            source_id="npc",
            target_id="",
            relation_type=RelationType.LOCATED_AT,
        )])

        assert kg.edge_count() == 0
        kg._repo.delete_edges_by_source.assert_awaited_once_with(
            "test-campaign", "npc", "located_at",
        )

    async def test_a_relocate_batch_applies_in_the_order_it_was_written(self, kg):
        """Within one batch the sequence IS the intent.

        The bridge relocates an NPC by removing every LOCATED_AT edge from
        them and then adding the new one. A type-priority sort ranked
        add_edge ahead of remove_edge and inverted exactly that pair, so the
        wildcard removal deleted the residency edge just written and the NPC
        was left with none — invisible to ``residents_of``, so scene
        hydration had nothing to put back (lore_recall 20260726_011328).
        """
        await kg.load()
        await kg.apply_operations([
            AddNode(entity=_make_entity("npc")),
            AddNode(entity=_make_entity("loc-a", EntityType.LOCATION)),
            AddNode(entity=_make_entity("loc-b", EntityType.LOCATION)),
            AddEdge(relationship=_make_relationship(
                "npc", "loc-a", RelationType.LOCATED_AT,
            )),
        ])

        # The bridge's relocate pair, in bridge order.
        await kg.apply_operations([
            RemoveEdge(
                source_id="npc",
                target_id="",
                relation_type=RelationType.LOCATED_AT,
            ),
            AddEdge(relationship=_make_relationship(
                "npc", "loc-b", RelationType.LOCATED_AT,
            )),
        ])

        assert [e.node_id for e in kg.residents_of("loc-b")] == ["npc"]
        assert kg.residents_of("loc-a") == []

    async def test_new_nodes_still_land_before_edges_that_reference_them(self, kg):
        """The one reordering that survives: an edge never references an
        endpoint that does not exist yet, whatever order the producer used."""
        await kg.load()
        rejections = await kg.apply_operations([
            AddEdge(relationship=_make_relationship(
                "npc", "loc-a", RelationType.LOCATED_AT,
            )),
            AddNode(entity=_make_entity("npc")),
            AddNode(entity=_make_entity("loc-a", EntityType.LOCATION)),
        ])

        assert not rejections
        assert [e.node_id for e in kg.residents_of("loc-a")] == ["npc"]

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

    async def test_one_radius_excludes_npc_beyond_adjacent_location(self):
        """Narrator radius must not leak an NPC through a neighboring place."""
        repo = _make_mock_repo()
        kg = KnowledgeGraph("test-campaign", repo)
        await kg.load()
        alley = _make_entity("spoke-alley", EntityType.LOCATION, "Spoke Alley")
        tavern = _make_entity("rusted-cog", EntityType.LOCATION, "Rusted Cog")
        roran = _make_entity("roran", name="Roran Hale")
        await kg.apply_operations([
            AddNode(entity=alley),
            AddNode(entity=tavern),
            AddNode(entity=roran),
            AddEdge(relationship=_make_relationship(
                "spoke-alley", "rusted-cog", RelationType.CONNECTED_TO
            )),
            AddEdge(relationship=_make_relationship(
                "roran", "rusted-cog", RelationType.LOCATED_AT
            )),
        ])

        ambient = kg.get_context_subgraph(["spoke-alley"], radius=1.0)
        explicit = kg.get_context_subgraph(["roran"], radius=1.0)

        assert "Roran Hale" not in {entry["name"] for entry in ambient}
        assert "Roran Hale" in {entry["name"] for entry in explicit}

    async def test_no_expand_seed_included_without_neighbors(self):
        """A speculative (vector-matched) seed must not drag in its owner NPC.

        Models the soak 20260722_230128 turn-55 leak: examining a parchment
        vector-matched a stale note whose 1-hop neighborhood contained the
        washed-out seed NPC.
        """
        repo = _make_mock_repo()
        kg = KnowledgeGraph("test-campaign", repo)
        await kg.load()
        note = _make_entity("crumpled-note", EntityType.ITEM, "crumpled note")
        sera = _make_entity("sera", name="Sera Vellik")
        await kg.apply_operations([
            AddNode(entity=note),
            AddNode(entity=sera),
            AddEdge(relationship=_make_relationship(
                "sera", "crumpled-note", RelationType.OWNS
            )),
        ])

        frozen = kg.get_context_subgraph(
            ["crumpled-note"], radius=1.0, no_expand_ids={"crumpled-note"}
        )
        expanded = kg.get_context_subgraph(["crumpled-note"], radius=1.0)

        assert {entry["name"] for entry in frozen} == {"crumpled note"}
        assert "Sera Vellik" in {entry["name"] for entry in expanded}

    async def test_no_expand_only_freezes_named_seeds(self, populated_kg):
        """Seeds outside no_expand_ids keep their normal BFS neighborhood."""
        result = populated_kg.get_context_subgraph(
            ["grimjaw"], no_expand_ids={"unrelated-seed"}
        )
        names = {entry["name"] for entry in result}
        assert "Grimjaw" in names
        assert "The Tavern" in names

    async def test_empty_seeds_returns_empty(self, populated_kg):
        result = populated_kg.get_context_subgraph([])
        assert result == []

    async def test_nonexistent_seed_returns_empty(self, populated_kg):
        result = populated_kg.get_context_subgraph(["nonexistent"])
        assert result == []

    async def test_max_entities_cap(self, populated_kg):
        result = populated_kg.get_context_subgraph(["grimjaw"], max_entities=2)
        assert len(result) <= 2

    async def test_explicit_seed_order_survives_context_cap(self):
        repo = _make_mock_repo()
        kg = KnowledgeGraph("test-campaign", repo)
        await kg.load()
        seed_ids = ["zz-explicit"] + [f"ambient-{i:02d}" for i in range(20)]
        await kg.apply_operations([
            AddNode(entity=_make_entity(node_id)) for node_id in seed_ids
        ])

        result = kg.get_context_subgraph(seed_ids, max_entities=3)

        assert result[0]["name"] == "Zz Explicit"
        assert len(result) == 3

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
        assert grimjaw["id"] == "grimjaw"
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
        """NPC ids are now UUIDs (NPCState.id) — the matcher emits the
        UUID directly as the seed. NPCState.id must match the KG node_id
        (cross-layer identity anchor). The fixture pre-creates a KG node
        with id 'grimjaw'; we set NPCState.id to match.
        """
        matcher = EntityNameMatcher(matcher_kg)
        ws = WorldState()
        ws.current_location = "Ironforge Tavern"
        npc = NPCState(id="grimjaw", name="Grimjaw", location="Ironforge Tavern")
        ws.npcs[npc.id] = npc
        seeds = matcher.scene_seeds(ws)
        assert "grimjaw" in seeds
        assert "ironforge-tavern" in seeds

    async def test_scene_seeds_empty_world_state(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        assert matcher.scene_seeds(None) == []

    async def test_scene_seeds_npc_at_different_sublocation_is_excluded(self, matcher_kg):
        """Free-form location similarity must not imply scene presence."""
        matcher = EntityNameMatcher(matcher_kg)
        ws = WorldState()
        ws.current_location = "Ironforge Tavern"
        npc = NPCState(id="grimjaw", name="Grimjaw", location="back room of the tavern")
        ws.npcs[npc.id] = npc
        seeds = matcher.scene_seeds(ws)
        assert "grimjaw" not in seeds
        assert "ironforge-tavern" in seeds

    async def test_scene_seeds_important_offscene_npc_and_history_excluded(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        ws = WorldState(
            current_location="Ironforge Tavern",
            connected_locations=["Goblin Caves"],
        )
        npc = NPCState(
            id="grimjaw",
            name="Grimjaw",
            location="Distant Castle",
            important=True,
        )
        ws.npcs[npc.id] = npc

        seeds = matcher.scene_seeds(ws)

        assert seeds == ["ironforge-tavern"]

    async def test_scene_seeds_dead_npc_excluded(self, matcher_kg):
        """Dead NPCs should not be seeded."""
        matcher = EntityNameMatcher(matcher_kg)
        ws = WorldState()
        ws.current_location = "Ironforge Tavern"
        npc = NPCState(id="grimjaw", name="Grimjaw", location="Ironforge Tavern", alive=False)
        ws.npcs[npc.id] = npc
        seeds = matcher.scene_seeds(ws)
        assert "grimjaw" not in seeds

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

    async def test_vector_match_requires_distinctive_lexical_grounding(self, matcher_kg):
        matcher = EntityNameMatcher(matcher_kg)
        mock_store = MagicMock()
        mock_store.search_entities.return_value = [
            {"node_id": "grimjaw", "name": "Grimjaw", "distance": 0.8}
        ]

        result = matcher.vector_match(
            "I meet the man's wary gaze and ask about the mark",
            "test-campaign",
            mock_store,
        )

        assert result == []

    async def test_vector_match_rejects_single_description_word_overlap(self, matcher_kg):
        entity = matcher_kg.get_entity("grimjaw")
        entity.properties["description"] = "A scarred dwarf with one clouded eye"
        matcher = EntityNameMatcher(matcher_kg)
        mock_store = MagicMock()
        mock_store.search_entities.return_value = [
            {"node_id": "grimjaw", "name": "Grimjaw", "distance": 0.8}
        ]

        result = matcher.vector_match(
            "I offer one note fragment to the nearest guard",
            "test-campaign",
            mock_store,
        )

        assert result == []

    async def test_vector_match_accepts_two_description_anchors(self, matcher_kg):
        entity = matcher_kg.get_entity("grimjaw")
        entity.properties["description"] = "A scarred dwarf in a crimson coat"
        matcher = EntityNameMatcher(matcher_kg)
        mock_store = MagicMock()
        mock_store.search_entities.return_value = [
            {"node_id": "grimjaw", "name": "Grimjaw", "distance": 0.8}
        ]

        result = matcher.vector_match(
            "I approach the scarred stranger wearing crimson",
            "test-campaign",
            mock_store,
        )

        assert result == ["grimjaw"]

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

    async def test_graph_reference_resolves_unique_human_readable_slug(self, matcher_kg):
        assert matcher_kg.resolve_entity_reference("old_grim").node_id == "grimjaw"

    async def test_graph_reference_abstains_on_ambiguous_alias(self, matcher_kg):
        other = _make_entity("other-dwarf", name="Other Dwarf")
        other.aliases = ["the dwarf"]
        await matcher_kg.apply_operations([AddNode(entity=other)])

        assert matcher_kg.resolve_entity_reference("the-dwarf") is None

    async def test_action_entity_names_returns_every_name_of_each_match(
        self, matcher_kg
    ):
        # "the dwarf" is a placeholder, not an identity — see the
        # placeholder test below for why it must not come back.
        assert sorted(action_entity_names(
            matcher_kg, "I talk to Grimjaw at the Ironforge Tavern"
        )) == ["Grimjaw", "Ironforge Tavern", "old grim"]

    async def test_action_entity_names_resolves_an_alias_to_its_name(
        self, matcher_kg
    ):
        """The anchor is the entity, not the wording that found it."""
        assert sorted(action_entity_names(matcher_kg, "old grim owes me")) == [
            "Grimjaw", "old grim",
        ]

    async def test_action_entity_names_ignores_entities_nobody_named(
        self, matcher_kg
    ):
        assert action_entity_names(matcher_kg, "I sharpen my sword") == []

    async def test_action_entity_names_requires_a_token_boundary(
        self, matcher_kg
    ):
        """`match` is bare substring; anchoring facts needs more than that.

        A speculative graph seed costs an entity card. Anchoring durable
        campaign facts is a stronger claim, so it re-tests the hit.
        """
        assert "grimjaw" in matcher_kg.get_all_names()
        assert EntityNameMatcher(matcher_kg).match(
            "I study the Ironforge Tavernkeeper's ledger"
        ) == ["ironforge-tavern"]

        assert action_entity_names(
            matcher_kg, "I study the Ironforge Tavernkeeper's ledger"
        ) == []

    async def test_a_placeholder_name_does_not_count_as_naming_an_entity(
        self, matcher_kg
    ):
        """"the dwarf" recurs forever; it must not open Grimjaw's file."""
        assert action_entity_names(matcher_kg, "I ask the dwarf for ale") == []

    async def test_a_placeholder_alias_never_escorts_a_real_name_in(
        self, matcher_kg
    ):
        """The filter is per NAME, not per entity.

        An entity found only through its placeholder would otherwise hand
        back its canonical name as an anchor — the generic gate bypassed by
        going the long way round.
        """
        hidden = _make_entity("sera-vellian", name="the woman")
        hidden.aliases = ["Sera Vellian"]
        await matcher_kg.apply_operations([AddNode(entity=hidden)])

        assert action_entity_names(
            matcher_kg, "I watch the woman by the door"
        ) == []

    async def test_action_entity_names_is_empty_without_a_graph(self):
        assert action_entity_names(None, "I talk to Grimjaw") == []

    async def test_action_entity_names_is_empty_without_text(self, matcher_kg):
        assert action_entity_names(matcher_kg, "") == []

    async def test_action_entity_names_degrades_instead_of_raising(self):
        """Retrieval widening must never cost the player their turn."""
        broken = MagicMock()
        broken.get_all_names.side_effect = RuntimeError("graph is unreadable")

        assert action_entity_names(broken, "I talk to Grimjaw") == []


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

    def test_quest_giver_not_in_known_skips_edge(self, bridge, world_state):
        """Audit #10: previously `if giver_id in known or giver_id` was always truthy,
        so a giver who didn't have a node yet would still produce an orphan
        QUEST_GIVER edge that `_apply_add_edge` later rejected with
        "Source node not found". With the fix (`and` instead of `or`), no edge
        is emitted when the giver isn't already a node.
        """
        delta = StateDelta(new_quests=[QuestState(
            name="Find the Amulet", giver="Unknown Stranger",
        )])
        # Note: existing_node_ids does NOT contain "unknown-stranger"
        ops = bridge.convert(delta, world_state, existing_node_ids={"some-other-npc"})

        giver_edges = [
            o for o in ops
            if isinstance(o, AddEdge)
            and o.relationship.relation_type == RelationType.QUEST_GIVER
        ]
        assert giver_edges == [], (
            "No QUEST_GIVER edge should be created when the giver has no node yet"
        )

    def test_quest_giver_empty_string_skips_edge(self, bridge, world_state):
        """Defensive: an empty-string giver should also produce no edge."""
        delta = StateDelta(new_quests=[QuestState(
            name="Find the Amulet", giver="",
        )])
        ops = bridge.convert(delta, world_state, existing_node_ids=set())

        giver_edges = [
            o for o in ops
            if isinstance(o, AddEdge)
            and o.relationship.relation_type == RelationType.QUEST_GIVER
        ]
        assert giver_edges == []


# ======================================================================
# Effect bridge — convert_effects
# ======================================================================

class TestEffectBridge:
    """Tests for DeltaBridge.convert_effects (tool-effect → KG path)."""

    @pytest.fixture
    def bridge(self):
        return DeltaBridge("test-campaign")

    @pytest.fixture
    def world_state(self):
        ws = WorldState()
        ws.current_location = "Market Square"
        return ws

    @staticmethod
    def _make_effect(**kwargs):
        """Build a minimal ProposedEffect for testing."""
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        defaults = {"effect_type": EffectType.ADD_NPC}
        defaults.update(kwargs)
        return ProposedEffect(**defaults)

    def test_empty_effects_no_ops(self, bridge, world_state):
        ops, promotions = bridge.convert_effects([], world_state)
        assert ops == []
        assert promotions == []

    def test_add_npc_creates_node_and_edge(self, bridge, world_state):
        from dnd_bot.llm.effects import EffectType
        effect = self._make_effect(
            effect_type=EffectType.ADD_NPC,
            npc_name="Grimjaw",
            npc_description="A gruff dwarf",
            npc_disposition="hostile",
        )
        ops, promotions = bridge.convert_effects([effect], world_state)

        add_nodes = [o for o in ops if isinstance(o, AddNode)]
        add_edges = [o for o in ops if isinstance(o, AddEdge)]

        npc_nodes = [n for n in add_nodes if n.entity.entity_type == EntityType.NPC]
        assert len(npc_nodes) == 1
        assert npc_nodes[0].entity.name == "Grimjaw"
        assert npc_nodes[0].entity.properties["disposition"] == "hostile"
        assert npc_nodes[0].entity.properties["description"] == "A gruff dwarf"

        located_at = [e for e in add_edges if e.relationship.relation_type == RelationType.LOCATED_AT]
        assert len(located_at) == 1
        assert located_at[0].relationship.source_id == "grimjaw"
        assert located_at[0].relationship.target_id == "market-square"

    def test_add_npc_unnamed_detection(self, bridge, world_state):
        from dnd_bot.llm.effects import EffectType
        effect = self._make_effect(
            effect_type=EffectType.ADD_NPC,
            npc_name="the cloaked stranger",
            npc_description="A mysterious figure",
        )
        ops, _ = bridge.convert_effects([effect], world_state)
        npc_nodes = [o for o in ops if isinstance(o, AddNode) and o.entity.entity_type == EntityType.NPC]
        assert npc_nodes[0].entity.properties.get("named") == "false"

    def test_spawn_object_creates_item_node(self, bridge, world_state):
        from dnd_bot.llm.effects import EffectType
        effect = self._make_effect(
            effect_type=EffectType.SPAWN_OBJECT,
            object_name="Iron Key",
            object_description="A heavy iron key with rust spots",
        )
        ops, _ = bridge.convert_effects([effect], world_state)

        item_nodes = [o for o in ops if isinstance(o, AddNode) and o.entity.entity_type == EntityType.ITEM]
        assert len(item_nodes) == 1
        assert item_nodes[0].entity.name == "Iron Key"
        assert item_nodes[0].entity.properties["description"] == "A heavy iron key with rust spots"

        located_at = [o for o in ops if isinstance(o, AddEdge)]
        assert len(located_at) == 1

    def test_ref_entity_proper_name_promotion(self, bridge, world_state):
        from dnd_bot.llm.effects import EffectType
        effect = self._make_effect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id="the cloaked stranger",
            ref_alias_used="Silas Vane",
        )
        ops, promotions = bridge.convert_effects([effect], world_state)
        assert ops == []  # ref_entity produces no graph ops
        assert len(promotions) == 1
        assert promotions[0].node_id == "the-cloaked-stranger"
        assert promotions[0].new_name == "Silas Vane"

    def test_ref_entity_descriptor_alias_no_promotion(self, bridge, world_state):
        from dnd_bot.llm.effects import EffectType
        effect = self._make_effect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id="Grimjaw",
            ref_alias_used="the dwarf",
        )
        _, promotions = bridge.convert_effects([effect], world_state)
        assert promotions == []

    def test_ref_entity_same_name_no_promotion(self, bridge, world_state):
        from dnd_bot.llm.effects import EffectType
        effect = self._make_effect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id="Grimjaw",
            ref_alias_used="Grimjaw",
        )
        _, promotions = bridge.convert_effects([effect], world_state)
        assert promotions == []

    def test_remove_entity_clears_location(self, bridge, world_state):
        from dnd_bot.llm.effects import EffectType
        effect = self._make_effect(
            effect_type=EffectType.REMOVE_ENTITY,
            target="Grimjaw",
        )
        ops, _ = bridge.convert_effects([effect], world_state)
        assert len(ops) == 2
        assert isinstance(ops[0], UpdateNode)
        assert ops[0].properties == {"location": ""}
        assert isinstance(ops[1], RemoveEdge)
        assert ops[1].source_id == "grimjaw"

    def test_remove_entity_resolves_uuid_anchored_npc(self, bridge, world_state):
        """NPC nodes are UUID-anchored (_effect_add_npc), so the removal ops
        must target the NPCState UUID — a raw slugify(target) produced a
        node id no NPC node has, the UpdateNode was rejected at debug level
        and LOCATED_AT edges survived (final review). The target here uses
        the roster's [id: slug] dialect for a multi-word name."""
        from dnd_bot.llm.effects import EffectType
        from dnd_bot.game.world_state import NPCState

        npc = NPCState(name="Old Bram", description="A weathered farmer")
        world_state.npcs[npc.id] = npc

        effect = self._make_effect(
            effect_type=EffectType.REMOVE_ENTITY,
            target="old-bram",
        )
        ops, _ = bridge.convert_effects([effect], world_state)
        assert len(ops) == 2
        assert isinstance(ops[0], UpdateNode)
        assert ops[0].node_id == npc.id
        assert isinstance(ops[1], RemoveEdge)
        assert ops[1].source_id == npc.id

    def test_mixed_effects_batch(self, bridge, world_state):
        from dnd_bot.llm.effects import EffectType
        effects = [
            self._make_effect(
                effect_type=EffectType.ADD_NPC,
                npc_name="Captain Elara",
                npc_description="A stern guard captain",
            ),
            self._make_effect(
                effect_type=EffectType.SPAWN_OBJECT,
                object_name="Wanted Poster",
                object_description="A poster on the wall",
            ),
            self._make_effect(
                effect_type=EffectType.REF_ENTITY,
                ref_entity_id="the hooded figure",
                ref_alias_used="Marcus Grey",
            ),
        ]
        ops, promotions = bridge.convert_effects(effects, world_state)

        # 2 entity AddNodes + 2 location placeholders possible + 2 LOCATED_AT edges
        add_nodes = [o for o in ops if isinstance(o, AddNode)]
        add_edges = [o for o in ops if isinstance(o, AddEdge)]
        assert len(add_nodes) >= 2  # at least the NPC and the item
        assert len(add_edges) == 2  # LOCATED_AT for each
        assert len(promotions) == 1
        assert promotions[0].new_name == "Marcus Grey"


# ======================================================================
# _is_proper_name helper
# ======================================================================

class TestIsProperName:

    def test_proper_name(self):
        assert _is_proper_name("Silas Vane") is True

    def test_titled_name(self):
        assert _is_proper_name("Captain Elara") is True

    def test_article_prefix(self):
        assert _is_proper_name("the cloaked stranger") is False

    def test_all_lowercase(self):
        assert _is_proper_name("old merchant") is False

    def test_single_capital(self):
        assert _is_proper_name("Grimjaw") is True

    def test_indefinite_article(self):
        assert _is_proper_name("a mysterious figure") is False
        assert _is_proper_name("an armored knight") is False


# ======================================================================
# Proper-name uniqueness at the write seam (cross-store naming promotion)
# ======================================================================

class TestProperNameUniquenessSeam:
    """canonical_npc_identity_unique enforced where writes land: AddNode
    merges into (or abstains from) an existing NPC node that already
    carries the same proper name; promote_entity_name abstains on
    collision instead of minting a second node with the canonical name."""

    @pytest.fixture
    def kg(self):
        repo = _make_mock_repo()
        return KnowledgeGraph("test-campaign", repo)

    async def test_add_node_merges_duplicate_proper_name(self, kg):
        await kg.load()
        await kg.apply_operations([AddNode(entity=_make_entity(
            "uuid-1", name="Orris", description="an older woman",
        ))])
        await kg.apply_operations([AddNode(entity=_make_entity(
            "uuid-2", name="Orris", disposition="friendly",
        ))])

        assert kg.node_count() == 1
        assert kg.get_entity("uuid-2") is None
        merged = kg.get_entity("uuid-1")
        assert merged.properties["disposition"] == "friendly"
        assert merged.properties["description"] == "an older woman"

    async def test_add_node_matches_existing_alias(self, kg):
        """A node holding the proper name as an ALIAS also claims it."""
        await kg.load()
        entity = _make_entity("uuid-1", name="the older woman")
        entity.aliases = ["Orris"]
        await kg.apply_operations([AddNode(entity=entity)])
        await kg.apply_operations([AddNode(entity=_make_entity(
            "uuid-2", name="Orris",
        ))])

        assert kg.node_count() == 1
        assert kg.get_entity("uuid-2") is None

    async def test_add_node_generic_labels_stay_distinct(self, kg):
        """Role labels are archetypes, not identities — no dedup."""
        await kg.load()
        await kg.apply_operations([AddNode(entity=_make_entity("uuid-1", name="the guard"))])
        await kg.apply_operations([AddNode(entity=_make_entity("uuid-2", name="the guard"))])
        assert kg.node_count() == 2

    async def test_add_node_distinct_proper_names_unaffected(self, kg):
        await kg.load()
        await kg.apply_operations([AddNode(entity=_make_entity("uuid-1", name="Orris"))])
        await kg.apply_operations([AddNode(entity=_make_entity("uuid-2", name="Elara Venn"))])
        assert kg.node_count() == 2

    async def test_add_node_non_npc_types_unaffected(self, kg):
        """Two locations may share a name — the invariant is NPC-only."""
        await kg.load()
        await kg.apply_operations([AddNode(entity=_make_entity(
            "loc-1", entity_type=EntityType.LOCATION, name="Riverside",
        ))])
        await kg.apply_operations([AddNode(entity=_make_entity(
            "loc-2", entity_type=EntityType.LOCATION, name="Riverside",
        ))])
        assert kg.node_count() == 2

    async def test_promotion_abstains_when_name_taken(self, kg):
        await kg.load()
        await kg.apply_operations([
            AddNode(entity=_make_entity("uuid-hooded", name="the hooded stranger")),
            AddNode(entity=_make_entity("uuid-orris", name="Orris")),
        ])

        promoted = await kg.promote_entity_name("uuid-hooded", "Orris")

        assert promoted is False
        assert kg.get_entity("uuid-hooded").name == "the hooded stranger"

    async def test_promotion_succeeds_when_name_free(self, kg):
        await kg.load()
        await kg.apply_operations([
            AddNode(entity=_make_entity("uuid-hooded", name="the hooded stranger")),
        ])

        promoted = await kg.promote_entity_name("uuid-hooded", "Orris")

        assert promoted is True
        entity = kg.get_entity("uuid-hooded")
        assert entity.name == "Orris"
        assert "the hooded stranger" in entity.aliases

    async def test_promotion_abstains_on_label_fragment(self, kg):
        """'Choir' offered as the new name for 'a Choir acolyte' is an
        excerpt of the descriptive label (the faction's name), not a
        newly revealed personal name (live case: run 20260723_120152
        T15 renamed the acolyte node to 'Choir', misbinding T16's
        legitimate 'the acolyte' ref)."""
        await kg.load()
        await kg.apply_operations([
            AddNode(entity=_make_entity("uuid-acolyte", name="a Choir acolyte")),
        ])

        promoted = await kg.promote_entity_name("uuid-acolyte", "Choir")

        assert promoted is False
        assert kg.get_entity("uuid-acolyte").name == "a Choir acolyte"

    async def test_promotion_accepts_name_extension(self, kg):
        """Gaining words is a real naming event ('Elara' -> 'Elara Venn')."""
        await kg.load()
        await kg.apply_operations([
            AddNode(entity=_make_entity("uuid-elara", name="Elara")),
        ])

        promoted = await kg.promote_entity_name("uuid-elara", "Elara Venn")

        assert promoted is True
        entity = kg.get_entity("uuid-elara")
        assert entity.name == "Elara Venn"
        assert "Elara" in entity.aliases
