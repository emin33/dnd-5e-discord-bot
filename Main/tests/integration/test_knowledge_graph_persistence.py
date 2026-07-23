"""Cross-store graph invariants against a real temporary SQLite database."""

from pathlib import Path

import pytest

from dnd_bot.data.database import Database
from dnd_bot.game.knowledge.graph import KnowledgeGraph
from dnd_bot.game.knowledge.models import (
    AddEdge,
    AddNode,
    Entity,
    EntityType,
    RelationType,
    Relationship,
    RemoveEdge,
)
from dnd_bot.game.knowledge.repository import KnowledgeGraphRepository


@pytest.fixture
async def graph_db(tmp_path: Path):
    db = Database(db_path=tmp_path / "graph.db")
    await db.connect()
    await db.execute(
        "INSERT INTO campaign (id, guild_id, name, dm_user_id) VALUES (?, ?, ?, ?)",
        ("campaign", 1, "Graph Test", 1),
    )
    await db.commit()
    yield db
    await db.disconnect()


def _node(node_id: str, entity_type: EntityType) -> AddNode:
    return AddNode(entity=Entity(
        node_id=node_id,
        entity_type=entity_type,
        name=node_id,
        campaign_id="campaign",
    ))


def _located_at(target_id: str) -> AddEdge:
    return AddEdge(relationship=Relationship(
        source_id="npc",
        target_id=target_id,
        relation_type=RelationType.LOCATED_AT,
        campaign_id="campaign",
    ))


@pytest.mark.asyncio
async def test_wildcard_edge_removal_matches_sqlite_after_reload(graph_db):
    repo = KnowledgeGraphRepository(graph_db)
    live = KnowledgeGraph("campaign", repo)
    await live.load()
    await live.apply_operations([
        _node("npc", EntityType.NPC),
        _node("old-location", EntityType.LOCATION),
        _node("other-old-location", EntityType.LOCATION),
        _located_at("old-location"),
        _located_at("other-old-location"),
    ])

    rejections = await live.apply_operations([RemoveEdge(
        source_id="npc",
        target_id="",
        relation_type=RelationType.LOCATED_AT,
    )])

    reloaded = KnowledgeGraph("campaign", repo)
    await reloaded.load()
    assert rejections == []
    assert live.node_count() == reloaded.node_count() == 3
    assert live.edge_count() == reloaded.edge_count() == 0
