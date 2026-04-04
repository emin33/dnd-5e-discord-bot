"""Knowledge graph SQLite persistence following the project's repository pattern."""

import json
from datetime import datetime
from typing import Optional

import structlog

from ...data.database import Database, get_database
from .models import Entity, EntityType, Relationship, RelationType

logger = structlog.get_logger()


class KnowledgeGraphRepository:
    """Async SQLite persistence for knowledge graph nodes and edges."""

    def __init__(self, db: Optional[Database] = None):
        self._db = db

    async def _get_db(self) -> Database:
        if self._db:
            return self._db
        return await get_database()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    async def load_nodes(self, campaign_id: str) -> list[Entity]:
        """Load all nodes for a campaign."""
        db = await self._get_db()
        rows = await db.fetch_all(
            "SELECT * FROM kg_node WHERE campaign_id = ?",
            (campaign_id,),
        )
        return [self._row_to_entity(row) for row in rows]

    async def upsert_node(self, entity: Entity) -> None:
        """Insert or update a node."""
        db = await self._get_db()
        await db.execute(
            """INSERT INTO kg_node (campaign_id, node_id, entity_type, name,
                                    aliases, properties, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (campaign_id, node_id) DO UPDATE SET
                   entity_type = excluded.entity_type,
                   name = excluded.name,
                   aliases = excluded.aliases,
                   properties = excluded.properties,
                   updated_at = excluded.updated_at""",
            (
                entity.campaign_id,
                entity.node_id,
                entity.entity_type.value,
                entity.name,
                json.dumps(entity.aliases),
                json.dumps(entity.properties),
                entity.created_at.isoformat(),
                entity.updated_at.isoformat(),
            ),
        )
        await db.commit()

    async def delete_node(self, campaign_id: str, node_id: str) -> None:
        """Delete a node and its edges (FK cascade)."""
        db = await self._get_db()
        await db.execute(
            "DELETE FROM kg_node WHERE campaign_id = ? AND node_id = ?",
            (campaign_id, node_id),
        )
        await db.commit()

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    async def load_edges(self, campaign_id: str) -> list[Relationship]:
        """Load all edges for a campaign."""
        db = await self._get_db()
        rows = await db.fetch_all(
            "SELECT * FROM kg_edge WHERE campaign_id = ?",
            (campaign_id,),
        )
        return [self._row_to_relationship(row) for row in rows]

    async def upsert_edge(self, rel: Relationship) -> None:
        """Insert or update an edge."""
        db = await self._get_db()
        await db.execute(
            """INSERT INTO kg_edge (campaign_id, source_id, target_id,
                                    relation_type, weight, properties, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (campaign_id, source_id, target_id, relation_type)
               DO UPDATE SET
                   weight = excluded.weight,
                   properties = excluded.properties""",
            (
                rel.campaign_id,
                rel.source_id,
                rel.target_id,
                rel.relation_type.value,
                rel.weight,
                json.dumps(rel.properties),
                rel.created_at.isoformat(),
            ),
        )
        await db.commit()

    async def delete_edge(
        self,
        campaign_id: str,
        source_id: str,
        target_id: str,
        relation_type: str,
    ) -> None:
        """Delete a specific edge."""
        db = await self._get_db()
        await db.execute(
            """DELETE FROM kg_edge
               WHERE campaign_id = ? AND source_id = ? AND target_id = ?
                     AND relation_type = ?""",
            (campaign_id, source_id, target_id, relation_type),
        )
        await db.commit()

    async def delete_edges_by_source(
        self,
        campaign_id: str,
        source_id: str,
        relation_type: str,
    ) -> None:
        """Delete all edges of a given type from a source node."""
        db = await self._get_db()
        await db.execute(
            """DELETE FROM kg_edge
               WHERE campaign_id = ? AND source_id = ? AND relation_type = ?""",
            (campaign_id, source_id, relation_type),
        )
        await db.commit()

    # ------------------------------------------------------------------
    # Row conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entity(row) -> Entity:
        return Entity(
            campaign_id=row[0],
            node_id=row[1],
            entity_type=EntityType(row[2]),
            name=row[3],
            aliases=json.loads(row[4]) if row[4] else [],
            properties=json.loads(row[5]) if row[5] else {},
            created_at=datetime.fromisoformat(row[6]) if row[6] else datetime.utcnow(),
            updated_at=datetime.fromisoformat(row[7]) if row[7] else datetime.utcnow(),
        )

    @staticmethod
    def _row_to_relationship(row) -> Relationship:
        return Relationship(
            campaign_id=row[0],
            source_id=row[1],
            target_id=row[2],
            relation_type=RelationType(row[3]),
            weight=row[4] if row[4] is not None else 1.0,
            properties=json.loads(row[5]) if row[5] else {},
            created_at=datetime.fromisoformat(row[6]) if row[6] else datetime.utcnow(),
        )


# ---------------------------------------------------------------------------
# Lazy singleton (project pattern)
# ---------------------------------------------------------------------------

_kg_repo: Optional[KnowledgeGraphRepository] = None


async def get_kg_repo(db: Optional[Database] = None) -> KnowledgeGraphRepository:
    """Get or create the knowledge graph repository singleton."""
    global _kg_repo
    if _kg_repo is None:
        _kg_repo = KnowledgeGraphRepository(db)
    return _kg_repo
