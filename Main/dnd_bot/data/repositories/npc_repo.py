"""NPC repository for database operations."""

from datetime import datetime
from typing import Optional

from ...models.npc import NPC, NPCRelationship, Disposition
from ..database import Database, get_database


class NPCRepository:
    """Repository for NPC database operations."""

    def __init__(self, db: Optional[Database] = None):
        self._db = db

    async def _get_db(self) -> Database:
        if self._db:
            return self._db
        return await get_database()

    # ==================== NPC Operations ====================

    async def create(self, npc: NPC) -> NPC:
        """Create a new NPC."""
        db = await self._get_db()
        await db.execute(
            """
            INSERT INTO npc (id, campaign_id, name, description, location,
                           monster_index, base_disposition, voice_notes,
                           is_alive, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                npc.id,
                npc.campaign_id,
                npc.name,
                npc.description,
                npc.location,
                npc.monster_index,
                self._disposition_to_int(npc.base_disposition),
                npc.voice_notes,
                1 if npc.is_alive else 0,
                npc.created_at.isoformat(),
            ),
        )
        await db.commit()
        return npc

    async def get_by_id(self, npc_id: str) -> Optional[NPC]:
        """Get an NPC by ID."""
        db = await self._get_db()
        row = await db.fetch_one(
            "SELECT * FROM npc WHERE id = ?", (npc_id,)
        )
        return self._row_to_npc(row) if row else None

    async def get_by_name(self, campaign_id: str, name: str) -> Optional[NPC]:
        """Get an NPC by name (case-insensitive partial match)."""
        db = await self._get_db()
        row = await db.fetch_one(
            """SELECT * FROM npc
               WHERE campaign_id = ? AND LOWER(name) LIKE LOWER(?)
               LIMIT 1""",
            (campaign_id, f"%{name}%"),
        )
        return self._row_to_npc(row) if row else None

    async def get_by_exact_name(self, campaign_id: str, name: str) -> Optional[NPC]:
        """Get an NPC by exact name (case-insensitive)."""
        db = await self._get_db()
        row = await db.fetch_one(
            """SELECT * FROM npc
               WHERE campaign_id = ? AND LOWER(name) = LOWER(?)
               LIMIT 1""",
            (campaign_id, name),
        )
        return self._row_to_npc(row) if row else None

    async def get_all_by_campaign(self, campaign_id: str) -> list[NPC]:
        """Get all NPCs for a campaign."""
        db = await self._get_db()
        rows = await db.fetch_all(
            "SELECT * FROM npc WHERE campaign_id = ? ORDER BY name",
            (campaign_id,),
        )
        return [self._row_to_npc(row) for row in rows]

    async def get_alive_by_campaign(self, campaign_id: str) -> list[NPC]:
        """Get all living NPCs for a campaign."""
        db = await self._get_db()
        rows = await db.fetch_all(
            "SELECT * FROM npc WHERE campaign_id = ? AND is_alive = 1 ORDER BY name",
            (campaign_id,),
        )
        return [self._row_to_npc(row) for row in rows]

    async def get_at_location(self, campaign_id: str, location: str) -> list[NPC]:
        """Get NPCs at a specific location (case-insensitive partial match)."""
        db = await self._get_db()
        rows = await db.fetch_all(
            """SELECT * FROM npc
               WHERE campaign_id = ? AND LOWER(location) LIKE LOWER(?)
               AND is_alive = 1""",
            (campaign_id, f"%{location}%"),
        )
        return [self._row_to_npc(row) for row in rows]

    async def update(self, npc: NPC) -> bool:
        """Update an NPC."""
        db = await self._get_db()
        await db.execute(
            """
            UPDATE npc SET name = ?, description = ?, location = ?,
                          monster_index = ?, base_disposition = ?,
                          voice_notes = ?, is_alive = ?
            WHERE id = ?
            """,
            (
                npc.name,
                npc.description,
                npc.location,
                npc.monster_index,
                self._disposition_to_int(npc.base_disposition),
                npc.voice_notes,
                1 if npc.is_alive else 0,
                npc.id,
            ),
        )
        await db.commit()
        return True

    async def update_location(self, npc_id: str, location: str) -> bool:
        """Update NPC location."""
        db = await self._get_db()
        await db.execute(
            "UPDATE npc SET location = ? WHERE id = ?",
            (location, npc_id),
        )
        await db.commit()
        return True

    async def mark_dead(self, npc_id: str) -> bool:
        """Mark an NPC as dead."""
        db = await self._get_db()
        await db.execute(
            "UPDATE npc SET is_alive = 0 WHERE id = ?",
            (npc_id,),
        )
        await db.commit()
        return True

    async def delete(self, npc_id: str) -> bool:
        """Delete an NPC and their relationships."""
        db = await self._get_db()
        # Relationships are cascade deleted by FK constraint
        await db.execute("DELETE FROM npc WHERE id = ?", (npc_id,))
        await db.commit()
        return True

    # ==================== Relationship Operations ====================

    async def get_relationship(
        self, npc_id: str, character_id: str
    ) -> Optional[NPCRelationship]:
        """Get relationship between NPC and character."""
        db = await self._get_db()
        row = await db.fetch_one(
            """SELECT * FROM npc_relationship
               WHERE npc_id = ? AND character_id = ?""",
            (npc_id, character_id),
        )
        return self._row_to_relationship(row) if row else None

    async def get_all_relationships(self, npc_id: str) -> list[NPCRelationship]:
        """Get all relationships for an NPC."""
        db = await self._get_db()
        rows = await db.fetch_all(
            "SELECT * FROM npc_relationship WHERE npc_id = ?",
            (npc_id,),
        )
        return [self._row_to_relationship(row) for row in rows]

    async def get_character_relationships(
        self, character_id: str
    ) -> list[NPCRelationship]:
        """Get all NPC relationships for a character."""
        db = await self._get_db()
        rows = await db.fetch_all(
            "SELECT * FROM npc_relationship WHERE character_id = ?",
            (character_id,),
        )
        return [self._row_to_relationship(row) for row in rows]

    async def set_relationship(self, relationship: NPCRelationship) -> bool:
        """Create or update a relationship."""
        db = await self._get_db()
        await db.execute(
            """
            INSERT OR REPLACE INTO npc_relationship
            (npc_id, character_id, sentiment, interaction_count, notes, last_interaction)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                relationship.npc_id,
                relationship.character_id,
                relationship.sentiment,
                relationship.interaction_count,
                relationship.notes,
                datetime.utcnow().isoformat(),
            ),
        )
        await db.commit()
        return True

    async def adjust_sentiment(
        self, npc_id: str, character_id: str, delta: int, note: str = ""
    ) -> int:
        """
        Adjust sentiment and return new value.

        Creates relationship if it doesn't exist.
        Clamps sentiment to -100 to 100.
        """
        rel = await self.get_relationship(npc_id, character_id)
        if not rel:
            rel = NPCRelationship(npc_id=npc_id, character_id=character_id)

        # Clamp sentiment to -100 to 100
        rel.sentiment = max(-100, min(100, rel.sentiment + delta))
        rel.interaction_count += 1
        if note:
            # Append note with timestamp
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
            new_note = f"[{timestamp}] {note}"
            rel.notes = f"{rel.notes}\n{new_note}".strip() if rel.notes else new_note

        await self.set_relationship(rel)
        return rel.sentiment

    async def record_interaction(self, npc_id: str, character_id: str) -> int:
        """Record an interaction and return new count."""
        rel = await self.get_relationship(npc_id, character_id)
        if not rel:
            rel = NPCRelationship(npc_id=npc_id, character_id=character_id)

        rel.interaction_count += 1
        await self.set_relationship(rel)
        return rel.interaction_count

    # ==================== Conversion Helpers ====================

    def _row_to_npc(self, row) -> NPC:
        """Convert database row to NPC model."""
        return NPC(
            id=row[0],
            campaign_id=row[1],
            name=row[2],
            description=row[3] or "",
            location=row[4],
            monster_index=row[5],
            base_disposition=self._int_to_disposition(row[6]),
            voice_notes=row[7],
            is_alive=bool(row[8]),
            created_at=datetime.fromisoformat(row[9]) if row[9] else datetime.utcnow(),
        )

    def _row_to_relationship(self, row) -> NPCRelationship:
        """Convert database row to NPCRelationship model."""
        return NPCRelationship(
            npc_id=row[0],
            character_id=row[1],
            sentiment=row[2] or 0,
            interaction_count=row[3] or 0,
            notes=row[4] or "",
            last_interaction=datetime.fromisoformat(row[5]) if row[5] else None,
        )

    def _disposition_to_int(self, disposition: Disposition) -> int:
        """Convert disposition enum to integer for DB storage."""
        mapping = {
            Disposition.HOSTILE: -75,
            Disposition.UNFRIENDLY: -25,
            Disposition.NEUTRAL: 0,
            Disposition.FRIENDLY: 25,
            Disposition.ALLIED: 75,
        }
        return mapping.get(disposition, 0)

    def _int_to_disposition(self, value: int) -> Disposition:
        """Convert integer from DB to disposition enum."""
        if value is None:
            return Disposition.NEUTRAL
        if value <= -50:
            return Disposition.HOSTILE
        elif value < 0:
            return Disposition.UNFRIENDLY
        elif value == 0:
            return Disposition.NEUTRAL
        elif value <= 50:
            return Disposition.FRIENDLY
        else:
            return Disposition.ALLIED


# Global repository instance
_repo: Optional[NPCRepository] = None


async def get_npc_repo() -> NPCRepository:
    """Get the global NPC repository."""
    global _repo
    if _repo is None:
        _repo = NPCRepository()
    return _repo
