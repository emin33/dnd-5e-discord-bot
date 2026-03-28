"""Campaign repository for database operations."""

from datetime import datetime
from typing import Optional

from ...models import Campaign
from ..database import Database, get_database


class CampaignRepository:
    """Repository for Campaign database operations."""

    def __init__(self, db: Optional[Database] = None):
        self._db = db

    async def _get_db(self) -> Database:
        if self._db:
            return self._db
        return await get_database()

    async def create(self, campaign: Campaign) -> Campaign:
        """Create a new campaign in the database."""
        db = await self._get_db()

        await db.execute(
            """
            INSERT INTO campaign (id, guild_id, name, description, world_setting, dm_user_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                campaign.id,
                campaign.guild_id,
                campaign.name,
                campaign.description,
                campaign.world_setting,
                campaign.dm_user_id,
                campaign.created_at.isoformat(),
            ),
        )
        await db.commit()
        return campaign

    async def get_by_id(self, campaign_id: str) -> Optional[Campaign]:
        """Get a campaign by ID."""
        db = await self._get_db()

        row = await db.fetch_one(
            "SELECT * FROM campaign WHERE id = ?",
            (campaign_id,),
        )

        if not row:
            return None

        return self._row_to_campaign(row)

    async def get_by_name_and_guild(
        self, name: str, guild_id: int
    ) -> Optional[Campaign]:
        """Get a campaign by name and guild."""
        db = await self._get_db()

        row = await db.fetch_one(
            "SELECT * FROM campaign WHERE name = ? AND guild_id = ?",
            (name, guild_id),
        )

        if not row:
            return None

        return self._row_to_campaign(row)

    async def get_all_by_guild(self, guild_id: int) -> list[Campaign]:
        """Get all campaigns in a guild."""
        db = await self._get_db()

        rows = await db.fetch_all(
            "SELECT * FROM campaign WHERE guild_id = ? ORDER BY last_played_at DESC NULLS LAST, created_at DESC",
            (guild_id,),
        )

        return [self._row_to_campaign(row) for row in rows]

    async def get_by_dm(self, dm_user_id: int, guild_id: int) -> list[Campaign]:
        """Get all campaigns where user is DM."""
        db = await self._get_db()

        rows = await db.fetch_all(
            "SELECT * FROM campaign WHERE dm_user_id = ? AND guild_id = ? ORDER BY last_played_at DESC NULLS LAST",
            (dm_user_id, guild_id),
        )

        return [self._row_to_campaign(row) for row in rows]

    async def update(self, campaign: Campaign) -> bool:
        """Update an existing campaign."""
        db = await self._get_db()

        await db.execute(
            """
            UPDATE campaign
            SET name = ?, description = ?, world_setting = ?, dm_user_id = ?, last_played_at = ?
            WHERE id = ?
            """,
            (
                campaign.name,
                campaign.description,
                campaign.world_setting,
                campaign.dm_user_id,
                campaign.last_played_at.isoformat() if campaign.last_played_at else None,
                campaign.id,
            ),
        )
        await db.commit()
        return True

    async def update_last_played(self, campaign_id: str) -> bool:
        """Update the last_played_at timestamp."""
        db = await self._get_db()

        await db.execute(
            "UPDATE campaign SET last_played_at = CURRENT_TIMESTAMP WHERE id = ?",
            (campaign_id,),
        )
        await db.commit()
        return True

    async def update_world_setting(self, campaign_id: str, world_setting: str) -> bool:
        """Update the world setting for a campaign."""
        db = await self._get_db()

        await db.execute(
            "UPDATE campaign SET world_setting = ? WHERE id = ?",
            (world_setting, campaign_id),
        )
        await db.commit()
        return True

    async def delete(self, campaign_id: str) -> bool:
        """Delete a campaign and all related data."""
        db = await self._get_db()

        await db.execute("DELETE FROM campaign WHERE id = ?", (campaign_id,))
        await db.commit()
        return True

    def _row_to_campaign(self, row) -> Campaign:
        """Convert a database row to a Campaign model.

        Column order from schema:
        0: id, 1: guild_id, 2: name, 3: description, 4: world_setting,
        5: dm_user_id, 6: created_at, 7: last_played_at
        """
        created_at = datetime.utcnow()
        if row[6]:
            try:
                created_at = datetime.fromisoformat(row[6])
            except (ValueError, TypeError):
                pass

        last_played = None
        if len(row) > 7 and row[7]:
            try:
                last_played = datetime.fromisoformat(row[7])
            except (ValueError, TypeError):
                pass

        return Campaign(
            id=row[0],
            guild_id=row[1],
            name=row[2],
            description=row[3],
            world_setting=row[4] or "A classic high fantasy world.",
            dm_user_id=row[5],
            created_at=created_at,
            last_played_at=last_played,
        )


# Global repository instance
_repo: Optional[CampaignRepository] = None


async def get_campaign_repo() -> CampaignRepository:
    """Get the global campaign repository."""
    global _repo
    if _repo is None:
        _repo = CampaignRepository()
    return _repo
