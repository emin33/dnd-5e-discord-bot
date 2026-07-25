"""Repository for durable class-feature resource counters."""

from typing import Optional

from ..database import Database, get_database
from ...models.feature_resource import FeatureResource


class FeatureResourceRepository:
    """CRUD for ``feature_resources`` rows.

    The rest flow is the primary writer: load (seed lazily if empty),
    let the RestManager mutate ``current`` in memory, save back.
    """

    def __init__(self, db: Optional[Database] = None):
        self._db = db

    async def _get_db(self) -> Database:
        if self._db:
            return self._db
        return await get_database()

    async def list_for_character(self, character_id: str) -> list[FeatureResource]:
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT character_id, resource_key, name, current, maximum,
                   recharge_rule, source
            FROM feature_resources
            WHERE character_id = ?
            ORDER BY resource_key
            """,
            (character_id,),
        )
        return [
            FeatureResource(
                character_id=row[0],
                resource_key=row[1],
                name=row[2],
                current=row[3],
                maximum=row[4],
                recharge_rule=row[5],
                source=row[6],
            )
            for row in rows or []
        ]

    async def save_all(self, resources: list[FeatureResource]) -> None:
        """Upsert every row; the (character_id, resource_key) PK dedups."""
        if not resources:
            return
        db = await self._get_db()
        for resource in resources:
            await db.execute(
                """
                INSERT INTO feature_resources
                    (character_id, resource_key, name, current, maximum,
                     recharge_rule, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(character_id, resource_key) DO UPDATE SET
                    name = excluded.name,
                    current = excluded.current,
                    maximum = excluded.maximum,
                    recharge_rule = excluded.recharge_rule,
                    source = excluded.source
                """,
                (
                    resource.character_id,
                    resource.resource_key,
                    resource.name,
                    resource.current,
                    resource.maximum,
                    resource.recharge_rule,
                    resource.source,
                ),
            )

    async def set_current(
        self, character_id: str, resource_key: str, current: int
    ) -> bool:
        """Targeted counter write (the spend path). False if no such row."""
        db = await self._get_db()
        cursor = await db.execute(
            """
            UPDATE feature_resources SET current = ?
            WHERE character_id = ? AND resource_key = ?
            """,
            (max(0, current), character_id, resource_key),
        )
        return bool(getattr(cursor, "rowcount", 0))

    async def delete_for_character(self, character_id: str) -> None:
        db = await self._get_db()
        await db.execute(
            "DELETE FROM feature_resources WHERE character_id = ?",
            (character_id,),
        )


_repo: Optional[FeatureResourceRepository] = None


async def get_feature_resource_repo() -> FeatureResourceRepository:
    """Get the global feature-resource repository."""
    global _repo
    if _repo is None:
        _repo = FeatureResourceRepository()
    return _repo
