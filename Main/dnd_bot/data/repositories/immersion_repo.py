"""Repository for immersion feature settings and voice catalog."""

import json
from pathlib import Path
from typing import Optional

from ...models.immersion import (
    GuildImmersionSettings,
    ImageFrequency,
    VoiceCatalogEntry,
)
from ..database import Database, get_database


class ImmersionRepository:
    """Repository for guild immersion settings and voice catalog."""

    _catalog_seeded: bool = False  # Class-level: shared across all instances

    def __init__(self, db: Optional[Database] = None):
        self._db = db

    async def _get_db(self) -> Database:
        if self._db:
            return self._db
        return await get_database()

    # ==================== Guild Settings ====================

    async def get_guild_settings(self, guild_id: int) -> Optional[GuildImmersionSettings]:
        """Get immersion settings for a guild."""
        db = await self._get_db()
        row = await db.fetch_one(
            "SELECT * FROM guild_immersion_settings WHERE guild_id = ?",
            (guild_id,),
        )
        if not row:
            return None
        return GuildImmersionSettings(
            guild_id=row[0],
            tts_enabled=bool(row[1]),
            image_enabled=bool(row[2]),
            image_frequency=ImageFrequency(row[3]),
            narrator_tts_provider=row[4] or "kokoro",
            narrator_tts_voice=row[5] or "af_heart",
            character_tts_provider=row[6] if len(row) > 6 else "",
        )

    async def upsert_guild_settings(self, settings: GuildImmersionSettings) -> None:
        """Create or update guild immersion settings."""
        db = await self._get_db()
        await db.execute(
            """
            INSERT OR REPLACE INTO guild_immersion_settings
            (guild_id, tts_enabled, image_enabled, image_frequency,
             narrator_tts_provider, narrator_tts_voice, character_tts_provider)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                settings.guild_id,
                1 if settings.tts_enabled else 0,
                1 if settings.image_enabled else 0,
                settings.image_frequency.value,
                settings.narrator_tts_provider,
                settings.narrator_tts_voice,
                settings.character_tts_provider,
            ),
        )
        await db.commit()

    async def get_or_create_guild_settings(
        self, guild_id: int
    ) -> GuildImmersionSettings:
        """Get settings, creating defaults if none exist."""
        settings = await self.get_guild_settings(guild_id)
        if settings is None:
            settings = GuildImmersionSettings(guild_id=guild_id)
            await self.upsert_guild_settings(settings)
        return settings

    # ==================== Voice Catalog ====================

    async def _ensure_catalog_seeded(self) -> None:
        """Seed the voice catalog from JSON on first access.

        Uses INSERT OR IGNORE so new voices added to the JSON (e.g. a new
        provider like Kokoro) are picked up on existing databases without
        duplicating existing entries.
        """
        if ImmersionRepository._catalog_seeded:
            return
        ImmersionRepository._catalog_seeded = True

        catalog_path = Path(__file__).parent.parent / "voice_catalog.json"
        if not catalog_path.exists():
            return

        # Compare DB count vs JSON count — reseed if JSON has more entries
        db = await self._get_db()
        row = await db.fetch_one("SELECT COUNT(*) FROM voice_catalog")
        db_count = row[0] if row else 0
        json_data = json.loads(catalog_path.read_text())
        json_count = len(json_data)

        if db_count >= json_count:
            return  # DB already has all entries

        count = await self.seed_voice_catalog(catalog_path)
        if count > 0:
            import structlog
            structlog.get_logger().info(
                "voice_catalog_seeded", total=count, new=json_count - db_count
            )

    async def get_all_voices(self) -> list[VoiceCatalogEntry]:
        """Get all voices in the catalog."""
        await self._ensure_catalog_seeded()
        db = await self._get_db()
        rows = await db.fetch_all(
            "SELECT voice_id, name, provider, gender, age, style_tags FROM voice_catalog"
        )
        return [self._row_to_voice(row) for row in rows]

    async def get_available_voices(
        self,
        gender: Optional[str] = None,
        age: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> list[VoiceCatalogEntry]:
        """Get voices matching gender/age/provider criteria."""
        await self._ensure_catalog_seeded()
        db = await self._get_db()
        query = "SELECT voice_id, name, provider, gender, age, style_tags FROM voice_catalog WHERE 1=1"
        params: list = []

        if gender:
            query += " AND gender = ?"
            params.append(gender)
        if age:
            query += " AND age = ?"
            params.append(age)
        if provider:
            query += " AND provider = ?"
            params.append(provider)

        rows = await db.fetch_all(query, tuple(params))
        return [self._row_to_voice(row) for row in rows]

    async def get_voice_by_id(self, voice_id: str) -> Optional[VoiceCatalogEntry]:
        """Get a specific voice by ID."""
        db = await self._get_db()
        row = await db.fetch_one(
            "SELECT voice_id, name, provider, gender, age, style_tags FROM voice_catalog WHERE voice_id = ?",
            (voice_id,),
        )
        return self._row_to_voice(row) if row else None

    async def seed_voice_catalog(self, catalog_path: Path) -> int:
        """Seed the voice catalog from a JSON file. Returns count of voices added."""
        db = await self._get_db()

        data = json.loads(catalog_path.read_text())
        count = 0

        for entry in data:
            await db.execute(
                """
                INSERT OR IGNORE INTO voice_catalog (voice_id, name, provider, gender, age, style_tags)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["voice_id"],
                    entry["name"],
                    entry.get("provider", "elevenlabs"),
                    entry.get("gender", "neutral"),
                    entry.get("age", "mature"),
                    json.dumps(entry.get("style_tags", [])),
                ),
            )
            count += 1

        await db.commit()
        return count

    def _row_to_voice(self, row) -> VoiceCatalogEntry:
        """Convert database row to VoiceCatalogEntry."""
        style_tags = []
        if row[5]:
            try:
                style_tags = json.loads(row[5])
            except (json.JSONDecodeError, TypeError):
                pass
        return VoiceCatalogEntry(
            voice_id=row[0],
            name=row[1],
            provider=row[2] or "elevenlabs",
            gender=row[3] or "neutral",
            age=row[4] or "mature",
            style_tags=style_tags,
        )


# Global repository instance
_repo: Optional[ImmersionRepository] = None


async def get_immersion_repo() -> ImmersionRepository:
    """Get the global immersion repository."""
    global _repo
    if _repo is None:
        _repo = ImmersionRepository()
    return _repo
