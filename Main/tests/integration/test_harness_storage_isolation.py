"""The standalone longform harness must never write to development stores."""

from pathlib import Path

import pytest

from dnd_bot.config import get_settings
from dnd_bot.data.database import get_database
from test_harness import TestSession as HarnessSession


@pytest.mark.asyncio
async def test_harness_uses_and_removes_run_unique_storage():
    settings = get_settings()
    original_paths = (
        settings.database_path,
        settings.chroma_persist_path,
    )
    session = HarnessSession(isolated_storage=True)

    assert await session.setup()
    storage_root = session.storage_root
    assert storage_root is not None
    assert settings.db_path == storage_root / "dnd_bot.db"
    assert settings.chroma_path == storage_root / "chroma"

    db = await get_database()
    row = await db.fetch_one(
        "SELECT COUNT(*) FROM campaign WHERE id = ?",
        (session.campaign_id,),
    )
    assert row and row[0] == 1

    await session.cleanup()

    assert (settings.database_path, settings.chroma_persist_path) == original_paths
    assert not Path(storage_root).exists()
