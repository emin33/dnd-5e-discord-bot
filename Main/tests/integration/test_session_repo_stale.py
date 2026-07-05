"""Integration tests for SessionRepository.end_stale_sessions against a real (tmp) SQLite DB.

Covers audit P0 #10 (dead session resume): the recovery path was deleted, so
rows left active by a crashed previous run must be marked ended at startup.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnd_bot.data.database import Database
from dnd_bot.data.repositories.session_repo import SessionRepository


@pytest.fixture
async def db(tmp_path: Path):
    db = Database(db_path=tmp_path / "test.db")
    await db.connect()

    # Seed a campaign to satisfy the game_session FK.
    await db.execute(
        "INSERT INTO campaign (id, guild_id, name, dm_user_id) VALUES (?, ?, ?, ?)",
        ("test-campaign", 99999, "Test Campaign", 12345),
    )
    await db.commit()

    yield db

    await db.disconnect()


@pytest.fixture
def repo(db: Database) -> SessionRepository:
    return SessionRepository(db=db)


async def _seed_session(db: Database, session_id: str, state: str) -> None:
    await db.execute(
        """
        INSERT INTO game_session (id, campaign_id, channel_id, session_number, state)
        VALUES (?, 'test-campaign', 123, 1, ?)
        """,
        (session_id, state),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_stale_active_rows_marked_ended(db: Database, repo: SessionRepository):
    """Rows left in any non-'ended' state by a crash become 'ended'.

    'paused' is included: nothing can write that state (pause_session was
    deleted as dead scaffolding), so any legacy 'paused' row is stale too.
    """
    for sid, state in [
        ("s-lobby", "lobby"),
        ("s-explore", "exploration"),
        ("s-combat", "combat"),
        ("s-paused", "paused"),
    ]:
        await _seed_session(db, sid, state)

    count = await repo.end_stale_sessions()
    assert count == 4

    rows = await db.fetch_all("SELECT id, state, ended_at FROM game_session")
    assert len(rows) == 4
    for row in rows:
        assert row[1] == "ended"
        assert row[2] is not None  # ended_at stamped


@pytest.mark.asyncio
async def test_ended_rows_untouched(db: Database, repo: SessionRepository):
    """'ended' rows are not modified; sweep reports zero."""
    await _seed_session(db, "s-ended", "ended")

    count = await repo.end_stale_sessions()
    assert count == 0

    rows = {
        row[0]: row[1]
        for row in await db.fetch_all("SELECT id, state FROM game_session")
    }
    assert rows == {"s-ended": "ended"}
