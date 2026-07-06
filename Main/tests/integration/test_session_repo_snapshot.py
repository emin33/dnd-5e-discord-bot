"""Integration tests for SessionRepository's ROOT-3 persistence surface.

Real (tmp) SQLite via migration 001 — the session_snapshot table existed
there since the initial schema but was dead until this slice. Covers:
save_world_snapshot's replace semantics (one 'world' row per session,
per-turn saves can't grow the table), get_latest_snapshot, and
load_active_sessions (the recovery candidate query, including legacy
'paused' rows).
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


async def _seed_session(db: Database, session_id: str, state: str = "exploration") -> None:
    await db.execute(
        """
        INSERT INTO game_session (id, campaign_id, channel_id, session_number, state)
        VALUES (?, 'test-campaign', 123, 1, ?)
        """,
        (session_id, state),
    )
    await db.commit()


class TestWorldSnapshot:
    async def test_save_then_get_round_trips(self, db, repo):
        await _seed_session(db, "s-1")

        await repo.save_world_snapshot("s-1", '{"version": 1, "turn": 3}')

        assert await repo.get_latest_snapshot("s-1") == '{"version": 1, "turn": 3}'

    async def test_get_without_snapshot_returns_none(self, db, repo):
        await _seed_session(db, "s-1")

        assert await repo.get_latest_snapshot("s-1") is None

    async def test_save_replaces_do_not_accumulate(self, db, repo):
        """Per-turn saves keep exactly ONE 'world' row per session."""
        await _seed_session(db, "s-1")

        for turn in range(1, 6):
            await repo.save_world_snapshot("s-1", f'{{"turn": {turn}}}')

        rows = await db.fetch_all(
            "SELECT game_state FROM session_snapshot WHERE session_id = 's-1'"
        )
        assert len(rows) == 1
        assert rows[0][0] == '{"turn": 5}'
        assert await repo.get_latest_snapshot("s-1") == '{"turn": 5}'

    async def test_snapshots_are_per_session(self, db, repo):
        await _seed_session(db, "s-1")
        await _seed_session(db, "s-2")

        await repo.save_world_snapshot("s-1", '{"who": "one"}')
        await repo.save_world_snapshot("s-2", '{"who": "two"}')

        assert await repo.get_latest_snapshot("s-1") == '{"who": "one"}'
        assert await repo.get_latest_snapshot("s-2") == '{"who": "two"}'

    async def test_non_world_snapshot_rows_survive_and_are_ignored(self, db, repo):
        """Replace semantics touch only 'world' rows; other types are
        neither deleted nor served (forward-compat with manual snapshots)."""
        await _seed_session(db, "s-1")
        await db.execute(
            """
            INSERT INTO session_snapshot (id, session_id, snapshot_type, game_state)
            VALUES ('manual-1', 's-1', 'manual', '{"manual": true}')
            """,
        )
        await db.commit()

        await repo.save_world_snapshot("s-1", '{"world": true}')
        await repo.save_world_snapshot("s-1", '{"world": 2}')

        assert await repo.get_latest_snapshot("s-1") == '{"world": 2}'
        rows = await db.fetch_all(
            "SELECT snapshot_type FROM session_snapshot WHERE session_id = 's-1' ORDER BY snapshot_type"
        )
        assert [r[0] for r in rows] == ["manual", "world"]


class TestLoadActiveSessions:
    async def test_returns_non_ended_rows_with_fields(self, db, repo):
        await _seed_session(db, "s-active", "exploration")
        await _seed_session(db, "s-combat", "combat")
        await _seed_session(db, "s-paused", "paused")
        await _seed_session(db, "s-ended", "ended")

        rows = await repo.load_active_sessions()

        by_id = {r["id"]: r for r in rows}
        assert set(by_id) == {"s-active", "s-combat", "s-paused"}
        active = by_id["s-active"]
        assert active["campaign_id"] == "test-campaign"
        assert active["channel_id"] == 123
        assert active["session_number"] == 1
        assert active["state"] == "exploration"
        assert active["active_combat_id"] is None
        assert active["started_at"] is not None

    async def test_empty_table_returns_empty_list(self, db, repo):
        assert await repo.load_active_sessions() == []
