"""Session repository for game session persistence."""

from typing import Any, Optional

from ..database import Database, get_database


class SessionRepository:
    """
    Repository for GameSession database operations.

    Handles saving and loading game sessions to/from the database
    so sessions persist across bot restarts.
    """

    def __init__(self, db: Optional[Database] = None):
        self._db = db

    async def _get_db(self) -> Database:
        if self._db:
            return self._db
        return await get_database()

    async def save_session(
        self,
        session_id: str,
        campaign_id: str,
        channel_id: int,
        session_number: int,
        state: str,
        active_combat_id: Optional[str] = None,
    ) -> None:
        """Save or update a game session."""
        db = await self._get_db()

        # Check if session exists
        existing = await db.fetch_one(
            "SELECT id FROM game_session WHERE id = ?",
            (session_id,),
        )

        if existing:
            await db.execute(
                """
                UPDATE game_session
                SET state = ?, active_combat_id = ?
                WHERE id = ?
                """,
                (state, active_combat_id, session_id),
            )
        else:
            await db.execute(
                """
                INSERT INTO game_session (id, campaign_id, channel_id, session_number, state, active_combat_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, campaign_id, channel_id, session_number, state, active_combat_id),
            )

        await db.commit()

    async def end_session(self, session_id: str) -> None:
        """Mark a session as ended."""
        db = await self._get_db()

        await db.execute(
            """
            UPDATE game_session
            SET state = 'ended', ended_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (session_id,),
        )
        await db.commit()

    async def load_active_sessions(self) -> list[dict[str, Any]]:
        """Load every non-ended session row (bot-restart recovery, ROOT-3).

        Candidates for recovery; rows the recovery pass cannot rebuild
        (no world snapshot, missing campaign) are ended individually by
        the caller. Legacy 'paused' rows are included — nothing can write
        that state anymore (pause_session was deleted), so they resolve
        the same way: no snapshot, swept.
        """
        db = await self._get_db()

        rows = await db.fetch_all(
            """
            SELECT id, campaign_id, channel_id, session_number, state, active_combat_id, started_at
            FROM game_session
            WHERE state != 'ended'
            ORDER BY started_at DESC
            """,
        )

        return [
            {
                "id": row[0],
                "campaign_id": row[1],
                "channel_id": row[2],
                "session_number": row[3],
                "state": row[4],
                "active_combat_id": row[5],
                "started_at": row[6],
            }
            for row in rows
        ]

    async def save_world_snapshot(self, session_id: str, game_state: str) -> None:
        """Persist the session's world snapshot (replace semantics).

        One 'world' row per session under the stable id
        ``world:<session_id>``, replaced in a single statement — per-turn
        saves can't grow the table, and there is no multi-statement window
        in which an interleaved commit on the shared connection could
        strand a deleted-but-not-reinserted snapshot. ``game_state`` is
        the already-serialized JSON envelope — the session layer owns its
        shape, this layer just stores bytes.
        """
        db = await self._get_db()

        await db.execute(
            """
            INSERT OR REPLACE INTO session_snapshot (id, session_id, snapshot_type, game_state)
            VALUES (?, ?, 'world', ?)
            """,
            (f"world:{session_id}", session_id, game_state),
        )
        await db.commit()

    async def get_latest_snapshot(self, session_id: str) -> Optional[str]:
        """The session's most recent 'world' snapshot JSON, or None."""
        db = await self._get_db()

        row = await db.fetch_one(
            """
            SELECT game_state
            FROM session_snapshot
            WHERE session_id = ? AND snapshot_type = 'world'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id,),
        )

        return row[0] if row else None

    async def get_session_number(self, campaign_id: str) -> int:
        """Get the next session number for a campaign."""
        db = await self._get_db()

        row = await db.fetch_one(
            """
            SELECT COALESCE(MAX(session_number), 0) + 1
            FROM game_session
            WHERE campaign_id = ?
            """,
            (campaign_id,),
        )

        return row[0] if row else 1


# Global repository instance
_repo: Optional[SessionRepository] = None


async def get_session_repo() -> SessionRepository:
    """Get the global session repository."""
    global _repo
    if _repo is None:
        _repo = SessionRepository()
    return _repo
