"""Session repository for game session persistence."""

from datetime import datetime
from typing import Optional
import json

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
        players_json: Optional[str] = None,
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

    async def load_session(self, channel_id: int) -> Optional[dict]:
        """
        Load the most recent active session for a channel.

        Returns dict with session data or None if no active session.
        """
        db = await self._get_db()

        row = await db.fetch_one(
            """
            SELECT id, campaign_id, channel_id, session_number, state, active_combat_id, started_at
            FROM game_session
            WHERE channel_id = ? AND state NOT IN ('ended', 'paused')
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (channel_id,),
        )

        if not row:
            return None

        return {
            "id": row[0],
            "campaign_id": row[1],
            "channel_id": row[2],
            "session_number": row[3],
            "state": row[4],
            "active_combat_id": row[5],
            "started_at": row[6],
        }

    async def load_active_sessions(self) -> list[dict]:
        """
        Load all active sessions (for bot restart recovery).

        Returns list of session dicts.
        """
        db = await self._get_db()

        rows = await db.fetch_all(
            """
            SELECT id, campaign_id, channel_id, session_number, state, active_combat_id, started_at
            FROM game_session
            WHERE state NOT IN ('ended', 'paused')
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

    async def pause_session(self, session_id: str) -> None:
        """Mark a session as paused (can be resumed)."""
        db = await self._get_db()

        await db.execute(
            """
            UPDATE game_session
            SET state = 'paused'
            WHERE id = ?
            """,
            (session_id,),
        )
        await db.commit()

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

    async def create_snapshot(
        self,
        session_id: str,
        snapshot_type: str,
        game_state: dict,
    ) -> str:
        """Create a session snapshot for rollback support."""
        import uuid

        db = await self._get_db()
        snapshot_id = str(uuid.uuid4())

        await db.execute(
            """
            INSERT INTO session_snapshot (id, session_id, snapshot_type, game_state)
            VALUES (?, ?, ?, ?)
            """,
            (snapshot_id, session_id, snapshot_type, json.dumps(game_state)),
        )
        await db.commit()

        return snapshot_id

    async def get_latest_snapshot(self, session_id: str) -> Optional[dict]:
        """Get the most recent snapshot for a session."""
        db = await self._get_db()

        row = await db.fetch_one(
            """
            SELECT id, snapshot_type, game_state, created_at
            FROM session_snapshot
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id,),
        )

        if not row:
            return None

        return {
            "id": row[0],
            "snapshot_type": row[1],
            "game_state": json.loads(row[2]) if row[2] else {},
            "created_at": row[3],
        }


# Global repository instance
_repo: Optional[SessionRepository] = None


async def get_session_repo() -> SessionRepository:
    """Get the global session repository."""
    global _repo
    if _repo is None:
        _repo = SessionRepository()
    return _repo
