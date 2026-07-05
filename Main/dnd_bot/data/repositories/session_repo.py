"""Session repository for game session persistence."""

from typing import Optional

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

    async def end_stale_sessions(self) -> int:
        """
        Mark lingering non-terminal sessions as ended (startup hygiene).

        Session resume is not supported, so rows left active by a crashed
        or killed previous run would otherwise stay active forever. Called
        once at bot startup. Returns the number of rows updated.
        """
        db = await self._get_db()

        cursor = await db.execute(
            """
            UPDATE game_session
            SET state = 'ended', ended_at = CURRENT_TIMESTAMP
            WHERE state != 'ended'
            """,
        )
        await db.commit()
        return cursor.rowcount

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
