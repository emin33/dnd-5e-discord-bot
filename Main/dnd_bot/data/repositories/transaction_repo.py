"""Transaction repository for idempotency pattern."""

from datetime import datetime, timedelta
from typing import Any, Optional
import json

from ..database import Database, get_database


class TransactionRepository:
    """
    Repository for managing idempotent transactions.

    This prevents duplicate state changes from LLM retries by:
    1. Generating a unique transaction key for each operation
    2. Checking if the transaction was already applied
    3. Returning the cached result if already applied
    4. Recording new transactions with their results

    Transaction keys follow the pattern: "{operation}:{target}:{unique_id}"
    Example: "hp:char_123:abc123"
    """

    DEFAULT_TTL_HOURS = 24

    def __init__(self, db: Optional[Database] = None):
        self._db = db

    async def _get_db(self) -> Database:
        if self._db:
            return self._db
        return await get_database()

    async def exists(self, transaction_key: str) -> bool:
        """Check if a transaction has already been applied."""
        db = await self._get_db()

        row = await db.fetch_one(
            """
            SELECT 1 FROM transaction_log
            WHERE transaction_key = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
            (transaction_key,),
        )

        return row is not None

    async def get_result(self, transaction_key: str) -> Optional[dict]:
        """Get the cached result of a previously applied transaction."""
        db = await self._get_db()

        row = await db.fetch_one(
            """
            SELECT result FROM transaction_log
            WHERE transaction_key = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
            (transaction_key,),
        )

        if not row or not row[0]:
            return None

        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return None

    async def record(
        self,
        transaction_key: str,
        operation_type: str,
        target_id: str,
        result: Any,
        payload: Optional[dict] = None,
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ) -> None:
        """
        Record a completed transaction.

        Args:
            transaction_key: Unique key for this transaction
            operation_type: Type of operation (e.g., "update_hp", "apply_condition")
            target_id: ID of the target entity
            result: The result to cache
            payload: Optional input payload for debugging
            ttl_hours: Hours until this record expires
        """
        db = await self._get_db()

        expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)

        await db.execute(
            """
            INSERT OR REPLACE INTO transaction_log
            (transaction_key, operation_type, target_id, payload, result, applied_at, expires_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            """,
            (
                transaction_key,
                operation_type,
                target_id,
                json.dumps(payload) if payload else None,
                json.dumps(result) if result else None,
                expires_at.isoformat(),
            ),
        )
        await db.commit()

    async def check_and_record(
        self,
        transaction_key: str,
        operation_type: str,
        target_id: str,
        result: Any,
        payload: Optional[dict] = None,
    ) -> tuple[bool, Optional[dict]]:
        """
        Check if transaction exists and record if not.

        Returns:
            (was_new, result): If was_new is False, result is the cached result.
                               If was_new is True, the transaction was recorded.
        """
        # Check if already exists
        existing = await self.get_result(transaction_key)
        if existing is not None:
            return (False, existing)

        # Record new transaction
        await self.record(
            transaction_key=transaction_key,
            operation_type=operation_type,
            target_id=target_id,
            result=result,
            payload=payload,
        )

        return (True, None)

    async def cleanup_expired(self) -> int:
        """Remove expired transaction records. Returns count of deleted records."""
        db = await self._get_db()

        cursor = await db.execute(
            "DELETE FROM transaction_log WHERE expires_at < CURRENT_TIMESTAMP"
        )
        await db.commit()

        return cursor.rowcount

    async def get_recent(
        self,
        operation_type: Optional[str] = None,
        target_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get recent transactions for debugging/audit."""
        db = await self._get_db()

        query = "SELECT * FROM transaction_log WHERE 1=1"
        params = []

        if operation_type:
            query += " AND operation_type = ?"
            params.append(operation_type)

        if target_id:
            query += " AND target_id = ?"
            params.append(target_id)

        query += " ORDER BY applied_at DESC LIMIT ?"
        params.append(limit)

        rows = await db.fetch_all(query, tuple(params))

        results = []
        for row in rows:
            results.append({
                "transaction_key": row[0],
                "operation_type": row[1],
                "target_id": row[2],
                "payload": json.loads(row[3]) if row[3] else None,
                "result": json.loads(row[4]) if row[4] else None,
                "applied_at": row[5],
                "expires_at": row[6],
            })

        return results


def generate_transaction_key(
    operation: str,
    target_id: str,
    unique_id: Optional[str] = None,
) -> str:
    """
    Generate a transaction key.

    Args:
        operation: The operation type (e.g., "hp", "condition", "spell_slot")
        target_id: The target entity ID
        unique_id: Optional unique identifier (if not provided, uses timestamp)

    Returns:
        A transaction key string
    """
    if unique_id is None:
        import uuid
        unique_id = str(uuid.uuid4())[:8]

    return f"{operation}:{target_id}:{unique_id}"


# Global repository instance
_repo: Optional[TransactionRepository] = None


async def get_transaction_repo() -> TransactionRepository:
    """Get the global transaction repository."""
    global _repo
    if _repo is None:
        _repo = TransactionRepository()
    return _repo
