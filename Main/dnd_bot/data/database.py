"""Database connection and migration management."""

import asyncio
from pathlib import Path
from typing import Optional
import aiosqlite
import structlog

from ..config import get_settings

logger = structlog.get_logger()

# Path to migrations directory
MIGRATIONS_DIR = Path(__file__).parent.parent.parent / "migrations"


class Database:
    """Async SQLite database manager."""

    def __init__(self, db_path: Optional[Path] = None):
        settings = get_settings()
        self.db_path = db_path or settings.db_path
        self._connection: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Connect to the database and run migrations."""
        if self._connection is not None:
            return

        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("connecting_to_database", path=str(self.db_path))
        self._connection = await aiosqlite.connect(self.db_path)

        # Enable foreign keys
        await self._connection.execute("PRAGMA foreign_keys = ON")

        # WAL mode: allows concurrent reads while writing, prevents "database is locked"
        await self._connection.execute("PRAGMA journal_mode = WAL")

        # Run migrations
        await self._run_migrations()

        logger.info("database_connected")

    async def disconnect(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
            logger.info("database_disconnected")

    @property
    def connection(self) -> aiosqlite.Connection:
        """Get the database connection. Raises if not connected."""
        if self._connection is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._connection

    async def execute(
        self, sql: str, parameters: tuple = ()
    ) -> aiosqlite.Cursor:
        """Execute a SQL statement."""
        return await self.connection.execute(sql, parameters)

    async def execute_many(
        self, sql: str, parameters: list[tuple]
    ) -> aiosqlite.Cursor:
        """Execute a SQL statement with multiple parameter sets."""
        return await self.connection.executemany(sql, parameters)

    async def fetch_one(
        self, sql: str, parameters: tuple = ()
    ) -> Optional[aiosqlite.Row]:
        """Fetch a single row."""
        cursor = await self.execute(sql, parameters)
        return await cursor.fetchone()

    async def fetch_all(
        self, sql: str, parameters: tuple = ()
    ) -> list[aiosqlite.Row]:
        """Fetch all rows."""
        cursor = await self.execute(sql, parameters)
        return await cursor.fetchall()

    async def commit(self) -> None:
        """Commit the current transaction."""
        await self.connection.commit()

    async def rollback(self) -> None:
        """Rollback the current transaction."""
        await self.connection.rollback()

    async def _run_migrations(self) -> None:
        """Run pending database migrations."""
        # Ensure schema_migrations table exists
        await self.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await self.commit()

        # Get current version
        row = await self.fetch_one(
            "SELECT MAX(version) as version FROM schema_migrations"
        )
        current_version = row[0] if row and row[0] else 0

        # Find migration files
        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

        for migration_file in migration_files:
            # Extract version from filename (e.g., 001_initial_schema.sql -> 1)
            version = int(migration_file.stem.split("_")[0])

            if version > current_version:
                logger.info(
                    "applying_migration",
                    version=version,
                    file=migration_file.name,
                )

                # Read and execute migration
                sql = migration_file.read_text()

                # Execute the migration (may contain multiple statements)
                await self.connection.executescript(sql)
                await self.commit()

                logger.info("migration_applied", version=version)

    async def transaction(self):
        """Context manager for transactions with automatic commit/rollback."""
        return TransactionContext(self)


class TransactionContext:
    """Context manager for database transactions."""

    def __init__(self, db: Database):
        self.db = db
        self._savepoint_name: Optional[str] = None

    async def __aenter__(self):
        # Use savepoint for nested transactions
        import uuid

        self._savepoint_name = f"sp_{uuid.uuid4().hex[:8]}"
        await self.db.execute(f"SAVEPOINT {self._savepoint_name}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Rollback on exception
            await self.db.execute(f"ROLLBACK TO SAVEPOINT {self._savepoint_name}")
            logger.warning("transaction_rolled_back", error=str(exc_val))
        else:
            # Release savepoint (commit)
            await self.db.execute(f"RELEASE SAVEPOINT {self._savepoint_name}")
            await self.db.commit()
        return False  # Don't suppress exceptions


# Global database instance
_db: Optional[Database] = None


async def get_database() -> Database:
    """Get the global database instance, connecting if needed."""
    global _db
    if _db is None:
        _db = Database()
        await _db.connect()
    return _db


async def close_database() -> None:
    """Close the global database connection."""
    global _db
    if _db is not None:
        await _db.disconnect()
        _db = None
