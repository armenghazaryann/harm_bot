"""Shared DB utilities across API and workers - Single source of truth."""

from typing import Optional

from core.settings import SETTINGS
from infra.resources import DatabaseResource


class DatabaseManager:
    """Centralized database connection management following Single Responsibility Principle."""

    _instance: Optional[DatabaseResource] = None

    @classmethod
    async def get_resource(cls) -> DatabaseResource:
        """Get initialized database resource (singleton pattern)."""
        if cls._instance is None:
            cls._instance = DatabaseResource(
                database_url=str(SETTINGS.DATABASE.DATABASE_URL)
            )
            await cls._instance.init()
        return cls._instance

    @classmethod
    async def reset(cls) -> None:
        """Reset singleton instance (for testing)."""
        if cls._instance is not None:
            await cls._instance.shutdown()
            cls._instance = None


def convert_async_to_sync_dsn(dsn: str) -> str:
    """Convert asyncpg DSN to psycopg for sync clients."""
    return dsn.replace("+asyncpg", "+psycopg")
