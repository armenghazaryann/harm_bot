"""Infrastructure resources: DB, Redis, MinIO.

This module is part of the infra layer and must not import from application features.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from urllib.parse import urlparse

from minio import Minio


class DatabaseResource:
    """Database resource for dependency injection."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.session_factory = None

    async def init(self):
        """Initialize database connection."""
        self.engine = create_async_engine(
            self.database_url,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        return self

    def get_session(self) -> AsyncSession:
        """Get database session (synchronous accessor)."""
        return self.session_factory()

    async def shutdown(self):
        """Shutdown database connection."""
        if self.engine:
            await self.engine.dispose()


class RedisResource:
    """Redis resource for dependency injection."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client = None

    async def init(self):
        """Initialize Redis connection."""
        # TODO: Implement actual Redis client
        return self

    async def connect(self):
        """Connect to Redis."""
        pass

    async def disconnect(self):
        """Disconnect from Redis."""
        pass


class MinIOResource:
    """MinIO resource for dependency injection."""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket_name: str):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.client = None

    async def init(self):
        """Initialize MinIO client."""
        # Parse endpoint to determine secure flag
        parsed = urlparse(self.endpoint if "://" in self.endpoint else f"http://{self.endpoint}")
        secure = parsed.scheme == "https"
        netloc = parsed.netloc or parsed.path  # handle cases like "minio:9000"

        self.client = Minio(
            endpoint=netloc,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=secure,
        )

        # Ensure bucket exists
        await self.ensure_bucket()
        return self

    async def ensure_bucket(self):
        """Ensure bucket exists."""
        assert self.client is not None, "MinIO client not initialized"
        found = self.client.bucket_exists(self.bucket_name)
        if not found:
            self.client.make_bucket(self.bucket_name)
