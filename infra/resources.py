"""Infrastructure resources: DB, Redis, MinIO.

This module is part of the infra layer and must not import from application features.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from urllib.parse import urlparse

from minio import Minio
from neo4j import GraphDatabase


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
        if self.session_factory is None:
            raise RuntimeError("Database not initialized. Call init() first.")
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

    def __init__(
        self, endpoint: str, access_key: str, secret_key: str, bucket_name: str
    ):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.client = None

    async def init(self):
        """Initialize MinIO client."""
        # Parse endpoint to determine secure flag
        parsed = urlparse(
            self.endpoint if "://" in self.endpoint else f"http://{self.endpoint}"
        )
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

    async def get_object_bytes(self, bucket_name: str, object_name: str) -> bytes:
        """Get object bytes from MinIO storage."""
        assert self.client is not None, "MinIO client not initialized"

        try:
            response = self.client.get_object(bucket_name, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except Exception as e:
            raise RuntimeError(
                f"Failed to get object {object_name} from bucket {bucket_name}: {e}"
            )

    async def put_object_bytes(
        self,
        bucket_name: str,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Put object bytes to MinIO storage."""
        assert self.client is not None, "MinIO client not initialized"

        from io import BytesIO

        try:
            self.client.put_object(
                bucket_name,
                object_name,
                BytesIO(data),
                len(data),
                content_type=content_type,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to put object {object_name} to bucket {bucket_name}: {e}"
            )

    async def object_exists(self, bucket_name: str, object_name: str) -> bool:
        """Check if object exists in MinIO storage."""
        assert self.client is not None, "MinIO client not initialized"

        try:
            self.client.stat_object(bucket_name, object_name)
            return True
        except Exception:
            return False

    async def shutdown(self):
        """Shutdown MinIO client."""
        self.client = None
        return self


class Neo4jResource:
    """Neo4j driver resource."""

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    async def init(self):
        # Driver is synchronous factory; keep API symmetric
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        # Verify connectivity
        try:
            self.driver.verify_connectivity()
        except Exception:
            # Let caller handle; keep resource constructed
            pass
        return self

    async def shutdown(self):
        if self.driver:
            self.driver.close()
