"""Worker initialization module - ensures proper DI container setup."""
import structlog
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from workers.container import WorkerContainer

logger = structlog.get_logger("workers.initialization")


class WorkerInitializer:
    """Manages worker container lifecycle - follows SOLID principles."""

    _container: WorkerContainer = None

    @classmethod
    def get_container(cls) -> WorkerContainer:
        """Get the worker container singleton."""
        if cls._container is None:
            cls._container = WorkerContainer()
        return cls._container

    @classmethod
    @asynccontextmanager
    async def worker_context(cls) -> AsyncGenerator[WorkerContainer, None]:
        """
        Async context manager for worker container lifecycle.

        Yields:
            Initialized worker container
        """
        container = cls.get_container()

        try:
            # Initialize all resources - explicitly call init() methods
            db_resource = container.infrastructure.database()
            minio_resource = container.infrastructure.minio_client()

            # Ensure resources are properly initialized
            if hasattr(db_resource, "init") and callable(db_resource.init):
                await db_resource.init()
            if hasattr(minio_resource, "init") and callable(minio_resource.init):
                await minio_resource.init()

            logger.info("worker.initialization.complete")
            yield container

        except Exception as e:
            logger.error("worker.initialization.failed", error=str(e))
            raise
        finally:
            # Cleanup resources
            try:
                db_resource = container.infrastructure.database()
                minio_resource = container.infrastructure.minio_client()

                if hasattr(db_resource, "shutdown") and callable(db_resource.shutdown):
                    await db_resource.shutdown()
                if hasattr(minio_resource, "shutdown") and callable(
                    minio_resource.shutdown
                ):
                    await minio_resource.shutdown()

                logger.info("worker.cleanup.complete")
            except Exception as e:
                logger.warning("worker.cleanup.error", error=str(e))


# Global worker initializer instance
worker_initializer = WorkerInitializer()
