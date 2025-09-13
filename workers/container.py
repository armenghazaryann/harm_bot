"""Worker dependency injection container - follows SOLID principles."""
import structlog
from dependency_injector import containers, providers

from core.settings import SETTINGS
from infra.resources import DatabaseResource, MinIOResource

logger = structlog.get_logger("workers")


class WorkerInfrastructureContainer(containers.DeclarativeContainer):
    """Infrastructure dependencies for workers - Single Responsibility Principle."""

    config = providers.Configuration()
    settings = providers.Object(SETTINGS)
    logger = providers.Object(logger)

    # Database
    database = providers.Resource(
        DatabaseResource,
        database_url=str(SETTINGS.DATABASE.DATABASE_URL),
    )

    # MinIO
    minio_client = providers.Resource(
        MinIOResource,
        endpoint=SETTINGS.MINIO.MINIO_ENDPOINT,
        access_key=SETTINGS.MINIO.MINIO_ACCESS_KEY,
        secret_key=SETTINGS.MINIO.MINIO_SECRET_KEY.get_secret_value(),
        bucket_name=SETTINGS.MINIO.MINIO_BUCKET,
    )


class WorkerServiceContainer(containers.DeclarativeContainer):
    """Worker services - Dependency Inversion Principle."""

    infrastructure = providers.DependenciesContainer()

    # Database session factory
    db_session = providers.Factory(
        lambda db: db.get_session(),
        db=infrastructure.database,
    )


class WorkerContainer(containers.DeclarativeContainer):
    """Main worker container - composing all dependencies."""

    infrastructure = providers.Container(WorkerInfrastructureContainer)
    services = providers.Container(
        WorkerServiceContainer, infrastructure=infrastructure
    )
