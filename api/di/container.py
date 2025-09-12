import structlog
from dependency_injector import containers, providers

from core.settings import SETTINGS
from infra.resources import DatabaseResource, RedisResource, MinIOResource


logger = structlog.get_logger("rag")


class DependencyContainer(containers.DeclarativeContainer):
    """Centralized dependency injection container."""

    config = providers.Configuration()

    wiring_config = containers.WiringConfiguration(
        modules=[
            "api.main",
            "api.features.documents.controller",
            "api.features.documents.router",
            "api.features.query.controller",
            "api.features.query.router",
            "api.features.jobs.controller",
            "api.features.jobs.router",
        ]
    )

    # Configuration
    settings = providers.Object(SETTINGS)

    # Logging
    logger = providers.Object(logger)

    # Database
    database = providers.Resource(
        DatabaseResource,
        database_url=str(SETTINGS.DATABASE.DATABASE_URL),
    )

    db_session = providers.Factory(
        lambda db: db.get_session(),
        db=database,
    )

    # Redis
    redis_db = providers.Resource(
        RedisResource,
        redis_url=str(SETTINGS.REDIS.REDIS_URL),
    )

    # MinIO
    minio_client = providers.Resource(
        MinIOResource,
        endpoint=SETTINGS.MINIO.MINIO_ENDPOINT,
        access_key=SETTINGS.MINIO.MINIO_ACCESS_KEY,
        secret_key=SETTINGS.MINIO.MINIO_SECRET_KEY.get_secret_value(),
        bucket_name=SETTINGS.MINIO.MINIO_BUCKET,
    )

    # Services
    document_service = providers.Factory(
        "api.features.documents.service.DocumentService",
        storage_client=minio_client,
    )

    query_service = providers.Factory(
        "api.features.query.service.QueryService",
        embedding_client=None,  # TODO: Add embedding client
        llm_client=None,  # TODO: Add LLM client
    )

    job_service = providers.Factory(
        "api.features.jobs.service.JobService",
        celery_app=None,  # TODO: Add Celery app
    )

    # Controllers
    document_controller = providers.Factory(
        "api.features.documents.controller.DocumentController",
        document_service=document_service,
    )

    query_controller = providers.Factory(
        "api.features.query.controller.QueryController"
    )

    job_controller = providers.Factory(
        "api.features.jobs.controller.JobController"
    )
