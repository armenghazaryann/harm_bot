from __future__ import annotations

import structlog
from dependency_injector import containers, providers

from core.settings import SETTINGS
from infra.resources import (
    DatabaseResource,
    RedisResource,
    MinIOResource,
    Neo4jResource,
)


logger = structlog.get_logger("rag")


class InfrastructureContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    settings = providers.Object(SETTINGS)
    logger = providers.Object(logger)

    # Database
    database = providers.Resource(
        DatabaseResource,
        database_url=str(SETTINGS.DATABASE.DATABASE_URL),
    )

    # Redis (optional)
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

    # Neo4j (optional)
    neo4j = providers.Resource(
        Neo4jResource,
        uri=SETTINGS.NEO4J.NEO4J_URI,
        user=SETTINGS.NEO4J.NEO4J_USER,
        password=SETTINGS.NEO4J.NEO4J_PASSWORD.get_secret_value(),
    )


class ServiceContainer(containers.DeclarativeContainer):
    """Application services - depends on infrastructure."""

    infrastructure = providers.DependenciesContainer()

    # Database session factory
    db_session = providers.Factory(
        lambda db: db.get_session(),
        db=infrastructure.database,
    )

    # Services
    document_service = providers.Factory(
        "api.features.documents.service.DocumentService",
        storage_client=infrastructure.minio_client,
    )

    query_service = providers.Factory(
        "api.features.query.service.QueryService",
    )

    # Query pipeline (RAG orchestrator)
    query_pipeline = providers.Factory(
        "rag.pipeline.query_pipeline.QueryPipeline",
    )


class ControllerContainer(containers.DeclarativeContainer):
    """Controller-specific dependencies."""

    services = providers.DependenciesContainer()

    # Controllers
    document_controller = providers.Factory(
        "api.features.documents.controller.DocumentController",
        document_service=services.document_service,
    )

    query_controller = providers.Factory(
        "api.features.query.controller.QueryController",
        query_service=services.query_service,
        query_pipeline=services.query_pipeline,
    )

    conversation_controller = providers.Factory(
        "api.features.conversation.controller.ConversationController",
    )


class ApplicationContainer(containers.DeclarativeContainer):
    """Main application container composing all sub-containers."""

    wiring_config = containers.WiringConfiguration(
        modules=[
            "api.main",
            "api.shared.db",
            "api.features.documents.controller",
            "api.features.documents.router",
            "api.features.query.controller",
            "api.features.query.router",
            "api.features.conversation.controller",
            "api.features.conversation.router",
        ]
    )

    infrastructure = providers.Container(InfrastructureContainer)
    services = providers.Container(ServiceContainer, infrastructure=infrastructure)
    controllers = providers.Container(ControllerContainer, services=services)
