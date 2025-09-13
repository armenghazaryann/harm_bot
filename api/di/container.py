"""Centralized dependency injection container following SOLID principles."""
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
    """Infrastructure layer dependencies - follows Single Responsibility Principle."""

    config = providers.Configuration()
    settings = providers.Object(SETTINGS)
    logger = providers.Object(logger)

    # Database
    database = providers.Resource(
        DatabaseResource,
        database_url=str(SETTINGS.DATABASE.DATABASE_URL),
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

    # Neo4j
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


class WorkerContainer(containers.DeclarativeContainer):
    """Worker-specific dependencies - MIGRATED TO LANGCHAIN.

    All deprecated worker functions have been replaced by LangChain components:
    - workers.transcripts.* → workers.langchain_processor.process_document_langchain_task
    - workers.embeddings.* → LangChain OpenAIEmbeddings + PGVector automatic handling
    - workers.indexing.* → LangChain PGVector automatic indexing
    """

    infrastructure = providers.DependenciesContainer()

    # LangChain Document Processor - replaces all deprecated workers
    langchain_processor = providers.Callable(
        "workers.langchain_processor.process_document_with_langchain"
    )

    # LangChain Query Service - production-grade hybrid search
    langchain_query_service = providers.Callable(
        "workers.langchain_query_service.query_documents"
    )

    # Note: Indexing is now handled automatically by LangChain PGVector
    # No manual indexing workers needed - LangChain handles HNSW indexing automatically


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
    )


class ApplicationContainer(containers.DeclarativeContainer):
    """Main application container composing all sub-containers."""

    wiring_config = containers.WiringConfiguration(
        modules=[
            "api.main",
            "api.features.documents.controller",
            "api.features.documents.router",
            "api.features.query.controller",
            "api.features.query.router",
        ]
    )

    infrastructure = providers.Container(InfrastructureContainer)
    services = providers.Container(ServiceContainer, infrastructure=infrastructure)
    workers = providers.Container(WorkerContainer, infrastructure=infrastructure)
    controllers = providers.Container(ControllerContainer, services=services)
