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
        embedding_client=None,
        llm_client=None,
    )


class WorkerContainer(containers.DeclarativeContainer):
    """Worker-specific dependencies."""

    infrastructure = providers.DependenciesContainer()

    # Worker tasks
    transcript_create_jsonl = providers.Callable(
        "workers.transcripts.create_utterances_jsonl"
    )
    transcript_ingest_pg = providers.Callable(
        "workers.transcripts.ingest_transcript_pg_from_minio"
    )
    transcript_ingest_neo4j = providers.Callable(
        "workers.transcripts.ingest_transcript_neo4j_from_minio"
    )
    transcript_materialize_chunks = providers.Callable(
        "workers.transcripts.materialize_transcript_chunks_from_pg"
    )

    # Workers: Embeddings
    embed_document_chunks = providers.Callable(
        "workers.embeddings.embed_document_chunks"
    )

    # Workers: Indexing with LangChain VectorStore (PGVector)
    indexing_pgvector = providers.Callable("workers.indexing.index_chunks_pgvector")


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
