from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, PostgresDsn, RedisDsn, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CustomSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.docker",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


class AppSettings(CustomSettings):
    ENVIRONMENT: Literal["local", "dev", "prod"] = Field(default="local")
    LOG_LEVEL: str = Field(default="INFO")
    JSON_LOGS: bool = Field(default=True)


class PgDbSettings(CustomSettings):
    POSTGRES_ENGINE: str = Field(default="postgresql+asyncpg")
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: SecretStr = Field(default="postgres")
    POSTGRES_DB: str = Field(default="rag")
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    DATABASE_URL: PostgresDsn | str = Field(default="")

    @model_validator(mode="before")
    def validate_postgres_dsn(cls, data: dict):
        if isinstance(data, dict) and not data.get("DATABASE_URL"):
            _built_uri = PostgresDsn.build(
                scheme=data.get("POSTGRES_ENGINE", "postgresql+asyncpg"),
                username=data.get("POSTGRES_USER", "postgres"),
                password=data.get("POSTGRES_PASSWORD", "postgres"),
                host=data.get("POSTGRES_HOST", "localhost"),
                port=int(data.get("POSTGRES_PORT", 5432)),
                path=data.get("POSTGRES_DB", "rag"),
            ).unicode_string()
            data["DATABASE_URL"] = _built_uri
        return data


class RedisSettings(CustomSettings):
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0)
    REDIS_PASSWORD: SecretStr = Field(default="")
    REDIS_URL: RedisDsn | str = Field(default="")
    CELERY_BROKER_URL: str = Field(default="")
    CELERY_RESULT_BACKEND: str = Field(default="")

    @model_validator(mode="before")
    def validate_redis_url(cls, data: dict):
        if isinstance(data, dict) and not data.get("REDIS_URL"):
            password = data.get("REDIS_PASSWORD", "")
            _built_uri = RedisDsn.build(
                scheme="redis",
                host=data.get("REDIS_HOST", "localhost"),
                port=int(data.get("REDIS_PORT", 6379)),
                path=f"/{data.get('REDIS_DB', 0)}",
                password=password if password else None,
            ).unicode_string()
            data["REDIS_URL"] = _built_uri
        # Set Celery URLs if not provided
        redis_url = data.get("REDIS_URL", "redis://localhost:6379/0")
        if not data.get("CELERY_BROKER_URL"):
            data["CELERY_BROKER_URL"] = redis_url
        if not data.get("CELERY_RESULT_BACKEND"):
            data["CELERY_RESULT_BACKEND"] = redis_url
        return data


class MinIOSettings(CustomSettings):
    MINIO_ENDPOINT: str = Field(default="http://localhost:9000")
    MINIO_ACCESS_KEY: str = Field(default="minioadmin")
    MINIO_SECRET_KEY: SecretStr = Field(default="minioadmin")
    MINIO_BUCKET: str = Field(default="rag")


class OpenAISettings(CustomSettings):
    OPENAI_API_KEY: SecretStr = Field(default="")
    USE_OPENAI: bool = Field(default=True)


class Neo4jSettings(CustomSettings):
    NEO4J_URI: str = Field(default="bolt://localhost:7687")
    NEO4J_USER: str = Field(default="neo4j")
    NEO4J_PASSWORD: SecretStr = Field(default="neo4j")


class JinaSettings(CustomSettings):
    """Configuration for Jina AI APIs (Reranker, Embeddings, etc.).

    Env vars:
    - JINA_API_KEY
    - JINA_RERANKER_ENABLED
    - JINA_RERANKER_MODEL
    """

    JINA_API_KEY: SecretStr = Field(default="")
    RERANKER_ENABLED: bool = Field(default=True)
    RERANKER_MODEL: str = Field(default="jina-reranker-v2-base-multilingual")


class ProcessingSettings(CustomSettings):
    PAGE_DPI: int = Field(default=300)
    EMBED_BATCH_SIZE: int = Field(default=128)
    MAX_CONCURRENCY: int = Field(default=4)
    TOLERANCE_REL: float = Field(default=0.001)
    # Chunking configuration for LangChain splitters (token-based if tiktoken available)
    CHUNK_TOKENS: int = Field(default=512)
    CHUNK_OVERLAP_TOKENS: int = Field(default=80)


class VectorSettings(CustomSettings):
    """Configuration for PGVector-backed vector store.

    Set via env vars:
    - VECTOR_COLLECTION_NAME
    - VECTOR_USE_JSONB
    """

    COLLECTION_NAME: str = Field(default="langchain_pgvector")
    VECTOR_USE_JSONB: bool = Field(default=True)


class UiSettings(CustomSettings):
    """Configuration for Streamlit UI to reach API endpoints.

    Set via env vars:
    - API_BASE_URL
    - ENDPOINT_QUERY_ANSWER
    - ENDPOINT_DOCUMENT_UPLOAD
    """

    API_BASE_URL: str = Field(default="http://localhost:8000")
    ENDPOINT_QUERY_ANSWER: str = Field(default="/api/v1/query/answer")
    ENDPOINT_DOCUMENT_UPLOAD: str = Field(default="/api/v1/documents/upload")


class SelfRagSettings(CustomSettings):
    """Configuration for Self-RAG Lite query flow.

    Set via env vars (optional):
    - SELF_RAG_ENABLED
    - SELF_RAG_TOP_K
    - SELF_RAG_TOP_M
    - SELF_RAG_RELEVANCE_THRESHOLD
    - SELF_RAG_SUPPORT_THRESHOLD
    - SELF_RAG_MAX_REWRITE
    - SELF_RAG_MAX_CLAIMS
    - SELF_RAG_RAG_FUSION_QUERIES
    """

    SELF_RAG_ENABLED: bool = Field(default=True)
    TOP_K: int = Field(default=8)
    TOP_M: int = Field(default=5)
    RELEVANCE_THRESHOLD: float = Field(default=0.6)
    SUPPORT_THRESHOLD: float = Field(default=0.5)
    MAX_REWRITE: int = Field(default=1)
    MAX_CLAIMS: int = Field(default=6)
    RAG_FUSION_QUERIES: int = Field(default=3)


class Settings(BaseModel):
    APP: AppSettings = Field(default_factory=AppSettings)
    DATABASE: PgDbSettings = Field(default_factory=PgDbSettings)
    REDIS: RedisSettings = Field(default_factory=RedisSettings)
    MINIO: MinIOSettings = Field(default_factory=MinIOSettings)
    OPENAI: OpenAISettings = Field(default_factory=OpenAISettings)
    JINA: JinaSettings = Field(default_factory=JinaSettings)
    PROCESSING: ProcessingSettings = Field(default_factory=ProcessingSettings)
    NEO4J: Neo4jSettings = Field(default_factory=Neo4jSettings)
    VECTOR: VectorSettings = Field(default_factory=VectorSettings)
    UI: UiSettings = Field(default_factory=UiSettings)
    SELF_RAG: SelfRagSettings = Field(default_factory=SelfRagSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()


SETTINGS = get_settings()
