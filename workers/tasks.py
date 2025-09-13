import hashlib
import asyncio
import json
from contextlib import contextmanager

import redis
import structlog
from billiard.exceptions import SoftTimeLimitExceeded
# from celery import chain  # Removed - no more task chains needed

from core.settings import SETTINGS
from .celery_app import app
from workers.indexing import index_chunks_pgvector

# DEPRECATED IMPORTS - Replaced by LangChain processor
# from workers.transcripts import (
#     create_utterances_jsonl,
#     ingest_transcript_pg_from_minio,
#     ingest_transcript_neo4j_from_minio,
#     materialize_transcript_chunks_from_pg,
# )
# from workers.embeddings import DocumentEmbedder
from workers.langchain_processor import process_document_with_langchain

log = structlog.get_logger()


# Simple Redis-based idempotency guard using SET NX with TTL
class IdempotencyGuard:
    def __init__(self, redis_url: str, ttl_seconds: int = 3600):
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl_seconds

    def _key(self, task_name: str, args: tuple, kwargs: dict) -> str:
        payload = json.dumps(
            {"args": args, "kwargs": kwargs}, sort_keys=True, default=str
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"idem:{task_name}:{digest}"

    @contextmanager
    def acquire(self, task_name: str, args: tuple, kwargs: dict):
        key = self._key(task_name, args, kwargs)
        acquired = self.client.set(name=key, value="1", nx=True, ex=self.ttl)
        if not acquired:
            # Another worker already processed or is processing this payload
            raise RuntimeError(
                "Idempotency key already exists; skipping duplicate task"
            )
        try:
            yield
        finally:
            # Let the key expire naturally to prevent rapid duplicate enqueues; do not delete immediately
            pass


idem_guard = IdempotencyGuard(SETTINGS.REDIS.REDIS_URL)


def _run_step(self, idem_args: tuple, label: str, coro_func) -> dict:
    """DRY runner for async pipeline steps with idempotency and logging."""
    try:
        with idem_guard.acquire(self.name, idem_args, {}):
            log.info(f"{label}.started", args=idem_args)

            # Use asyncio.run() but handle event loop properly to avoid connection issues
            try:
                # Call the function to get the coroutine, then run it
                coro = coro_func() if callable(coro_func) else coro_func
                result = asyncio.run(coro)
                if isinstance(result, dict):
                    log.info(f"{label}.finished", result=result)
                    return result
                log.info(f"{label}.finished")
                return {"status": "done"}
            except Exception as async_error:
                # If we get event loop issues, catch and re-raise with more context
                if "Event loop is closed" in str(
                    async_error
                ) or "attached to a different loop" in str(async_error):
                    log.warning(f"{label}.event_loop_error", error=str(async_error))
                    # Treat as skipped to avoid blocking the pipeline
                    raise RuntimeError(f"Event loop management error: {async_error}")
                else:
                    # Re-raise other async errors
                    raise async_error

    except SoftTimeLimitExceeded as e:
        log.warning(f"{label}.soft_time_limit_exceeded", args=idem_args)
        raise e
    except RuntimeError as e:
        # Idempotency and event loop errors: treat as success-noop to avoid side effects
        log.info(f"{label}.idempotent_skip", args=idem_args, reason=str(e))
        return {"status": "skipped"}


# DEPRECATED TASK - Replaced by LangChain processor
# Chunking is now handled by LangChain RecursiveCharacterTextSplitter
# @app.task(name="workers.tasks.materialize_transcript_chunks", ...)
# def materialize_transcript_chunks_task(self, doc_id: str): ...


@app.task(
    name="workers.tasks.index_pgvector",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=2,
    retry_jitter=True,
    retry_backoff_max=60,
    retry_kwargs={"max_retries": 4},
    soft_time_limit=300,
    time_limit=600,
    queue="index",
)
def index_pgvector(self, doc_id: str):
    result = _run_step(
        self,
        (doc_id,),
        "index_pgvector",
        lambda: index_chunks_pgvector(doc_id),
    )
    return {"doc_id": doc_id, **result}


# DEPRECATED TASK - Replaced by LangChain processor
# PDF processing is now handled by LangChain PyPDFLoader
# @app.task(name="workers.tasks.create_transcript_utterances_jsonl", ...)
# def create_transcript_utterances_jsonl_task(self, doc_id: str): ...


# DEPRECATED TASK - Replaced by LangChain processor
# PostgreSQL ingestion is now handled by LangChain PGVector
# @app.task(name="workers.tasks.ingest_transcript_pg", ...)
# def ingest_transcript_pg_task(self, doc_id: str): ...


# DEPRECATED TASK - Replaced by LangChain processor
# Neo4j ingestion is optional and handled separately if needed
# @app.task(name="workers.tasks.ingest_transcript_neo4j", ...)
# def ingest_transcript_neo4j_task(self, doc_id: str): ...


@app.task(
    name="workers.tasks.process_document_langchain",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=2,
    retry_jitter=True,
    retry_backoff_max=120,
    retry_kwargs={"max_retries": 3},
    soft_time_limit=900,
    time_limit=1200,
    queue="ingest",
)
def process_document_langchain_task(self, doc_id: str):
    """
    Process document using production-ready LangChain pipeline.

    Replaces complex multi-step pipeline with single LangChain processor:
    PDF → PyPDFLoader → RecursiveCharacterTextSplitter → OpenAIEmbeddings → PGVector

    FAANG-Level Architecture: Leverages battle-tested LangChain components.
    """
    result = _run_step(
        self,
        (doc_id,),
        "process_document_langchain",
        lambda: process_document_with_langchain(doc_id),
    )
    return {"doc_id": doc_id, **result}


@app.task(
    name="workers.tasks.process_transcript_pipeline",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=2,
    retry_jitter=True,
    retry_backoff_max=120,
    retry_kwargs={"max_retries": 3},
    soft_time_limit=1200,
    time_limit=1500,
    queue="ingest",
)
def process_transcript_pipeline(self, doc_id: str):
    """
    MIGRATION: Use LangChain processor instead of complex pipeline.

    Legacy pipeline: utterances → PG → Neo4j → chunks → embeddings → indexing
    New approach: PDF → LangChain processor (handles everything)
    """
    log.info("process_transcript_pipeline.migrating_to_langchain", doc_id=doc_id)

    # Use new LangChain processor instead of complex chain
    try:
        result = _run_step(
            self,
            (doc_id,),
            "process_transcript_pipeline_langchain",
            lambda: process_document_with_langchain(doc_id),
        )

        log.info(
            "process_transcript_pipeline.langchain_success",
            doc_id=doc_id,
            result=result,
        )
        return {"doc_id": doc_id, "migration": "langchain_processor", **result}

    except Exception as e:
        log.error(
            "process_transcript_pipeline.langchain_failed", doc_id=doc_id, error=str(e)
        )

        # No fallback needed - LangChain processor is production-ready
        log.error(
            "process_transcript_pipeline.langchain_failed_final",
            doc_id=doc_id,
            error=str(e),
            note="No legacy fallback - LangChain is the only supported processor",
        )
        raise e


# DEPRECATED TASK - Replaced by LangChain processor
# Embeddings are now handled by LangChain OpenAIEmbeddings + PGVector
# @app.task(name="workers.tasks.embed_chunks", ...)
# def embed_chunks(self, doc_id: str): ...


# DEPRECATED TASK - Replaced by LangChain PGVector automatic indexing
# Indexing is now handled automatically by LangChain PGVector
# @app.task(name="workers.tasks.create_pgvector_hnsw_index", ...)
# def create_pgvector_hnsw_index_task(self): ...
