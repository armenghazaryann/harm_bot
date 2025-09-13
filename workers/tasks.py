import hashlib
import asyncio
import json
from contextlib import contextmanager

import redis
import structlog
from billiard.exceptions import SoftTimeLimitExceeded
from celery import chain

from core.settings import SETTINGS
from .celery_app import app
from workers.indexing import index_chunks_pgvector
from workers.transcripts import (
    create_utterances_jsonl,
    ingest_transcript_pg_from_minio,
    ingest_transcript_neo4j_from_minio,
    materialize_transcript_chunks_from_pg,
)
from workers.embeddings import DocumentEmbedder

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


def _run_step(self, idem_args: tuple, label: str, coro) -> dict:
    """DRY runner for async pipeline steps with idempotency and logging."""
    try:
        with idem_guard.acquire(self.name, idem_args, {}):
            log.info(f"{label}.started", args=idem_args)
            result = asyncio.run(coro)
            if isinstance(result, dict):
                log.info(f"{label}.finished", result=result)
                return result
            log.info(f"{label}.finished")
            return {"status": "done"}
    except SoftTimeLimitExceeded as e:
        log.warning(f"{label}.soft_time_limit_exceeded", args=idem_args)
        raise e
    except RuntimeError as e:
        # Idempotency: treat as success-noop to avoid side effects
        log.info(f"{label}.idempotent_skip", args=idem_args, reason=str(e))
        return {"status": "skipped"}


@app.task(
    name="workers.tasks.materialize_transcript_chunks",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=2,
    retry_jitter=True,
    retry_backoff_max=120,
    retry_kwargs={"max_retries": 5},
    soft_time_limit=600,
    time_limit=900,
    queue="chunk",
)
def materialize_transcript_chunks_task(self, doc_id: str):
    result = _run_step(
        self,
        (doc_id,),
        "materialize_transcript_chunks",
        materialize_transcript_chunks_from_pg(doc_id),
    )
    return {"doc_id": doc_id, **result}


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
        index_chunks_pgvector(doc_id),
    )
    return {"doc_id": doc_id, **result}


@app.task(
    name="workers.tasks.create_transcript_utterances_jsonl",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=2,
    retry_jitter=True,
    retry_backoff_max=120,
    retry_kwargs={"max_retries": 5},
    soft_time_limit=600,
    time_limit=900,
    queue="chunk",
)
def create_transcript_utterances_jsonl_task(self, doc_id: str):
    result = _run_step(
        self,
        (doc_id,),
        "create_transcript_utterances_jsonl",
        create_utterances_jsonl(doc_id),
    )
    return {"doc_id": doc_id, **result}


@app.task(
    name="workers.tasks.ingest_transcript_pg",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=2,
    retry_jitter=True,
    retry_backoff_max=120,
    retry_kwargs={"max_retries": 5},
    soft_time_limit=300,
    time_limit=600,
    queue="ingest",
)
def ingest_transcript_pg_task(self, doc_id: str):
    result = _run_step(
        self,
        (doc_id,),
        "ingest_transcript_pg",
        ingest_transcript_pg_from_minio(doc_id),
    )
    return {"doc_id": doc_id, **result}


@app.task(
    name="workers.tasks.ingest_transcript_neo4j",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=2,
    retry_jitter=True,
    retry_backoff_max=120,
    retry_kwargs={"max_retries": 5},
    soft_time_limit=300,
    time_limit=600,
    queue="ingest",
)
def ingest_transcript_neo4j_task(self, doc_id: str):
    result = _run_step(
        self,
        (doc_id,),
        "ingest_transcript_neo4j",
        ingest_transcript_neo4j_from_minio(doc_id),
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
    """Schedule a distributed Celery chain for the transcript pipeline and return immediately."""
    try:
        log.info("process_transcript_pipeline.schedule", doc_id=doc_id)
        ch = chain(
            create_transcript_utterances_jsonl_task.si(doc_id),
            ingest_transcript_pg_task.si(doc_id),
            ingest_transcript_neo4j_task.si(doc_id),
            materialize_transcript_chunks_task.si(doc_id),
            index_pgvector.si(doc_id),
            create_pgvector_hnsw_index_task.si(),
        )
        async_result = ch.apply_async()
        log.info(
            "process_transcript_pipeline.scheduled",
            doc_id=doc_id,
            chain_id=async_result.id,
        )
        return {"doc_id": doc_id, "status": "scheduled", "chain_id": async_result.id}
    except SoftTimeLimitExceeded as e:
        log.warning(
            "process_transcript_pipeline.soft_time_limit_exceeded", doc_id=doc_id
        )
        raise e
    except RuntimeError as e:
        log.info(
            "process_transcript_pipeline.idempotent_skip", doc_id=doc_id, reason=str(e)
        )
        return {"doc_id": doc_id, "status": "skipped"}


@app.task(
    name="workers.tasks.embed_chunks",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=2,
    retry_jitter=True,
    retry_backoff_max=60,
    retry_kwargs={"max_retries": 4},
    soft_time_limit=120,
    time_limit=180,
    queue="embed",
)
def embed_chunks(self, doc_id: str):
    result = _run_step(
        self,
        (doc_id,),
        "embed_chunks",
        DocumentEmbedder().embed_document_chunks(doc_id),
    )
    return {"doc_id": doc_id, **result}


@app.task(
    name="workers.tasks.create_pgvector_hnsw_index",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=2,
    retry_jitter=True,
    retry_backoff_max=120,
    retry_kwargs={"max_retries": 3},
    soft_time_limit=300,
    time_limit=600,
    queue="index",
)
def create_pgvector_hnsw_index_task(self):
    """Create HNSW index on embeddings table - DISABLED for LangChain PGVector."""
    log.info(
        "create_pgvector_hnsw_index.skipped",
        reason="Using LangChain PGVector instead of custom embeddings table",
    )
    return {"status": "skipped", "reason": "Using LangChain PGVector"}
