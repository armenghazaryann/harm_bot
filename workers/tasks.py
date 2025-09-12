import hashlib
import asyncio
import json
from contextlib import contextmanager

import redis
import structlog
from billiard.exceptions import SoftTimeLimitExceeded

from core.settings import SETTINGS
from .celery_app import app
from api.di.container import DependencyContainer

log = structlog.get_logger()
container = DependencyContainer()


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
                log.info(f"{label}.finished", **result)
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
)
def materialize_transcript_chunks_task(self, doc_id: str):
    result = _run_step(
        self,
        (doc_id,),
        "materialize_transcript_chunks",
        container.transcript_materialize_chunks(doc_id),
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
)
def create_transcript_utterances_jsonl_task(self, doc_id: str):
    result = _run_step(
        self,
        (doc_id,),
        "create_transcript_utterances_jsonl",
        container.transcript_create_jsonl(doc_id),
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
)
def ingest_transcript_pg_task(self, doc_id: str):
    result = _run_step(
        self, (doc_id,), "ingest_transcript_pg", container.transcript_ingest_pg(doc_id)
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
)
def ingest_transcript_neo4j_task(self, doc_id: str):
    result = _run_step(
        self,
        (doc_id,),
        "ingest_transcript_neo4j",
        container.transcript_ingest_neo4j(doc_id),
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
)
def process_transcript_pipeline(self, doc_id: str):
    """Orchestrate JSONL creation -> Postgres ingest -> Neo4j ingest.

    Uses synchronous calls to async functions for simplicity in MVP.
    """
    # Orchestrate the three async steps sequentially; no idempotency guard here to allow Celery autoretry
    try:
        log.info("process_transcript_pipeline.started", doc_id=doc_id)
        r1 = asyncio.run(container.transcript_create_jsonl(doc_id))
        r2 = asyncio.run(container.transcript_ingest_pg(doc_id))
        r3 = asyncio.run(container.transcript_ingest_neo4j(doc_id))
        result = {**r1, **r2, **r3}
        log.info("process_transcript_pipeline.finished", doc_id=doc_id, **result)
        return result
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
)
def embed_chunks(self, doc_id: str):
    result = _run_step(
        self, (doc_id,), "embed_chunks", container.embed_document_chunks(doc_id)
    )
    return {"doc_id": doc_id, **result}
