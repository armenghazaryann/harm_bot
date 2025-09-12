import hashlib
import json
import time
from contextlib import contextmanager

import redis
import structlog
from billiard.exceptions import SoftTimeLimitExceeded

from core.settings import SETTINGS
from .celery_app import app

log = structlog.get_logger()


# Simple Redis-based idempotency guard using SET NX with TTL
class IdempotencyGuard:
    def __init__(self, redis_url: str, ttl_seconds: int = 3600):
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl_seconds

    def _key(self, task_name: str, args: tuple, kwargs: dict) -> str:
        payload = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"idem:{task_name}:{digest}"

    @contextmanager
    def acquire(self, task_name: str, args: tuple, kwargs: dict):
        key = self._key(task_name, args, kwargs)
        acquired = self.client.set(name=key, value="1", nx=True, ex=self.ttl)
        if not acquired:
            # Another worker already processed or is processing this payload
            raise RuntimeError("Idempotency key already exists; skipping duplicate task")
        try:
            yield
        finally:
            # Let the key expire naturally to prevent rapid duplicate enqueues; do not delete immediately
            pass


idem_guard = IdempotencyGuard(SETTINGS.REDIS.REDIS_URL)


@app.task(
    name="workers.tasks.process_document",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=2,  # exponential base
    retry_jitter=True,
    retry_backoff_max=120,
    retry_kwargs={"max_retries": 5},
    soft_time_limit=120,
    time_limit=180,
)
def process_document(self, doc_id: str, checksum: str):
    try:
        with idem_guard.acquire(self.name, (doc_id, checksum), {}):
            log.info("process_document.started", doc_id=doc_id, checksum=checksum)
            # TODO: call processing pipeline here
            time.sleep(0.1)
            log.info("process_document.finished", doc_id=doc_id)
            return {"doc_id": doc_id, "status": "done"}
    except SoftTimeLimitExceeded as e:
        log.warning("process_document.soft_time_limit_exceeded", doc_id=doc_id)
        raise e
    except RuntimeError as e:
        # Idempotency: treat as success-noop to avoid side effects
        log.info("process_document.idempotent_skip", doc_id=doc_id, reason=str(e))
        return {"doc_id": doc_id, "status": "skipped"}


@app.task(
    name="workers.tasks.chunk_document",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=2,
    retry_jitter=True,
    retry_backoff_max=60,
    retry_kwargs={"max_retries": 5},
    soft_time_limit=90,
    time_limit=120,
)
def chunk_document(self, doc_id: str):
    try:
        with idem_guard.acquire(self.name, (doc_id,), {}):
            log.info("chunk_document.started", doc_id=doc_id)
            # TODO: chunking pipeline
            time.sleep(0.1)
            log.info("chunk_document.finished", doc_id=doc_id)
            return {"doc_id": doc_id, "status": "done"}
    except SoftTimeLimitExceeded as e:
        log.warning("chunk_document.soft_time_limit_exceeded", doc_id=doc_id)
        raise e
    except RuntimeError as e:
        log.info("chunk_document.idempotent_skip", doc_id=doc_id, reason=str(e))
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
    try:
        with idem_guard.acquire(self.name, (doc_id,), {}):
            log.info("embed_chunks.started", doc_id=doc_id)
            # TODO: embeddings writer
            time.sleep(0.1)
            log.info("embed_chunks.finished", doc_id=doc_id)
            return {"doc_id": doc_id, "status": "done"}
    except SoftTimeLimitExceeded as e:
        log.warning("embed_chunks.soft_time_limit_exceeded", doc_id=doc_id)
        raise e
    except RuntimeError as e:
        log.info("embed_chunks.idempotent_skip", doc_id=doc_id, reason=str(e))
        return {"doc_id": doc_id, "status": "skipped"}


@app.task(
    name="workers.tasks.index_build",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=2,
    retry_jitter=True,
    retry_backoff_max=60,
    retry_kwargs={"max_retries": 3},
    soft_time_limit=300,
    time_limit=360,
)
def index_build(self):
    try:
        with idem_guard.acquire(self.name, tuple(), {}):
            log.info("index_build.started")
            # TODO: FTS / index builds
            time.sleep(0.1)
            log.info("index_build.finished")
            return {"status": "done"}
    except SoftTimeLimitExceeded as e:
        log.warning("index_build.soft_time_limit_exceeded")
        raise e
    except RuntimeError as e:
        log.info("index_build.idempotent_skip", reason=str(e))
        return {"status": "skipped"}


@app.task(
    name="workers.tasks.eval_run",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=2,
    retry_jitter=True,
    retry_backoff_max=60,
    retry_kwargs={"max_retries": 3},
    soft_time_limit=300,
    time_limit=360,
)
def eval_run(self, run_label: str | None = None):
    try:
        with idem_guard.acquire(self.name, (run_label or "__none__",), {}):
            log.info("eval_run.started", run_label=run_label)
            # TODO: evaluation orchestration
            time.sleep(0.1)
            log.info("eval_run.finished", run_label=run_label)
            return {"run_label": run_label, "status": "done"}
    except SoftTimeLimitExceeded as e:
        log.warning("eval_run.soft_time_limit_exceeded", run_label=run_label)
        raise e
    except RuntimeError as e:
        log.info("eval_run.idempotent_skip", run_label=run_label, reason=str(e))
        return {"run_label": run_label, "status": "skipped"}
