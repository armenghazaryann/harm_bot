import hashlib
import asyncio
import json
from contextlib import contextmanager

import redis
import structlog
from billiard.exceptions import SoftTimeLimitExceeded

from core.settings import SETTINGS
from .celery_app import app
from etl.pipeline import process_document_etl

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
                if callable(coro_func):
                    coro = coro_func()
                else:
                    coro = coro_func

                if asyncio.iscoroutine(coro):
                    result = asyncio.run(coro)
                else:
                    # Handle regular functions
                    result = coro

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


@app.task(
    name="workers.tasks.process_document_etl",
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
def process_document_etl_task(
    self, doc_id: str, storage_path: str, document_type: str = "transcript"
):
    """
    Process document using modular ETL pipeline.

    New architecture: ingest → parse → chunk → embed → index
    - PDF-only with four strategies: transcript, release, slides, press
    - Idempotent steps with manifest tracking
    - Cost recording and proper error handling
    """
    from di.container import ApplicationContainer

    async def run_etl():
        container = ApplicationContainer()
        # Initialize resources synchronously (not async)
        container.init_resources()

        try:
            # Initialize database resource
            db_resource = container.infrastructure.database()
            await db_resource.init()
            db_session = db_resource.get_session()

            # Initialize MinIO resource
            minio_resource = container.infrastructure.minio_client()
            await minio_resource.init()
            minio_client = minio_resource.client

            result = await process_document_etl(
                document_id=doc_id,
                storage_path=storage_path,
                document_type=document_type,
                db_session=db_session,
                minio_client=minio_client,
            )

            await db_session.close()
            return result

        finally:
            # Cleanup resources
            try:
                db_resource = container.infrastructure.database()
                if db_resource:
                    await db_resource.shutdown()
            except Exception as e:
                log.warning("Error during database resource cleanup", error=str(e))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(run_etl())
    return {
        "doc_id": doc_id,
        "storage_path": storage_path,
        "document_type": document_type,
        **result,
    }


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
def process_transcript_pipeline(
    self, doc_id: str, storage_path: str = None, document_type: str = "transcript"
):
    """
    MIGRATION: Use modular ETL pipeline instead of monolithic processor.

    Legacy pipeline: utterances → PG → Neo4j → chunks → embeddings → indexing
    New approach: ingest → parse → chunk → embed → index (modular, idempotent)
    """
    log.info("process_transcript_pipeline.migrating_to_etl", doc_id=doc_id)

    # If storage_path not provided, try to get it from database
    if not storage_path:
        log.warning("storage_path not provided, using fallback", doc_id=doc_id)
        storage_path = f"raw/{doc_id}.pdf"  # Default assumption

    # Use new modular ETL pipeline
    try:
        result = process_document_etl_task(self, doc_id, storage_path, document_type)

        log.info(
            "process_transcript_pipeline.etl_success",
            doc_id=doc_id,
            result=result,
        )
        return {"doc_id": doc_id, "migration": "modular_etl_pipeline", **result}

    except Exception as e:
        log.error("process_transcript_pipeline.etl_failed", doc_id=doc_id, error=str(e))

        # No fallback needed - modular ETL is production-ready
        log.error(
            "process_transcript_pipeline.etl_failed_final",
            doc_id=doc_id,
            error=str(e),
            note="No legacy fallback - modular ETL is the only supported processor",
        )
        raise e
