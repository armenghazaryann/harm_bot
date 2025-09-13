from celery import Celery
from kombu import Queue

from core.settings import SETTINGS

app = Celery(
    "rag",
    broker=SETTINGS.REDIS.CELERY_BROKER_URL,
    backend=SETTINGS.REDIS.CELERY_RESULT_BACKEND,
    include=["workers.tasks"],
)

app.conf.update(
    task_default_queue="default",
    task_queues=(
        Queue("default"),
        Queue("ingest"),
        Queue("chunk"),
        Queue("embed"),
        Queue("index"),
    ),
    task_routes={
        # Transcript pipeline (distributed MVP)
        "workers.tasks.create_transcript_utterances_jsonl": {"queue": "chunk"},
        "workers.tasks.ingest_transcript_pg": {"queue": "ingest"},
        "workers.tasks.ingest_transcript_neo4j": {"queue": "ingest"},
        "workers.tasks.materialize_transcript_chunks": {"queue": "chunk"},
        "workers.tasks.process_transcript_pipeline": {"queue": "ingest"},
        # Embedding
        "workers.tasks.embed_chunks": {"queue": "embed"},
        # Vector indexing
        "workers.tasks.index_pgvector": {"queue": "index"},
        "workers.tasks.create_pgvector_hnsw_index": {"queue": "index"},
    },
    # Reliability defaults
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Global time limits (can be overridden per task)
    task_soft_time_limit=60,  # seconds
    task_time_limit=90,  # seconds
    worker_hijack_root_logger=False,
)
