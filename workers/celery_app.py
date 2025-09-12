from celery import Celery
from kombu import Queue

from core.settings import SETTINGS

app = Celery(
    "rag",
    broker=SETTINGS.REDIS.CELERY_BROKER_URL,
    backend=SETTINGS.REDIS.CELERY_RESULT_BACKEND,
    include=["workers.tasks"]
)

app.conf.update(
    task_default_queue="default",
    task_queues=(
        Queue("default"),
        Queue("ingest"),
        Queue("chunk"),
        Queue("embed"),
        Queue("index"),
        Queue("eval"),
    ),
    task_routes={
        "workers.tasks.process_document": {"queue": "ingest"},
        "workers.tasks.chunk_document": {"queue": "chunk"},
        "workers.tasks.embed_chunks": {"queue": "embed"},
        "workers.tasks.index_build": {"queue": "index"},
        "workers.tasks.eval_run": {"queue": "eval"},
    },
    # Reliability defaults
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Global time limits (can be overridden per task)
    task_soft_time_limit=60,  # seconds
    task_time_limit=90,       # seconds
    worker_hijack_root_logger=False,
)
