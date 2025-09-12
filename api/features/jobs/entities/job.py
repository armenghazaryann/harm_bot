"""Job entity following SOLID principles and clean architecture."""
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import String, Integer, Text, JSON, DateTime, Index
from sqlalchemy.orm import Mapped, mapped_column

from api.shared.entities.base import BaseEntity


class JobStatus(str, Enum):
    """Job status enumeration for tracking execution state."""
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    RETRYING = "retrying"


class JobType(str, Enum):
    """Job type enumeration for different background tasks."""
    PROCESS_DOCUMENT = "process_document"
    CHUNK_DOCUMENT = "chunk_document"
    EMBED_CHUNKS = "embed_chunks"
    INDEX_BUILD = "index_build"
    EVAL_RUN = "eval_run"


class Job(BaseEntity):
    """Job entity for tracking background job execution with idempotency."""
    
    # Job identification and type
    job_type: Mapped[JobType] = mapped_column(String(50), nullable=False)
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Idempotency key for preventing duplicate jobs
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Status and retry tracking
    status: Mapped[JobStatus] = mapped_column(String(20), default=JobStatus.QUEUED)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    
    # Job payload and results
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    result: Mapped[Optional[dict]] = mapped_column(JSON)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    # Execution timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    __table_args__ = (
        Index("idx_job_status", "status"),
        Index("idx_job_type", "job_type"),
        Index("idx_job_idempotency", "job_type", "idempotency_key", unique=True),
        Index("idx_job_celery_task", "celery_task_id"),
    )

    def is_retryable(self) -> bool:
        """Check if job can be retried."""
        return (
            self.status == JobStatus.FAILED and 
            self.attempts < self.max_attempts
        )
    
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in [JobStatus.DONE, JobStatus.FAILED]
    
    def can_be_cancelled(self) -> bool:
        """Check if job can be cancelled."""
        return self.status in [JobStatus.QUEUED, JobStatus.RUNNING]
