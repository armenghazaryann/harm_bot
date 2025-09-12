"""DTOs for the Jobs feature."""
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field, validator

from api.shared.dtos import BaseDTO, TimestampMixin


class JobStatusResponse(BaseDTO, TimestampMixin):
    """Response DTO for job status."""
    job_id: UUID = Field(description="Job identifier")
    job_type: str = Field(description="Type of job")
    status: str = Field(description="Current job status")
    progress: Optional[Dict[str, Any]] = Field(default=None, description="Job progress information")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Job result if completed")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    started_at: Optional[datetime] = Field(default=None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Job completion timestamp")
    duration_ms: Optional[float] = Field(default=None, description="Job duration in milliseconds")


class JobListResponse(BaseDTO):
    """Response DTO for job listing."""
    jobs: List[JobStatusResponse] = Field(description="List of jobs")
    total: int = Field(description="Total number of jobs")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Number of items per page")


class EvalRunRequest(BaseDTO):
    """Request DTO for evaluation run."""
    run_label: Optional[str] = Field(default=None, description="Label for the evaluation run")
    dataset_name: Optional[str] = Field(default=None, description="Dataset to evaluate against")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Evaluation configuration")
    models_to_test: Optional[List[str]] = Field(default=None, description="Models to include in evaluation")
    
    @validator('run_label')
    def validate_run_label(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError("Run label cannot be empty if provided")
        return v.strip() if v else None


class EvalRunResponse(BaseDTO):
    """Response DTO for evaluation run."""
    run_id: UUID = Field(description="Evaluation run identifier")
    status: str = Field(description="Run status")
    message: str = Field(description="Status message")
    estimated_duration_minutes: Optional[int] = Field(default=None, description="Estimated duration")


class JobCancelRequest(BaseDTO):
    """Request DTO for job cancellation."""
    job_id: UUID = Field(description="Job identifier to cancel")
    reason: Optional[str] = Field(default=None, description="Reason for cancellation")


class JobRetryRequest(BaseDTO):
    """Request DTO for job retry."""
    job_id: UUID = Field(description="Job identifier to retry")
    reset_progress: bool = Field(default=True, description="Whether to reset job progress")


class JobStatsResponse(BaseDTO):
    """Response DTO for job statistics."""
    total_jobs: int = Field(description="Total number of jobs")
    running_jobs: int = Field(description="Number of running jobs")
    completed_jobs: int = Field(description="Number of completed jobs")
    failed_jobs: int = Field(description="Number of failed jobs")
    queued_jobs: int = Field(description="Number of queued jobs")
    average_duration_ms: float = Field(description="Average job duration in milliseconds")
    success_rate: float = Field(description="Job success rate as percentage")


class ProcessingJobRequest(BaseDTO):
    """Request DTO for document processing job."""
    document_id: UUID = Field(description="Document to process")
    processing_options: Optional[Dict[str, Any]] = Field(default=None, description="Processing options")
    priority: str = Field(default="normal", description="Job priority")
    
    @validator('priority')
    def validate_priority(cls, v):
        allowed_priorities = ['low', 'normal', 'high', 'urgent']
        if v not in allowed_priorities:
            raise ValueError(f"Priority must be one of: {allowed_priorities}")
        return v


class EmbeddingJobRequest(BaseDTO):
    """Request DTO for embedding generation job."""
    chunk_ids: List[UUID] = Field(description="Chunk IDs to generate embeddings for")
    model_name: str = Field(default="text-embedding-ada-002", description="Embedding model to use")
    batch_size: int = Field(default=100, ge=1, le=1000, description="Batch size for processing")


class JobProgressUpdate(BaseDTO):
    """DTO for job progress updates."""
    job_id: UUID = Field(description="Job identifier")
    progress: Dict[str, Any] = Field(description="Progress information")
    status: Optional[str] = Field(default=None, description="Updated status")
    message: Optional[str] = Field(default=None, description="Progress message")


class JobResultUpdate(BaseDTO):
    """DTO for job result updates."""
    job_id: UUID = Field(description="Job identifier")
    status: str = Field(description="Final job status")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Job result")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    completed_at: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")
