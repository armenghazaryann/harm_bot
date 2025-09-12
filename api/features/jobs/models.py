"""Models for the Jobs feature."""
from datetime import datetime
from typing import Dict, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field

from api.features.jobs.entities.job import Job as JobEntity, JobStatus, JobType


class JobModel(BaseModel):
    """Domain model for Job."""
    
    id: UUID = Field(description="Job identifier")
    job_type: JobType = Field(description="Type of job")
    status: JobStatus = Field(description="Current job status")
    idempotency_key: str = Field(description="Idempotency key")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Job payload")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Job result")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    progress: Optional[Dict[str, Any]] = Field(default=None, description="Job progress")
    created_at: datetime = Field(description="Creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    
    class Config:
        from_attributes = True
    
    @classmethod
    def from_entity(cls, entity: JobEntity) -> "JobModel":
        """Create model from database entity."""
        return cls(
            id=entity.id,
            job_type=entity.job_type,
            status=entity.status,
            idempotency_key=entity.idempotency_key,
            payload=entity.payload or {},
            result=entity.result,
            error_message=entity.error_message,
            progress=entity.progress,
            created_at=entity.created_at,
            started_at=entity.started_at,
            completed_at=entity.completed_at
        )
    
    def to_entity(self) -> JobEntity:
        """Convert model to database entity."""
        return JobEntity(
            id=self.id,
            job_type=self.job_type,
            status=self.status,
            idempotency_key=self.idempotency_key,
            payload=self.payload,
            result=self.result,
            error_message=self.error_message,
            progress=self.progress,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at
        )
    
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status in [JobStatus.QUEUED, JobStatus.RUNNING]
    
    def is_completed(self) -> bool:
        """Check if job is completed (success or failure)."""
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
    
    def get_duration_ms(self) -> Optional[float]:
        """Get job duration in milliseconds."""
        if self.started_at and self.completed_at:
            duration = self.completed_at - self.started_at
            return duration.total_seconds() * 1000
        return None
    
    def get_progress_percentage(self) -> float:
        """Get progress as percentage."""
        if not self.progress:
            return 0.0
        
        if self.status == JobStatus.COMPLETED:
            return 100.0
        
        # Try to extract progress from common progress formats
        if isinstance(self.progress, dict):
            if 'percentage' in self.progress:
                return float(self.progress['percentage'])
            elif 'completed' in self.progress and 'total' in self.progress:
                total = self.progress['total']
                if total > 0:
                    return (self.progress['completed'] / total) * 100.0
        
        return 0.0


class JobCreateModel(BaseModel):
    """Model for creating a new job."""
    
    job_type: JobType = Field(description="Type of job")
    idempotency_key: str = Field(description="Idempotency key")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Job payload")
    
    def to_entity(self) -> JobEntity:
        """Convert to database entity."""
        return JobEntity(
            job_type=self.job_type,
            status=JobStatus.QUEUED,
            idempotency_key=self.idempotency_key,
            payload=self.payload
        )


class JobUpdateModel(BaseModel):
    """Model for updating a job."""
    
    status: Optional[JobStatus] = Field(default=None, description="Job status")
    progress: Optional[Dict[str, Any]] = Field(default=None, description="Job progress")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Job result")
    error_message: Optional[str] = Field(default=None, description="Error message")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    
    def apply_to_entity(self, entity: JobEntity) -> JobEntity:
        """Apply updates to an existing entity."""
        if self.status is not None:
            entity.status = self.status
        if self.progress is not None:
            entity.progress = self.progress
        if self.result is not None:
            entity.result = self.result
        if self.error_message is not None:
            entity.error_message = self.error_message
        if self.started_at is not None:
            entity.started_at = self.started_at
        if self.completed_at is not None:
            entity.completed_at = self.completed_at
        
        entity.updated_at = datetime.utcnow()
        return entity


class EvaluationRunModel(BaseModel):
    """Model for evaluation runs."""
    
    run_id: UUID = Field(description="Evaluation run identifier")
    run_label: Optional[str] = Field(default=None, description="Run label")
    dataset_name: Optional[str] = Field(default=None, description="Dataset name")
    config: Dict[str, Any] = Field(default_factory=dict, description="Evaluation configuration")
    status: str = Field(description="Run status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_job_payload(self) -> Dict[str, Any]:
        """Convert to job payload format."""
        return {
            "run_id": str(self.run_id),
            "run_label": self.run_label,
            "dataset_name": self.dataset_name,
            "config": self.config
        }
