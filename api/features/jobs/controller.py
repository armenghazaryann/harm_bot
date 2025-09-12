"""Controller for the Jobs feature."""
import logging
from typing import Optional
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.features.jobs.dtos import (
    JobStatusResponse,
    JobListResponse,
    EvalRunRequest,
    EvalRunResponse,
    JobCancelRequest,
    JobRetryRequest,
    JobStatsResponse,
    ProcessingJobRequest,
    EmbeddingJobRequest
)
from api.features.jobs.exceptions import (
    JobNotFoundError,
    JobExecutionError,
    JobInvalidStateError,
    JobCancellationError
)
from api.features.jobs.service import JobService
from api.shared.dtos import PaginationRequest
from api.shared.response import ResponseModel
from workers.tasks import eval_run

logger = logging.getLogger("rag.jobs")


class JobController:
    """Controller for job operations."""
    
    def __init__(self, job_service: JobService):
        self.job_service = job_service
    
    async def get_job_status(
        self,
        job_id: UUID,
        db_session: AsyncSession
    ) -> ResponseModel[JobStatusResponse]:
        """Get the status of a specific job."""
        try:
            job = await self.job_service.get_job_by_id(job_id, db_session)
            
            if not job:
                raise JobNotFoundError(str(job_id))
            
            response_data = JobStatusResponse(
                job_id=job.id,
                job_type=job.job_type.value,
                status=job.status.value,
                progress=job.progress,
                result=job.result,
                error_message=job.error_message,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                updated_at=None,  # Will be set by TimestampMixin if needed
                duration_ms=job.get_duration_ms()
            )
            
            return ResponseModel.success(
                data=response_data,
                message="Job status retrieved successfully"
            )
            
        except JobNotFoundError as e:
            logger.warning(f"Job not found: {e.message}")
            raise HTTPException(status_code=404, detail=e.message)
        except Exception as e:
            logger.exception("Unexpected error retrieving job status")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def list_jobs(
        self,
        pagination: PaginationRequest,
        job_type: Optional[str] = None,
        status: Optional[str] = None,
        db_session: AsyncSession = None
    ) -> ResponseModel[JobListResponse]:
        """List jobs with optional filtering."""
        try:
            jobs, total = await self.job_service.list_jobs(
                skip=pagination.skip,
                limit=pagination.limit,
                job_type_filter=job_type,
                status_filter=status,
                db_session=db_session
            )
            
            job_responses = [
                JobStatusResponse(
                    job_id=job.id,
                    job_type=job.job_type.value,
                    status=job.status.value,
                    progress=job.progress,
                    result=job.result,
                    error_message=job.error_message,
                    created_at=job.created_at,
                    started_at=job.started_at,
                    completed_at=job.completed_at,
                    updated_at=None,
                    duration_ms=job.get_duration_ms()
                )
                for job in jobs
            ]
            
            response_data = JobListResponse(
                jobs=job_responses,
                total=total,
                page=pagination.skip // pagination.limit + 1,
                page_size=pagination.limit
            )
            
            return ResponseModel.success(
                data=response_data,
                message="Jobs retrieved successfully"
            )
            
        except Exception as e:
            logger.exception("Unexpected error listing jobs")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def run_evaluation(
        self,
        request: EvalRunRequest,
        db_session: AsyncSession
    ) -> ResponseModel[EvalRunResponse]:
        """Run an evaluation suite."""
        try:
            # Create evaluation run
            eval_run_model = await self.job_service.create_evaluation_run(
                run_label=request.run_label,
                dataset_name=request.dataset_name,
                config=request.config,
                db_session=db_session
            )
            
            # Queue evaluation task
            task = eval_run.delay(run_label=request.run_label)
            
            response_data = EvalRunResponse(
                run_id=eval_run_model.run_id,
                status="queued",
                message="Evaluation run queued successfully",
                estimated_duration_minutes=30  # Mock estimate
            )
            
            logger.info(f"Evaluation run started: {eval_run_model.run_id}")
            return ResponseModel.success(
                data=response_data,
                message="Evaluation started successfully"
            )
            
        except Exception as e:
            logger.exception("Unexpected error starting evaluation")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def cancel_job(
        self,
        job_id: UUID,
        cancel_request: JobCancelRequest,
        db_session: AsyncSession
    ) -> ResponseModel[str]:
        """Cancel a running job."""
        try:
            success = await self.job_service.cancel_job(
                job_id=job_id,
                reason=cancel_request.reason,
                db_session=db_session
            )
            
            if not success:
                raise JobNotFoundError(str(job_id))
            
            logger.info(f"Job cancelled successfully: {job_id}")
            return ResponseModel.success(
                data=f"Job {job_id} cancelled successfully",
                message="Job cancelled successfully"
            )
            
        except JobNotFoundError as e:
            logger.warning(f"Job not found for cancellation: {e.message}")
            raise HTTPException(status_code=404, detail=e.message)
        except JobInvalidStateError as e:
            logger.warning(f"Job in invalid state for cancellation: {e.message}")
            raise HTTPException(status_code=400, detail=e.message)
        except JobCancellationError as e:
            logger.error(f"Job cancellation failed: {e.message}")
            raise HTTPException(status_code=500, detail=e.message)
        except Exception as e:
            logger.exception("Unexpected error cancelling job")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def retry_job(
        self,
        job_id: UUID,
        retry_request: JobRetryRequest,
        db_session: AsyncSession
    ) -> ResponseModel[JobStatusResponse]:
        """Retry a failed job."""
        try:
            job = await self.job_service.retry_job(
                job_id=job_id,
                reset_progress=retry_request.reset_progress,
                db_session=db_session
            )
            
            response_data = JobStatusResponse(
                job_id=job.id,
                job_type=job.job_type.value,
                status=job.status.value,
                progress=job.progress,
                result=job.result,
                error_message=job.error_message,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                updated_at=None,
                duration_ms=job.get_duration_ms()
            )
            
            logger.info(f"Job retried successfully: {job_id}")
            return ResponseModel.success(
                data=response_data,
                message="Job retried successfully"
            )
            
        except JobNotFoundError as e:
            logger.warning(f"Job not found for retry: {e.message}")
            raise HTTPException(status_code=404, detail=e.message)
        except JobInvalidStateError as e:
            logger.warning(f"Job in invalid state for retry: {e.message}")
            raise HTTPException(status_code=400, detail=e.message)
        except Exception as e:
            logger.exception("Unexpected error retrying job")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def get_job_stats(
        self,
        db_session: AsyncSession
    ) -> ResponseModel[JobStatsResponse]:
        """Get job statistics."""
        try:
            stats = await self.job_service.get_job_stats(db_session)
            
            response_data = JobStatsResponse(
                total_jobs=stats["total_jobs"],
                running_jobs=stats["running_jobs"],
                completed_jobs=stats["completed_jobs"],
                failed_jobs=stats["failed_jobs"],
                queued_jobs=stats["queued_jobs"],
                average_duration_ms=stats["average_duration_ms"],
                success_rate=stats["success_rate"]
            )
            
            return ResponseModel.success(
                data=response_data,
                message="Job statistics retrieved successfully"
            )
            
        except Exception as e:
            logger.exception("Unexpected error retrieving job statistics")
            raise HTTPException(status_code=500, detail="Internal server error")
