"""Service layer for the Jobs feature."""
import logging
from typing import List, Optional, Tuple
from uuid import UUID, uuid4

from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from api.features.jobs.exceptions import (
    JobNotFoundError,
    JobExecutionError,
    JobInvalidStateError,
    JobCancellationError
)
from api.features.jobs.models import JobModel, JobCreateModel, JobUpdateModel, EvaluationRunModel
from api.features.jobs.entities.job import Job as JobEntity, JobStatus, JobType

logger = logging.getLogger("rag.jobs.service")


class JobService:
    """Service for job operations."""
    
    def __init__(self, celery_app=None):
        self.celery_app = celery_app
    
    async def create_job(
        self, 
        create_model: JobCreateModel, 
        db_session: AsyncSession
    ) -> JobModel:
        """Create a new job."""
        try:
            # Check for existing job with same idempotency key
            existing = await self.get_job_by_idempotency_key(
                create_model.idempotency_key, 
                db_session
            )
            if existing:
                logger.info(f"Job with idempotency key {create_model.idempotency_key} already exists")
                return existing
            
            # Create entity
            entity = create_model.to_entity()
            db_session.add(entity)
            await db_session.commit()
            await db_session.refresh(entity)
            
            logger.info(f"Job created: {entity.id}")
            return JobModel.from_entity(entity)
            
        except Exception as e:
            await db_session.rollback()
            logger.error(f"Failed to create job: {str(e)}")
            raise
    
    async def get_job_by_id(
        self, 
        job_id: UUID, 
        db_session: AsyncSession
    ) -> Optional[JobModel]:
        """Get job by ID."""
        try:
            stmt = select(JobEntity).where(JobEntity.id == job_id)
            result = await db_session.execute(stmt)
            entity = result.scalar_one_or_none()
            
            if entity:
                return JobModel.from_entity(entity)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {str(e)}")
            raise
    
    async def get_job_by_idempotency_key(
        self, 
        idempotency_key: str, 
        db_session: AsyncSession
    ) -> Optional[JobModel]:
        """Get job by idempotency key."""
        try:
            stmt = select(JobEntity).where(JobEntity.idempotency_key == idempotency_key)
            result = await db_session.execute(stmt)
            entity = result.scalar_one_or_none()
            
            if entity:
                return JobModel.from_entity(entity)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get job by idempotency key {idempotency_key}: {str(e)}")
            raise
    
    async def update_job(
        self, 
        job_id: UUID, 
        update_model: JobUpdateModel, 
        db_session: AsyncSession
    ) -> JobModel:
        """Update job."""
        try:
            stmt = select(JobEntity).where(JobEntity.id == job_id)
            result = await db_session.execute(stmt)
            entity = result.scalar_one_or_none()
            
            if not entity:
                raise JobNotFoundError(str(job_id))
            
            # Apply updates
            entity = update_model.apply_to_entity(entity)
            await db_session.commit()
            await db_session.refresh(entity)
            
            logger.info(f"Job updated: {job_id}")
            return JobModel.from_entity(entity)
            
        except Exception as e:
            await db_session.rollback()
            logger.error(f"Failed to update job {job_id}: {str(e)}")
            raise
    
    async def list_jobs(
        self,
        skip: int = 0,
        limit: int = 100,
        job_type_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
        db_session: AsyncSession = None
    ) -> Tuple[List[JobModel], int]:
        """List jobs with pagination and filtering."""
        try:
            # Build query
            stmt = select(JobEntity)
            count_stmt = select(func.count(JobEntity.id))
            
            # Apply filters
            if job_type_filter:
                try:
                    job_type_enum = JobType(job_type_filter)
                    stmt = stmt.where(JobEntity.job_type == job_type_enum)
                    count_stmt = count_stmt.where(JobEntity.job_type == job_type_enum)
                except ValueError:
                    logger.warning(f"Invalid job type filter: {job_type_filter}")
            
            if status_filter:
                try:
                    status_enum = JobStatus(status_filter)
                    stmt = stmt.where(JobEntity.status == status_enum)
                    count_stmt = count_stmt.where(JobEntity.status == status_enum)
                except ValueError:
                    logger.warning(f"Invalid status filter: {status_filter}")
            
            # Apply pagination
            stmt = stmt.offset(skip).limit(limit).order_by(JobEntity.created_at.desc())
            
            # Execute queries
            result = await db_session.execute(stmt)
            count_result = await db_session.execute(count_stmt)
            
            entities = result.scalars().all()
            total = count_result.scalar()
            
            jobs = [JobModel.from_entity(entity) for entity in entities]
            
            return jobs, total
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {str(e)}")
            raise
    
    async def cancel_job(
        self,
        job_id: UUID,
        reason: Optional[str] = None,
        db_session: AsyncSession = None
    ) -> bool:
        """Cancel a job."""
        try:
            job = await self.get_job_by_id(job_id, db_session)
            if not job:
                raise JobNotFoundError(str(job_id))
            
            # Check if job can be cancelled
            if not job.is_running():
                raise JobInvalidStateError(
                    str(job_id), 
                    job.status.value, 
                    "queued or running"
                )
            
            # Cancel in Celery if available
            if self.celery_app:
                try:
                    self.celery_app.control.revoke(str(job_id), terminate=True)
                except Exception as e:
                    logger.warning(f"Failed to cancel Celery task {job_id}: {str(e)}")
            
            # Update job status
            update_model = JobUpdateModel(
                status=JobStatus.CANCELLED,
                error_message=f"Job cancelled: {reason}" if reason else "Job cancelled"
            )
            
            await self.update_job(job_id, update_model, db_session)
            
            logger.info(f"Job cancelled: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {str(e)}")
            raise JobCancellationError(str(job_id), str(e))
    
    async def retry_job(
        self,
        job_id: UUID,
        reset_progress: bool = True,
        db_session: AsyncSession = None
    ) -> JobModel:
        """Retry a failed job."""
        try:
            job = await self.get_job_by_id(job_id, db_session)
            if not job:
                raise JobNotFoundError(str(job_id))
            
            # Check if job can be retried
            if job.status != JobStatus.FAILED:
                raise JobInvalidStateError(
                    str(job_id), 
                    job.status.value, 
                    "failed"
                )
            
            # Reset job state
            update_model = JobUpdateModel(
                status=JobStatus.QUEUED,
                error_message=None,
                started_at=None,
                completed_at=None
            )
            
            if reset_progress:
                update_model.progress = None
            
            updated_job = await self.update_job(job_id, update_model, db_session)
            
            # Re-queue job if Celery is available
            if self.celery_app:
                try:
                    # TODO: Re-queue the job based on job type
                    pass
                except Exception as e:
                    logger.warning(f"Failed to re-queue job {job_id}: {str(e)}")
            
            logger.info(f"Job retried: {job_id}")
            return updated_job
            
        except Exception as e:
            logger.error(f"Failed to retry job {job_id}: {str(e)}")
            raise
    
    async def get_job_stats(self, db_session: AsyncSession) -> dict:
        """Get job statistics."""
        try:
            # Count jobs by status
            total_stmt = select(func.count(JobEntity.id))
            running_stmt = select(func.count(JobEntity.id)).where(
                JobEntity.status.in_([JobStatus.QUEUED, JobStatus.RUNNING])
            )
            completed_stmt = select(func.count(JobEntity.id)).where(
                JobEntity.status == JobStatus.COMPLETED
            )
            failed_stmt = select(func.count(JobEntity.id)).where(
                JobEntity.status == JobStatus.FAILED
            )
            queued_stmt = select(func.count(JobEntity.id)).where(
                JobEntity.status == JobStatus.QUEUED
            )
            
            # Calculate average duration for completed jobs
            avg_duration_stmt = select(
                func.avg(
                    func.extract('epoch', JobEntity.completed_at - JobEntity.started_at) * 1000
                )
            ).where(
                JobEntity.status == JobStatus.COMPLETED,
                JobEntity.started_at.isnot(None),
                JobEntity.completed_at.isnot(None)
            )
            
            # Execute queries
            total_result = await db_session.execute(total_stmt)
            running_result = await db_session.execute(running_stmt)
            completed_result = await db_session.execute(completed_stmt)
            failed_result = await db_session.execute(failed_stmt)
            queued_result = await db_session.execute(queued_stmt)
            avg_duration_result = await db_session.execute(avg_duration_stmt)
            
            total_jobs = total_result.scalar()
            completed_jobs = completed_result.scalar()
            failed_jobs = failed_result.scalar()
            
            # Calculate success rate
            success_rate = 0.0
            if total_jobs > 0:
                success_rate = (completed_jobs / total_jobs) * 100.0
            
            return {
                "total_jobs": total_jobs,
                "running_jobs": running_result.scalar(),
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "queued_jobs": queued_result.scalar(),
                "average_duration_ms": avg_duration_result.scalar() or 0.0,
                "success_rate": success_rate
            }
            
        except Exception as e:
            logger.error(f"Failed to get job stats: {str(e)}")
            raise
    
    async def create_evaluation_run(
        self,
        run_label: Optional[str] = None,
        dataset_name: Optional[str] = None,
        config: Optional[dict] = None,
        db_session: AsyncSession = None
    ) -> EvaluationRunModel:
        """Create an evaluation run job."""
        try:
            run_id = uuid4()
            
            eval_run = EvaluationRunModel(
                run_id=run_id,
                run_label=run_label,
                dataset_name=dataset_name,
                config=config or {},
                status="queued"
            )
            
            # Create job for the evaluation run
            job_create = JobCreateModel(
                job_type=JobType.EVALUATION,
                idempotency_key=f"eval_run_{run_id}",
                payload=eval_run.to_job_payload()
            )
            
            await self.create_job(job_create, db_session)
            
            logger.info(f"Evaluation run created: {run_id}")
            return eval_run
            
        except Exception as e:
            logger.error(f"Failed to create evaluation run: {str(e)}")
            raise
