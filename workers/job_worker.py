"""Job-based worker system for MVP - uses jobs module instead of Celery."""
import asyncio
import logging
from typing import Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession

from api.features.jobs.entities.job import JobType, JobStatus
from api.features.jobs.models import JobCreateModel
from api.features.jobs.service import JobService
from infra.db_utils import DatabaseManager
from workers.document_processor import DocumentProcessor

logger = logging.getLogger("rag.job_worker")


class JobWorker:
    """Background worker that processes jobs from the database queue."""

    def __init__(self):
        self.job_service = JobService()
        self.document_processor = DocumentProcessor()

    async def process_jobs(self) -> None:
        """Main worker loop - processes pending jobs."""
        logger.info("Job worker started")

        while True:
            try:
                await self._process_pending_jobs()
                await asyncio.sleep(1)  # Poll every second
            except Exception as e:
                logger.error(f"Error in job worker: {e}")
                await asyncio.sleep(5)  # Back off on error

    async def _process_pending_jobs(self) -> None:
        """Process all pending jobs."""
        db = await DatabaseManager.get_resource()

        async with db.get_session() as session:
            pending_jobs = await self.job_service.get_pending_jobs(db_session=session)

            for job_model in pending_jobs:
                await self._execute_job(job_model.id, session)

    async def _execute_job(self, job_id: str, session: AsyncSession) -> None:
        """Execute a single job."""
        try:
            job = await self.job_service.get_job(job_id, session)

            if job.status != JobStatus.QUEUED:
                return

            # Mark as running
            await self.job_service.update_job_status(
                job_id, JobStatus.RUNNING, db_session=session
            )

            # Execute based on job type
            result = await self._handle_job_type(job.job_type, job.payload, session)

            # Mark as completed
            await self.job_service.update_job_status(
                job_id, JobStatus.DONE, result=result, db_session=session
            )

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            await self.job_service.update_job_status(
                job_id, JobStatus.FAILED, error_message=str(e), db_session=session
            )

    async def _handle_job_type(
        self, job_type: JobType, payload: Dict[str, Any], session: AsyncSession
    ) -> Dict[str, Any]:
        """Handle different job types."""
        if job_type == JobType.PROCESS_DOCUMENT:
            return await self.document_processor.process_document(
                payload["document_id"], session
            )
        elif job_type == JobType.CHUNK_DOCUMENT:
            return await self.document_processor.chunk_document(
                payload["document_id"], session
            )
        elif job_type == JobType.EMBED_CHUNKS:
            return await self.document_processor.embed_document(
                payload["document_id"], session
            )
        else:
            raise ValueError(f"Unknown job type: {job_type}")

    async def enqueue_document_processing(self, document_id: str) -> str:
        """Enqueue a document processing job."""
        db = await DatabaseManager.get_resource()

        async with db.get_session() as session:
            job_create = JobCreateModel(
                job_type=JobType.PROCESS_DOCUMENT,
                idempotency_key=f"process_{document_id}",
                payload={"document_id": document_id},
            )

            job = await self.job_service.create_job(job_create, db_session=session)
            return str(job.id)

    async def enqueue_chunking(self, document_id: str) -> str:
        """Enqueue a chunking job."""
        db = await DatabaseManager.get_resource()

        async with db.get_session() as session:
            job_create = JobCreateModel(
                job_type=JobType.CHUNK_DOCUMENT,
                idempotency_key=f"chunk_{document_id}",
                payload={"document_id": document_id},
            )

            job = await self.job_service.create_job(job_create, db_session=session)
            return str(job.id)

    async def enqueue_embedding(self, document_id: str) -> str:
        """Enqueue an embedding job."""
        db = await DatabaseManager.get_resource()

        async with db.get_session() as session:
            job_create = JobCreateModel(
                job_type=JobType.EMBED_CHUNKS,
                idempotency_key=f"embed_{document_id}",
                payload={"document_id": document_id},
            )

            job = await self.job_service.create_job(job_create, db_session=session)
            return str(job.id)


async def start_job_worker() -> None:
    """Start the job worker."""
    worker = JobWorker()
    await worker.process_jobs()
