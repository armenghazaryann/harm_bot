import hashlib
import logging
from typing import Optional
from uuid import UUID, uuid4

from fastapi import UploadFile, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.features.documents.dtos import (
    DocumentUploadResponse,
    DocumentStatusResponse,
    DocumentListResponse,
    DocumentDeleteRequest,
)
from api.shared.dtos import PaginationRequest
from api.features.documents.exceptions import (
    DocumentNotFoundError,
)
from api.features.documents.models import DocumentCreateModel
from api.features.documents.service import DocumentService
from api.features.documents.validators import DocumentValidator
from workers.tasks import process_document_etl_task

logger = logging.getLogger(__name__)


class StoragePathGenerator:
    """Generates storage paths for documents."""

    @staticmethod
    def generate_path(filename: str, doc_id: UUID) -> str:
        """Generate storage path for document."""
        return f"documents/{doc_id}/{filename}"


class DocumentModelFactory:
    """Factory for creating document models."""

    @staticmethod
    def create_document_model(
        filename: str, content: bytes, content_type: Optional[str], storage_path: str
    ) -> DocumentCreateModel:
        """Create DocumentCreateModel from file data."""
        return DocumentCreateModel(
            filename=filename,
            document_type=DocumentValidator.determine_document_type(
                filename, content_type
            ),
            size=len(content),
            checksum=hashlib.sha256(content).hexdigest(),
            storage_path=storage_path,
            content_type=content_type or "application/octet-stream",
            metadata={
                "content_type": content_type or "application/octet-stream",
                "original_filename": filename,
            },
        )


class DocumentController:
    """Controller for document operations - orchestrates other components."""

    def __init__(self, document_service: DocumentService):
        self.document_service = document_service

    async def upload_document(
        self, file: UploadFile, *, db_session: AsyncSession
    ) -> DocumentUploadResponse:
        """Upload a document for processing."""
        try:
            # Validate file
            content = await DocumentValidator.validate_upload(file)

            # Generate document ID and storage path
            doc_id = uuid4()
            storage_path = StoragePathGenerator.generate_path(file.filename, doc_id)

            # Create document model
            document_model = DocumentModelFactory.create_document_model(
                filename=file.filename,
                content=content,
                content_type=file.content_type,
                storage_path=storage_path,
            )

            # Create document record
            document = await self.document_service.create_document(
                document_model, db_session=db_session
            )

            # Commit the transaction so Celery workers can see the document
            await db_session.commit()

            # Upload to storage
            await self.document_service.upload_to_storage(
                storage_path=storage_path,
                content=content,
                content_type=file.content_type,
            )

            # Get download URL
            download_url = await self.document_service.get_download_url(
                storage_path=storage_path
            )

            # Trigger processing using new modular ETL pipeline
            try:
                # Use the already-determined document type from the validator (no duplicate logic!)
                doc_type = (
                    document.document_type
                )  # This was already correctly determined by DocumentValidator

                # Queue ETL task with Celery (convert enum to string for task serialization)
                task_result = process_document_etl_task.delay(
                    str(document.id),
                    storage_path,
                    doc_type.value,  # Use .value to get the string representation
                )
                processing_job_id = task_result.id
            except Exception as e:
                logger.warning(f"Failed to queue ETL task: {e}")
                processing_job_id = None

            return DocumentUploadResponse(
                doc_id=document.id,
                filename=document.filename,
                size=document.size,
                checksum=document.checksum,
                status=(
                    document.status.value
                    if hasattr(document.status, "value")
                    else str(document.status)
                ),
                storage_path=document.storage_path,
                download_url=download_url,
                processing_job_id=processing_job_id,
            )

        except Exception as e:
            logger.error(f"Failed to upload document: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_document_status(
        self, doc_id: UUID, *, db_session: AsyncSession
    ) -> DocumentStatusResponse:
        """Get document processing status."""
        document = await self.document_service.get_document(
            document_id=doc_id, db_session=db_session
        )
        if not document:
            raise DocumentNotFoundError(f"Document {doc_id} not found")
        return DocumentStatusResponse.from_entity(document)

    async def list_documents(
        self, pagination: PaginationRequest, *, db_session: AsyncSession
    ) -> DocumentListResponse:
        """List documents with pagination."""
        documents, total = await self.document_service.list_documents(
            offset=pagination.skip, limit=pagination.limit, db_session=db_session
        )
        return DocumentListResponse(
            documents=[DocumentStatusResponse.from_entity(doc) for doc in documents],
            total=total,
            page=pagination.skip // pagination.limit + 1,
            page_size=pagination.limit,
        )

    async def delete_document(
        self, request: DocumentDeleteRequest, *, db_session: AsyncSession
    ) -> str:
        """Delete a document."""
        document = await self.document_service.get_document(
            document_id=request.doc_id, db_session=db_session
        )
        if not document:
            raise DocumentNotFoundError(f"Document {request.doc_id} not found")

        # TODO: Implement actual document deletion logic in service
        # This should handle deletion of document, chunks, embeddings based on request flags
        # await self.document_service.delete_document(
        #     document_id=request.doc_id,
        #     delete_chunks=request.delete_chunks,
        #     delete_embeddings=request.delete_embeddings,
        #     db_session=db_session
        # )

        return f"Document {request.doc_id} deleted successfully"
