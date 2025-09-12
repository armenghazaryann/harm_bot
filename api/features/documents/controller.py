"""Controller for the Documents feature."""
import hashlib
import logging
from typing import Optional
from uuid import UUID, uuid4

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from api.features.documents.dtos import (
    DocumentUploadResponse,
    DocumentStatusResponse,
    DocumentListResponse,
    DocumentDeleteRequest,
)
from api.features.documents.exceptions import (
    DocumentNotFoundError,
    DocumentValidationError,
)
from api.features.documents.models import DocumentCreateModel
from api.features.documents.service import DocumentService
from api.shared.dtos import PaginationRequest
from api.features.documents.entities.document import DocumentType
from workers.celery_app import app as celery_app

logger = logging.getLogger("rag.documents")


class DocumentController:
    """Controller for document operations."""

    def __init__(self, document_service: DocumentService):
        self.document_service = document_service

    async def upload_document(
        self, file: UploadFile, db_session: AsyncSession
    ) -> DocumentUploadResponse:
        """Upload a document for processing."""
        # Validate file
        if not file.filename:
            raise DocumentValidationError("No filename provided")

        # Read file content
        file_content = await file.read()
        file_size = len(file_content)

        if file_size == 0:
            raise DocumentValidationError("Empty file provided")

        # Calculate checksum
        checksum = hashlib.sha256(file_content).hexdigest()

        # Determine document type
        document_type = self._determine_document_type(file.filename, file.content_type)

        # Create document model
        doc_id = uuid4()
        storage_path = f"documents/{doc_id}/{file.filename}"

        create_model = DocumentCreateModel(
            filename=file.filename,
            document_type=document_type,
            size=file_size,
            checksum=checksum,
            storage_path=storage_path,
            content_type=file.content_type or "application/octet-stream",
            metadata={
                "content_type": file.content_type or "application/octet-stream",
                "original_filename": file.filename,
            },
        )

        # Save document to database
        document = await self.document_service.create_document(create_model, db_session)

        # Upload to storage using resolved storage path and content type
        await self.document_service.upload_to_storage(
            storage_path, file_content, file.content_type
        )
        download_url = await self.document_service.get_download_url(storage_path)

        # Queue processing task (distributed MVP pipeline) on our configured Celery app
        try:
            celery_app.send_task(
                name="workers.tasks.process_transcript_pipeline",
                args=[str(document.id)],
                queue="ingest",
            )
            logger.info(
                "Queued process_transcript_pipeline", extra={"doc_id": str(document.id)}
            )
        except Exception:
            logger.exception(
                "Failed to enqueue process_transcript_pipeline",
                extra={"doc_id": str(document.id)},
            )

        response_data = DocumentUploadResponse(
            doc_id=document.id,
            filename=document.filename,
            size=document.size,
            checksum=document.checksum,
            status=document.status.value,
            storage_path=storage_path,
            download_url=download_url,
        )

        logger.info(f"Document uploaded successfully: {document.id}")
        return response_data

    async def get_document_status(
        self, doc_id: UUID, db_session: AsyncSession
    ) -> DocumentStatusResponse:
        """Get document processing status."""
        document = await self.document_service.get_document_by_id(doc_id, db_session)

        if not document:
            raise DocumentNotFoundError(str(doc_id))

        response_data = DocumentStatusResponse(
            doc_id=document.id,
            filename=document.filename,
            status=document.status.value,
            document_type=document.document_type.value,
            size=document.size,
            checksum=document.checksum,
            progress=document.get_processing_progress(),
            error_message=document.error_message,
            metadata=document.metadata,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

        return response_data

    async def list_documents(
        self,
        pagination: PaginationRequest,
        status_filter: Optional[str],
        db_session: AsyncSession,
    ) -> DocumentListResponse:
        """List documents with optional filtering."""
        documents, total = await self.document_service.list_documents(
            skip=pagination.skip,
            limit=pagination.limit,
            status_filter=status_filter,
            db_session=db_session,
        )

        document_responses = [
            DocumentStatusResponse(
                doc_id=doc.id,
                filename=doc.filename,
                status=doc.status.value,
                document_type=doc.document_type.value,
                size=doc.size,
                checksum=doc.checksum,
                progress=doc.get_processing_progress(),
                error_message=doc.error_message,
                metadata=doc.metadata,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
            )
            for doc in documents
        ]

        response_data = DocumentListResponse(
            documents=document_responses,
            total=total,
            page=pagination.skip // pagination.limit + 1,
            page_size=pagination.limit,
        )

        return response_data

    async def delete_document(
        self,
        doc_id: UUID,
        delete_request: DocumentDeleteRequest,
        db_session: AsyncSession,
    ) -> str:
        """Delete a document and optionally its associated data."""
        success = await self.document_service.delete_document(
            doc_id=doc_id,
            delete_chunks=delete_request.delete_chunks,
            delete_embeddings=delete_request.delete_embeddings,
            db_session=db_session,
        )

        if not success:
            raise DocumentNotFoundError(str(doc_id))

        logger.info(f"Document deleted successfully: {doc_id}")
        return f"Document {doc_id} deleted successfully"

    def _determine_document_type(
        self, filename: str, content_type: Optional[str]
    ) -> DocumentType:
        """Determine document type from filename and content type."""
        filename_lower = filename.lower()

        # Map based on filename patterns for financial documents
        if "transcript" in filename_lower:
            return DocumentType.TRANSCRIPT
        elif "earnings_release" in filename_lower or "release" in filename_lower:
            return DocumentType.EARNINGS_RELEASE
        elif (
            "slide" in filename_lower
            or "presentation" in filename_lower
            or "deck" in filename_lower
        ):
            return DocumentType.SLIDE_DECK
        elif (
            "press" in filename_lower
            or "announcement" in filename_lower
            or "news" in filename_lower
        ):
            return DocumentType.PRESS_ANNOUNCEMENT
        else:
            # Default to transcript for PDF files (most earnings documents are transcripts)
            if filename_lower.endswith(".pdf"):
                return DocumentType.TRANSCRIPT
            else:
                # For other file types, default to earnings release
                return DocumentType.EARNINGS_RELEASE
