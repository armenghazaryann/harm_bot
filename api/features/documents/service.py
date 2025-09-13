"""Service layer for the Documents feature - refactored with repository pattern."""
import logging
from typing import List, Optional, Tuple
from uuid import UUID
from io import BytesIO
from datetime import timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from api.features.documents.exceptions import (
    DocumentNotFoundError,
    DocumentAlreadyExistsError,
)
from api.features.documents.models import (
    DocumentModel,
    DocumentCreateModel,
    DocumentUpdateModel,
)
from api.features.documents.entities.document import DocumentStatus, DocumentType
from api.features.documents.repositories.document_repository import DocumentRepository
from infra.resources import MinIOResource

logger = logging.getLogger("rag.documents.service")


class DocumentService:
    """Service for document operations using repository pattern."""

    def __init__(self, storage_client: MinIOResource):
        self.storage_client = storage_client

    async def create_document(
        self, create_model: DocumentCreateModel, *, db_session: AsyncSession
    ) -> DocumentModel:
        """Create a new document."""
        repository = DocumentRepository(db_session)

        # Check if document with same checksum already exists
        existing = await repository.get_by_checksum(create_model.checksum)
        if existing:
            raise DocumentAlreadyExistsError(create_model.checksum, "checksum")

        # Create entity
        entity = create_model.to_entity()
        entity = await repository.create(entity)

        logger.info(f"Document created: {entity.id}")
        return DocumentModel.from_entity(entity)

    async def get_document(
        self, document_id: UUID, *, db_session: AsyncSession
    ) -> Optional[DocumentModel]:
        """Get document by ID."""
        repository = DocumentRepository(db_session)
        entity = await repository.get_by_id(document_id)
        return DocumentModel.from_entity(entity) if entity else None

    async def get_document_by_checksum(
        self, checksum: str, *, db_session: AsyncSession
    ) -> Optional[DocumentModel]:
        """Get document by checksum."""
        repository = DocumentRepository(db_session)
        entity = await repository.get_by_checksum(checksum)
        return DocumentModel.from_entity(entity) if entity else None

    async def list_documents(
        self, offset: int, limit: int, *, db_session: AsyncSession
    ) -> Tuple[List[DocumentModel], int]:
        """List documents with pagination."""
        repository = DocumentRepository(db_session)
        entities, total = await repository.list(offset, limit, order_by="-created_at")
        return [DocumentModel.from_entity(entity) for entity in entities], total

    async def update_document(
        self,
        document_id: UUID,
        update_model: DocumentUpdateModel,
        *,
        db_session: AsyncSession,
    ) -> DocumentModel:
        """Update document."""
        repository = DocumentRepository(db_session)

        entity = await repository.get_by_id(document_id)
        if not entity:
            raise DocumentNotFoundError(f"Document {document_id} not found")

        # Update fields from model
        update_data = update_model.dict(exclude_unset=True)
        entity = await repository.update_by_id(document_id, **update_data)

        return DocumentModel.from_entity(entity)

    async def update_document_status(
        self, document_id: UUID, status: DocumentStatus, *, db_session: AsyncSession
    ) -> DocumentModel:
        """Update document status."""
        repository = DocumentRepository(db_session)
        entity = await repository.update_by_id(document_id, status=status)

        if not entity:
            raise DocumentNotFoundError(f"Document {document_id} not found")

        return DocumentModel.from_entity(entity)

    async def delete_document(
        self, document_id: UUID, *, db_session: AsyncSession
    ) -> bool:
        """Delete document."""
        repository = DocumentRepository(db_session)
        return await repository.delete(document_id)

    async def get_documents_by_status(
        self,
        status: DocumentStatus,
        *,
        offset: int = 0,
        limit: int = 100,
        db_session: AsyncSession,
    ) -> List[DocumentModel]:
        """Get documents by processing status."""
        repository = DocumentRepository(db_session)
        entities = await repository.get_by_status(status, offset=offset, limit=limit)
        return [DocumentModel.from_entity(entity) for entity in entities]

    async def get_documents_by_type(
        self,
        document_type: DocumentType,
        *,
        offset: int = 0,
        limit: int = 100,
        db_session: AsyncSession,
    ) -> List[DocumentModel]:
        """Get documents by type."""
        repository = DocumentRepository(db_session)
        entities = await repository.get_by_type(
            document_type, offset=offset, limit=limit
        )
        return [DocumentModel.from_entity(entity) for entity in entities]

    async def get_processing_stats(self, *, db_session: AsyncSession) -> dict:
        """Get document processing statistics."""
        repository = DocumentRepository(db_session)
        return await repository.get_processing_stats()

    # Storage methods
    async def upload_to_storage(
        self, storage_path: str, content: bytes, content_type: Optional[str] = None
    ) -> None:
        """Upload document to storage."""
        try:
            # Wrap bytes in BytesIO for MinIO client compatibility
            data_stream = BytesIO(content)
            self.storage_client.client.put_object(
                bucket_name=self.storage_client.bucket_name,
                object_name=storage_path,
                data=data_stream,
                length=len(content),
                content_type=content_type or "application/octet-stream",
            )
            logger.info(f"Uploaded document to storage: {storage_path}")
        except Exception as e:
            logger.error(f"Failed to upload document: {e}")
            raise

    async def get_download_url(self, storage_path: str) -> str:
        """Get download URL for document."""
        try:
            return self.storage_client.client.presigned_get_object(
                bucket_name=self.storage_client.bucket_name,
                object_name=storage_path,
                expires=timedelta(hours=1),
            )
        except Exception as e:
            logger.error(f"Failed to get download URL: {e}")
            raise
