"""Service layer for the Documents feature."""
import logging
from typing import List, Optional, Tuple
from datetime import timedelta
from uuid import UUID

from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from api.features.documents.exceptions import (
    DocumentNotFoundError,
    DocumentStorageError,
    DocumentAlreadyExistsError
)
from api.features.documents.models import DocumentModel, DocumentCreateModel, DocumentUpdateModel
from api.features.documents.entities.document import (
    Document as DocumentEntity,
    DocumentStatus,
)

logger = logging.getLogger("rag.documents.service")


class DocumentService:
    """Service for document operations."""
    
    def __init__(self, storage_client=None):
        self.storage_client = storage_client
    
    async def create_document(
        self, 
        create_model: DocumentCreateModel, 
        db_session: AsyncSession
    ) -> DocumentModel:
        """Create a new document."""
        try:
            # Check if document with same checksum already exists
            existing = await self.get_document_by_checksum(create_model.checksum, db_session)
            if existing:
                raise DocumentAlreadyExistsError(create_model.checksum, "checksum")
            
            # Create entity
            entity = create_model.to_entity()
            db_session.add(entity)
            await db_session.commit()
            await db_session.refresh(entity)
            
            logger.info(f"Document created: {entity.id}")
            return DocumentModel.from_entity(entity)
            
        except Exception as e:
            await db_session.rollback()
            logger.error(f"Failed to create document: {str(e)}")
            raise
    
    async def get_document_by_id(
        self, 
        doc_id: UUID, 
        db_session: AsyncSession
    ) -> Optional[DocumentModel]:
        """Get document by ID."""
        try:
            stmt = select(DocumentEntity).where(DocumentEntity.id == doc_id)
            result = await db_session.execute(stmt)
            entity = result.scalar_one_or_none()
            
            if entity:
                return DocumentModel.from_entity(entity)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {str(e)}")
            raise
    
    async def get_document_by_checksum(
        self, 
        checksum: str, 
        db_session: AsyncSession
    ) -> Optional[DocumentModel]:
        """Get document by checksum."""
        try:
            stmt = select(DocumentEntity).where(DocumentEntity.checksum == checksum)
            result = await db_session.execute(stmt)
            entity = result.scalar_one_or_none()
            
            if entity:
                return DocumentModel.from_entity(entity)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document by checksum {checksum}: {str(e)}")
            raise
    
    async def update_document(
        self, 
        doc_id: UUID, 
        update_model: DocumentUpdateModel, 
        db_session: AsyncSession
    ) -> DocumentModel:
        """Update document."""
        try:
            stmt = select(DocumentEntity).where(DocumentEntity.id == doc_id)
            result = await db_session.execute(stmt)
            entity = result.scalar_one_or_none()
            
            if not entity:
                raise DocumentNotFoundError(str(doc_id))
            
            # Apply updates
            entity = update_model.apply_to_entity(entity)
            await db_session.commit()
            await db_session.refresh(entity)
            
            logger.info(f"Document updated: {doc_id}")
            return DocumentModel.from_entity(entity)
            
        except Exception as e:
            await db_session.rollback()
            logger.error(f"Failed to update document {doc_id}: {str(e)}")
            raise
    
    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 100,
        status_filter: Optional[str] = None,
        db_session: AsyncSession = None
    ) -> Tuple[List[DocumentModel], int]:
        """List documents with pagination and filtering."""
        try:
            # Build query
            stmt = select(DocumentEntity)
            count_stmt = select(func.count(DocumentEntity.id))
            
            # Apply status filter
            if status_filter:
                try:
                    status_enum = DocumentStatus(status_filter)
                    stmt = stmt.where(DocumentEntity.status == status_enum)
                    count_stmt = count_stmt.where(DocumentEntity.status == status_enum)
                except ValueError:
                    logger.warning(f"Invalid status filter: {status_filter}")
            
            # Apply pagination
            stmt = stmt.offset(skip).limit(limit).order_by(DocumentEntity.created_at.desc())
            
            # Execute queries
            result = await db_session.execute(stmt)
            count_result = await db_session.execute(count_stmt)
            
            entities = result.scalars().all()
            total = count_result.scalar()
            
            documents = [DocumentModel.from_entity(entity) for entity in entities]
            
            return documents, total
            
        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            raise
    
    async def delete_document(
        self,
        doc_id: UUID,
        delete_chunks: bool = True,
        delete_embeddings: bool = True,
        db_session: AsyncSession = None
    ) -> bool:
        """Delete document and optionally associated data."""
        try:
            # Check if document exists
            document = await self.get_document_by_id(doc_id, db_session)
            if not document:
                return False
            
            # Delete from storage
            if self.storage_client:
                try:
                    if document.storage_path:
                        await self.delete_from_storage(document.storage_path)
                except Exception as e:
                    logger.warning(f"Failed to delete from storage: {str(e)}")
            
            # Delete associated chunks and embeddings if requested
            if delete_chunks:
                # TODO: Implement chunk deletion
                pass
            
            if delete_embeddings:
                # TODO: Implement embedding deletion
                pass
            
            # Delete document entity
            stmt = delete(DocumentEntity).where(DocumentEntity.id == doc_id)
            await db_session.execute(stmt)
            await db_session.commit()
            
            logger.info(f"Document deleted: {doc_id}")
            return True
            
        except Exception as e:
            await db_session.rollback()
            logger.error(f"Failed to delete document {doc_id}: {str(e)}")
            raise
    
    async def upload_to_storage(self, storage_path: str, content: bytes, content_type: Optional[str] = None) -> str:
        """Upload document content to object storage.

        Args:
            storage_path: Object key/path to store the content under.
            content: Bytes content of the file.
            content_type: Optional MIME type.

        Returns:
            The storage path used.
        """
        try:
            if not self.storage_client or getattr(self.storage_client, "client", None) is None:
                raise DocumentStorageError("Storage client not configured")

            client = self.storage_client.client
            bucket = self.storage_client.bucket_name

            # Upload bytes using put_object with a stream
            import io

            data_stream = io.BytesIO(content)
            data_len = len(content)
            client.put_object(
                bucket_name=bucket,
                object_name=storage_path,
                data=data_stream,
                length=data_len,
                content_type=content_type or "application/octet-stream",
            )

            logger.info(f"Uploaded to storage: {storage_path}")
            return storage_path

        except Exception as e:
            logger.error(f"Failed to upload to storage {storage_path}: {str(e)}")
            raise DocumentStorageError(f"Failed to upload document: {str(e)}")
    
    async def delete_from_storage(self, storage_path: str) -> bool:
        """Delete document from object storage by path."""
        try:
            if not self.storage_client or getattr(self.storage_client, "client", None) is None:
                logger.warning("Storage client not configured, skipping storage deletion")
                return True

            client = self.storage_client.client
            bucket = self.storage_client.bucket_name

            client.remove_object(bucket, storage_path)

            logger.info(f"Deleted from storage: {storage_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete from storage {storage_path}: {str(e)}")
            raise DocumentStorageError(f"Failed to delete document from storage: {str(e)}")

    async def get_download_url(self, storage_path: str, expires_seconds: int = 3600) -> Optional[str]:
        """Generate a presigned GET URL for downloading the object.

        Returns None if storage client not configured.
        """
        if not self.storage_client or getattr(self.storage_client, "client", None) is None:
            return None
        client = self.storage_client.client
        bucket = self.storage_client.bucket_name
        url = client.presigned_get_object(bucket, storage_path, expires=timedelta(seconds=expires_seconds))
        return url
    
    async def get_document_stats(self, db_session: AsyncSession) -> dict:
        """Get document statistics."""
        try:
            # Count documents by status
            total_stmt = select(func.count(DocumentEntity.id))
            processing_stmt = select(func.count(DocumentEntity.id)).where(
                DocumentEntity.status == DocumentStatus.PROCESSING
            )
            completed_stmt = select(func.count(DocumentEntity.id)).where(
                DocumentEntity.status == DocumentStatus.COMPLETED
            )
            failed_stmt = select(func.count(DocumentEntity.id)).where(
                DocumentEntity.status == DocumentStatus.FAILED
            )
            
            # Execute queries
            total_result = await db_session.execute(total_stmt)
            processing_result = await db_session.execute(processing_stmt)
            completed_result = await db_session.execute(completed_stmt)
            failed_result = await db_session.execute(failed_stmt)
            
            return {
                "total_documents": total_result.scalar(),
                "processing_documents": processing_result.scalar(),
                "completed_documents": completed_result.scalar(),
                "failed_documents": failed_result.scalar(),
                "total_chunks": 0,  # TODO: Implement chunk counting
                "total_embeddings": 0,  # TODO: Implement embedding counting
                "storage_used_bytes": 0  # TODO: Implement storage usage calculation
            }
            
        except Exception as e:
            logger.error(f"Failed to get document stats: {str(e)}")
            raise
