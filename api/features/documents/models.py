"""Models for the Documents feature."""
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from api.features.documents.entities.document import (
    Document as DocumentEntity,
    DocumentStatus,
    DocumentType,
)


class DocumentModel(BaseModel):
    """Domain model for Document."""
    
    id: UUID = Field(description="Document identifier")
    filename: str = Field(description="Stored filename")
    document_type: DocumentType = Field(description="Document type")
    status: DocumentStatus = Field(description="Processing status")
    size: int = Field(description="File size in bytes")
    checksum: str = Field(description="File checksum")
    storage_path: str = Field(description="Storage path in object store")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Document metadata")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")
    processed_at: Optional[datetime] = Field(default=None, description="Processing completion timestamp (if tracked)")
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")
    
    class Config:
        from_attributes = True
    
    @classmethod
    def from_entity(cls, entity: DocumentEntity) -> "DocumentModel":
        """Create model from database entity."""
        return cls(
            id=entity.id,
            filename=entity.filename,
            document_type=entity.doc_type,
            status=entity.status,
            size=entity.file_size,
            checksum=entity.checksum,
            storage_path=entity.raw_path if entity.raw_path else (entity.processed_path or ""),
            metadata=(entity.processing_metadata or {}),
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            processed_at=None,
            error_message=entity.error_message,
        )
    
    def to_entity(self) -> DocumentEntity:
        """Convert model to database entity."""
        # Convert back to entity; fields not present on entity will be ignored or approximated
        entity = DocumentEntity(
            filename=self.filename,
            original_filename=self.filename,
            content_type="application/octet-stream",
            file_size=self.size,
            checksum=self.checksum,
            doc_type=self.document_type,
            status=self.status,
            raw_path=self.storage_path,
            processed_path=None,
            processing_metadata=self.metadata,
            error_message=self.error_message,
        )
        return entity
    
    def is_processing_complete(self) -> bool:
        """Check if document processing is complete."""
        return self.status in [DocumentStatus.INDEXED, DocumentStatus.FAILED]
    
    def is_ready_for_query(self) -> bool:
        """Check if document is ready for querying."""
        return self.status in [DocumentStatus.CHUNKED, DocumentStatus.EMBEDDED, DocumentStatus.INDEXED]
    
    def get_processing_progress(self) -> Dict[str, any]:
        """Get processing progress information."""
        progress = {
            "status": self.status.value,
            "completed": self.is_processing_complete(),
            "ready_for_query": self.is_ready_for_query()
        }
        
        if self.processed_at:
            progress["processed_at"] = self.processed_at.isoformat()
        
        if self.error_message:
            progress["error"] = self.error_message
            
        return progress


class DocumentCreateModel(BaseModel):
    """Model for creating a new document."""
    
    filename: str = Field(description="Original filename")
    document_type: DocumentType = Field(description="Document type")
    size: int = Field(description="File size in bytes")
    checksum: str = Field(description="File checksum")
    storage_path: str = Field(description="Storage path in object store")
    content_type: str = Field(description="MIME type of the document")
    metadata: Optional[Dict[str, str]] = Field(default=None, description="Document metadata")
    
    def to_entity(self) -> DocumentEntity:
        """Convert to database entity."""
        return DocumentEntity(
            filename=self.filename,
            original_filename=self.filename,
            content_type=self.content_type,
            file_size=self.size,
            checksum=self.checksum,
            doc_type=self.document_type,
            status=DocumentStatus.UPLOADED,
            raw_path=self.storage_path,
            processed_path=None,
            processing_metadata=self.metadata or {},
        )


class DocumentUpdateModel(BaseModel):
    """Model for updating a document."""
    
    status: Optional[DocumentStatus] = Field(default=None, description="Processing status")
    processed_at: Optional[datetime] = Field(default=None, description="Processing completion timestamp")
    error_message: Optional[str] = Field(default=None, description="Error message")
    metadata: Optional[Dict[str, str]] = Field(default=None, description="Updated metadata")
    
    def apply_to_entity(self, entity: DocumentEntity) -> DocumentEntity:
        """Apply updates to an existing entity."""
        if self.status is not None:
            entity.status = self.status
        # processed_at is not tracked in current entity; ignore if provided
        if self.error_message is not None:
            entity.error_message = self.error_message
        if self.metadata is not None:
            entity.processing_metadata = self.metadata

        entity.updated_at = datetime.utcnow()
        return entity
