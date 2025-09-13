"""DTOs for the Documents feature."""
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import Field, validator

from api.shared.dtos import BaseDTO, TimestampMixin
from api.features.documents.models import DocumentModel


class DocumentUploadRequest(BaseDTO):
    """Request DTO for document upload."""

    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type of the document")
    metadata: Optional[Dict[str, str]] = Field(
        default=None, description="Additional metadata"
    )

    @validator("filename")
    def validate_filename(cls, v) -> str:
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        return v.strip()


class DocumentUploadResponse(BaseDTO):
    """Response DTO for document upload."""

    doc_id: UUID = Field(description="Document identifier")
    filename: str = Field(description="Original filename")
    size: int = Field(description="File size in bytes")
    checksum: str = Field(description="File checksum")
    status: str = Field(description="Upload status")
    upload_url: Optional[str] = Field(
        default=None, description="Presigned upload URL if applicable"
    )
    storage_path: Optional[str] = Field(
        default=None, description="Object storage path (key)"
    )
    download_url: Optional[str] = Field(
        default=None, description="Presigned download URL"
    )
    processing_job_id: Optional[UUID] = Field(
        default=None, description="Job identifier"
    )


class DocumentStatusResponse(BaseDTO, TimestampMixin):
    """Response DTO for document status."""

    doc_id: UUID = Field(description="Document identifier")
    filename: str = Field(description="Original filename")
    status: str = Field(description="Processing status")
    document_type: str = Field(description="Document type")
    size: int = Field(description="File size in bytes")
    checksum: str = Field(description="File checksum")
    progress: Optional[Dict[str, Any]] = Field(
        default=None, description="Processing progress"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None, description="Document metadata"
    )

    @classmethod
    def from_entity(cls, entity: DocumentModel) -> "DocumentStatusResponse":
        """Create DocumentStatusResponse from DocumentModel."""
        return cls(
            doc_id=entity.id,
            filename=entity.filename,
            status=entity.status.value
            if hasattr(entity.status, "value")
            else str(entity.status),
            document_type=entity.document_type.value
            if hasattr(entity.document_type, "value")
            else str(entity.document_type),
            size=entity.size,
            checksum=entity.checksum,
            progress=entity.get_processing_progress(),
            error_message=entity.error_message,
            metadata=entity.metadata,
        )


class DocumentListResponse(BaseDTO):
    """Response DTO for document listing."""

    documents: List[DocumentStatusResponse] = Field(description="List of documents")
    total: int = Field(description="Total number of documents")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Number of items per page")


class DocumentProcessingRequest(BaseDTO):
    """Request DTO for document processing."""

    doc_id: UUID = Field(description="Document identifier")
    processing_options: Optional[Dict[str, Any]] = Field(
        default=None, description="Processing options"
    )
    priority: str = Field(default="normal", description="Processing priority")

    @validator("priority")
    def validate_priority(cls, v) -> str:
        allowed_priorities = ["low", "normal", "high", "urgent"]
        if v not in allowed_priorities:
            raise ValueError(f"Priority must be one of: {allowed_priorities}")
        return v


class DocumentChunkResponse(BaseDTO):
    """Response DTO for document chunks."""

    chunk_id: UUID = Field(description="Chunk identifier")
    doc_id: UUID = Field(description="Document identifier")
    content: str = Field(description="Chunk content")
    position: int = Field(description="Chunk position in document")
    metadata: Dict[str, Any] = Field(description="Chunk metadata")
    embedding_status: str = Field(description="Embedding generation status")


class DocumentDeleteRequest(BaseDTO):
    """Request DTO for document deletion."""

    doc_id: UUID = Field(description="Document identifier")
    delete_chunks: bool = Field(
        default=True, description="Whether to delete associated chunks"
    )
    delete_embeddings: bool = Field(
        default=True, description="Whether to delete embeddings"
    )


class DocumentStatsResponse(BaseDTO):
    """Response DTO for document statistics."""

    total_documents: int = Field(description="Total number of documents")
    processing_documents: int = Field(description="Number of documents being processed")
    completed_documents: int = Field(description="Number of completed documents")
    failed_documents: int = Field(description="Number of failed documents")
    total_chunks: int = Field(description="Total number of chunks")
    total_embeddings: int = Field(description="Total number of embeddings")
    storage_used_bytes: int = Field(description="Storage used in bytes")
