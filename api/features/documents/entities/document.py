"""Document entity following SOLID principles and clean architecture."""
from enum import Enum
from typing import Dict, Any, Optional

from sqlalchemy import String, Integer, Text, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from api.shared.entities.base import BaseEntity


class DocumentType(str, Enum):
    """Document type enumeration for classification."""

    TRANSCRIPT = "transcript"
    EARNINGS_RELEASE = "earnings_release"
    SLIDE_DECK = "slide_deck"
    GENERAL = "general"


class DocumentStatus(str, Enum):
    """Document processing status enumeration."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    FAILED = "failed"


class Document(BaseEntity):
    """Document entity for storing document metadata and state."""

    # File metadata
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    # Document classification and status
    doc_type: Mapped[DocumentType] = mapped_column(
        SQLEnum(DocumentType), nullable=False
    )
    status: Mapped[DocumentStatus] = mapped_column(
        SQLEnum(DocumentStatus), default=DocumentStatus.UPLOADED
    )

    # Storage paths
    raw_path: Mapped[str] = mapped_column(String(500), nullable=False)
    processed_path: Mapped[Optional[str]] = mapped_column(String(500))

    # Processing metadata
    page_count: Mapped[Optional[int]] = mapped_column(Integer)
    processing_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    def is_processing_complete(self) -> bool:
        """Check if document processing is complete."""
        return self.status in [DocumentStatus.INDEXED, DocumentStatus.FAILED]

    def can_be_chunked(self) -> bool:
        """Check if document can be chunked."""
        return self.status == DocumentStatus.PROCESSED

    def can_be_embedded(self) -> bool:
        """Check if document can be embedded."""
        return self.status == DocumentStatus.CHUNKED
