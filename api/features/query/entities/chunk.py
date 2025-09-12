"""Chunk entity following SOLID principles and clean architecture."""
from enum import Enum
from typing import Optional

from sqlalchemy import String, Integer, Text, JSON, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.shared.entities.base import BaseEntity


class ChunkType(str, Enum):
    """Chunk type enumeration for different content types."""
    TEXT = "text"
    TABLE = "table"
    PAGE = "page"
    SLIDE = "slide"
    SECTION = "section"


class Chunk(BaseEntity):
    """Chunk entity for storing document chunks with semantic meaning."""
    
    document_id: Mapped[str] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), 
        nullable=False
    )
    
    # Chunk identification and type
    chunk_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    chunk_type: Mapped[ChunkType] = mapped_column(String(20), nullable=False)
    
    # Content (original and normalized for search)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_normalized: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Position and structure metadata
    page_number: Mapped[Optional[int]] = mapped_column(Integer)
    section_title: Mapped[Optional[str]] = mapped_column(String(500))
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Token count for LLM context management
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Additional metadata (flexible JSON field)
    # NOTE: attribute name 'metadata' is reserved by SQLAlchemy; keep column name but change attribute
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSON, name="metadata")
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    embeddings = relationship("Embedding", back_populates="chunk", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_chunk_document_sequence", "document_id", "sequence_number"),
        Index("idx_chunk_type", "chunk_type"),
        Index("idx_chunk_page", "page_number"),
    )

    def is_embeddable(self) -> bool:
        """Check if chunk is ready for embedding."""
        return len(self.content.strip()) > 0 and self.token_count > 0
    
    def get_context_window_size(self) -> int:
        """Get the token count for context window calculations."""
        return self.token_count
