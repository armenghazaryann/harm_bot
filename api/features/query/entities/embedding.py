"""Embedding entity following SOLID principles and clean architecture."""
from typing import Optional

from sqlalchemy import String, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from api.shared.entities.base import BaseEntity


class Embedding(BaseEntity):
    """Embedding entity for storing vector embeddings with metadata."""
    
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("chunk.id", ondelete="CASCADE"), 
        nullable=False
    )
    
    # Embedding space classification (text, table, page)
    embedding_space: Mapped[str] = mapped_column(String(20), nullable=False)
    
    # Vector embedding (1536 dimensions for OpenAI text-embedding-3-small)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    
    # Model metadata for versioning and tracking
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Relationships
    chunk = relationship("Chunk", back_populates="embeddings")
    
    __table_args__ = (
        Index("idx_embedding_space", "embedding_space"),
        Index("idx_embedding_vector_cosine", "embedding", postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"}),
        Index("idx_embedding_vector_ip", "embedding", postgresql_using="hnsw", postgresql_ops={"embedding": "vector_ip_ops"}),
    )

    def get_similarity_search_params(self) -> dict:
        """Get parameters for similarity search optimization."""
        return {
            "embedding_space": self.embedding_space,
            "model_name": self.model_name,
            "vector_dimension": len(self.embedding)
        }
