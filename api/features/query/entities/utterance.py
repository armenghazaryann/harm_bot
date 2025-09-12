"""Utterance entity for transcript turns (MVP)."""
from typing import Optional

from sqlalchemy import String, Integer, Text, ForeignKey, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.shared.entities.base import BaseEntity


class Utterance(BaseEntity):
    """Stores transcript utterances to bridge Neo4j and retrieval."""

    document_id: Mapped[str] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Stable external id for idempotency (hash of doc_id|speaker|speech|turn_index)
    utterance_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    # Ordering and labels
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    speaker: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[Optional[str]] = mapped_column(
        String(50)
    )  # management|analyst|operator|unknown
    section: Mapped[Optional[str]] = mapped_column(
        String(50)
    )  # prepared_remarks|qa|participants|other

    # Content
    speech: Mapped[str] = mapped_column(Text, nullable=False)

    # Optional metadata
    page_spans: Mapped[Optional[dict]] = mapped_column(JSON)
    extraction_method: Mapped[Optional[str]] = mapped_column(
        String(30)
    )  # text|ocr_cloud|ocr_tesseract

    # Relationships
    document = relationship("Document")

    __table_args__ = (
        Index("idx_utterance_document_turn", "document_id", "turn_index", unique=False),
        Index("idx_utterance_speaker", "speaker", unique=False),
    )
