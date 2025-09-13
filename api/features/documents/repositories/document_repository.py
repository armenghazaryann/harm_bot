"""Document repository using base repository pattern."""
from typing import List, Optional

from api.features.documents.entities.document import (
    Document as DocumentEntity,
    DocumentStatus,
    DocumentType,
    Document,
)
from api.shared.base import BaseRepository


class DocumentRepository(BaseRepository[DocumentEntity]):
    """Repository for document entities with specialized queries."""

    model = Document

    async def get_by_checksum(self, checksum: str) -> Optional[DocumentEntity]:
        """Get document by checksum for deduplication."""
        entities = await self.get_by_field("checksum", checksum, limit=1)
        return entities[0] if entities else None

    async def get_by_status(
        self, status: DocumentStatus, *, offset: int = 0, limit: int = 100
    ) -> List[DocumentEntity]:
        """Get documents by processing status."""
        entities, _ = await self.list(
            offset=offset, limit=limit, order_by="created_at", status=status
        )
        return entities

    async def get_by_type(
        self, document_type: DocumentType, *, offset: int = 0, limit: int = 100
    ) -> List[DocumentEntity]:
        """Get documents by type."""
        entities, _ = await self.list(
            offset=offset,
            limit=limit,
            order_by="created_at",
            document_type=document_type,
        )
        return entities

    async def get_by_filename(self, filename: str) -> Optional[DocumentEntity]:
        """Get document by filename."""
        entities = await self.get_by_field("filename", filename, limit=1)
        return entities[0] if entities else None

    async def get_processing_stats(self) -> dict:
        """Get document processing statistics."""

        # Get counts by status
        status_counts = {}
        for status in DocumentStatus:
            count = await self.count(status=status)
            status_counts[status.value] = count

        # Get counts by type
        type_counts = {}
        for doc_type in DocumentType:
            count = await self.count(document_type=doc_type)
            type_counts[doc_type.value] = count

        return {
            "by_status": status_counts,
            "by_type": type_counts,
            "total": sum(status_counts.values()),
        }
