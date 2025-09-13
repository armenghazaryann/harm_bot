"""Chunk repository using base repository pattern."""
from typing import List

from api.features.query.entities.chunk import Chunk as ChunkEntity
from api.shared.base import BaseRepository


class ChunkRepository(BaseRepository[ChunkEntity]):
    """Repository for chunk entities with specialized queries."""

    model = ChunkEntity

    async def get_by_document_id(
        self, document_id: str, offset: int = 0, limit: int = 100
    ) -> List[ChunkEntity]:
        """Get all chunks for a document ordered by sequence."""
        return await self.get_by_fields(
            document_id=document_id, order_by="sequence_number"
        )

    async def get_by_document_and_sequence_range(
        self, document_id: str, start_sequence: int, end_sequence: int
    ) -> List[ChunkEntity]:
        """Get chunks within a sequence range for a document."""
        from sqlalchemy import select, and_

        stmt = (
            select(self.model)
            .where(
                and_(
                    self.model.document_id == document_id,
                    self.model.sequence_number >= start_sequence,
                    self.model.sequence_number <= end_sequence,
                )
            )
            .order_by(self.model.sequence_number.asc())
        )

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_chunk_count_by_document(self, document_id: str) -> int:
        """Get total chunk count for a document."""
        return await self.count(document_id=document_id)

    async def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        return await self.delete_by_field("document_id", document_id)
