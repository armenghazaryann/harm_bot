"""Document processing service integrated with job system and repository pattern."""
import logging
from typing import Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession

from api.features.documents.entities.document import DocumentStatus, DocumentType
from api.features.documents.repositories.document_repository import DocumentRepository
from workers.embeddings import DocumentEmbedder

logger = logging.getLogger("rag.document_processor")


class DocumentProcessor:
    """Handles document processing pipeline using repository pattern."""

    def __init__(self):
        self.embedder = DocumentEmbedder()

    async def process_document(
        self, document_id: str, session: AsyncSession
    ) -> Dict[str, Any]:
        """Process document: parse, extract text, and prepare for chunking."""
        repository = DocumentRepository(session)

        # Get document
        document = await repository.get_by_id(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")

        # Update document status
        await repository.update_by_id(document_id, status=DocumentStatus.PROCESSING)

        try:
            # Parse document based on type
            if document.document_type == DocumentType.PDF:
                result = await self._process_pdf(document_id, session)
            elif document.document_type == DocumentType.CSV:
                result = await self._process_csv(document_id, session)
            elif document.document_type == DocumentType.TEXT:
                result = await self._process_text(document_id, session)
            else:
                raise ValueError(f"Unsupported document type: {document.document_type}")

            # Update status to ready for chunking
            await repository.update_by_id(
                document_id, status=DocumentStatus.READY_FOR_CHUNKING
            )

            return {"document_id": document_id, "pages_processed": result}

        except Exception:
            await repository.update_by_id(document_id, status=DocumentStatus.FAILED)
            raise

    async def chunk_document(
        self, document_id: str, session: AsyncSession
    ) -> Dict[str, Any]:
        """Chunk document into smaller pieces."""
        repository = DocumentRepository(session)

        # Update document status
        await repository.update_by_id(document_id, status=DocumentStatus.CHUNKING)

        try:
            # Chunk based on document type
            document = await repository.get_by_id(document_id)

            if document.document_type == DocumentType.PDF:
                chunks = await self._chunk_pdf(document_id, session)
            elif document.document_type == DocumentType.CSV:
                chunks = await self._chunk_csv(document_id, session)
            elif document.document_type == DocumentType.TEXT:
                chunks = await self._chunk_text(document_id, session)
            else:
                raise ValueError(f"Unsupported document type: {document.document_type}")

            # Update status to ready for embedding
            await repository.update_by_id(
                document_id, status=DocumentStatus.READY_FOR_EMBEDDING
            )

            return {
                "document_id": document_id,
                "chunks_created": len(chunks),
                "total_chunks": len(chunks),
            }

        except Exception:
            await repository.update_by_id(document_id, status=DocumentStatus.FAILED)
            raise

    async def embed_document(
        self, document_id: str, session: AsyncSession
    ) -> Dict[str, Any]:
        """Embed document chunks."""
        repository = DocumentRepository(session)

        # Update document status
        await repository.update_by_id(document_id, status=DocumentStatus.EMBEDDING)

        try:
            # Use the DocumentEmbedder service
            result = await self.embedder.embed_document_chunks(document_id)

            # Update status to completed
            await repository.update_by_id(document_id, status=DocumentStatus.COMPLETED)

            return result

        except Exception:
            await repository.update_by_id(document_id, status=DocumentStatus.FAILED)
            raise

    async def _process_pdf(self, document_id: str, session: AsyncSession) -> int:
        """Process PDF document."""
        logger.info(f"Processing PDF: {document_id}")
        return 1  # Placeholder

    async def _process_csv(self, document_id: str, session: AsyncSession) -> int:
        """Process CSV document."""
        logger.info(f"Processing CSV: {document_id}")
        return 1  # Placeholder

    async def _process_text(self, document_id: str, session: AsyncSession) -> int:
        """Process text document."""
        logger.info(f"Processing text: {document_id}")
        return 1  # Placeholder

    async def _chunk_pdf(self, document_id: str, session: AsyncSession) -> list:
        """Chunk PDF document."""
        logger.info(f"Chunking PDF: {document_id}")
        return []  # Placeholder

    async def _chunk_csv(self, document_id: str, session: AsyncSession) -> list:
        """Chunk CSV document."""
        logger.info(f"Chunking CSV: {document_id}")
        return []  # Placeholder

    async def _chunk_text(self, document_id: str, session: AsyncSession) -> list:
        """Chunk text document."""
        logger.info(f"Chunking text: {document_id}")
        return []  # Placeholder
