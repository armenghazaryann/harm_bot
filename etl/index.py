"""Chunk indexing for vector and full-text search.

Handles storing embeddings in PGVector and ensuring proper FTS indexing
for hybrid retrieval.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import structlog
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from core.settings import SETTINGS
from infra.db_utils import convert_async_to_sync_dsn

logger = structlog.get_logger("etl.index")


async def index_chunks(
    *,
    document_id: str,
    chunks_with_embeddings: List[Dict[str, Any]],
    db_session: AsyncSession,
    force: bool = False,
) -> Dict[str, Any]:
    """Index chunks in both vector store and FTS.

    Args:
        document_id: Document UUID
        chunks_with_embeddings: Chunks with embedding vectors
        db_session: Database session
        force: Skip idempotency check

    Returns:
        Indexing result with status and metrics
    """
    start_time = time.time()

    if not chunks_with_embeddings:
        return {
            "status": "skipped",
            "reason": "no_chunks_with_embeddings",
            "processing_time": time.time() - start_time,
        }

    try:
        # Index in PGVector for vector search
        vector_result = await _index_in_pgvector(chunks_with_embeddings, document_id)

        # Ensure FTS indexes are in place on LangChain table
        await _ensure_fts_indexes(db_session)

        processing_time = time.time() - start_time

        logger.info(
            "Chunk indexing completed",
            document_id=document_id,
            chunk_count=len(chunks_with_embeddings),
            vector_indexed=vector_result.get("indexed_count", 0),
            processing_time=processing_time,
        )

        return {
            "status": "completed",
            "document_id": document_id,
            "chunk_count": len(chunks_with_embeddings),
            "vector_indexed": vector_result.get("indexed_count", 0),
            "fts_ready": True,
            "processing_time": processing_time,
        }

    except Exception as e:
        logger.error(
            "Chunk indexing failed",
            document_id=document_id,
            error=str(e),
        )
        raise


async def _index_in_pgvector(
    chunks_with_embeddings: List[Dict[str, Any]], document_id: str
) -> Dict[str, Any]:
    """Index chunks in PGVector for similarity search."""
    try:
        # Convert to LangChain Documents
        documents = []
        embeddings_list = []

        for chunk in chunks_with_embeddings:
            # Create Document with metadata
            metadata = chunk.get("metadata", {}).copy()
            metadata.update(
                {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": document_id,
                    "sequence": chunk["sequence"],
                }
            )

            doc = Document(
                page_content=chunk["content_normalized"],
                metadata=metadata,
            )
            documents.append(doc)
            embeddings_list.append(chunk["embedding"])

        # Initialize PGVector store
        connection_string = convert_async_to_sync_dsn(
            str(SETTINGS.DATABASE.DATABASE_URL)
        )
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=SETTINGS.OPENAI.OPENAI_API_KEY.get_secret_value(),
        )

        # Create or get existing collection
        collection_name = "rag_chunks_v1"
        vector_store = PGVector(
            connection=connection_string,
            embeddings=embeddings_model,
            collection_name=collection_name,
            use_jsonb=True,
        )

        # Add documents with pre-computed embeddings
        # Use add_embeddings to provide pre-computed vectors and avoid re-embedding
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate unique IDs for each document to avoid null ID constraint violation
        ids = [chunk["chunk_id"] for chunk in chunks_with_embeddings]

        vector_store.add_embeddings(
            texts=texts, embeddings=embeddings_list, metadatas=metadatas, ids=ids
        )

        logger.info(
            "Documents indexed in PGVector",
            document_id=document_id,
            collection=collection_name,
            count=len(documents),
        )

        return {
            "status": "completed",
            "indexed_count": len(documents),
            "collection_name": collection_name,
        }

    except Exception as e:
        logger.error("PGVector indexing failed", document_id=document_id, error=str(e))
        raise


async def _ensure_fts_indexes(db_session: AsyncSession) -> None:
    """Ensure FTS indexes are in place for hybrid search on LangChain table."""
    try:
        # Check if FTS index exists on langchain_pg_embedding table
        index_check = await db_session.execute(
            text(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE tablename = 'langchain_pg_embedding'
                AND indexname LIKE '%fts%'
            """
            )
        )

        existing_indexes = [row[0] for row in index_check.fetchall()]

        if not existing_indexes:
            # Create GIN index for FTS on document content
            logger.info("Creating FTS index on langchain_pg_embedding table")
            # Later can be moved to concurrent index creation
            await db_session.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS langchain_pg_embedding_document_fts_idx "
                    "ON langchain_pg_embedding USING gin(to_tsvector('english', document))"
                )
            )
            await db_session.commit()
            logger.info(
                "FTS index created successfully on langchain_pg_embedding.document"
            )
        else:
            logger.debug("FTS indexes already exist", indexes=existing_indexes)

    except Exception as e:
        logger.warning("Failed to ensure FTS indexes", error=str(e))
        # Don't raise - indexing can continue without this optimization
