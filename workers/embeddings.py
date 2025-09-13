"""Embedding utilities for documents (MVP) - refactored with repository pattern.

Implements embedding for transcript chunks using OpenAI text-embedding-3-small (1536-dim).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import structlog
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from api.features.query.repositories.chunk_repository import ChunkRepository
from api.features.documents.repositories.document_repository import DocumentRepository
from core.settings import SETTINGS
from infra.db_utils import DatabaseManager, convert_async_to_sync_dsn

logger = structlog.get_logger("workers.embeddings")


class DocumentEmbedder:
    """High-level orchestrator for document embedding using repository pattern."""

    def __init__(self):
        if not SETTINGS.OPENAI.USE_OPENAI:
            raise RuntimeError("OpenAI usage disabled by settings")
        # Ensure API key available for langchain_openai
        api_key = SETTINGS.OPENAI.OPENAI_API_KEY.get_secret_value() or os.getenv(
            "OPENAI_API_KEY"
        )
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    async def embed_document_chunks(self, doc_id: str) -> Dict[str, Any]:
        """Embed all chunks for a document via LangChain PGVector.

        This writes vectors into LangChain's tables (langchain_pg_collection, langchain_pg_embedding)
        for collection 'rag_chunks_v1', keeping retrieval consistent with query.retrievers.
        """
        db = await DatabaseManager.get_resource()

        async with db.get_session() as session:
            # Repositories
            chunk_repo = ChunkRepository(session)
            doc_repo = DocumentRepository(session)

            # Load document and its chunks
            document = await doc_repo.get_by_id(doc_id)
            chunks = await chunk_repo.get_by_document_id(doc_id)
            if not chunks:
                return {"doc_id": doc_id, "embedded": 0}

            # Prepare PGVector store
            conn = convert_async_to_sync_dsn(str(SETTINGS.DATABASE.DATABASE_URL))
            store = PGVector(
                connection_string=conn,
                embedding_function=self.embeddings,
                collection_name="rag_chunks_v1",
            )

            # Build payloads
            texts: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            ids: List[str] = []
            for ch in chunks:
                text = (
                    getattr(ch, "content_normalized", None) or ch.content or ""
                ).strip()
                if not text:
                    continue
                texts.append(text)
                metadatas.append(
                    {
                        "chunk_id": str(ch.id),
                        "doc_id": str(ch.document_id),
                        "document_filename": getattr(document, "filename", ""),
                        "sequence": int(getattr(ch, "sequence_number", 0)),
                        "retriever": "vector",
                    }
                )
                ids.append(str(ch.id))

            if not texts:
                return {"doc_id": doc_id, "embedded": 0}

            # Idempotency: remove any existing vectors with same IDs, then add
            try:
                store.delete(ids=ids)
            except Exception as e:
                logger.warning("pgvector_delete_failed", error=str(e))

            store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

            return {
                "doc_id": doc_id,
                "embedded": len(ids),
                "model": "text-embedding-3-small",
                "dim": 1536,
                "collection": "rag_chunks_v1",
            }


# Module-level function for backward compatibility
def embed_document_chunks(doc_id: str) -> Dict[str, Any]:
    """Embed all chunks for a document that are missing embeddings.

    This is a convenience wrapper around DocumentEmbedder for backward compatibility.
    """
    embedder = DocumentEmbedder()
    return embedder.embed_document_chunks(doc_id)
