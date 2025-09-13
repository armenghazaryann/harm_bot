"""LangChain-based retrievers for Vector (PGVector) and Postgres FTS.

This module centralizes retrieval so QueryService remains clean and focused.
"""
from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from core.settings import SETTINGS
from infra.db_utils import convert_async_to_sync_dsn


def get_pgvector_store(collection_name: str = "rag_chunks_v1") -> PGVector:
    conn = convert_async_to_sync_dsn(str(SETTINGS.DATABASE.DATABASE_URL))
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = PGVector(
        connection_string=conn,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    return store


def vector_search(
    query: str, top_k: int = 10, collection: str = "rag_chunks_v1"
) -> List[Tuple[Document, float]]:
    store = get_pgvector_store(collection)
    # Returns list[(Document, score)] where score is similarity (higher is better)
    return store.similarity_search_with_relevance_scores(query, k=top_k)


async def fts_search(
    session: AsyncSession, query: str, top_k: int = 10
) -> List[Tuple[Document, float]]:
    """Full-text search over chunk.content_normalized using Postgres FTS.

    Returns Documents with metadata containing at least: chunk_id, doc_id, filename, sequence.
    Score returned is ts_rank (higher is better).
    """
    sql = text(
        """
        SELECT
            c.id::text as chunk_id,
            c.document_id::text as doc_id,
            d.filename as filename,
            c.sequence_number as sequence,
            c.content_normalized as content,
            c.metadata as metadata,
            ts_rank_cd(to_tsvector('english', coalesce(c.content_normalized, '')),
                       plainto_tsquery('english', :q)) AS score
        FROM chunk c
        JOIN document d ON d.id = c.document_id
        WHERE to_tsvector('english', coalesce(c.content_normalized, '')) @@ plainto_tsquery('english', :q)
        ORDER BY score DESC
        LIMIT :k
        """
    )
    res = await session.execute(sql, {"q": query, "k": top_k})
    rows = res.mappings().all()
    docs: List[Tuple[Document, float]] = []
    for r in rows:
        meta = r.get("metadata") or {}
        meta.update(
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "document_filename": r["filename"],
                "sequence": int(r["sequence"]),
                "retriever": "fts",
            }
        )
        docs.append(
            (Document(page_content=r["content"], metadata=meta), float(r["score"]))
        )
    return docs
