"""Postgres FTS retriever for chunk content.

Returns LangChain Documents with metadata including chunk_id, doc_id, document_name, sequence.
"""
from __future__ import annotations

from typing import List, Tuple, Optional

from langchain_core.documents import Document
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


async def fts_search(
    session: AsyncSession,
    query: str,
    top_k: int = 10,
    *,
    doc_id: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    """Full-text search over chunk content using Postgres FTS.

    Returns Documents with metadata containing at least: chunk_id, doc_id, document_name, sequence.
    Score returned is ts_rank (higher is better).
    """
    base_sql = """
        SELECT
            e.id::text as chunk_id,
            e.cmetadata->>'document_id' as doc_id,
            e.cmetadata->>'filename' as document_name,
            (e.cmetadata->>'sequence')::int as sequence,
            e.document as content,
            e.cmetadata as metadata,
            ts_rank_cd(
                to_tsvector('english', coalesce(e.document, '')),
                plainto_tsquery('english', :q)
            ) AS score
        FROM langchain_pg_embedding e
        WHERE to_tsvector('english', coalesce(e.document, '')) @@ plainto_tsquery('english', :q)
        {doc_filter}
        ORDER BY score DESC
        LIMIT :k
        """
    doc_filter = ""
    params = {"q": query, "k": top_k}
    if doc_id:
        doc_filter = " AND e.cmetadata->>'document_id' = :doc_id"
        params["doc_id"] = doc_id

    sql = text(base_sql.format(doc_filter=doc_filter))
    res = await session.execute(sql, params)
    rows = res.mappings().all()
    docs: List[Tuple[Document, float]] = []
    for r in rows:
        meta = r.get("metadata") or {}
        meta.update(
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "document_name": r["document_name"],
                "sequence": int(r["sequence"] or 0),
                "retriever": "fts",
            }
        )
        docs.append(
            (Document(page_content=r["content"], metadata=meta), float(r["score"]))
        )
    return docs
