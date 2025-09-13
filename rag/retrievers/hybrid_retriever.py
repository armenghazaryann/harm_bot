"""Hybrid retriever: PGVector + Postgres FTS with Reciprocal Rank Fusion (RRF).

- Vector search via PGVector + OpenAIEmbeddings (LangChain)
- FTS search via Postgres `ts_rank_cd` over normalized content
- Merge results with RRF: score += 1 / (k + rank)
"""
from __future__ import annotations

from typing import List, Optional, Dict

from langchain_core.documents import Document

from .vector_retriever import vector_search
from .fts_retriever import fts_search

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


def _doc_key(d: Document) -> str:
    md = d.metadata or {}
    # Prefer chunk_id if present, else doc_id + sequence
    cid = md.get("chunk_id")
    if cid:
        return str(cid)
    return f"{md.get('doc_id','')}/{md.get('sequence','')}"


async def _doc_name_from_id(session: AsyncSession, doc_id: str) -> str:
    """Fetch the document filename for a given document UUID.
    Returns empty string if not found.
    """
    if not doc_id:
        return ""
    sql = text("SELECT filename FROM document WHERE id = :doc_id")
    # AsyncSession.execute returns a coroutine; await it.
    result = await session.execute(sql, {"doc_id": doc_id})
    row = result.fetchone()
    return row[0] if row else ""


async def hybrid_rrf(
    *,
    query: str,
    k: int = 8,
    doc_id: Optional[str] = None,
    collection_name: Optional[str] = None,
    session=None,
    fts_k: int = 30,
    vec_k: int = 30,
    rrf_k: int = 60,
) -> List[Document]:
    """Run vector and FTS searches, then merge results with RRF.

    Args:
      query: query text
      k: final top-k to return
      doc_id: optional filter to a single document
      collection_name: pgvector collection name
      session: AsyncSession for FTS
      fts_k, vec_k: candidate depth per modality
      rrf_k: constant in 1/(rrf_k + rank)
    """
    # Vector search (sync client) – returns (Document, relevance_score)
    vec_docs = vector_search(
        query, top_k=vec_k, collection=collection_name or "rag_chunks_v1"
    )
    # Enrich vector docs with human‑readable document name if a DB session is provided
    if session is not None:
        enriched_vec = []
        for doc, score in vec_docs:
            doc_id = doc.metadata.get("doc_id") or doc.metadata.get("document_id")
            if doc_id:
                # Await the async DB lookup
                name = await _doc_name_from_id(session, str(doc_id))
                if name:
                    doc.metadata["document_name"] = name
            enriched_vec.append((doc, score))
        vec_docs = enriched_vec
    # vec_docs -> List[Tuple[Document, relevance_score]]

    # FTS search (async)
    fts_docs = []
    if session is not None:
        fts_docs = await fts_search(session, query, top_k=fts_k, doc_id=doc_id)

    # Build rank lists
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    # Vector ranks
    for rank, (doc, score) in enumerate(vec_docs, start=1):
        key = _doc_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + score + 1.0 / (rrf_k + rank)
        if key not in doc_map:
            doc_map[key] = doc

    # FTS ranks
    for rank, (doc, score) in enumerate(fts_docs, start=1):
        key = _doc_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + score + 1.0 / (rrf_k + rank)
        if key not in doc_map:
            doc_map[key] = doc

    ranked = sorted(
        doc_map.items(), key=lambda x: rrf_scores.get(x[0], 0.0), reverse=True
    )
    # Attach the final RRF score to each document's metadata for the controller to expose.
    final_docs: List[Document] = []
    for key, doc in ranked[:k]:
        # Ensure metadata dict exists
        meta = doc.metadata or {}
        meta["score"] = rrf_scores.get(key, 0.0)
        doc.metadata = meta
        final_docs.append(doc)
    return final_docs
