"""Vector retriever backed by PGVector and OpenAI embeddings.

KISS/DRY: single source of truth for vector search used across the app.
"""
from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

from core.settings import SETTINGS
from infra.db_utils import convert_async_to_sync_dsn


def get_pgvector_store(collection_name: str | None = None) -> PGVector:
    """Create a PGVector store with OpenAI embeddings.

    Uses a synchronous DSN because PGVector expects a sync driver.
    """
    conn = convert_async_to_sync_dsn(str(SETTINGS.DATABASE.DATABASE_URL))
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = PGVector(
        connection=conn,
        embeddings=embeddings,
        collection_name=collection_name or SETTINGS.VECTOR.COLLECTION_NAME,
    )
    return store


def vector_search(
    query: str, top_k: int = 10, collection: str | None = None
) -> List[Tuple[Document, float]]:
    """Perform vector similarity search and return relevance scores.

    PGVector's ``similarity_search_with_score`` returns a distance metric. We
    convert it to a bounded relevance score using ``1 / (1 + distance)`` which
    yields values in the (0, 1] range and avoids zero scores for close matches.
    The function also enriches each document's metadata with a human‑readable
    ``document_name`` extracted from the stored ``cmetadata`` (the ``filename``
    field). This satisfies the request to show document names instead of UUIDs.
    """
    store = get_pgvector_store(collection)
    raw_results = store.similarity_search_with_score(query, k=top_k)
    processed: List[Tuple[Document, float]] = []
    for doc, distance in raw_results:
        # Convert distance to relevance score (higher is better)
        relevance = 1.0 / (1.0 + float(distance)) if distance is not None else 0.0
        # Enrich metadata with document name if available
        meta = doc.metadata or {}
        if "filename" in meta:
            meta["document_name"] = meta["filename"]
        elif "document_name" not in meta and "document_id" in meta:
            # fallback to document_id (UI can resolve later)
            meta["document_name"] = meta.get("document_id", "")
        doc.metadata = meta
        processed.append((doc, relevance))
    return processed
