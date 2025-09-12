"""Embedding utilities for documents (MVP).

Implements embedding for transcript chunks using OpenAI text-embedding-3-small (1536-dim).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import structlog
from openai import OpenAI
from sqlalchemy import select

from core.settings import SETTINGS
from infra.resources import DatabaseResource
from api.features.query.entities.chunk import Chunk as ChunkEntity
from api.features.query.entities.embedding import Embedding as EmbeddingEntity

logger = structlog.get_logger("workers.embeddings")


def _get_openai_client() -> OpenAI:
    api_key = SETTINGS.OPENAI.OPENAI_API_KEY.get_secret_value() or os.getenv(
        "OPENAI_API_KEY"
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)


async def _get_db() -> DatabaseResource:
    db = DatabaseResource(database_url=str(SETTINGS.DATABASE.DATABASE_URL))
    await db.init()
    return db


async def embed_document_chunks(
    doc_id: str, batch_size: int | None = None
) -> Dict[str, Any]:
    """Embed all chunks for a document that are missing embeddings.

    Uses text-embedding-3-small (1536 dims). Writes to embeddings table.
    """
    if not SETTINGS.OPENAI.USE_OPENAI:
        raise RuntimeError("OpenAI usage disabled by settings")

    model_name = "text-embedding-3-small"
    vector_dim = 1536
    client = _get_openai_client()
    db = await _get_db()

    total_embedded = 0
    async with db.get_session() as session:
        # Fetch chunks without embeddings for this doc
        # We'll do a left join in two queries for simplicity in SQLAlchemy ORM
        chunk_stmt = (
            select(ChunkEntity)
            .where(ChunkEntity.document_id == doc_id)
            .order_by(ChunkEntity.sequence_number.asc())
        )
        res = await session.execute(chunk_stmt)
        chunks: List[ChunkEntity] = list(res.scalars().all())
        if not chunks:
            return {"doc_id": doc_id, "embedded": 0}

        # Preload existing embedding presence
        emb_stmt = select(EmbeddingEntity.chunk_id)
        emb_res = await session.execute(emb_stmt)
        have_emb = set(emb_res.scalars().all())

        # Prepare batches
        bs = batch_size or SETTINGS.PROCESSING.EMBED_BATCH_SIZE
        texts: List[str] = []
        cur_chunks: List[ChunkEntity] = []

        def flush_batch() -> int:
            nonlocal texts, cur_chunks, total_embedded
            if not texts:
                return 0
            resp = client.embeddings.create(model=model_name, input=texts)
            for ch, datum in zip(cur_chunks, resp.data):
                vec = datum.embedding
                # Insert row
                emb = EmbeddingEntity(
                    chunk_id=ch.id,
                    embedding_space="text",
                    embedding=vec,  # pgvector will coerce list[float]
                    model_name=model_name,
                    model_version=None,
                )
                session.add(emb)
            # Commit per batch to avoid large transactions
            # Note: This is inside async context; use await
            # mypy/ruff may flag; but SQLAlchemy AsyncSession supports await commit
            # noinspection PyUnresolvedReferences
            return len(cur_chunks)

        # stream through chunks
        for ch in chunks:
            if ch.id in have_emb:
                continue
            txt = (ch.content_normalized or ch.content or "").strip()
            if not txt:
                continue
            texts.append(txt)
            cur_chunks.append(ch)
            if len(texts) >= bs:
                _ = client.embeddings.create(model=model_name, input=texts)
                resp = _
                for ch_i, datum in zip(cur_chunks, resp.data):
                    emb = EmbeddingEntity(
                        chunk_id=ch_i.id,
                        embedding_space="text",
                        embedding=datum.embedding,
                        model_name=model_name,
                        model_version=None,
                    )
                    session.add(emb)
                await session.commit()
                total_embedded += len(cur_chunks)
                texts, cur_chunks = [], []
        # flush tail
        if texts:
            resp = client.embeddings.create(model=model_name, input=texts)
            for ch_i, datum in zip(cur_chunks, resp.data):
                emb = EmbeddingEntity(
                    chunk_id=ch_i.id,
                    embedding_space="text",
                    embedding=datum.embedding,
                    model_name=model_name,
                    model_version=None,
                )
                session.add(emb)
            await session.commit()
            total_embedded += len(cur_chunks)

    return {
        "doc_id": doc_id,
        "embedded": total_embedded,
        "model": model_name,
        "dim": vector_dim,
    }
