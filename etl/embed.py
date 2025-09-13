"""Chunk embedding with batch processing and rate limiting.

Handles OpenAI embedding generation with proper error handling, retries,
and cost tracking.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List

import structlog
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.ext.asyncio import AsyncSession

from core.settings import SETTINGS
from infra.costs.recorder import record_cost_event

logger = structlog.get_logger("etl.embed")

# Embedding configuration
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 1.0


async def embed_chunks(
    *,
    document_id: str,
    chunks: List[Dict[str, Any]],
    db_session: AsyncSession,
    force: bool = False,
) -> Dict[str, Any]:
    """Generate embeddings for document chunks with batch processing.

    Args:
        document_id: Document UUID
        chunks: List of chunk dictionaries
        db_session: Database session
        force: Skip idempotency check

    Returns:
        Embedding result with chunks and vectors
    """
    start_time = time.time()

    if not chunks:
        return {
            "status": "skipped",
            "reason": "no_chunks",
            "chunks_with_embeddings": [],
            "processing_time": time.time() - start_time,
        }

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # Cost-effective model
        openai_api_key=SETTINGS.OPENAI.OPENAI_API_KEY.get_secret_value(),
    )

    chunks_with_embeddings = []
    total_tokens = 0
    request_id = str(uuid.uuid4())

    try:
        # Process chunks in batches
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            batch_texts = [chunk["content"] for chunk in batch]

            logger.info(
                "Processing embedding batch",
                document_id=document_id,
                batch_start=i,
                batch_size=len(batch),
            )

            # Generate embeddings with retry logic
            batch_embeddings = await _generate_embeddings_with_retry(
                embeddings, batch_texts, max_retries=MAX_RETRIES
            )

            # Combine chunks with their embeddings
            for chunk, embedding in zip(batch, batch_embeddings):
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding["embedding"] = embedding
                chunks_with_embeddings.append(chunk_with_embedding)

            # Estimate token usage (rough approximation)
            batch_tokens = sum(
                len(text.split()) * 1.3 for text in batch_texts
            )  # ~1.3 tokens per word
            total_tokens += int(batch_tokens)

            # Rate limiting between batches
            if i + BATCH_SIZE < len(chunks):
                await asyncio.sleep(0.1)  # Small delay to respect rate limits

        processing_time = time.time() - start_time

        # Record cost event
        try:
            await record_cost_event(
                db_session,
                provider="openai",
                model="text-embedding-3-small",
                route="etl.embed.chunks",
                request_id=request_id,
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens,
                latency_ms=int(processing_time * 1000),
                status="completed",
                metadata={
                    "document_id": document_id,
                    "chunk_count": len(chunks),
                    "batch_count": (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE,
                },
            )
            await db_session.commit()
        except Exception as e:
            logger.warning("Failed to record embedding cost event", error=str(e))

        logger.info(
            "Chunk embedding completed",
            document_id=document_id,
            chunk_count=len(chunks),
            total_tokens=total_tokens,
            processing_time=processing_time,
        )

        return {
            "status": "completed",
            "chunks_with_embeddings": chunks_with_embeddings,
            "chunk_count": len(chunks_with_embeddings),
            "total_tokens": total_tokens,
            "processing_time": processing_time,
        }

    except Exception as e:
        logger.error(
            "Chunk embedding failed",
            document_id=document_id,
            error=str(e),
            request_id=request_id,
        )

        # Record failure cost event
        try:
            await record_cost_event(
                db_session,
                provider="openai",
                model="text-embedding-3-small",
                route="etl.embed.chunks",
                request_id=request_id,
                latency_ms=int((time.time() - start_time) * 1000),
                status="failed",
                metadata={"error": str(e), "document_id": document_id},
            )
            await db_session.commit()
        except Exception:
            pass

        raise


async def _generate_embeddings_with_retry(
    embeddings: OpenAIEmbeddings,
    texts: List[str],
    max_retries: int = 3,
) -> List[List[float]]:
    """Generate embeddings with exponential backoff retry logic."""
    for attempt in range(max_retries + 1):
        try:
            # Use asyncio executor to run sync embedding generation
            vectors = await asyncio.get_event_loop().run_in_executor(
                None, embeddings.embed_documents, texts
            )
            return vectors

        except Exception as e:
            if attempt == max_retries:
                logger.error(
                    "Embedding generation failed after all retries", error=str(e)
                )
                raise

            delay = RETRY_DELAY * (2**attempt)  # Exponential backoff
            logger.warning(
                "Embedding generation failed, retrying",
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=delay,
                error=str(e),
            )
            await asyncio.sleep(delay)

    raise Exception("Embedding generation failed after all retries")
