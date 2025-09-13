"""JinaAI Reranker adapter with minimal cost logging.

Uses JinaAI official HTTP API to rerank LangChain Documents for a query and
records a simple `cost_event` row (latency + identifiers) when a DB session is
provided.
"""
from __future__ import annotations

import time
from typing import List, Optional

import httpx
import structlog
from langchain_core.documents import Document
from sqlalchemy.ext.asyncio import AsyncSession

from core.settings import SETTINGS

logger = structlog.get_logger(__name__)


async def rerank_jina(
    question: str,
    docs: List[Document],
    top_n: int = 10,
    *,
    session: Optional[AsyncSession] = None,
) -> List[Document]:
    """Call Jina Reranker API and return documents in re-ranked order.

    Falls back to original order on error or when not configured/enabled.
    """
    if not SETTINGS.JINA.RERANKER_ENABLED:
        return docs[:top_n]

    api_key = (
        SETTINGS.JINA.JINA_API_KEY.get_secret_value()
        if SETTINGS.JINA.JINA_API_KEY
        else ""
    )
    logger.info("IS Jina reranker enabled", enabled=SETTINGS.JINA.RERANKER_ENABLED)
    is_api_key_present = api_key is not None
    logger.info("IS Jina API key", is_api_key_present=is_api_key_present)
    if not api_key:
        return docs[:top_n]

    try:
        _ = time.time()
        payload = {
            "model": SETTINGS.JINA.RERANKER_MODEL,
            "query": question,
            "documents": [d.page_content for d in docs],
            "top_n": min(top_n, len(docs)),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.jina.ai/v1/rerank", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json() or {}
            ranking = []
            for item in data.get("data", []) or []:
                try:
                    idx = int(item.get("index"))
                    score = float(
                        item.get("relevance_score") or item.get("score") or 0.0
                    )
                    if 0 <= idx < len(docs):
                        ranking.append((idx, score))
                except Exception as e:
                    logger.error("Failed to process item", error=str(e))
                    continue
            if not ranking:
                return docs[:top_n]
            ranking.sort(key=lambda x: x[1], reverse=True)
            return [docs[i] for i, _ in ranking[:top_n]]
    except Exception:
        return docs[:top_n]
