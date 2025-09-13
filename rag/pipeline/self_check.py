"""Self-RAG Lite: fast chunk grading with compact JSON prompt.

This module implements a latency-optimized grading pass over a small top subset
of retrieved documents and filters by relevance/support thresholds.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from rag.prompts.grading.grade_chunk import build_grade_prompt

logger = logging.getLogger(__name__)


def _safe_parse_scores(text: str) -> Dict[str, float]:
    try:
        data = json.loads(text)
        return {
            "relevance": float(data.get("relevance", 0.0)),
            "supportiveness": float(data.get("supportiveness", 0.0)),
            "usefulness": float(data.get("usefulness", 0.0)),
        }
    except Exception:
        return {"relevance": 0.0, "supportiveness": 0.0, "usefulness": 0.0}


def _score_doc(scores: Dict[str, float]) -> float:
    # Weighted sum; can be tuned via settings later
    return (
        0.6 * scores["relevance"]
        + 0.3 * scores["supportiveness"]
        + 0.1 * scores["usefulness"]
    )


def summarize_docs(
    docs: List[Document], scores: List[Dict[str, float]]
) -> Dict[str, Any]:
    top = [
        {
            "chunk_id": d.metadata.get("chunk_id") or d.metadata.get("id"),
            "doc_id": d.metadata.get("doc_id"),
            "score": _score_doc(s),
            "r": s.get("relevance", 0.0),
            "s": s.get("supportiveness", 0.0),
            "u": s.get("usefulness", 0.0),
        }
        for d, s in zip(docs, scores)
    ]
    return {"graded": top}


def filter_docs_fast(
    *,
    question: str,
    docs: List[Document],
    llm,
    top_m: int = 5,
    relevance_threshold: float = 0.6,
    support_threshold: float = 0.5,
) -> Tuple[List[Document], Dict[str, Any]]:
    """Grade up to top_m docs and filter.

    ChatOpenAI is sync; call predict() sequentially with short prompts for reliability.
    """
    logger.info("filter_docs_fast called with %d docs, top_m=%d", len(docs), top_m)
    if not docs:
        return [], {"graded": []}

    candidates = docs[:top_m]
    logger.info("Selected %d candidate docs for grading", len(candidates))
    scores: List[Dict[str, float]] = []
    for d in candidates:
        prompt = build_grade_prompt(question=question, chunk=d.page_content[:2000])
        try:
            out = llm.predict(prompt)
            logger.debug(
                "LLM output for doc %s: %s",
                d.metadata.get("chunk_id") or d.metadata.get("id"),
                out,
            )
            s = _safe_parse_scores(out)
        except Exception:
            logger.exception(
                "Error during LLM prediction for doc %s",
                d.metadata.get("chunk_id") or d.metadata.get("id"),
            )
            s = {"relevance": 0.0, "supportiveness": 0.0, "usefulness": 0.0}
        scores.append(s)
    logger.debug("Collected scores: %s", scores)

    # Filter by thresholds
    filtered: List[Document] = []
    for d, s in zip(candidates, scores):
        if (
            s.get("relevance", 0.0) >= relevance_threshold
            and s.get("supportiveness", 0.0) >= support_threshold
        ):
            filtered.append(d)
    logger.info("Filtered docs count after thresholds: %d", len(filtered))

    # If everything filtered out, fall back to original top_k
    final_docs = filtered if filtered else candidates
    logger.info(
        "Returning %d final docs (fallback used: %s)",
        len(final_docs),
        "yes" if not filtered else "no",
    )

    diagnostics = summarize_docs(candidates, scores)
    return final_docs, diagnostics
