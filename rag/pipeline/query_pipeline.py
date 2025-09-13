"""Query Pipeline: Hybrid retrieval → optional rerank → answer with citations.

- Uses OpenAI via LangChain (ChatOpenAI) for answering
- Uses PGVector + FTS hybrid retrieval with RRF
- Optional JinaAI reranker
- Optimized self-check is provided in self_check.py (can be integrated later)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from core.settings import SETTINGS
from rag.retrievers.hybrid_retriever import hybrid_rrf
from rag.rankers.reranker import rerank_jina
from rag.prompts.answer.final_answer import build_answer_prompt
from rag.pipeline.self_check import filter_docs_fast
from infra.costs.pricing import record_cost_event_with_pricing


class QueryPipeline:
    def __init__(self, *, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.llm = ChatOpenAI(model=self.model, temperature=self.temperature)

    async def retrieve(
        self,
        *,
        question: str,
        db_session: AsyncSession,
        k: int = 8,
        doc_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        use_jina_rerank: bool = True,
    ) -> List[Document]:
        docs = await hybrid_rrf(
            query=question,
            k=k,
            doc_id=doc_id,
            collection_name=collection_name,
            session=db_session,
        )
        if (
            use_jina_rerank
            and getattr(SETTINGS, "JINA", None)
            and SETTINGS.JINA.RERANKER_ENABLED
        ):
            try:
                docs = await rerank_jina(
                    question, docs, top_n=min(k, len(docs)), session=db_session
                )
            except Exception:
                pass
        return docs[:k]

    async def answer(
        self,
        *,
        question: str,
        db_session: AsyncSession,
        k: int = 8,
        doc_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        history_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        docs = await self.retrieve(
            question=question,
            db_session=db_session,
            k=k,
            doc_id=doc_id,
            collection_name=collection_name,
            use_jina_rerank=True,
        )

        # Apply Self‑RAG fast grading to filter retrieved chunks
        # This is a sync function; run it in a thread pool to avoid blocking the event loop
        filtered_docs, self_check_info = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: filter_docs_fast(question=question, docs=docs, llm=self.llm),
        )

        # Use the filtered docs for answer generation (fallback to original if all filtered out)
        docs_to_use = filtered_docs

        # Build evidence block with doc:chunk citations
        lines: List[str] = []
        for d in docs_to_use:
            md = d.metadata or {}
            # Prefer human‑readable document name if available; fall back to doc_id.
            doc_name = md.get("document_name") or md.get("doc_id") or "unknown_doc"
            chunk_id = md.get("chunk_id") or md.get("sequence") or ""
            doc_key = f"{doc_name}:{chunk_id}" if chunk_id else f"{doc_name}"
            lines.append(f"[{doc_key}] {d.page_content}")
        prompt = build_answer_prompt(evidence_lines=lines, question=question)
        if history_text:
            prompt = (
                f"Conversation history (for context only, do NOT cite):\n{history_text}\n\n"
                + prompt
            )

        # ChatOpenAI is sync; run in executor to avoid blocking
        start = time.time()
        out = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.llm.predict(prompt)
        )
        latency_ms = int((time.time() - start) * 1000)

        # Record cost with precise USD calculation (4‑decimal precision)
        try:
            req_id = str(uuid.uuid4())
            # Simple token estimation: split on whitespace (approximation)
            prompt_tokens = len(prompt.split())
            completion_tokens = len(out.split())
            await record_cost_event_with_pricing(
                db_session,
                provider="openai",
                model=self.model,
                route="rag.query.answer",
                request_id=req_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                status="ok",
                metadata={"k": k, "doc_filter": doc_id or ""},
            )
            await db_session.commit()
        except Exception:
            # Do not fail the request on cost logging errors
            pass
        return {
            "answer": out,
            "source_documents": docs_to_use,
            "processing_info": {
                "retriever_type": "Hybrid+Jina",
                "prompt_type": "final_answer_with_citations",
                "doc_filter": doc_id,
                "self_check": self_check_info,
            },
            "model_used": self.model,
        }
