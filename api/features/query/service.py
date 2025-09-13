"""
LangChain-powered query service for the Query feature.

This module consolidates the LangChain retrieval + LLM answering logic and
exposes simple async functions for controllers to call.

Also retains a minimal `QueryService` class for suggestions to satisfy DI.
"""
import asyncio
import logging
import time
import json
import re
from collections import defaultdict
from hashlib import md5
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional

import structlog
import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from api.features.query.exceptions import SearchError
from core.settings import SETTINGS

# LangChain / Retrieval imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from typing import Dict as _DictAlias  # avoid confusion in type hints

logger = logging.getLogger("rag.query.service")
lc_logger = structlog.get_logger("rag.langchain")


def timing_decorator(func):
    """Decorator to add performance timing to async functions for production monitoring."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            end_time = time.time()
            processing_time_ms = (end_time - start_time) * 1000

            if isinstance(result, dict):
                result["performance_metrics"] = {
                    "processing_time_ms": processing_time_ms,
                    "timestamp": int(time.time()),
                    "function": func.__name__,
                }
            lc_logger.info(
                f"{func.__name__}_performance",
                processing_time_ms=processing_time_ms,
                function=func.__name__,
            )
            return result
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            lc_logger.error(
                f"{func.__name__}_error",
                processing_time_ms=processing_time_ms,
                error=str(e),
            )
            raise

    return wrapper


def calculate_confidence_score(
    answer: str, source_docs: List[Document], query: str
) -> float:
    """Heuristic confidence score for observability."""
    try:
        base_score = 0.5
        source_count = len(source_docs) if source_docs else 0
        source_factor = min(source_count / 5.0, 1.0) * 0.2
        answer_length = len(answer.split()) if answer else 0
        length_factor = (
            0.2
            if 20 <= answer_length <= 200
            else 0.1
            if 5 <= answer_length < 10 or 200 < answer_length <= 300
            else 0.0
        )
        uncertainty_phrases = [
            "i don't know",
            "not sure",
            "unclear",
            "maybe",
            "possibly",
        ]
        uncertainty_penalty = 0.0
        if answer:
            al = answer.lower()
            for p in uncertainty_phrases:
                if p in al:
                    uncertainty_penalty += 0.1
        confidence = base_score + source_factor + length_factor - uncertainty_penalty
        return max(0.1, min(1.0, confidence))
    except Exception:
        return 0.5


class SearchType(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    SELF_QUERY = "self_query"


class LangChainQueryService:
    """Production-ready LangChain query service for RAG applications."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            openai_api_key=SETTINGS.OPENAI.OPENAI_API_KEY.get_secret_value(),
            max_tokens=1000,
        )
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=SETTINGS.OPENAI.OPENAI_API_KEY.get_secret_value(),
            dimensions=1536,
        )
        self.connection_string = (
            f"postgresql+psycopg://{SETTINGS.DATABASE.POSTGRES_USER}:"
            f"{SETTINGS.DATABASE.POSTGRES_PASSWORD.get_secret_value()}@{SETTINGS.DATABASE.POSTGRES_HOST}:"
            f"{SETTINGS.DATABASE.POSTGRES_PORT}/{SETTINGS.DATABASE.POSTGRES_DB}"
        )
        self.qa_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an expert assistant for question-answering tasks.\n"
                "Use the following pieces of retrieved context to answer the question.\n"
                "If you don't know the answer, just say that you don't know.\n"
                "Use three sentences maximum and keep the answer concise.\n\n"
                "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            ),
        )
        self.detailed_qa_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an expert research assistant. Use the following retrieved context to provide a comprehensive, well-structured answer to the question.\n\n"
                "Instructions:\n- Use only information from the provided context\n- Structure your response with clear sections if appropriate\n- Include relevant details and examples from the context\n- If the context doesn't contain enough information, clearly state what's missing\n- Cite specific parts of the context when making claims\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\nComprehensive Answer:"
            ),
        )
        self.metadata_field_info = [
            {
                "name": "doc_id",
                "description": "The document ID that the chunk belongs to",
                "type": "string",
            },
            {
                "name": "page_number",
                "description": "The page number in the original document",
                "type": "integer",
            },
            {
                "name": "extraction_method",
                "description": "Method used to extract the text (PyPDFLoader, UnstructuredPDFLoader)",
                "type": "string",
            },
            {
                "name": "processing_pipeline",
                "description": "Processing pipeline used (langchain, legacy)",
                "type": "string",
            },
        ]
        self.document_content_description = (
            "Document chunks from processed PDFs containing text content"
        )

    # ---------------- Self-RAG Lite helpers ----------------
    def _safe_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except Exception:
            # Try to extract the first JSON object/array present
            try:
                m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
                if m:
                    return json.loads(m.group(1))
            except Exception:
                return None
        return None

    def _doc_key(self, d: Document) -> str:
        md = d.metadata or {}
        cid = md.get("chunk_id") or md.get("id") or ""
        did = md.get("doc_id") or ""
        if cid or did:
            return f"{did}:{cid}"
        # fallback to content hash
        return md5((d.page_content or "").encode("utf-8")).hexdigest()

    async def _classify_retrieval_needed(self, question: str) -> Dict[str, Any]:
        prompt = (
            "Decide if external retrieval is needed to answer the question factually.\n"
            'Return JSON only: {"retrieval_needed": true|false, "reason": "short"}.\n'
            f"Question: {question}\n"
        )
        out = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.llm.predict(prompt)
        )
        js = self._safe_json(out) or {}
        needed = bool(js.get("retrieval_needed", True))
        return {"retrieval_needed": needed, "raw": out, "parsed": js}

    async def _grade_document(self, question: str, doc: Document) -> Dict[str, Any]:
        # Truncate for safety
        text = (doc.page_content or "")[:2000]
        prompt = (
            "You are grading a document chunk for answering a question.\n"
            "Grade in [0..1] the following: relevance, supportiveness, usefulness.\n"
            'JSON only: {"relevance": n, "supportiveness": n, "usefulness": n, "rationale": "short"}.\n'
            f"Question: {question}\nChunk: {text}\n"
        )
        out = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.llm.predict(prompt)
        )
        js = self._safe_json(out) or {}
        rel = float(js.get("relevance", 0))
        sup = float(js.get("supportiveness", 0))
        use = float(js.get("usefulness", 0))
        score = rel + sup + 0.5 * use
        return {
            "doc": doc,
            "relevance": rel,
            "supportiveness": sup,
            "usefulness": use,
            "score": score,
            "raw": out,
        }

    async def _grade_documents(
        self, question: str, docs: List[Document]
    ) -> List[Dict[str, Any]]:
        graded: List[Dict[str, Any]] = []
        for d in docs:
            try:
                g = await self._grade_document(question, d)
                graded.append(g)
            except Exception as e:
                lc_logger.warning("grading_failed", error=str(e))
        graded.sort(key=lambda x: x.get("score", 0), reverse=True)
        return graded

    def _select_evidence(self, graded: List[Dict[str, Any]]) -> List[Document]:
        top_m = SETTINGS.SELF_RAG.TOP_M
        rel_th = SETTINGS.SELF_RAG.RELEVANCE_THRESHOLD
        sup_th = SETTINGS.SELF_RAG.SUPPORT_THRESHOLD
        selected: List[Document] = []
        for g in graded:
            if g["relevance"] >= rel_th and g["supportiveness"] >= sup_th:
                selected.append(g["doc"])
            if len(selected) >= top_m:
                break
        # Ensure at least 2 if available
        if len(selected) < min(2, len(graded)):
            selected = [
                x["doc"] for x in graded[: min(SETTINGS.SELF_RAG.TOP_M, len(graded))]
            ]
        return selected

    async def _generate_query_variants(self, question: str, n: int) -> List[str]:
        prompt = (
            "Generate distinct alternative search queries (no numbering).\n"
            f"Return JSON list of {n} strings covering different phrasings/aspects of: {question}\n"
        )
        out = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.llm.predict(prompt)
        )
        js = self._safe_json(out)
        if isinstance(js, list) and js:
            return [str(x)[:200] for x in js][:n]
        return [question]

    async def _hyde_query(self, question: str) -> str:
        prompt = (
            "Write a short (3-5 sentences) hypothetical but plausible answer to the question.\n"
            "Do not fabricate specifics; keep generic structure.\n"
            f"Question: {question}\n"
            'Then output a JSON object: {"keywords": ['
            """comma-separated keywords"""
            "]}."
        )
        out = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.llm.predict(prompt)
        )
        js = self._safe_json(out) or {}
        kws = js.get("keywords") or []
        if isinstance(kws, list) and kws:
            return ", ".join([str(k) for k in kws][:12])
        return question

    async def _rrf_merge(
        self,
        queries: List[str],
        k: int,
        doc_id: Optional[str],
        collection_name: Optional[str],
    ) -> List[Document]:
        # Retrieve per query and merge using Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Document] = {}
        for q in queries:
            retriever = await self._get_hybrid_retriever(
                q, doc_id=doc_id, k=k, collection_name=collection_name
            )
            try:
                docs = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: retriever.get_relevant_documents(q)
                )
            except Exception:
                docs = []
            for rank, d in enumerate(docs):
                key = self._doc_key(d)
                rrf_scores[key] += 1.0 / (60.0 + rank)
                if key not in doc_map:
                    doc_map[key] = d
        ranked = sorted(
            doc_map.items(), key=lambda x: rrf_scores.get(x[0], 0), reverse=True
        )
        return [doc for key, doc in ranked[:k]]

    async def _jina_rerank(
        self, question: str, docs: List[Document], top_n: int
    ) -> List[Document]:
        """Call Jina Reranker API to reorder documents by cross-encoder relevance.

        Falls back to original order on error or when not configured.
        """
        try:
            api_key = (
                SETTINGS.JINA.JINA_API_KEY.get_secret_value()
                if hasattr(SETTINGS.JINA, "JINA_API_KEY")
                else ""
            )
        except Exception:
            api_key = ""
        if not SETTINGS.JINA.RERANKER_ENABLED or not api_key or not docs:
            return docs
        try:
            payload = {
                "model": SETTINGS.JINA.RERANKER_MODEL,
                "query": question,
                "top_n": max(1, min(int(top_n), len(docs))),
                "documents": [(d.page_content or "")[:4000] for d in docs],
                "return_documents": False,
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.jina.ai/v1/rerank", headers=headers, json=payload
                )
                resp.raise_for_status()
                data = resp.json()
            results = data.get("results") or data.get("data") or []
            ranking: List[tuple[int, float]] = []
            for item in results:
                try:
                    idx = int(item.get("index"))
                    score = float(
                        item.get("relevance_score") or item.get("score") or 0.0
                    )
                    if 0 <= idx < len(docs):
                        ranking.append((idx, score))
                except Exception:
                    continue
            if not ranking:
                return docs
            ranking.sort(key=lambda x: x[1], reverse=True)
            ordered = [docs[i] for i, _ in ranking]
            return ordered + [d for d in docs if d not in ordered]
        except Exception as e:
            lc_logger.warning("jina_rerank_failed", error=str(e))
            return docs

    async def _generate_answer_with_citations(
        self, question: str, evidence: List[Document]
    ) -> str:
        lines = []
        for d in evidence:
            md = d.metadata or {}
            did = md.get("doc_id", "unknown")
            cid = md.get("chunk_id") or md.get("id") or md.get("sequence") or "0"
            lines.append(f"- ({did}:{cid}) {d.page_content[:800]}")
        prompt = (
            "Use ONLY the evidence to answer. Every factual sentence MUST include citations like [doc_id:chunk_id].\n"
            "If evidence is insufficient, say so.\n\n"
            "Evidence:\n"
            + "\n".join(lines)
            + "\n\n"
            + f"Question: {question}\n\nFinal answer with citations:"
        )
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.llm.predict(prompt)
        )

    async def _chain_of_verification(
        self, question: str, answer: str, evidence: List[Document]
    ) -> Dict[str, Any]:
        lines = []
        for d in evidence:
            md = d.metadata or {}
            did = md.get("doc_id", "unknown")
            cid = md.get("chunk_id") or md.get("id") or md.get("sequence") or "0"
            lines.append(f"- ({did}:{cid}) {d.page_content[:600]}")
        prompt = (
            "Extract up to 6 atomic factual claims from the answer; verify each against the evidence.\n"
            "Label each as Verified/Unsupported/Contradicted and list supporting citations.\n"
            'Return JSON: {"claims":[{"text":...,"label":...,"evidence":["doc:chunk"]}],"needs_revision":bool,"notes":"..."}.\n\n'
            f"Question: {question}\nAnswer: {answer}\n\nEvidence:\n" + "\n".join(lines)
        )
        raw = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.llm.predict(prompt)
        )
        parsed = self._safe_json(raw) or {
            "claims": [],
            "needs_revision": False,
            "notes": "",
        }
        return {"raw": raw, "parsed": parsed}

    async def self_rag_answer(
        self,
        question: str,
        doc_id: Optional[str] = None,
        k: int = 5,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {"self_rag": True}
        # Step 0: classify retrieval need
        try:
            cls = await self._classify_retrieval_needed(question)
        except Exception:
            cls = {"retrieval_needed": True}
        diagnostics["retrieval_needed"] = cls
        if not cls.get("retrieval_needed", True):
            # Direct short answer, low confidence cap
            direct = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm.predict(f"Answer concisely: {question}")
            )
            return {
                "answer": direct,
                "question": question,
                "source_documents": [],
                "processing_info": {
                    "retriever_type": "none",
                    "prompt_type": "direct",
                    "doc_filter": doc_id,
                },
                "retrieval_diagnostics": diagnostics,
                "model_used": "gpt-4-turbo-preview",
            }

        # Step 1: initial hybrid retrieval (retrieve more if reranker enabled)
        retrieve_k = max(k, SETTINGS.SELF_RAG.TOP_K)
        if getattr(SETTINGS, "JINA", None) and SETTINGS.JINA.RERANKER_ENABLED:
            retrieve_k = max(retrieve_k, 20)
        retriever = await self._get_hybrid_retriever(
            question, doc_id=doc_id, k=retrieve_k, collection_name=collection_name
        )
        try:
            initial_docs = await asyncio.get_event_loop().run_in_executor(
                None, lambda: retriever.get_relevant_documents(question)
            )
        except Exception:
            initial_docs = []
        diagnostics["initial_docs_count"] = len(initial_docs)
        # Optional: Jina rerank before grading
        used_jina = False
        if getattr(SETTINGS, "JINA", None) and SETTINGS.JINA.RERANKER_ENABLED:
            try:
                initial_docs = await self._jina_rerank(
                    question,
                    initial_docs,
                    top_n=min(SETTINGS.SELF_RAG.TOP_K * 2, len(initial_docs)),
                )
                used_jina = True
            except Exception:
                used_jina = False
        diagnostics["jina_reranker_used"] = used_jina
        if used_jina:
            diagnostics["jina_model"] = SETTINGS.JINA.RERANKER_MODEL

        # Step 2: grade and select evidence
        graded = await self._grade_documents(
            question, initial_docs[: SETTINGS.SELF_RAG.TOP_K]
        )
        selected = self._select_evidence(graded)
        diagnostics["grades"] = [
            {
                "relevance": g["relevance"],
                "supportiveness": g["supportiveness"],
                "usefulness": g["usefulness"],
                "key": self._doc_key(g["doc"]),
            }
            for g in graded
        ]
        diagnostics["selected_count"] = len(selected)

        # Step 3: if weak coverage -> generate query variants and HyDE rewrite, then RRF merge
        need_rewrite = len(selected) < 2
        if need_rewrite and SETTINGS.SELF_RAG.MAX_REWRITE > 0:
            qvars = await self._generate_query_variants(
                question, SETTINGS.SELF_RAG.RAG_FUSION_QUERIES
            )
            hyde_q = await self._hyde_query(question)
            all_q = list(dict.fromkeys([question] + qvars + [hyde_q]))
            diagnostics["query_variants"] = all_q
            merged = await self._rrf_merge(
                all_q,
                k=max(k, SETTINGS.SELF_RAG.TOP_K),
                doc_id=doc_id,
                collection_name=collection_name,
            )
            if getattr(SETTINGS, "JINA", None) and SETTINGS.JINA.RERANKER_ENABLED:
                try:
                    merged = await self._jina_rerank(
                        question,
                        merged,
                        top_n=min(SETTINGS.SELF_RAG.TOP_K * 2, len(merged)),
                    )
                except Exception:
                    pass
            graded2 = await self._grade_documents(question, merged)
            selected = self._select_evidence(graded2)
            diagnostics["selected_count_after_rewrite"] = len(selected)

        # Step 4: answer with citations
        answer = await self._generate_answer_with_citations(question, selected)

        # Step 5: chain-of-verification and possible revision
        cov = await self._chain_of_verification(question, answer, selected)
        verification = cov.get("parsed", {})
        if verification.get("needs_revision"):
            # one revision loop
            rev_prompt = (
                "Revise the answer to ensure every claim is supported by the evidence.\n"
                "Keep citations [doc:chunk] and remove unsupported claims.\n"
                f"Question: {question}\nOriginal Answer: {answer}\nVerification: {json.dumps(verification)[:1500]}\n"
            )
            answer = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm.predict(rev_prompt)
            )

        response = {
            "answer": answer,
            "question": question,
            "search_type": "hybrid",
            "source_count": len(selected),
            "source_documents": selected,
            "processing_info": {
                "retriever_type": "SelfRAG-Hybrid+Jina"
                if used_jina
                else "SelfRAG-Hybrid",
                "chunks_retrieved": len(selected),
                "prompt_type": "self_rag",
                "doc_filter": doc_id,
            },
            "verification_report": verification,
            "retrieval_diagnostics": diagnostics,
            "model_used": "gpt-4-turbo-preview",
        }
        return response

    def _get_vector_store(self, collection_name: Optional[str] = None) -> PGVector:
        if not collection_name:
            collection_name = SETTINGS.VECTOR.COLLECTION_NAME
        return PGVector(
            embeddings=self.embeddings,
            connection=self.connection_string,
            collection_name=collection_name,
            use_jsonb=SETTINGS.VECTOR.VECTOR_USE_JSONB,
        )

    async def _get_hybrid_retriever(
        self,
        query: str,
        doc_id: Optional[str] = None,
        k: int = 5,
        collection_name: Optional[str] = None,
    ) -> EnsembleRetriever:
        vector_store = self._get_vector_store(collection_name)
        try:
            lc_logger.info(
                "hybrid_retriever.collection",
                collection=collection_name or SETTINGS.VECTOR.COLLECTION_NAME,
            )
        except Exception:
            pass
        search_kwargs: _DictAlias[str, Any] = {"k": k}
        if doc_id:
            search_kwargs["filter"] = {"doc_id": doc_id}
        vector_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
        all_docs = await asyncio.get_event_loop().run_in_executor(
            None, lambda: vector_store.similarity_search(query, k=max(k * 20, 200))
        )
        if doc_id:
            all_docs = [d for d in all_docs if d.metadata.get("doc_id") == doc_id]
        if not all_docs:
            lc_logger.info(
                "BM25 corpus empty, falling back to vector-only retriever", query=query
            )
            return vector_retriever
        try:
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = k
        except Exception as e:
            lc_logger.warning(
                "BM25 retriever creation failed, using vector-only retriever",
                error=str(e),
            )
            return vector_retriever
        return EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever], weights=[0.6, 0.4]
        )

    async def semantic_search(
        self,
        query: str,
        doc_id: Optional[str] = None,
        k: int = 5,
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        try:
            # Initialize vector store (defensive: catch engine issues)
            try:
                vector_store = self._get_vector_store(collection_name)
            except Exception as e:
                lc_logger.error("vector_store_init_failed", error=str(e))
                raise
            try:
                lc_logger.info(
                    "semantic_search.collection",
                    collection=collection_name or SETTINGS.VECTOR.COLLECTION_NAME,
                )
            except Exception:
                pass
            search_kwargs: _DictAlias[str, Any] = {"k": k}
            if doc_id:
                search_kwargs["filter"] = {"doc_id": doc_id}
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: vector_store.similarity_search_with_score(
                    query, **search_kwargs
                ),
            )
            docs: List[Document] = []
            for doc, distance in results:
                try:
                    md = dict(doc.metadata or {})
                    d = float(distance)
                    sim = 1.0 - d if 0.0 <= d <= 1.0 else 1.0 / (1.0 + max(d, 0.0))
                    md["distance"], md["score"] = d, round(sim, 6)
                    doc.metadata = md
                except Exception:
                    pass
                docs.append(doc)
            lc_logger.info("semantic_search.results", count=len(docs), k=k)
            return docs
        except Exception as e:
            lc_logger.error(f"Semantic search failed: {e}", query=query, doc_id=doc_id)
            return []

    async def answer_question(
        self,
        question: str,
        doc_id: Optional[str] = None,
        search_type: SearchType = SearchType.SEMANTIC,
        detailed: bool = False,
        k: int = 5,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            vector_store = self._get_vector_store(collection_name)
            if search_type == SearchType.HYBRID:
                retriever = await self._get_hybrid_retriever(
                    query=question, doc_id=doc_id, k=k, collection_name=collection_name
                )
            elif search_type == SearchType.SELF_QUERY:
                retriever = SelfQueryRetriever.from_llm(
                    llm=self.llm,
                    vectorstore=vector_store,
                    document_contents=self.document_content_description,
                    metadata_field_info=self.metadata_field_info,
                    search_kwargs={"k": k},
                )
            else:
                skw = {"k": k}
                if doc_id:
                    skw["filter"] = {"doc_id": doc_id}
                retriever = vector_store.as_retriever(search_kwargs=skw)

            prompt_template = (
                self.detailed_qa_prompt_template
                if detailed
                else self.qa_prompt_template
            )
            # We call LLM directly with formatted prompt to stay compatible across LC versions
            # First, ensure we have candidate source docs
            try:
                # Prefer synchronous call in thread to avoid async engine issues
                source_docs = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: retriever.get_relevant_documents(question),
                )
            except Exception as e:
                lc_logger.error("retriever_invoke_failed", error=str(e))
                source_docs = []
            if not source_docs:
                # fallback to semantic
                fallback_scored = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: vector_store.similarity_search_with_score(question, k=k),
                )
                source_docs = [d for (d, _s) in fallback_scored] or []
                for d, dist in fallback_scored:
                    try:
                        md = dict(d.metadata or {})
                        dd = float(dist)
                        sim = (
                            1.0 - dd if 0.0 <= dd <= 1.0 else 1.0 / (1.0 + max(dd, 0.0))
                        )
                        md["distance"], md["score"] = dd, round(sim, 6)
                        d.metadata = md
                    except Exception:
                        pass
            else:
                # Re-score to attach semantic scores
                rescored = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: vector_store.similarity_search_with_score(
                        question, k=len(source_docs)
                    ),
                )
                for idx, (d, dist) in enumerate(rescored):
                    if idx < len(source_docs):
                        try:
                            md = dict(source_docs[idx].metadata or {})
                            dd = float(dist)
                            sim = (
                                1.0 - dd
                                if 0.0 <= dd <= 1.0
                                else 1.0 / (1.0 + max(dd, 0.0))
                            )
                            md["distance"], md["score"] = dd, round(sim, 6)
                            source_docs[idx].metadata = md
                        except Exception:
                            pass

            context = "\n\n".join([doc.page_content for doc in source_docs])
            answer = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.predict(
                    prompt_template.format(context=context, question=question)
                ),
            )
            response = {
                "answer": answer,
                "question": question,
                "search_type": search_type.value,
                "source_count": len(source_docs),
                "source_documents": source_docs,
                "processing_info": {
                    "retriever_type": type(retriever).__name__,
                    "chunks_retrieved": k,
                    "prompt_type": "detailed" if detailed else "concise",
                    "doc_filter": doc_id,
                },
            }
            lc_logger.info(
                "answer_question.success",
                question=question,
                source_count=len(source_docs),
                search_type=search_type.value,
            )
            return response
        except Exception as e:
            lc_logger.error(f"Question answering failed: {e}", question=question)
            return {
                "answer": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "question": question,
                "error": str(e),
                "search_type": search_type.value,
                "source_count": 0,
                "sources": [],
            }


# Singleton instance and module-level functions
_query_service_impl = LangChainQueryService()


@timing_decorator
async def query_documents(
    question: str,
    doc_id: Optional[str] = None,
    search_type: str = "semantic",
    detailed: bool = False,
    k: int = 5,
) -> Dict[str, Any]:
    # When detailed=True and Self-RAG is enabled, use the Self-RAG Lite flow.
    if SETTINGS.SELF_RAG.SELF_RAG_ENABLED and detailed:
        result = await _query_service_impl.self_rag_answer(
            question=question,
            doc_id=doc_id,
            k=k,
            collection_name=None,
        )
    else:
        result = await _query_service_impl.answer_question(
            question=question,
            doc_id=doc_id,
            search_type=SearchType(search_type),
            detailed=detailed,
            k=k,
        )
    if isinstance(result, dict) and "answer" in result:
        confidence_score = calculate_confidence_score(
            answer=result.get("answer", ""),
            source_docs=result.get("source_documents", []),
            query=question,
        )
        result["confidence_score"] = confidence_score
    return result


@timing_decorator
async def query_multiple_documents(
    question: str, doc_ids: List[str], search_type: str = "semantic", k_per_doc: int = 3
) -> Dict[str, Any]:
    # Simple fan-in by aggregating semantic results per doc, then answering
    all_docs: List[Document] = []
    for did in doc_ids:
        docs = await _query_service_impl.semantic_search(
            query=question, doc_id=did, k=k_per_doc
        )
        all_docs.extend(docs)
    if not all_docs:
        return {
            "answer": "No relevant information found in the specified documents.",
            "question": question,
            "doc_ids": doc_ids,
            "source_count": 0,
            "sources": [],
        }
    context = "\n\n".join([doc.page_content for doc in all_docs])
    answer = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _query_service_impl.llm.predict(
            _query_service_impl.detailed_qa_prompt_template.format(
                context=context, question=question
            )
        ),
    )
    result = {
        "answer": answer,
        "question": question,
        "search_type": search_type,
        "doc_ids": doc_ids,
        "source_count": len(all_docs),
        "source_documents": all_docs,
    }
    confidence_score = calculate_confidence_score(
        answer=result.get("answer", ""), source_docs=all_docs, query=question
    )
    result["confidence_score"] = confidence_score
    return result


@timing_decorator
async def semantic_search_documents(
    query: str, doc_id: Optional[str] = None, k: int = 5
) -> List[Dict[str, Any]]:
    docs = await _query_service_impl.semantic_search(query=query, doc_id=doc_id, k=k)
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "doc_id": doc.metadata.get("doc_id"),
            "page_number": doc.metadata.get("page_number"),
        }
        for doc in docs
    ]


class QueryService:
    """Minimal QueryService: only suggestions are handled here."""

    async def get_query_suggestions(
        self, prefix: str = "", limit: int = 10, db_session: AsyncSession = None
    ) -> List[str]:
        """Get query suggestions based on prefix and popular queries."""
        try:
            suggestions = [
                "What was the revenue in Q1?",
                "How did the company perform compared to last year?",
                "What are the key risks mentioned?",
                "What is the guidance for next quarter?",
                "What were the main highlights?",
                "How much cash does the company have?",
                "What are the growth drivers?",
                "What challenges does the company face?",
            ]
            if prefix:
                suggestions = [
                    s for s in suggestions if s.lower().startswith(prefix.lower())
                ]
            return suggestions[:limit]
        except Exception as e:
            logger.error(f"Failed to get query suggestions: {str(e)}")
            raise SearchError(f"Failed to get suggestions: {str(e)}")
