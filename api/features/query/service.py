"""Service layer for the Query feature using LangChain retrievers and PGVector."""
import logging
import time
from typing import List, Optional, Tuple, Dict
from uuid import UUID
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.features.query.exceptions import (
    SearchError,
    AnswerGenerationError,
    InsufficientContextError,
)
from api.features.query.models import (
    ChunkModel,
    SearchResultModel,
    QueryContextModel,
    AnswerModel,
)
from api.features.query.entities.chunk import Chunk as ChunkEntity
from api.features.query.retrievers import vector_search, fts_search

logger = logging.getLogger("rag.query.service")


class QueryService:
    """Service for query and search operations."""

    def __init__(self, embedding_client=None, llm_client=None):
        # embedding_client is unused now; using LangChain inside retrievers.
        self.embedding_client = embedding_client
        self.llm_client = llm_client

    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        document_filters: Optional[List[UUID]] = None,
        db_session: AsyncSession = None,
        collection: str = "rag_chunks_v1",
    ) -> QueryContextModel:
        """Semantic search via LangChain PGVector."""
        start_time = time.time()
        try:
            pairs = vector_search(query, top_k=top_k, collection=collection)
            # Extract chunk_ids from metadata and batch-load entities
            chunk_ids: List[str] = []
            raw_items: List[Tuple[Dict, float]] = []
            for doc, score in pairs:
                meta = doc.metadata or {}
                cid = meta.get("chunk_id")
                if not cid:
                    continue
                if document_filters and meta.get("doc_id") not in {
                    str(d) for d in document_filters
                }:
                    continue
                chunk_ids.append(cid)
                raw_items.append((meta, float(score)))
            ent_map: Dict[str, ChunkEntity] = {}
            if chunk_ids and db_session is not None:
                q = select(ChunkEntity).where(ChunkEntity.id.in_(chunk_ids))
                res = await db_session.execute(q)
                for ent in res.scalars().all():
                    ent_map[str(ent.id)] = ent
            # Build results
            results: List[SearchResultModel] = []
            for rank, (meta, score) in enumerate(raw_items, start=1):
                ent = ent_map.get(meta.get("chunk_id"))
                if not ent:
                    # Fallback to lightweight model using metadata
                    doc_fn = meta.get("document_filename", "")
                    cm = ChunkModel(
                        id=UUID(meta.get("chunk_id")),
                        document_id=UUID(meta.get("doc_id")),
                        content="",  # will be empty without DB entity
                        position=int(meta.get("sequence", 0)),
                        metadata={},
                        created_at=datetime.fromtimestamp(
                            0, tz=timezone.utc
                        ),  # placeholder
                    )
                else:
                    cm = ChunkModel.from_entity(ent)
                    doc_fn = meta.get("document_filename", "")
                results.append(
                    SearchResultModel(
                        chunk=cm,
                        score=float(score),
                        document_filename=doc_fn,
                        rank=rank,
                    )
                )
            processing_time = (time.time() - start_time) * 1000
            return QueryContextModel(
                query=query,
                search_results=results,
                total_results=len(results),
                processing_time_ms=processing_time,
                search_type="semantic",
            )
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise SearchError(f"Semantic search failed: {e}")

    async def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        db_session: AsyncSession = None,
    ) -> QueryContextModel:
        """Perform keyword search using Postgres FTS."""
        start_time = time.time()
        try:
            pairs = await fts_search(db_session, query, top_k=top_k)
            # Load chunk entities for created_at and full content
            chunk_ids = [
                p[0].metadata.get("chunk_id")
                for p in pairs
                if p[0].metadata.get("chunk_id")
            ]
            ent_map: Dict[str, ChunkEntity] = {}
            if chunk_ids:
                q = select(ChunkEntity).where(ChunkEntity.id.in_(chunk_ids))
                res = await db_session.execute(q)
                for ent in res.scalars().all():
                    ent_map[str(ent.id)] = ent
            results: List[SearchResultModel] = []
            for rank, (doc, score) in enumerate(pairs, start=1):
                meta = doc.metadata or {}
                ent = ent_map.get(meta.get("chunk_id", ""))
                if ent:
                    cm = ChunkModel.from_entity(ent)
                    filename = meta.get("document_filename", "")
                else:
                    # Fallback minimal
                    cm = ChunkModel(
                        id=UUID(meta.get("chunk_id")),
                        document_id=UUID(meta.get("doc_id")),
                        content=doc.page_content,
                        position=int(meta.get("sequence", 0)),
                        metadata={},
                        created_at=datetime.fromtimestamp(0, tz=timezone.utc),
                    )
                    filename = meta.get("document_filename", "")
                results.append(
                    SearchResultModel(
                        chunk=cm,
                        score=float(score),
                        document_filename=filename,
                        rank=rank,
                    )
                )
            processing_time = (time.time() - start_time) * 1000
            return QueryContextModel(
                query=query,
                search_results=results,
                total_results=len(results),
                processing_time_ms=processing_time,
                search_type="keyword",
            )
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            raise SearchError(f"Keyword search failed: {e}")

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        db_session: AsyncSession = None,
    ) -> QueryContextModel:
        """Perform hybrid search combining vector and FTS (BM25-ish) using weighted fusion."""
        start_time = time.time()
        try:
            sem = await self.semantic_search(
                query=query, top_k=top_k * 2, db_session=db_session
            )
            kw = await self.keyword_search(
                query=query, top_k=top_k * 2, db_session=db_session
            )
            combined: Dict[str, Dict[str, float]] = {}
            selection: Dict[str, SearchResultModel] = {}
            for r in sem.search_results:
                cid = str(r.chunk.id)
                combined[cid] = {"sem": r.score, "kw": 0.0}
                selection[cid] = r
            for r in kw.search_results:
                cid = str(r.chunk.id)
                if cid in combined:
                    combined[cid]["kw"] = r.score
                    # keep existing selection (from sem)
                else:
                    combined[cid] = {"sem": 0.0, "kw": r.score}
                    selection[cid] = r
            fused: List[SearchResultModel] = []
            for cid, scores in combined.items():
                sc = semantic_weight * scores["sem"] + keyword_weight * scores["kw"]
                item = selection[cid]
                item.score = sc
                fused.append(item)
            fused.sort(key=lambda x: x.score, reverse=True)
            final_results = fused[:top_k]
            for i, r in enumerate(final_results):
                r.rank = i + 1
            processing_time = (time.time() - start_time) * 1000
            return QueryContextModel(
                query=query,
                search_results=final_results,
                total_results=len(final_results),
                processing_time_ms=processing_time,
                search_type="hybrid",
            )
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise SearchError(f"Hybrid search failed: {e}")

    async def generate_answer(
        self,
        question: str,
        context: QueryContextModel,
        max_tokens: int = 500,
        temperature: float = 0.1,
        model: str = "gpt-3.5-turbo",
    ) -> AnswerModel:
        """Generate an answer using RAG."""
        start_time = time.time()

        try:
            if not self.llm_client:
                # Keep working with a mock to avoid overengineering LLM selection here
                pass

            if not context.search_results:
                raise InsufficientContextError(1, 0)

            # Build context from search results
            context_text = context.get_context_text(max_chunks=5)

            # Create prompt
            _ = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say so clearly.

Context:
{context_text}

Question: {question}

Answer:"""

            # TODO: Plug in ChatOpenAI (LangChain) later. For now keep mock.
            answer = f"This is a mock answer for: '{question}'. A production path would call an LLM with the retrieved context."

            generation_time = (time.time() - start_time) * 1000

            return AnswerModel(
                question=question,
                answer=answer,
                context=context,
                confidence_score=0.85,  # Mock confidence score
                model_used=model,
                generation_time_ms=generation_time,
            )

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise AnswerGenerationError(str(e), model)

    async def get_query_suggestions(
        self, prefix: str = "", limit: int = 10, db_session: AsyncSession = None
    ) -> List[str]:
        """Get query suggestions based on prefix and popular queries."""
        try:
            # TODO: Implement actual suggestion logic
            # For now, return mock suggestions
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

            # Filter by prefix if provided
            if prefix:
                suggestions = [
                    s for s in suggestions if s.lower().startswith(prefix.lower())
                ]

            return suggestions[:limit]

        except Exception as e:
            logger.error(f"Failed to get query suggestions: {str(e)}")
            raise SearchError(f"Failed to get suggestions: {str(e)}")

    # Note: No direct embedding generation here; retrievers handle embeddings via LangChain.
