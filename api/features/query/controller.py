"""Controller for the Query feature."""
import logging
import uuid

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.features.query.dtos import (
    QueryRequest,
    SearchResponse,
    AnswerRequest,
    AnswerResponse,
    SuggestionRequest,
    SuggestionResponse,
    SemanticSearchRequest,
    KeywordSearchRequest,
)
from api.features.query.exceptions import (
    QueryValidationError,
    SearchError,
    AnswerGenerationError,
    NoResultsError,
)
from api.features.query.service import (
    QueryService,
    query_documents,
)
from rag.pipeline.query_pipeline import QueryPipeline
from api.shared.response import ResponseModel
from api.features.conversation.service import (
    fetch_recent_messages,
    append_message,
)
from rag.retrievers.hybrid_retriever import hybrid_rrf
from rag.retrievers.vector_retriever import vector_search
from rag.retrievers.fts_retriever import fts_search

logger = logging.getLogger("rag.query")


class QueryController:
    """Controller for query and search operations."""

    def __init__(
        self, query_service: QueryService, query_pipeline: QueryPipeline | None = None
    ):
        self.query_service = query_service
        self.query_pipeline = query_pipeline

    async def search_documents(
        self, request: QueryRequest, db_session: AsyncSession
    ) -> ResponseModel[SearchResponse]:
        """Search documents using hybrid RRF (Vector + FTS) with optional rerank."""
        try:
            docs = await hybrid_rrf(
                query=request.query,
                k=request.top_k,
                session=db_session,
            )

            # Convert to DTO list
            search_results = []
            for doc in docs:
                md = doc.metadata or {}
                chunk_id = md.get("chunk_id") or md.get("id") or str(uuid.uuid4())
                document_id = md.get("doc_id") or str(uuid.uuid4())
                document_filename = md.get("document_name") or ""
                position = md.get("chunk_index") or md.get("sequence") or 0
                score = float(md.get("score", 0.0))
                search_results.append(
                    {
                        "chunk_id": str(chunk_id),
                        "content": doc.page_content,
                        "score": score,
                        "metadata": md,
                        "document_id": str(document_id),
                        "document_filename": document_filename,
                        "position": int(position),
                    }
                )

            response_data = SearchResponse(
                query=request.query,
                results=search_results,
                total_results=len(search_results),
                processing_time_ms=0.0,
                search_metadata={
                    "search_type": "hybrid_rrf",
                    "retriever_type": "PGVector + Postgres FTS (RRF)",
                },
            )

            logger.info(
                f"Search completed: {len(search_results)} results for query '{request.query}'"
            )
            return ResponseModel.success(
                data=response_data, message="Search completed successfully"
            )

        except NoResultsError as e:
            logger.info(f"No results found: {e.message}")
            # Return empty results instead of error
            response_data = SearchResponse(
                query=request.query,
                results=[],
                total_results=0,
                processing_time_ms=0.0,
                search_metadata={"search_type": "hybrid"},
            )
            return ResponseModel.success(
                data=response_data, message="No results found for the given query"
            )
        except QueryValidationError as e:
            logger.warning(f"Query validation failed: {e.message}")
            raise HTTPException(status_code=400, detail=e.message)
        except SearchError as e:
            logger.error(f"Search failed: {e.message}")
            raise HTTPException(status_code=500, detail=e.message)
        except Exception:
            logger.exception("Unexpected error during search")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def semantic_search(
        self, request: SemanticSearchRequest, db_session: AsyncSession
    ) -> ResponseModel[SearchResponse]:
        """Perform semantic search using PGVector similarity."""
        try:
            vec = vector_search(request.query, top_k=request.top_k)
            search_results = []
            for doc, score in vec:
                md = doc.metadata or {}
                chunk_id = md.get("chunk_id") or md.get("id") or str(uuid.uuid4())
                document_id = md.get("doc_id") or str(uuid.uuid4())
                document_filename = md.get("document_name") or ""
                position = md.get("chunk_index") or md.get("sequence") or 0
                search_results.append(
                    {
                        "chunk_id": str(chunk_id),
                        "content": doc.page_content,
                        "score": float(score or 0.0),
                        "metadata": md,
                        "document_id": str(document_id),
                        "document_filename": document_filename,
                        "position": int(position),
                    }
                )

            response_data = SearchResponse(
                query=request.query,
                results=search_results,
                total_results=len(search_results),
                processing_time_ms=0.0,
                search_metadata={
                    "search_type": "semantic_pgvector",
                    "retriever_type": "PGVector similarity search",
                    "embedding_model": "text-embedding-3-small",
                    "similarity_threshold": request.similarity_threshold,
                },
            )

            return ResponseModel.success(
                data=response_data, message="Semantic search completed successfully"
            )

        except Exception:
            logger.exception("Semantic search failed")
            raise HTTPException(status_code=500, detail="Semantic search failed")

    async def keyword_search(
        self, request: KeywordSearchRequest, db_session: AsyncSession
    ) -> ResponseModel[SearchResponse]:
        """Perform keyword search using Postgres FTS (ts_rank_cd)."""
        try:
            docs = await fts_search(db_session, request.query, top_k=10)
            search_results = []
            for doc, score in docs:
                md = doc.metadata or {}
                chunk_id = md.get("chunk_id") or md.get("id") or str(uuid.uuid4())
                document_id = md.get("doc_id") or str(uuid.uuid4())
                document_filename = md.get("document_name") or ""
                position = md.get("chunk_index") or md.get("sequence") or 0
                search_results.append(
                    {
                        "chunk_id": str(chunk_id),
                        "content": doc.page_content,
                        "score": float(score or 0.0),
                        "metadata": md,
                        "document_id": str(document_id),
                        "document_filename": document_filename,
                        "position": int(position),
                    }
                )

            response_data = SearchResponse(
                query=request.query,
                results=search_results,
                total_results=len(search_results),
                processing_time_ms=0.0,
                search_metadata={
                    "search_type": "keyword_fts",
                    "retriever_type": "Postgres FTS (ts_rank_cd)",
                    "use_stemming": request.use_stemming,
                    "use_fuzzy": request.use_fuzzy,
                },
            )

            return ResponseModel.success(
                data=response_data, message="Keyword search completed successfully"
            )

        except Exception:
            logger.exception("Keyword search failed")
            raise HTTPException(status_code=500, detail="Keyword search failed")

    async def answer_question(
        self, request: AnswerRequest, db_session: AsyncSession
    ) -> ResponseModel[AnswerResponse]:
        """Answer a question using RAG."""
        try:
            # Prepare optional conversation history text
            history_text = None
            if request.conversation_id:
                try:
                    recent = await fetch_recent_messages(
                        db_session,
                        conversation_id=str(request.conversation_id),
                        limit=6,
                    )
                    history_text = "\n".join(
                        [f"{m['role']}: {m['content']}" for m in (recent or [])]
                    )
                except Exception:
                    history_text = None

            # Prefer the new QueryPipeline if available (fast, modular)
            if self.query_pipeline is not None:
                result = await self.query_pipeline.answer(
                    question=request.question,
                    db_session=db_session,
                    k=request.context_limit,
                    doc_id=None,
                    history_text=history_text,
                )
            else:
                # Fallback to existing service-based flow
                result = await query_documents(
                    question=request.question,
                    doc_id=None,  # Search across all documents
                    k=request.context_limit,
                    search_type="hybrid",
                    detailed=True,  # Enable Self-RAG Lite path by default
                )

            # Propagate backend errors properly
            if isinstance(result, dict) and result.get("error"):
                raise HTTPException(status_code=500, detail=str(result.get("error")))

            if not result.get("answer"):
                raise NoResultsError(request.question)

            # Convert result to AnswerResponse format
            source_documents = result.get("source_documents", [])
            citations = []
            import uuid as _uuid  # local import scoped to function

            for doc in source_documents:
                md = doc.metadata or {}
                citations.append(
                    {
                        "chunk_id": md.get("chunk_id")
                        or md.get("id")
                        or str(_uuid.uuid4()),
                        "document_id": md.get("doc_id") or str(_uuid.uuid4()),
                        "document_filename": md.get("document_name") or "",
                        "content_excerpt": (doc.page_content or "")[:500],
                        "relevance_score": float(md.get("score", 0.0)),
                        "page_number": md.get("page_number"),
                    }
                )

            response_data = AnswerResponse(
                question=request.question,
                answer=str(result.get("answer", "")),
                citations=citations,
                confidence_score=float(result.get("confidence_score", 0.85)),
                processing_time_ms=float(
                    (result.get("performance_metrics", {}) or {}).get(
                        "processing_time_ms", 0.0
                    )
                ),
                model_used=str(result.get("model_used", "gpt-4o")),
                verification_report=result.get("verification_report", {}),
                retrieval_diagnostics=result.get("retrieval_diagnostics", {}),
            )

            logger.info(f"Answer generated for question: '{request.question}'")

            # Append messages to conversation if provided
            try:
                if request.conversation_id:
                    await append_message(
                        db_session,
                        conversation_id=str(request.conversation_id),
                        role="user",
                        content=request.question,
                    )
                    await append_message(
                        db_session,
                        conversation_id=str(request.conversation_id),
                        role="assistant",
                        content=response_data.answer,
                    )
                    await db_session.commit()
            except Exception:
                # Do not fail the response on conversation logging errors
                pass
            return ResponseModel.success(
                data=response_data, message="Answer generated successfully"
            )

        except NoResultsError as e:
            logger.info(f"No context found for question: {e.message}")
            raise HTTPException(
                status_code=404,
                detail="No relevant context found to answer the question",
            )
        except AnswerGenerationError as e:
            logger.error(f"Answer generation failed: {e.message}")
            raise HTTPException(status_code=500, detail=e.message)
        except Exception:
            logger.exception("Unexpected error during answer generation")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def get_query_suggestions(
        self, request: SuggestionRequest, db_session: AsyncSession
    ) -> ResponseModel[SuggestionResponse]:
        """Get query suggestions."""
        try:
            suggestions = await self.query_service.get_query_suggestions(
                prefix=request.prefix, limit=request.limit, db_session=db_session
            )

            response_data = SuggestionResponse(
                suggestions=suggestions,
                prefix=request.prefix,
                total_suggestions=len(suggestions),
            )

            return ResponseModel.success(
                data=response_data, message="Suggestions retrieved successfully"
            )

        except Exception:
            logger.exception("Failed to get query suggestions")
            raise HTTPException(status_code=500, detail="Failed to get suggestions")
