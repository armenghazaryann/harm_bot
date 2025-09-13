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
    semantic_search_documents,
)
from api.shared.response import ResponseModel

logger = logging.getLogger("rag.query")


class QueryController:
    """Controller for query and search operations."""

    def __init__(self, query_service: QueryService):
        self.query_service = query_service

    async def search_documents(
        self, request: QueryRequest, db_session: AsyncSession
    ) -> ResponseModel[SearchResponse]:
        """Search documents using production-grade hybrid search (Vector Store + BM25)."""
        try:
            # Use LangChain hybrid search with EnsembleRetriever (Vector + BM25)
            result = await query_documents(
                question=request.query,
                doc_id=None,  # Search across all documents
                k=request.top_k,
                search_type="hybrid",
            )

            # Treat presence of retrieved documents as success, regardless of LLM answer text
            if not result.get("source_documents"):
                # Fallback to pure semantic search to increase recall
                try:
                    semantic = await semantic_search_documents(
                        query=request.query, doc_id=None, k=request.top_k
                    )
                    fallback_docs = [
                        type(
                            "_Doc",
                            (),
                            {
                                "page_content": item.get("content", ""),
                                "metadata": item.get("metadata", {}),
                            },
                        )
                        for item in (semantic or [])
                    ]
                    result["source_documents"] = fallback_docs
                except Exception:
                    raise NoResultsError(request.query)

            # Convert LangChain result to SearchResponse format
            search_results = []
            if result.get("source_documents"):
                for i, doc in enumerate(result["source_documents"]):
                    md = doc.metadata or {}
                    # Derive required DTO fields
                    chunk_id = md.get("chunk_id") or md.get("id") or str(uuid.uuid4())
                    document_id = md.get("doc_id")
                    document_filename = md.get("document_filename") or (
                        md.get("source") or ""
                    )
                    position = md.get("chunk_index") or md.get("sequence") or 0

                    search_results.append(
                        {
                            "chunk_id": str(chunk_id),
                            "content": doc.page_content,
                            "score": md.get("score", 0.0),
                            "metadata": md,
                            "document_id": str(document_id)
                            if document_id
                            else str(uuid.uuid4()),
                            "document_filename": document_filename,
                            "position": int(position),
                        }
                    )

            response_data = SearchResponse(
                query=request.query,
                results=search_results,
                total_results=len(search_results),
                processing_time_ms=0.0,  # TODO: Add timing to LangChain service
                search_metadata={
                    "search_type": "hybrid_langchain",
                    "retriever_type": "EnsembleRetriever (Vector + BM25)",
                    "vector_weight": 0.6,
                    "bm25_weight": 0.4,
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
        """Perform semantic search using LangChain vector similarity."""
        try:
            # Use LangChain semantic search
            result = await semantic_search_documents(
                query=request.query,
                doc_id=(
                    str(request.document_filters[0])
                    if request.document_filters
                    else None
                ),
                k=request.top_k,
            )

            # semantic_search_documents returns a List[Dict]
            search_results = []
            for i, item in enumerate(result or []):
                metadata = item.get("metadata", {})
                chunk_id = (
                    metadata.get("chunk_id") or metadata.get("id") or str(uuid.uuid4())
                )
                document_id = metadata.get("doc_id") or item.get("doc_id")
                document_filename = metadata.get("document_filename") or (
                    metadata.get("source") or ""
                )
                position = metadata.get("chunk_index") or metadata.get("sequence") or 0

                search_results.append(
                    {
                        "chunk_id": str(chunk_id),
                        "content": item.get("content", ""),
                        "score": metadata.get("score", 0.0),
                        "metadata": metadata,
                        "document_id": str(document_id)
                        if document_id
                        else str(uuid.uuid4()),
                        "document_filename": document_filename,
                        "position": int(position),
                    }
                )

            response_data = SearchResponse(
                query=request.query,
                results=search_results,
                total_results=len(search_results),
                processing_time_ms=0.0,  # TODO: Add timing to LangChain service
                search_metadata={
                    "search_type": "semantic_langchain",
                    "retriever_type": "PGVector similarity search",
                    "embedding_model": "text-embedding-3-large",
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
        """Perform keyword search using LangChain BM25 retriever."""
        try:
            # Use LangChain BM25-based keyword search (part of hybrid search)
            result = await query_documents(
                question=request.query,
                doc_id=None,
                k=10,  # Default top_k for keyword search
                search_type="keyword",  # This will use BM25 component
            )

            search_results = []
            if result.get("source_documents"):
                for i, doc in enumerate(result["source_documents"]):
                    md = doc.metadata or {}
                    chunk_id = md.get("chunk_id") or md.get("id") or str(uuid.uuid4())
                    document_id = md.get("doc_id")
                    document_filename = md.get("document_filename") or (
                        md.get("source") or ""
                    )
                    position = md.get("chunk_index") or md.get("sequence") or 0

                    search_results.append(
                        {
                            "chunk_id": str(chunk_id),
                            "content": doc.page_content,
                            "score": md.get("score", 0.0),
                            "metadata": md,
                            "document_id": str(document_id)
                            if document_id
                            else str(uuid.uuid4()),
                            "document_filename": document_filename,
                            "position": int(position),
                        }
                    )

            response_data = SearchResponse(
                query=request.query,
                results=search_results,
                total_results=len(search_results),
                processing_time_ms=0.0,  # TODO: Add timing to LangChain service
                search_metadata={
                    "search_type": "keyword_langchain",
                    "retriever_type": "BM25Retriever",
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
            # Use LangChain RetrievalQA with hybrid search for answer generation
            result = await query_documents(
                question=request.question,
                doc_id=None,  # Search across all documents
                k=request.context_limit,
                search_type="hybrid",
                detailed=True,  # Enable Self-RAG Lite path by default
            )

            # Propagate backend errors properly
            if result.get("error"):
                raise HTTPException(status_code=500, detail=str(result.get("error")))

            if not result.get("answer"):
                raise NoResultsError(request.question)

            # Convert LangChain result to AnswerResponse format
            source_documents = result.get("source_documents", [])
            # Map to required citations DTO
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
                        "document_filename": md.get("document_filename")
                        or (md.get("source") or ""),
                        "content_excerpt": (doc.page_content or "")[:500],
                        "relevance_score": float(md.get("score", 0.0)),
                        "page_number": md.get("page_number"),
                    }
                )

            response_data = AnswerResponse(
                question=request.question,
                answer=result.get("answer", ""),
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
