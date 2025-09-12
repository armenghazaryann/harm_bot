"""Controller for the Query feature."""
import logging
from typing import List

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
    HybridSearchRequest
)
from api.features.query.exceptions import (
    QueryValidationError,
    SearchError,
    AnswerGenerationError,
    NoResultsError
)
from api.features.query.service import QueryService
from api.shared.response import ResponseModel

logger = logging.getLogger("rag.query")


class QueryController:
    """Controller for query and search operations."""
    
    def __init__(self, query_service: QueryService):
        self.query_service = query_service
    
    async def search_documents(
        self,
        request: QueryRequest,
        db_session: AsyncSession
    ) -> ResponseModel[SearchResponse]:
        """Search documents using hybrid search."""
        try:
            # Perform hybrid search by default
            context = await self.query_service.hybrid_search(
                query=request.query,
                top_k=request.top_k,
                db_session=db_session
            )
            
            if not context.search_results:
                raise NoResultsError(request.query)
            
            # Convert to DTOs
            search_results = [
                result.to_search_result_dto() 
                for result in context.search_results
            ]
            
            response_data = SearchResponse(
                query=request.query,
                results=search_results,
                total_results=context.total_results,
                processing_time_ms=context.processing_time_ms,
                search_metadata={
                    "search_type": context.search_type,
                    "filters_applied": request.filters is not None
                }
            )
            
            logger.info(f"Search completed: {len(search_results)} results for query '{request.query}'")
            return ResponseModel.success(
                data=response_data,
                message="Search completed successfully"
            )
            
        except NoResultsError as e:
            logger.info(f"No results found: {e.message}")
            # Return empty results instead of error
            response_data = SearchResponse(
                query=request.query,
                results=[],
                total_results=0,
                processing_time_ms=0.0,
                search_metadata={"search_type": "hybrid"}
            )
            return ResponseModel.success(
                data=response_data,
                message="No results found for the given query"
            )
        except QueryValidationError as e:
            logger.warning(f"Query validation failed: {e.message}")
            raise HTTPException(status_code=400, detail=e.message)
        except SearchError as e:
            logger.error(f"Search failed: {e.message}")
            raise HTTPException(status_code=500, detail=e.message)
        except Exception as e:
            logger.exception("Unexpected error during search")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def semantic_search(
        self,
        request: SemanticSearchRequest,
        db_session: AsyncSession
    ) -> ResponseModel[SearchResponse]:
        """Perform semantic search using vector similarity."""
        try:
            context = await self.query_service.semantic_search(
                query=request.query,
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold,
                document_filters=request.document_filters,
                db_session=db_session
            )
            
            search_results = [
                result.to_search_result_dto() 
                for result in context.search_results
            ]
            
            response_data = SearchResponse(
                query=request.query,
                results=search_results,
                total_results=context.total_results,
                processing_time_ms=context.processing_time_ms,
                search_metadata={
                    "search_type": "semantic",
                    "similarity_threshold": request.similarity_threshold
                }
            )
            
            return ResponseModel.success(
                data=response_data,
                message="Semantic search completed successfully"
            )
            
        except Exception as e:
            logger.exception("Semantic search failed")
            raise HTTPException(status_code=500, detail="Semantic search failed")
    
    async def keyword_search(
        self,
        request: KeywordSearchRequest,
        db_session: AsyncSession
    ) -> ResponseModel[SearchResponse]:
        """Perform keyword search using full-text search."""
        try:
            context = await self.query_service.keyword_search(
                query=request.query,
                top_k=10,  # Default top_k for keyword search
                use_stemming=request.use_stemming,
                use_fuzzy=request.use_fuzzy,
                db_session=db_session
            )
            
            search_results = [
                result.to_search_result_dto() 
                for result in context.search_results
            ]
            
            response_data = SearchResponse(
                query=request.query,
                results=search_results,
                total_results=context.total_results,
                processing_time_ms=context.processing_time_ms,
                search_metadata={
                    "search_type": "keyword",
                    "use_stemming": request.use_stemming,
                    "use_fuzzy": request.use_fuzzy
                }
            )
            
            return ResponseModel.success(
                data=response_data,
                message="Keyword search completed successfully"
            )
            
        except Exception as e:
            logger.exception("Keyword search failed")
            raise HTTPException(status_code=500, detail="Keyword search failed")
    
    async def answer_question(
        self,
        request: AnswerRequest,
        db_session: AsyncSession
    ) -> ResponseModel[AnswerResponse]:
        """Answer a question using RAG."""
        try:
            # First, search for relevant context
            context = await self.query_service.hybrid_search(
                query=request.question,
                top_k=request.context_limit,
                db_session=db_session
            )
            
            if not context.search_results:
                raise NoResultsError(request.question)
            
            # Generate answer using the context
            answer_model = await self.query_service.generate_answer(
                question=request.question,
                context=context,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            response_data = answer_model.to_answer_response_dto()
            
            logger.info(f"Answer generated for question: '{request.question}'")
            return ResponseModel.success(
                data=response_data,
                message="Answer generated successfully"
            )
            
        except NoResultsError as e:
            logger.info(f"No context found for question: {e.message}")
            raise HTTPException(
                status_code=404, 
                detail="No relevant context found to answer the question"
            )
        except AnswerGenerationError as e:
            logger.error(f"Answer generation failed: {e.message}")
            raise HTTPException(status_code=500, detail=e.message)
        except Exception as e:
            logger.exception("Unexpected error during answer generation")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def get_query_suggestions(
        self,
        request: SuggestionRequest,
        db_session: AsyncSession
    ) -> ResponseModel[SuggestionResponse]:
        """Get query suggestions."""
        try:
            suggestions = await self.query_service.get_query_suggestions(
                prefix=request.prefix,
                limit=request.limit,
                db_session=db_session
            )
            
            response_data = SuggestionResponse(
                suggestions=suggestions,
                prefix=request.prefix,
                total_suggestions=len(suggestions)
            )
            
            return ResponseModel.success(
                data=response_data,
                message="Suggestions retrieved successfully"
            )
            
        except Exception as e:
            logger.exception("Failed to get query suggestions")
            raise HTTPException(status_code=500, detail="Failed to get suggestions")
