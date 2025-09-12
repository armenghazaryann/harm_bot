"""Router for the Query feature."""
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Query as QueryParam

from api.features.query.controller import QueryController
from api.features.query.dtos import (
    QueryRequest,
    SearchResponse,
    AnswerRequest,
    AnswerResponse,
    SuggestionRequest,
    SuggestionResponse,
    SemanticSearchRequest,
    KeywordSearchRequest
)
from api.shared.dtos import HealthCheckResponse
from api.di.container import DependencyContainer
from api.shared.response import ResponseModel

router = APIRouter()


@router.get("/health", response_model=ResponseModel[HealthCheckResponse])
async def health_check():
    """Health check endpoint for query service."""
    return ResponseModel.success(
        data=HealthCheckResponse(
            status="healthy",
            dependencies={"embeddings": "ok", "llm": "ok", "database": "ok"}
        ),
        message="Query service is healthy"
    )


@router.post("/search", response_model=ResponseModel[SearchResponse])
@inject
async def search_documents(
    request: QueryRequest,
    controller: QueryController = Depends(Provide[DependencyContainer.query_controller])
):
    """Search documents using hybrid search (semantic + keyword)."""
    return await controller.search_documents(request)


@router.post("/search/semantic", response_model=ResponseModel[SearchResponse])
@inject
async def semantic_search(
    request: SemanticSearchRequest,
    controller: QueryController = Depends(Provide[DependencyContainer.query_controller])
):
    """Perform semantic search using vector similarity."""
    return await controller.semantic_search(request)


@router.post("/search/keyword", response_model=ResponseModel[SearchResponse])
@inject
async def keyword_search(
    request: KeywordSearchRequest,
    controller: QueryController = Depends(Provide[DependencyContainer.query_controller])
):
    """Perform keyword search using full-text search."""
    return await controller.keyword_search(request)


@router.post("/answer", response_model=ResponseModel[AnswerResponse])
@inject
async def answer_question(
    request: AnswerRequest,
    controller: QueryController = Depends(Provide[DependencyContainer.query_controller])
):
    """Answer a question using RAG (Retrieval-Augmented Generation)."""
    return await controller.answer_question(request)


@router.get("/suggestions", response_model=ResponseModel[SuggestionResponse])
@inject
async def get_query_suggestions(
    prefix: str = QueryParam("", description="Query prefix for suggestions"),
    limit: int = QueryParam(10, ge=1, le=50, description="Maximum number of suggestions"),
    controller: QueryController = Depends(Provide[DependencyContainer.query_controller])
):
    """Get query suggestions based on prefix."""
    request = SuggestionRequest(prefix=prefix, limit=limit)
    return await controller.get_query_suggestions(request)
