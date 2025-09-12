"""DTOs for the Query feature."""
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, validator

from api.shared.dtos import BaseDTO


class QueryRequest(BaseDTO):
    """Request DTO for document search."""
    query: str = Field(..., min_length=1, max_length=1000, description="The search query")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional search filters")
    include_metadata: bool = Field(default=True, description="Include chunk metadata in results")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SearchResult(BaseDTO):
    """Individual search result."""
    chunk_id: UUID = Field(description="Chunk identifier")
    content: str = Field(description="Chunk content")
    score: float = Field(description="Relevance score")
    metadata: Dict[str, Any] = Field(description="Chunk metadata")
    document_id: UUID = Field(description="Source document identifier")
    document_filename: str = Field(description="Source document filename")
    position: int = Field(description="Chunk position in document")


class SearchResponse(BaseDTO):
    """Response DTO for search results."""
    query: str = Field(description="Original query")
    results: List[SearchResult] = Field(description="Search results")
    total_results: int = Field(description="Total number of results found")
    processing_time_ms: float = Field(description="Query processing time in milliseconds")
    search_metadata: Dict[str, Any] = Field(default_factory=dict, description="Search metadata")


class AnswerRequest(BaseDTO):
    """Request DTO for RAG question answering."""
    question: str = Field(..., min_length=1, max_length=1000, description="The question to answer")
    context_limit: int = Field(default=5, ge=1, le=20, description="Number of context chunks to use")
    include_citations: bool = Field(default=True, description="Include citations in response")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=500, ge=50, le=2000, description="Maximum response tokens")
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class Citation(BaseDTO):
    """Citation information."""
    chunk_id: UUID = Field(description="Source chunk identifier")
    document_id: UUID = Field(description="Source document identifier")
    document_filename: str = Field(description="Source document filename")
    content_excerpt: str = Field(description="Relevant content excerpt")
    relevance_score: float = Field(description="Relevance score")
    page_number: Optional[int] = Field(default=None, description="Page number if available")


class AnswerResponse(BaseDTO):
    """Response DTO for RAG answers."""
    question: str = Field(description="Original question")
    answer: str = Field(description="Generated answer")
    citations: List[Citation] = Field(description="Supporting citations")
    confidence_score: float = Field(description="Answer confidence score")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    model_used: str = Field(description="LLM model used for generation")


class SuggestionRequest(BaseDTO):
    """Request DTO for query suggestions."""
    prefix: str = Field(default="", max_length=100, description="Query prefix for suggestions")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of suggestions")
    context: Optional[str] = Field(default=None, description="Context for better suggestions")


class SuggestionResponse(BaseDTO):
    """Response DTO for query suggestions."""
    suggestions: List[str] = Field(description="List of suggested queries")
    prefix: str = Field(description="Original prefix")
    total_suggestions: int = Field(description="Total number of suggestions available")


class SemanticSearchRequest(BaseDTO):
    """Request DTO for semantic search."""
    query: str = Field(..., min_length=1, description="Semantic search query")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    document_filters: Optional[List[UUID]] = Field(default=None, description="Filter by document IDs")


class KeywordSearchRequest(BaseDTO):
    """Request DTO for keyword search."""
    query: str = Field(..., min_length=1, description="Keyword search query")
    use_stemming: bool = Field(default=True, description="Use word stemming")
    use_fuzzy: bool = Field(default=False, description="Use fuzzy matching")
    boost_exact_match: bool = Field(default=True, description="Boost exact phrase matches")


class HybridSearchRequest(BaseDTO):
    """Request DTO for hybrid search (semantic + keyword)."""
    query: str = Field(..., min_length=1, description="Search query")
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for semantic search")
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for keyword search")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    
    @validator('semantic_weight', 'keyword_weight')
    def validate_weights(cls, v, values):
        if 'semantic_weight' in values:
            semantic_weight = values.get('semantic_weight', 0.7)
            if abs(semantic_weight + v - 1.0) > 0.001:  # Allow small floating point errors
                raise ValueError("Semantic weight and keyword weight must sum to 1.0")
        return v
