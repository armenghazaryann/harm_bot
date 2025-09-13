"""
DEPRECATED: Models for Query feature - REPLACED BY LANGCHAIN DOCUMENT OBJECTS

This file contains models that are now obsolete with our LangChain implementation:

REPLACED BY:
- ChunkModel/ChunkEntity → LangChain Document objects with metadata
- EmbeddingModel/EmbeddingEntity → LangChain PGVector automatic embeddings
- SearchResultModel → LangChain retriever results with Document objects
- QueryContextModel → LangChain RetrievalQA chain responses

All functionality has been migrated to LangChain components with superior performance.
This file should be deprecated after confirming LangChain pipeline works properly.
"""
from datetime import datetime
from typing import Dict, List, Any
from uuid import UUID

from pydantic import BaseModel, Field

# DEPRECATED IMPORTS - Replaced by LangChain Document objects
# from api.features.query.entities.chunk import Chunk as ChunkEntity
# from api.features.query.entities.embedding import Embedding as EmbeddingEntity


# Create stub classes to prevent runtime errors during transition to LangChain
class ChunkEntity:
    """Stub class to prevent runtime errors during transition to LangChain."""

    id = None
    document_id = None
    content = ""
    sequence_number = 0
    extra_metadata = {}
    created_at = datetime.now()


class EmbeddingEntity:
    """Stub class to prevent runtime errors during transition to LangChain."""

    id = None
    chunk_id = None
    model_name = ""
    embedding = []
    created_at = datetime.now()


class ChunkModel(BaseModel):
    """Domain model for Chunk."""

    id: UUID = Field(description="Chunk identifier")
    document_id: UUID = Field(description="Source document identifier")
    content: str = Field(description="Chunk content")
    position: int = Field(description="Chunk position in document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    created_at: datetime = Field(description="Creation timestamp")

    class Config:
        from_attributes = True

    @classmethod
    def from_entity(cls, entity: ChunkEntity) -> "ChunkModel":
        """Create model from database entity."""
        if entity is None:
            return None
        return cls(
            id=entity.id,
            document_id=entity.document_id,
            content=entity.content,
            position=entity.sequence_number,
            metadata=entity.extra_metadata or {},
            created_at=entity.created_at,
        )


class EmbeddingModel(BaseModel):
    """Domain model for Embedding."""

    id: UUID = Field(description="Embedding identifier")
    chunk_id: UUID = Field(description="Associated chunk identifier")
    model_name: str = Field(description="Embedding model used")
    vector: List[float] = Field(description="Embedding vector")
    created_at: datetime = Field(description="Creation timestamp")

    class Config:
        from_attributes = True

    @classmethod
    def from_entity(cls, entity: EmbeddingEntity) -> "EmbeddingModel":
        """Create model from database entity."""
        if entity is None:
            return None
        return cls(
            id=entity.id,
            chunk_id=entity.chunk_id,
            model_name=entity.model_name,
            vector=list(entity.embedding),
            created_at=entity.created_at,
        )


class SearchResultModel(BaseModel):
    """Model for search results."""

    chunk: ChunkModel = Field(description="Chunk information")
    score: float = Field(description="Relevance score")
    document_filename: str = Field(description="Source document filename")
    rank: int = Field(description="Result rank")

    def to_search_result_dto(self):
        """Convert to SearchResult DTO."""
        from api.features.query.dtos import SearchResult

        return SearchResult(
            chunk_id=self.chunk.id,
            content=self.chunk.content,
            score=self.score,
            metadata=self.chunk.metadata,
            document_id=self.chunk.document_id,
            document_filename=self.document_filename,
            position=self.chunk.position,
        )


class QueryContextModel(BaseModel):
    """Model for query context."""

    query: str = Field(description="Original query")
    search_results: List[SearchResultModel] = Field(description="Search results")
    total_results: int = Field(description="Total number of results")
    processing_time_ms: float = Field(description="Processing time")
    search_type: str = Field(description="Type of search performed")

    def get_context_text(self, max_chunks: int = 5) -> str:
        """Get concatenated context text from top results."""
        context_chunks = []
        for result in self.search_results[:max_chunks]:
            context_chunks.append(
                f"[Document: {result.document_filename}]\n{result.chunk.content}"
            )

        return "\n\n".join(context_chunks)

    def get_citations(self) -> List[Dict[str, Any]]:
        """Get citation information from results."""
        citations = []
        for i, result in enumerate(self.search_results):
            citations.append(
                {
                    "chunk_id": str(result.chunk.id),
                    "document_id": str(result.chunk.document_id),
                    "document_filename": result.document_filename,
                    "content_excerpt": result.chunk.content[:200] + "..."
                    if len(result.chunk.content) > 200
                    else result.chunk.content,
                    "relevance_score": result.score,
                    "rank": i + 1,
                    "page_number": result.chunk.metadata.get("page_number"),
                }
            )

        return citations


class AnswerModel(BaseModel):
    """Model for generated answers."""

    question: str = Field(description="Original question")
    answer: str = Field(description="Generated answer")
    context: QueryContextModel = Field(description="Query context used")
    confidence_score: float = Field(description="Answer confidence")
    model_used: str = Field(description="LLM model used")
    generation_time_ms: float = Field(description="Answer generation time")

    def to_answer_response_dto(self):
        """Convert to AnswerResponse DTO."""
        from api.features.query.dtos import AnswerResponse, Citation

        citations = [
            Citation(
                chunk_id=result.chunk.id,
                document_id=result.chunk.document_id,
                document_filename=result.document_filename,
                content_excerpt=result.chunk.content[:200] + "..."
                if len(result.chunk.content) > 200
                else result.chunk.content,
                relevance_score=result.score,
                page_number=result.chunk.metadata.get("page_number"),
            )
            for result in self.context.search_results
        ]

        return AnswerResponse(
            question=self.question,
            answer=self.answer,
            citations=citations,
            confidence_score=self.confidence_score,
            processing_time_ms=self.context.processing_time_ms
            + self.generation_time_ms,
            model_used=self.model_used,
        )
