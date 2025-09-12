"""Service layer for the Query feature."""
import logging
import time
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.features.query.exceptions import (
    SearchError,
    EmbeddingError,
    AnswerGenerationError,
    NoResultsError,
    InsufficientContextError
)
from api.features.query.models import (
    ChunkModel,
    EmbeddingModel,
    SearchResultModel,
    QueryContextModel,
    AnswerModel
)
from api.features.query.entities.chunk import Chunk as ChunkEntity
from api.features.query.entities.embedding import Embedding as EmbeddingEntity
from api.features.documents.entities.document import Document as DocumentEntity, DocumentStatus, DocumentType

logger = logging.getLogger("rag.query.service")


class QueryService:
    """Service for query and search operations."""
    
    def __init__(self, embedding_client=None, llm_client=None):
        self.embedding_client = embedding_client
        self.llm_client = llm_client
    
    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        document_filters: Optional[List[UUID]] = None,
        db_session: AsyncSession = None
    ) -> QueryContextModel:
        """Perform semantic search using vector similarity."""
        start_time = time.time()
        
        try:
            # Generate query embedding
            if not self.embedding_client:
                raise EmbeddingError("Embedding client not configured")
            
            # TODO: Implement actual embedding generation
            query_embedding = await self._generate_embedding(query)
            
            # Build search query
            search_query = """
            SELECT 
                c.id as chunk_id,
                c.document_id,
                c.content,
                c.position,
                c.metadata,
                c.created_at,
                d.filename,
                e.vector <=> %s as distance
            FROM chunks c
            JOIN embeddings e ON c.id = e.chunk_id
            JOIN documents d ON c.document_id = d.id
            WHERE e.vector <=> %s < %s
            """
            
            params = [query_embedding, query_embedding, 1.0 - similarity_threshold]
            
            # Add document filters
            if document_filters:
                placeholders = ','.join(['%s'] * len(document_filters))
                search_query += f" AND c.document_id IN ({placeholders})"
                params.extend([str(doc_id) for doc_id in document_filters])
            
            search_query += " ORDER BY distance LIMIT %s"
            params.append(top_k)
            
            # Execute search
            result = await db_session.execute(text(search_query), params)
            rows = result.fetchall()
            
            # Convert to models
            search_results = []
            for i, row in enumerate(rows):
                chunk = ChunkModel(
                    id=row.chunk_id,
                    document_id=row.document_id,
                    content=row.content,
                    position=row.position,
                    metadata=row.metadata or {},
                    created_at=row.created_at
                )
                
                search_result = SearchResultModel(
                    chunk=chunk,
                    score=1.0 - row.distance,  # Convert distance to similarity score
                    document_filename=row.filename,
                    rank=i + 1
                )
                search_results.append(search_result)
            
            processing_time = (time.time() - start_time) * 1000
            
            return QueryContextModel(
                query=query,
                search_results=search_results,
                total_results=len(search_results),
                processing_time_ms=processing_time,
                search_type="semantic"
            )
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            raise SearchError(f"Semantic search failed: {str(e)}")
    
    async def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        use_stemming: bool = True,
        use_fuzzy: bool = False,
        db_session: AsyncSession = None
    ) -> QueryContextModel:
        """Perform keyword search using full-text search."""
        start_time = time.time()
        
        try:
            # Build FTS query
            search_query = """
            SELECT 
                c.id as chunk_id,
                c.document_id,
                c.content,
                c.position,
                c.metadata,
                c.created_at,
                d.filename,
                ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', %s)) as rank
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
            """
            
            params = [query, query, top_k]
            
            # Execute search
            result = await db_session.execute(text(search_query), params)
            rows = result.fetchall()
            
            # Convert to models
            search_results = []
            for i, row in enumerate(rows):
                chunk = ChunkModel(
                    id=row.chunk_id,
                    document_id=row.document_id,
                    content=row.content,
                    position=row.position,
                    metadata=row.metadata or {},
                    created_at=row.created_at
                )
                
                search_result = SearchResultModel(
                    chunk=chunk,
                    score=float(row.rank),
                    document_filename=row.filename,
                    rank=i + 1
                )
                search_results.append(search_result)
            
            processing_time = (time.time() - start_time) * 1000
            
            return QueryContextModel(
                query=query,
                search_results=search_results,
                total_results=len(search_results),
                processing_time_ms=processing_time,
                search_type="keyword"
            )
            
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            raise SearchError(f"Keyword search failed: {str(e)}")
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        db_session: AsyncSession = None
    ) -> QueryContextModel:
        """Perform hybrid search combining semantic and keyword search."""
        start_time = time.time()
        
        try:
            # Perform both searches
            semantic_results = await self.semantic_search(
                query=query,
                top_k=top_k * 2,  # Get more results for better fusion
                db_session=db_session
            )
            
            keyword_results = await self.keyword_search(
                query=query,
                top_k=top_k * 2,
                db_session=db_session
            )
            
            # Combine and re-rank results using weighted scores
            combined_results = {}
            
            # Add semantic results
            for result in semantic_results.search_results:
                chunk_id = str(result.chunk.id)
                combined_results[chunk_id] = {
                    'result': result,
                    'semantic_score': result.score,
                    'keyword_score': 0.0
                }
            
            # Add keyword results
            for result in keyword_results.search_results:
                chunk_id = str(result.chunk.id)
                if chunk_id in combined_results:
                    combined_results[chunk_id]['keyword_score'] = result.score
                else:
                    combined_results[chunk_id] = {
                        'result': result,
                        'semantic_score': 0.0,
                        'keyword_score': result.score
                    }
            
            # Calculate hybrid scores and sort
            hybrid_results = []
            for chunk_id, data in combined_results.items():
                hybrid_score = (
                    semantic_weight * data['semantic_score'] +
                    keyword_weight * data['keyword_score']
                )
                
                result = data['result']
                result.score = hybrid_score
                hybrid_results.append(result)
            
            # Sort by hybrid score and take top_k
            hybrid_results.sort(key=lambda x: x.score, reverse=True)
            final_results = hybrid_results[:top_k]
            
            # Update ranks
            for i, result in enumerate(final_results):
                result.rank = i + 1
            
            processing_time = (time.time() - start_time) * 1000
            
            return QueryContextModel(
                query=query,
                search_results=final_results,
                total_results=len(final_results),
                processing_time_ms=processing_time,
                search_type="hybrid"
            )
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            raise SearchError(f"Hybrid search failed: {str(e)}")
    
    async def generate_answer(
        self,
        question: str,
        context: QueryContextModel,
        max_tokens: int = 500,
        temperature: float = 0.1,
        model: str = "gpt-3.5-turbo"
    ) -> AnswerModel:
        """Generate an answer using RAG."""
        start_time = time.time()
        
        try:
            if not self.llm_client:
                raise AnswerGenerationError("LLM client not configured", model)
            
            if not context.search_results:
                raise InsufficientContextError(1, 0)
            
            # Build context from search results
            context_text = context.get_context_text(max_chunks=5)
            
            # Create prompt
            prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say so clearly.

Context:
{context_text}

Question: {question}

Answer:"""
            
            # TODO: Implement actual LLM call
            # For now, return a mock answer
            answer = f"This is a mock answer for the question: '{question}'. In a real implementation, this would be generated by an LLM using the provided context."
            
            generation_time = (time.time() - start_time) * 1000
            
            return AnswerModel(
                question=question,
                answer=answer,
                context=context,
                confidence_score=0.85,  # Mock confidence score
                model_used=model,
                generation_time_ms=generation_time
            )
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            raise AnswerGenerationError(str(e), model)
    
    async def get_query_suggestions(
        self,
        prefix: str = "",
        limit: int = 10,
        db_session: AsyncSession = None
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
                "What challenges does the company face?"
            ]
            
            # Filter by prefix if provided
            if prefix:
                suggestions = [s for s in suggestions if s.lower().startswith(prefix.lower())]
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get query suggestions: {str(e)}")
            raise SearchError(f"Failed to get suggestions: {str(e)}")
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            # TODO: Implement actual embedding generation
            # For now, return a mock embedding
            return [0.1] * 1536  # Mock OpenAI embedding dimension
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")
