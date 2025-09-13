# RAG ETL Pipeline: Production-Grade Architecture

## Architecture Overview

This document outlines the final architecture of our production-grade RAG (Retrieval-Augmented Generation) ETL pipeline after completing the migration to a fully LangChain-based implementation. The architecture follows FAANG-level engineering standards with emphasis on scalability, maintainability, and performance.

## System Components

### 1. Document Processing Pipeline

**LangChain Document Processor**
- **Document Loading**: Uses `PyPDFLoader` and `UnstructuredPDFLoader` for robust PDF extraction
- **Text Chunking**: Implements `RecursiveCharacterTextSplitter` for optimal semantic chunking
- **Embedding Generation**: Utilizes `OpenAIEmbeddings` with text-embedding-3-large model
- **Vector Storage**: Leverages `PGVector` for PostgreSQL-native vector storage with HNSW indexing

**Celery Task Orchestration**
- Asynchronous document processing with idempotency guards
- Error handling with structured logging
- Progress tracking with document status updates

### 2. Hybrid Search System

**Production-Grade Retrieval**
- **EnsembleRetriever**: Combines vector similarity and BM25 keyword search
  - Vector Store (0.6 weight): Semantic understanding via embeddings
  - BM25 (0.4 weight): Keyword matching for terminology precision
- **Maximum Marginal Relevance (MMR)**: Ensures result diversity and reduces redundancy

**Performance Optimization**
- Timing decorator for performance monitoring
- Confidence scoring algorithm for result quality assessment
- Structured logging for observability

### 3. API Layer

**FastAPI Endpoints**
- RESTful API with OpenAPI documentation
- Dependency injection with proper resource lifecycle management
- Comprehensive validation and error handling

**Query Controller**
- Unified interface for semantic, keyword, and hybrid search
- Question answering with context-aware responses
- Citation tracking for result provenance

## Data Flow

1. **Document Upload**
   - Document uploaded via API and stored in MinIO
   - Document metadata recorded in PostgreSQL
   - Processing job created and queued

2. **Document Processing**
   - PDF extraction with fallback mechanisms
   - Text chunking with semantic boundaries
   - Embedding generation and vector storage

3. **Query Processing**
   - Query embedding generation
   - Hybrid retrieval (vector + keyword)
   - Context assembly and (optional) LLM answer generation

## Technical Stack

- **Backend**: Python 3.13+, FastAPI
- **Database**: PostgreSQL with pgvector extension
- **Vector Storage**: LangChain PGVector
- **Object Storage**: MinIO
- **Task Queue**: Celery with Redis broker
- **Embeddings**: OpenAI text-embedding-3-large
- **LLM**: OpenAI GPT-4 Turbo
- **Containerization**: Docker, docker-compose

## Performance Characteristics

- **Retrieval Quality**: Enhanced by hybrid search combining semantic and keyword matching
- **Response Time**: Optimized through vector indexing and caching
- **Scalability**: Horizontal scaling via containerization and message queues
- **Reliability**: Error handling, retries, and graceful degradation

## Security Considerations

- Environment-based configuration for sensitive values
- API key management for external services
- Input validation and sanitization
- Proper error handling to prevent information leakage

## Monitoring and Observability

- Performance timing for all critical operations
- Confidence scoring for retrieval quality assessment
- Structured logging with correlation IDs
- Health check endpoints for system status

## Migration Summary

The system has been successfully migrated from a custom implementation to a fully LangChain-based architecture, resulting in:

1. **Reduced Code Complexity**: Eliminated custom ML/model code in favor of battle-tested LangChain components
2. **Enhanced Retrieval Quality**: Implemented hybrid search combining vector similarity and BM25
3. **Improved Maintainability**: Standardized on LangChain's document and vector store abstractions
4. **Better Performance**: Leveraged optimized implementations for embedding and retrieval
5. **Production Readiness**: Added observability, error handling, and performance monitoring

## Future Enhancements

1. **Retrieval Augmentation**: Experiment with query expansion and rewriting
2. **Multi-Modal Support**: Extend to handle images and other document types
3. **Evaluation Framework**: Implement systematic evaluation of retrieval quality
4. **Caching Layer**: Add result caching for frequently asked queries
5. **Streaming Responses**: Implement streaming for large response payloads
