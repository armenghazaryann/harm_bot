"""
LangChain-based Query Service for Production RAG.

This module implements production-ready LangChain patterns for querying:
- RetrievalQA chains with proper prompt templates
- PGVector similarity search with metadata filtering
- Multi-document querying capabilities
- Semantic and hybrid search strategies
- Production error handling and response formatting

FAANG-Level Architecture:
- Leverage LangChain's battle-tested RetrievalQA chains
- Optimize for production performance and scalability
- Clean separation of concerns and error handling
"""
import asyncio
import time
import structlog
from typing import Dict, Any, List, Optional
from enum import Enum
from functools import wraps

# LangChain Core Components
from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Infrastructure
from core.settings import SETTINGS

logger = structlog.get_logger("workers.langchain_query_service")


def timing_decorator(func):
    """Decorator to add performance timing to async functions for production monitoring."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            end_time = time.time()
            processing_time_ms = (end_time - start_time) * 1000

            # Add timing to result if it's a dict
            if isinstance(result, dict):
                result["performance_metrics"] = {
                    "processing_time_ms": round(processing_time_ms, 2),
                    "timestamp": int(time.time()),
                    "function": func.__name__,
                }

            logger.info(
                f"{func.__name__}_performance",
                processing_time_ms=processing_time_ms,
                function=func.__name__,
            )
            return result
        except Exception as e:
            end_time = time.time()
            processing_time_ms = (end_time - start_time) * 1000
            logger.error(
                f"{func.__name__}_error",
                processing_time_ms=processing_time_ms,
                error=str(e),
            )
            raise

    return wrapper


def calculate_confidence_score(answer: str, source_docs: List, query: str) -> float:
    """Calculate confidence score based on answer quality and source relevance."""
    try:
        # Basic confidence scoring algorithm for production monitoring
        base_score = 0.5

        # Factor 1: Number of source documents (more sources = higher confidence)
        source_count = len(source_docs) if source_docs else 0
        source_factor = min(source_count / 5.0, 1.0) * 0.2

        # Factor 2: Answer length (moderate length indicates good coverage)
        answer_length = len(answer.split()) if answer else 0
        length_factor = 0.0
        if 10 <= answer_length <= 200:
            length_factor = 0.2
        elif 5 <= answer_length < 10 or 200 < answer_length <= 300:
            length_factor = 0.1

        # Factor 3: Presence of uncertainty phrases (lower confidence)
        uncertainty_phrases = [
            "i don't know",
            "not sure",
            "unclear",
            "maybe",
            "possibly",
        ]
        uncertainty_penalty = 0.0
        if answer:
            answer_lower = answer.lower()
            for phrase in uncertainty_phrases:
                if phrase in answer_lower:
                    uncertainty_penalty += 0.1

        # Calculate final confidence score
        confidence = base_score + source_factor + length_factor - uncertainty_penalty
        return max(0.1, min(1.0, confidence))  # Clamp between 0.1 and 1.0

    except Exception as e:
        logger.warning(f"Confidence calculation failed: {e}")
        return 0.5  # Default confidence


class SearchType(str, Enum):
    """Search type enumeration for different query strategies."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"  # Vector Store + BM25 ensemble
    SELF_QUERY = "self_query"


class LangChainQueryService:
    """Production-ready LangChain query service for RAG applications."""

    def __init__(self):
        """Initialize LangChain components with production settings."""
        # Chat Model - GPT-4 for production RAG
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,  # Low temperature for factual responses
            openai_api_key=SETTINGS.OPENAI.OPENAI_API_KEY.get_secret_value(),
            max_tokens=1000,  # Reasonable response length
        )

        # Embeddings - Same model as used in document processing
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=SETTINGS.OPENAI.OPENAI_API_KEY.get_secret_value(),
            dimensions=1536,
        )

        # PGVector connection string
        self.connection_string = (
            f"postgresql+psycopg://{SETTINGS.DATABASE.POSTGRES_USER}:"
            f"{SETTINGS.DATABASE.POSTGRES_PASSWORD.get_secret_value()}@{SETTINGS.DATABASE.POSTGRES_HOST}:"
            f"{SETTINGS.DATABASE.POSTGRES_PORT}/{SETTINGS.DATABASE.POSTGRES_DB}"
        )

        # Production-ready prompt templates
        self.qa_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:""",
        )

        # Detailed prompt for comprehensive responses
        self.detailed_qa_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert research assistant. Use the following retrieved context to provide a comprehensive, well-structured answer to the question.

Instructions:
- Use only information from the provided context
- Structure your response with clear sections if appropriate
- Include relevant details and examples from the context
- If the context doesn't contain enough information, clearly state what's missing
- Cite specific parts of the context when making claims

Context:
{context}

Question: {question}

Comprehensive Answer:""",
        )

        # Metadata attributes for self-query retrieval
        self.metadata_field_info = [
            AttributeInfo(
                name="doc_id",
                description="The document ID that the chunk belongs to",
                type="string",
            ),
            AttributeInfo(
                name="page_number",
                description="The page number in the original document",
                type="integer",
            ),
            AttributeInfo(
                name="extraction_method",
                description="Method used to extract the text (PyPDFLoader, UnstructuredPDFLoader)",
                type="string",
            ),
            AttributeInfo(
                name="processing_pipeline",
                description="Processing pipeline used (langchain, legacy)",
                type="string",
            ),
        ]

        self.document_content_description = (
            "Document chunks from processed PDFs containing text content"
        )

    def _get_vector_store(self, collection_name: str = None) -> PGVector:
        """Get PGVector store instance for a specific collection."""
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
        collection_name: str = None,
    ) -> EnsembleRetriever:
        """
        Create production-grade hybrid retriever combining Vector Store + BM25.

        This implements true hybrid search using:
        - Vector Store: Semantic similarity search
        - BM25: Keyword-based search
        - EnsembleRetriever: Combines both with optimal weighting

        Args:
            query: Search query
            doc_id: Optional document filter
            k: Number of results per retriever
            collection_name: Vector store collection

        Returns:
            Configured EnsembleRetriever for hybrid search
        """
        # Get vector store retriever
        vector_store = self._get_vector_store(collection_name)
        # Log which collection we're querying to avoid mismatch issues
        try:
            logger.info(
                "hybrid_retriever.collection",
                collection=collection_name or SETTINGS.VECTOR.COLLECTION_NAME,
            )
        except Exception:
            pass
        search_kwargs = {"k": k}
        if doc_id:
            search_kwargs["filter"] = {"doc_id": doc_id}

        vector_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

        # Get documents for BM25 retriever using query-aware sampling
        # Use the actual query to fetch a broader candidate set for BM25
        # In production, consider caching or maintaining a text corpus
        all_docs = await asyncio.get_event_loop().run_in_executor(
            None, lambda: vector_store.similarity_search(query, k=max(k * 20, 200))
        )

        # Filter documents by doc_id if specified
        if doc_id:
            all_docs = [doc for doc in all_docs if doc.metadata.get("doc_id") == doc_id]

        # If no documents for BM25, fallback to vector-only retriever
        if not all_docs:
            logger.info(
                "BM25 corpus empty, falling back to vector-only retriever", query=query
            )
            return vector_retriever

        try:
            # Create BM25 retriever from documents
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = k
        except Exception as e:
            logger.warning(
                "BM25 retriever creation failed, using vector-only retriever",
                error=str(e),
            )
            return vector_retriever

        # Create ensemble retriever with optimal weights
        # 0.6 for vector (semantic), 0.4 for BM25 (keyword)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.6, 0.4],
        )

        return ensemble_retriever

    async def semantic_search(
        self,
        query: str,
        doc_id: Optional[str] = None,
        k: int = 5,
        collection_name: str = None,
    ) -> List[Document]:
        """
        Perform semantic similarity search using PGVector.

        Args:
            query: Search query
            doc_id: Optional document ID filter
            k: Number of results to return
            collection_name: Vector store collection name

        Returns:
            List of relevant documents
        """
        try:
            vector_store = self._get_vector_store(collection_name)
            try:
                logger.info(
                    "semantic_search.collection",
                    collection=collection_name or SETTINGS.VECTOR.COLLECTION_NAME,
                )
            except Exception:
                pass

            # Build search kwargs
            search_kwargs = {"k": k}
            if doc_id:
                search_kwargs["filter"] = {"doc_id": doc_id}

            # Perform similarity search with scores
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: vector_store.similarity_search_with_score(
                    query, **search_kwargs
                ),
            )
            # Unpack and attach scores into metadata for downstream consumers
            docs = []
            for doc, distance in results:
                try:
                    doc.metadata = dict(doc.metadata or {})
                    d = float(distance)
                    # Convert distance to similarity (assume cosine distance if in [0,1])
                    if 0.0 <= d <= 1.0:
                        sim = 1.0 - d
                    else:
                        sim = 1.0 / (1.0 + max(d, 0.0))
                    doc.metadata["distance"] = d
                    doc.metadata["score"] = round(sim, 6)
                except Exception:
                    pass
                docs.append(doc)

            logger.info(
                f"Semantic search returned {len(docs)} documents",
                query=query,
                doc_id=doc_id,
                k=k,
            )

            return docs

        except Exception as e:
            logger.error(f"Semantic search failed: {e}", query=query, doc_id=doc_id)
            return []

    async def answer_question(
        self,
        question: str,
        doc_id: Optional[str] = None,
        search_type: SearchType = SearchType.SEMANTIC,
        detailed: bool = False,
        k: int = 5,
        collection_name: str = None,
    ) -> Dict[str, Any]:
        """
        Answer a question using LangChain RetrievalQA chain.

        Args:
            question: Question to answer
            doc_id: Optional document ID filter
            search_type: Search strategy to use
            detailed: Whether to use detailed prompt template
            k: Number of chunks to retrieve
            collection_name: Vector store collection name

        Returns:
            Answer with metadata and source documents
        """
        try:
            vector_store = self._get_vector_store(collection_name)

            # Configure retriever based on search type
            if search_type == SearchType.HYBRID:
                # Use production-grade hybrid search (Vector Store + BM25)
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
                # Standard similarity search retriever (semantic or keyword)
                search_kwargs = {"k": k}
                if doc_id:
                    search_kwargs["filter"] = {"doc_id": doc_id}

                retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

            # Choose prompt template
            prompt_template = (
                self.detailed_qa_prompt_template
                if detailed
                else self.qa_prompt_template
            )

            # Create RetrievalQA chain using factory to match latest LangChain schema
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt_template},
                return_source_documents=True,
            )

            # Execute query using modern invoke API
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: qa_chain.invoke({"query": question})
            )

            # Format response; add semantic fallback if no sources returned
            source_docs = result.get("source_documents", [])
            if not source_docs:
                try:
                    # Perform vector-only semantic search as a fallback
                    fallback_scored = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: vector_store.similarity_search_with_score(
                            question, k=k
                        ),
                    )
                    source_docs = [d for (d, _s) in fallback_scored] or []
                    # Attach scores for consistency
                    for d, dist in fallback_scored:
                        try:
                            md = dict(d.metadata or {})
                            dd = float(dist)
                            if 0.0 <= dd <= 1.0:
                                sim = 1.0 - dd
                            else:
                                sim = 1.0 / (1.0 + max(dd, 0.0))
                            md["distance"] = dd
                            md["score"] = round(sim, 6)
                            d.metadata = md
                        except Exception:
                            pass
                    # Keep original result answer, but attach sources for downstream consumers
                    result["source_documents"] = source_docs
                except Exception as e:
                    logger.info("semantic_fallback_failed", error=str(e))
            else:
                # We have sources from retriever (hybrid or otherwise). Attach semantic scores by re-scoring the question
                try:
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
                                if 0.0 <= dd <= 1.0:
                                    sim = 1.0 - dd
                                else:
                                    sim = 1.0 / (1.0 + max(dd, 0.0))
                                md["distance"] = dd
                                md["score"] = round(sim, 6)
                                source_docs[idx].metadata = md
                            except Exception:
                                pass
                except Exception as e:
                    logger.info("rescoring_failed", error=str(e))
            response = {
                "answer": result.get("result", ""),
                "question": question,
                "search_type": search_type.value,
                "source_count": len(source_docs),
                "source_documents": source_docs,
                "sources": [
                    {
                        "content": doc.page_content[:200]
                        + "...",  # Truncate for response
                        "metadata": doc.metadata,
                        "doc_id": doc.metadata.get("doc_id"),
                        "page_number": doc.metadata.get("page_number"),
                    }
                    for doc in source_docs
                ],
                "processing_info": {
                    "retriever_type": type(retriever).__name__,
                    "chunks_retrieved": k,
                    "prompt_type": "detailed" if detailed else "concise",
                    "doc_filter": doc_id,
                },
            }

            logger.info(
                "Question answered successfully",
                question=question,
                source_count=len(source_docs),
                search_type=search_type.value,
            )

            return response

        except Exception as e:
            logger.error(f"Question answering failed: {e}", question=question)
            return {
                "answer": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "question": question,
                "error": str(e),
                "search_type": search_type.value,
                "source_count": 0,
                "sources": [],
            }

    async def multi_document_query(
        self,
        question: str,
        doc_ids: List[str],
        search_type: SearchType = SearchType.SEMANTIC,
        k_per_doc: int = 3,
        collection_name: str = "langchain_pgvector",
    ) -> Dict[str, Any]:
        """
        Query across multiple documents with aggregated results.

        Args:
            question: Question to answer
            doc_ids: List of document IDs to search
            search_type: Search strategy
            k_per_doc: Number of chunks per document
            collection_name: Vector store collection name

        Returns:
            Aggregated answer with sources from multiple documents
        """
        try:
            all_docs = []
            doc_results = {}

            # Query each document separately
            for doc_id in doc_ids:
                docs = await self.semantic_search(
                    query=question,
                    doc_id=doc_id,
                    k=k_per_doc,
                    collection_name=collection_name,
                )
                all_docs.extend(docs)
                doc_results[doc_id] = len(docs)

            if not all_docs:
                return {
                    "answer": "No relevant information found in the specified documents.",
                    "question": question,
                    "doc_ids": doc_ids,
                    "source_count": 0,
                    "sources": [],
                }

            # Create retriever from collected documents

            # Use detailed prompt for multi-document queries (direct prompt below)

            # Manually provide the documents we retrieved
            context = "\n\n".join([doc.page_content for doc in all_docs])

            # Get answer from LLM
            answer = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.predict(
                    self.detailed_qa_prompt_template.format(
                        context=context, question=question
                    )
                ),
            )

            response = {
                "answer": answer,
                "question": question,
                "search_type": search_type.value,
                "doc_ids": doc_ids,
                "source_count": len(all_docs),
                "doc_results": doc_results,
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata,
                        "doc_id": doc.metadata.get("doc_id"),
                        "page_number": doc.metadata.get("page_number"),
                    }
                    for doc in all_docs
                ],
                "processing_info": {
                    "multi_document_query": True,
                    "documents_queried": len(doc_ids),
                    "total_chunks": len(all_docs),
                    "chunks_per_doc": k_per_doc,
                },
            }

            logger.info(
                "Multi-document query completed",
                question=question,
                doc_count=len(doc_ids),
                total_chunks=len(all_docs),
            )

            return response

        except Exception as e:
            logger.error(f"Multi-document query failed: {e}", question=question)
            return {
                "answer": f"Error processing multi-document query: {str(e)}",
                "question": question,
                "doc_ids": doc_ids,
                "error": str(e),
                "source_count": 0,
                "sources": [],
            }


# Singleton instance for application use
query_service = LangChainQueryService()


# Async wrapper functions for easy integration
@timing_decorator
async def query_documents(
    question: str,
    doc_id: Optional[str] = None,
    search_type: str = "semantic",
    detailed: bool = False,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Main query function for document question answering.

    Args:
        question: Question to answer
        doc_id: Optional document ID filter
        search_type: Search strategy ("semantic", "keyword", "hybrid", "self_query")
        detailed: Whether to provide detailed responses
        k: Number of chunks to retrieve

    Returns:
        Answer with sources and metadata
    """
    result = await query_service.answer_question(
        question=question,
        doc_id=doc_id,
        search_type=SearchType(search_type),
        detailed=detailed,
        k=k,
    )

    # Add confidence scoring for production monitoring
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
    """
    Query across multiple documents.

    Args:
        question: Question to answer
        doc_ids: List of document IDs
        search_type: Search strategy
        k_per_doc: Chunks per document

    Returns:
        Aggregated answer from multiple documents
    """
    result = await query_service.multi_document_query(
        question=question,
        doc_ids=doc_ids,
        search_type=SearchType(search_type),
        k_per_doc=k_per_doc,
    )

    # Add confidence scoring for production monitoring
    if isinstance(result, dict) and "answer" in result:
        confidence_score = calculate_confidence_score(
            answer=result.get("answer", ""),
            source_docs=result.get("source_documents", []),
            query=question,
        )
        result["confidence_score"] = confidence_score

    return result


@timing_decorator
async def semantic_search_documents(
    query: str, doc_id: Optional[str] = None, k: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform semantic search without LLM answer generation.

    Args:
        query: Search query
        doc_id: Optional document filter
        k: Number of results

    Returns:
        List of matching document chunks
    """
    docs = await query_service.semantic_search(query=query, doc_id=doc_id, k=k)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "doc_id": doc.metadata.get("doc_id"),
            "page_number": doc.metadata.get("page_number"),
        }
        for doc in docs
    ]
