"""
LangChain-based document processing pipeline for RAG ETL.

This module implements production-ready LangChain patterns:
- DocumentLoaders for reliable PDF processing
- RecursiveCharacterTextSplitter for optimal chunking
- OpenAI embeddings with PGVector integration
- Async processing patterns for Celery workers

FAANG-Level Architecture:
- Leverage battle-tested LangChain components
- Eliminate custom boilerplate code
- Follow LangChain best practices for production RAG
"""
import asyncio
import structlog
from typing import Dict, Any
from pathlib import Path

# LangChain Core Components
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

# Infrastructure
from core.settings import SETTINGS
from infra.resources import MinIOResource
from api.features.documents.entities.document import Document as DocumentEntity

logger = structlog.get_logger("workers.langchain_processor")


class LangChainDocumentProcessor:
    """Production-ready LangChain document processing pipeline."""

    def __init__(self):
        """Initialize LangChain components with production settings."""
        # Text Splitter - LangChain best practices
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Optimal chunk size for embeddings
            chunk_overlap=200,  # Prevent information loss at boundaries
            add_start_index=True,  # Track position in original document
            separators=["\n\n", "\n", " ", ""],  # Semantic splitting
        )

        # Embeddings - OpenAI text-embedding-3-large (production recommended)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=SETTINGS.OPENAI.OPENAI_API_KEY.get_secret_value(),
            dimensions=1536,  # Optimal for most use cases
        )

        # PGVector connection string
        self.connection_string = (
            f"postgresql+psycopg://{SETTINGS.DATABASE.POSTGRES_USER}:"
            f"{SETTINGS.DATABASE.POSTGRES_PASSWORD.get_secret_value()}@{SETTINGS.DATABASE.POSTGRES_HOST}:"
            f"{SETTINGS.DATABASE.POSTGRES_PORT}/{SETTINGS.DATABASE.POSTGRES_DB}"
        )

    async def process_pdf_from_minio(
        self, doc_id: str, storage_path: str, minio_resource: MinIOResource
    ) -> Dict[str, Any]:
        """
        Process PDF using LangChain DocumentLoaders.

        Args:
            doc_id: Document ID
            storage_path: MinIO storage path
            minio_resource: MinIO resource instance

        Returns:
            Processing results with chunk count and metadata
        """
        try:
            # Download PDF from MinIO to temporary file
            pdf_bytes = await minio_resource.get_object_bytes(
                bucket_name=SETTINGS.MINIO.MINIO_BUCKET, object_name=storage_path
            )

            # Save to temporary file for LangChain DocumentLoaders
            temp_path = f"/tmp/{doc_id}.pdf"
            with open(temp_path, "wb") as f:
                f.write(pdf_bytes)

            # Use LangChain PyPDFLoader for reliable PDF processing
            try:
                loader = PyPDFLoader(temp_path)
                documents = await asyncio.get_event_loop().run_in_executor(
                    None, loader.load
                )
                extraction_method = "PyPDFLoader"
            except Exception as e:
                logger.warning(f"PyPDFLoader failed, trying UnstructuredPDFLoader: {e}")
                # Fallback to UnstructuredPDFLoader
                loader = UnstructuredPDFLoader(temp_path)
                documents = await asyncio.get_event_loop().run_in_executor(
                    None, loader.load
                )
                extraction_method = "UnstructuredPDFLoader"

            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)

            logger.info(f"Loaded {len(documents)} pages with {extraction_method}")

            # Split documents using LangChain RecursiveCharacterTextSplitter
            chunks = self.text_splitter.split_documents(documents)

            logger.info(f"Split into {len(chunks)} chunks")

            # Add document metadata to each chunk
            for chunk in chunks:
                chunk.metadata.update(
                    {
                        "doc_id": doc_id,
                        "extraction_method": extraction_method,
                        "processing_pipeline": "langchain",
                    }
                )

            # Store embeddings in PGVector using LangChain
            vector_store = PGVector(
                embeddings=self.embeddings,
                connection=self.connection_string,
                collection_name=SETTINGS.VECTOR.COLLECTION_NAME,  # Shared collection for all documents
                use_jsonb=SETTINGS.VECTOR.VECTOR_USE_JSONB,  # Store metadata in JSONB for better querying
            )

            # Add documents to vector store (handles embedding generation automatically)
            document_ids = await asyncio.get_event_loop().run_in_executor(
                None, vector_store.add_documents, chunks
            )

            logger.info(f"Stored {len(document_ids)} embeddings in PGVector")

            return {
                "doc_id": doc_id,
                "chunks_created": len(chunks),
                "embeddings_stored": len(document_ids),
                "extraction_method": extraction_method,
                "processing_pipeline": "langchain",
                "vector_ids": document_ids[:5],  # Sample of vector IDs
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"LangChain processing failed for {doc_id}: {e}")
            return {
                "doc_id": doc_id,
                "status": "failed",
                "error": str(e),
                "processing_pipeline": "langchain",
            }


async def process_document_with_langchain(doc_id: str) -> Dict[str, Any]:
    """
    Celery task function: Process document using pure LangChain pipeline.

    This replaces all custom processing with LangChain best practices:
    - DocumentLoader → TextSplitter → Embeddings → VectorStore

    Args:
        doc_id: Document ID to process

    Returns:
        Processing results
    """
    from workers.initialization import worker_initializer

    async with worker_initializer.worker_context() as container:
        processor = LangChainDocumentProcessor()

        # Get database session - these are already initialized resources
        db_resource = container.infrastructure.database()
        minio_resource = container.infrastructure.minio_client()

        async with db_resource.get_session() as session:
            # Load document entity
            from sqlalchemy import select

            stmt = select(DocumentEntity).where(DocumentEntity.id == doc_id)
            result = await session.execute(stmt)
            document = result.scalar_one_or_none()

            if not document:
                return {
                    "doc_id": doc_id,
                    "status": "failed",
                    "error": "Document not found",
                }

            try:
                # Process with LangChain pipeline
                result = await processor.process_pdf_from_minio(
                    doc_id=doc_id,
                    storage_path=document.raw_path,
                    minio_resource=minio_resource,
                )

                # Update document processing metadata
                processing_metadata = document.processing_metadata or {}
                processing_metadata.update(
                    {"langchain_processing": result, "pipeline_version": "langchain_v1"}
                )
                document.processing_metadata = processing_metadata
                await session.commit()

                return result

            except Exception as e:
                await session.rollback()
                logger.error(
                    "process_document_with_langchain.error", doc_id=doc_id, error=str(e)
                )
                raise


# Legacy function compatibility - gradually phase out
async def extract_transcript_utterances_from_minio(doc_id: str) -> Dict[str, Any]:
    """
    DEPRECATED: Legacy transcript processing.
    Use process_document_with_langchain() instead.

    This function is kept for backward compatibility during migration.
    """
    logger.warning(
        f"Using deprecated extract_transcript_utterances_from_minio for {doc_id}. "
        "Migrate to process_document_with_langchain()"
    )

    # Fallback to new LangChain processor
    return await process_document_with_langchain(doc_id)
