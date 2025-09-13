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
import re
import json
import base64
from typing import Dict, Any, List
from pathlib import Path
from io import BytesIO

# LangChain Core Components
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

# Image processing and OpenAI Vision
try:
    import fitz  # PyMuPDF for PDF to image conversion

    _PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    _PYMUPDF_AVAILABLE = False

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:
    Image = None
    _PIL_AVAILABLE = False

import openai

# Table extraction libraries
try:
    import camelot  # type: ignore

    _CAMELOT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    camelot = None  # type: ignore
    _CAMELOT_AVAILABLE = False

import pdfplumber

# Infrastructure
from core.settings import SETTINGS
from infra.resources import MinIOResource
from api.features.documents.entities.document import (
    Document as DocumentEntity,
    DocumentType,
)

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

    async def process_earning_transcript_from_minio(
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
                        "document_id": doc_id,  # Ensure document_id is included for langchain_embedding table
                        "doc_id": doc_id,
                        "extraction_method": extraction_method,
                        "processing_pipeline": "langchain",
                        "content_type": "text",
                        "source_path": storage_path,
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
            logger.error(f"PDF processing failed for {doc_id}: {e}")
            return {
                "doc_id": doc_id,
                "status": "failed",
                "error": str(e),
                "processing_pipeline": "langchain",
            }

    # Earnings Release specific processing with table extraction
    async def process_earnings_release_from_minio(
        self, doc_id: str, storage_path: str, minio_resource: MinIOResource
    ) -> Dict[str, Any]:
        """
        Process an earnings release PDF with high-accuracy table extraction.

        Strategy (best practices):
        - Prefer Camelot 'lattice' for ruled financial tables (if available)
        - Fallback to Camelot 'stream' for whitespace-separated tables (if available)
        - Fallback to pdfplumber with lines/text strategies
        - Combine: text via PyPDFLoader (+ Unstructured fallback) and tables as
          standalone per-row JSON Documents to preserve schema and semantics
        """
        try:
            # Download PDF from MinIO
            pdf_bytes = await minio_resource.get_object_bytes(
                bucket_name=SETTINGS.MINIO.MINIO_BUCKET, object_name=storage_path
            )

            temp_path = f"/tmp/{doc_id}.pdf"
            with open(temp_path, "wb") as f:
                f.write(pdf_bytes)

            # Text extraction
            try:
                loader = PyPDFLoader(temp_path)
                text_documents = await asyncio.get_event_loop().run_in_executor(
                    None, loader.load
                )
                text_extractor = "PyPDFLoader"
            except Exception as e:
                logger.warning(
                    f"EarningsRelease: PyPDFLoader failed, trying UnstructuredPDFLoader: {e}"
                )
                loader = UnstructuredPDFLoader(temp_path)
                text_documents = await asyncio.get_event_loop().run_in_executor(
                    None, loader.load
                )
                text_extractor = "UnstructuredPDFLoader"

            # Table extraction (returns per-row JSON documents)
            (
                table_documents,
                table_extractor,
                tables_count,
                table_rows_count,
            ) = self._extract_tables(
                pdf_path=temp_path, doc_id=doc_id, storage_path=storage_path
            )

            # Remove temporary file
            Path(temp_path).unlink(missing_ok=True)

            # Split text into chunks; keep table row docs intact
            text_chunks = self.text_splitter.split_documents(text_documents)

            # Add metadata to text chunks
            for chunk in text_chunks:
                chunk.metadata.update(
                    {
                        "document_id": doc_id,  # Ensure document_id is included for langchain_embedding table
                        "doc_id": doc_id,
                        "extraction_method": text_extractor,
                        "processing_pipeline": "langchain",
                        "content_type": "text",
                        "source_path": storage_path,
                    }
                )

            # Merge text chunks with table documents
            all_chunks: List[Document] = [*text_chunks, *table_documents]

            # Store in PGVector
            vector_store = PGVector(
                embeddings=self.embeddings,
                connection=self.connection_string,
                collection_name=SETTINGS.VECTOR.COLLECTION_NAME,
                use_jsonb=SETTINGS.VECTOR.VECTOR_USE_JSONB,
            )

            document_ids = await asyncio.get_event_loop().run_in_executor(
                None, vector_store.add_documents, all_chunks
            )

            logger.info(
                "EarningsRelease.processed",
                doc_id=doc_id,
                text_chunks=len(text_chunks),
                tables=tables_count,
                table_rows=table_rows_count,
                embeddings=len(document_ids),
                table_extractor=table_extractor,
                text_extractor=text_extractor,
            )

            return {
                "doc_id": doc_id,
                "status": "completed",
                "processing_pipeline": "langchain",
                "text_extractor": text_extractor,
                "table_extractor": table_extractor,
                "tables_indexed": tables_count,
                "table_rows_indexed": table_rows_count,
                "chunks_created": len(all_chunks),
                "embeddings_stored": len(document_ids),
                "vector_ids": document_ids[:5],
            }

        except Exception as e:
            logger.error(f"EarningsRelease processing failed for {doc_id}: {e}")
            return {
                "doc_id": doc_id,
                "status": "failed",
                "error": str(e),
                "processing_pipeline": "langchain",
            }

    async def process_slide_deck_from_minio(
        self, doc_id: str, storage_path: str, minio_resource: MinIOResource
    ) -> Dict[str, Any]:
        """
        Process a slide deck PDF with comprehensive extraction.

        Strategy:
        - Extract text via PyPDFLoader (+ UnstructuredPDFLoader fallback)
        - Extract tables using same approach as earnings releases (per-row JSON)
        - Extract charts/visualizations and analyze with OpenAI Vision API
        - Combine all content types for embedding and storage
        """
        try:
            pdf_bytes = await minio_resource.get_object_bytes(
                bucket_name=SETTINGS.MINIO.MINIO_BUCKET, object_name=storage_path
            )

            temp_path = f"/tmp/{doc_id}.pdf"
            with open(temp_path, "wb") as f:
                f.write(pdf_bytes)

            # 1. Text extraction
            try:
                loader = PyPDFLoader(temp_path)
                text_documents = await asyncio.get_event_loop().run_in_executor(
                    None, loader.load
                )
                text_extractor = "PyPDFLoader"
            except Exception as e:
                logger.warning(
                    f"SlideDeck: PyPDFLoader failed, trying UnstructuredPDFLoader: {e}"
                )
                loader = UnstructuredPDFLoader(temp_path)
                text_documents = await asyncio.get_event_loop().run_in_executor(
                    None, loader.load
                )
                text_extractor = "UnstructuredPDFLoader"

            # 2. Table extraction (same as earnings releases)
            (
                table_documents,
                table_extractor,
                tables_count,
                table_rows_count,
            ) = self._extract_tables(
                pdf_path=temp_path, doc_id=doc_id, storage_path=storage_path
            )

            # 3. Chart/visualization extraction and analysis
            chart_documents, charts_count = await self._extract_and_analyze_charts(
                pdf_path=temp_path, doc_id=doc_id, storage_path=storage_path
            )

            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)

            # 4. Process text into chunks
            text_chunks = self.text_splitter.split_documents(text_documents)

            # Add metadata to text chunks
            total_chars = 0
            for chunk in text_chunks:
                page_chars = len(chunk.page_content or "")
                total_chars += page_chars
                chunk.metadata.update(
                    {
                        "document_id": doc_id,  # Ensure document_id is included for langchain_embedding table
                        "doc_id": doc_id,
                        "extraction_method": text_extractor,
                        "processing_pipeline": "langchain",
                        "content_type": "slide_text",
                        "source_path": storage_path,
                    }
                )

            # 5. Combine all content types
            all_chunks: List[Document] = [
                *text_chunks,
                *table_documents,
                *chart_documents,
            ]

            # 6. Store in PGVector
            vector_store = PGVector(
                embeddings=self.embeddings,
                connection=self.connection_string,
                collection_name=SETTINGS.VECTOR.COLLECTION_NAME,
                use_jsonb=SETTINGS.VECTOR.VECTOR_USE_JSONB,
            )

            document_ids = await asyncio.get_event_loop().run_in_executor(
                None, vector_store.add_documents, all_chunks
            )

            logger.info(
                "SlideDeck.processed",
                doc_id=doc_id,
                text_chunks=len(text_chunks),
                tables=tables_count,
                table_rows=table_rows_count,
                charts=charts_count,
                total_chars=total_chars,
                embeddings=len(document_ids),
                text_extractor=text_extractor,
                table_extractor=table_extractor,
            )

            return {
                "doc_id": doc_id,
                "status": "completed",
                "processing_pipeline": "langchain",
                "text_extractor": text_extractor,
                "table_extractor": table_extractor,
                "tables_indexed": tables_count,
                "table_rows_indexed": table_rows_count,
                "charts_analyzed": charts_count,
                "chunks_created": len(all_chunks),
                "embeddings_stored": len(document_ids),
                "vector_ids": document_ids[:5],
            }

        except Exception as e:
            logger.error(f"SlideDeck processing failed for {doc_id}: {e}")
            return {
                "doc_id": doc_id,
                "status": "failed",
                "error": str(e),
                "processing_pipeline": "langchain",
            }

    async def process_general_pdf_from_minio(
        self, doc_id: str, storage_path: str, minio_resource: MinIOResource
    ) -> Dict[str, Any]:
        """
        Process a general PDF with comprehensive extraction of all content types.

        Strategy:
        - Extract text via PyPDFLoader (+ UnstructuredPDFLoader fallback)
        - Extract tables using same approach as earnings releases (per-row JSON)
        - Extract and analyze charts/visualizations with OpenAI Vision API
        - Extract and describe general images with OpenAI Vision API
        - Combine all content types for embedding and storage
        """
        try:
            pdf_bytes = await minio_resource.get_object_bytes(
                bucket_name=SETTINGS.MINIO.MINIO_BUCKET, object_name=storage_path
            )

            temp_path = f"/tmp/{doc_id}.pdf"
            with open(temp_path, "wb") as f:
                f.write(pdf_bytes)

            # 1. Text extraction
            try:
                loader = PyPDFLoader(temp_path)
                text_documents = await asyncio.get_event_loop().run_in_executor(
                    None, loader.load
                )
                text_extractor = "PyPDFLoader"
            except Exception as e:
                logger.warning(
                    f"GeneralPDF: PyPDFLoader failed, trying UnstructuredPDFLoader: {e}"
                )
                loader = UnstructuredPDFLoader(temp_path)
                text_documents = await asyncio.get_event_loop().run_in_executor(
                    None, loader.load
                )
                text_extractor = "UnstructuredPDFLoader"

            # 2. Table extraction (same as earnings releases)
            (
                table_documents,
                table_extractor,
                tables_count,
                table_rows_count,
            ) = self._extract_tables(
                pdf_path=temp_path, doc_id=doc_id, storage_path=storage_path
            )

            # 3. Chart/visualization extraction and analysis
            chart_documents, charts_count = await self._extract_and_analyze_charts(
                pdf_path=temp_path, doc_id=doc_id, storage_path=storage_path
            )

            # 4. General image extraction and description
            image_documents, images_count = await self._extract_and_describe_images(
                pdf_path=temp_path, doc_id=doc_id, storage_path=storage_path
            )

            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)

            # 5. Process text into chunks
            text_chunks = self.text_splitter.split_documents(text_documents)

            # Add metadata to text chunks
            total_chars = 0
            for chunk in text_chunks:
                page_chars = len(chunk.page_content or "")
                total_chars += page_chars
                chunk.metadata.update(
                    {
                        "document_id": doc_id,  # Ensure document_id is included for langchain_embedding table
                        "doc_id": doc_id,
                        "extraction_method": text_extractor,
                        "processing_pipeline": "langchain",
                        "content_type": "general_text",
                        "source_path": storage_path,
                    }
                )

            # 6. Combine all content types
            all_chunks: List[Document] = [
                *text_chunks,
                *table_documents,
                *chart_documents,
                *image_documents,
            ]

            # 7. Store in PGVector
            vector_store = PGVector(
                embeddings=self.embeddings,
                connection=self.connection_string,
                collection_name=SETTINGS.VECTOR.COLLECTION_NAME,
                use_jsonb=SETTINGS.VECTOR.VECTOR_USE_JSONB,
            )

            document_ids = await asyncio.get_event_loop().run_in_executor(
                None, vector_store.add_documents, all_chunks
            )

            logger.info(
                "GeneralPDF.processed",
                doc_id=doc_id,
                text_chunks=len(text_chunks),
                tables=tables_count,
                table_rows=table_rows_count,
                charts=charts_count,
                images=images_count,
                total_chars=total_chars,
                embeddings=len(document_ids),
                text_extractor=text_extractor,
                table_extractor=table_extractor,
            )

            return {
                "doc_id": doc_id,
                "status": "completed",
                "processing_pipeline": "langchain",
                "text_extractor": text_extractor,
                "table_extractor": table_extractor,
                "tables_indexed": tables_count,
                "table_rows_indexed": table_rows_count,
                "charts_analyzed": charts_count,
                "images_described": images_count,
                "chunks_created": len(all_chunks),
                "embeddings_stored": len(document_ids),
                "vector_ids": document_ids[:5],
            }

        except Exception as e:
            logger.error(f"GeneralPDF processing failed for {doc_id}: {e}")
            return {
                "doc_id": doc_id,
                "status": "failed",
                "error": str(e),
                "processing_pipeline": "langchain",
            }

    def _extract_tables(
        self, *, pdf_path: str, doc_id: str, storage_path: str
    ) -> tuple[List[Document], str, int, int]:
        """
        Extract tables from a PDF using best-practice, layered fallbacks.

        Returns a tuple: (table_row_documents, extractor_used, tables_count, rows_count)
        """
        # 1) Try Camelot (lattice then stream) if available
        if _CAMELOT_AVAILABLE and camelot is not None:
            try:
                tables = camelot.read_pdf(
                    pdf_path, pages="all", flavor="lattice", strip_text="\n"
                )
                if getattr(tables, "n", 0) > 0:
                    docs, tbls, rows = self._camelot_tables_to_row_documents(
                        tables=tables,
                        doc_id=doc_id,
                        storage_path=storage_path,
                        flavor="camelot_lattice",
                    )
                    if rows > 0:
                        return docs, "camelot_lattice", tbls, rows

                # Fallback to stream if lattice found none
                tables = camelot.read_pdf(
                    pdf_path, pages="all", flavor="stream", strip_text="\n"
                )
                if getattr(tables, "n", 0) > 0:
                    docs, tbls, rows = self._camelot_tables_to_row_documents(
                        tables=tables,
                        doc_id=doc_id,
                        storage_path=storage_path,
                        flavor="camelot_stream",
                    )
                    if rows > 0:
                        return docs, "camelot_stream", tbls, rows
            except Exception as e:  # pragma: no cover - optional path
                logger.warning(
                    f"Camelot extraction failed, falling back to pdfplumber: {e}"
                )

        # 2) Fallback: pdfplumber with lines strategy then text strategy
        try:
            docs_lines, tables_lines, rows_lines = self._pdfplumber_extract(
                pdf_path=pdf_path,
                doc_id=doc_id,
                storage_path=storage_path,
                strategy="lines",
            )
            if rows_lines > 0:
                return docs_lines, "pdfplumber_lines", tables_lines, rows_lines

            docs_text, tables_text, rows_text = self._pdfplumber_extract(
                pdf_path=pdf_path,
                doc_id=doc_id,
                storage_path=storage_path,
                strategy="text",
            )
            return docs_text, "pdfplumber_text", tables_text, rows_text
        except Exception as e:  # pragma: no cover - fallback error path
            logger.error(f"pdfplumber extraction failed: {e}")
            return [], "none", 0, 0

    def _camelot_tables_to_row_documents(
        self, *, tables, doc_id: str, storage_path: str, flavor: str
    ) -> tuple[List[Document], int, int]:
        """Convert Camelot tables to per-row JSON Documents.

        Returns (row_documents, tables_count, rows_count).
        """
        docs: List[Document] = []
        tables_count = 0
        rows_count = 0
        try:
            for idx, table in enumerate(tables):
                try:
                    df = table.df  # pandas DataFrame
                    headers = [str(col) for col in list(df.columns)]
                    rows = [headers] + [[str(x) for x in r] for r in df.values.tolist()]

                    # Heuristics: require at least 2x2, and some numeric density or finance keywords
                    n_rows = len(rows)
                    n_cols = len(rows[0]) if rows else 0
                    if n_rows < 2 or n_cols < 2:
                        continue

                    flat_cells = [c for row in rows for c in row]
                    total_cells = max(len(flat_cells), 1)
                    numeric_pat = re.compile(r"^[\s\$\(\)\-\+\.,%\d]+$")
                    numeric_cells = sum(
                        1 for c in flat_cells if numeric_pat.match(str(c or "").strip())
                    )
                    numeric_ratio = numeric_cells / total_cells
                    finance_keywords = {
                        "revenue",
                        "income",
                        "cost",
                        "expense",
                        "eps",
                        "capex",
                        "operating",
                        "margin",
                        "tac",
                        "guidance",
                    }
                    has_keyword = any(
                        any(k in str(cell).lower() for k in finance_keywords)
                        for cell in rows[0]
                    )
                    if numeric_ratio < 0.15 and not has_keyword:
                        continue

                    page_num = (
                        int(getattr(table, "page", 0))
                        if str(getattr(table, "page", "0")).isdigit()
                        else None
                    )
                    title = (
                        " | ".join(headers)[:120]
                        if headers
                        else f"{flavor}_table_{idx}"
                    )
                    norm_headers = self._normalize_headers(headers)
                    row_docs = self._row_documents_from_rows(
                        rows=rows,
                        original_headers=headers,
                        normalized_headers=norm_headers,
                        doc_id=doc_id,
                        storage_path=storage_path,
                        extractor=flavor,
                        page=page_num,
                        table_index=idx,
                        table_confidence=numeric_ratio,
                        table_title=title,
                    )
                    if row_docs:
                        tables_count += 1
                        rows_count += len(row_docs)
                        docs.extend(row_docs)
                except Exception as e:
                    logger.warning(
                        f"Failed to convert Camelot table to row Documents: {e}"
                    )
        except Exception as e:  # pragma: no cover
            logger.warning(f"Camelot to row documents exception: {e}")
        return docs, tables_count, rows_count

    def _pdfplumber_extract(
        self, *, pdf_path: str, doc_id: str, storage_path: str, strategy: str
    ) -> tuple[List[Document], int, int]:
        """Extract tables using pdfplumber with the specified strategy ('lines' or 'text').

        Returns (row_documents, tables_count, rows_count).
        """
        docs: List[Document] = []
        tables_count = 0
        rows_count = 0
        table_settings_lines = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
        }
        table_settings_text = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
        }

        settings = table_settings_lines if strategy == "lines" else table_settings_text

        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables(table_settings=settings)
                except Exception as pe:
                    logger.warning(
                        f"pdfplumber extract_tables failed on page {page_idx}: {pe}"
                    )
                    continue
                for t_idx, table in enumerate(tables or []):
                    try:
                        rows = [
                            [
                                "" if cell is None else str(cell).strip()
                                for cell in (row or [])
                            ]
                            for row in (table or [])
                        ]
                        # Basic shape filter
                        if not rows or not rows[0]:
                            continue
                        n_cols = len(rows[0])
                        n_rows = len(rows)
                        if n_cols < 2 or n_rows < 2:
                            continue

                        # Content heuristics to avoid misclassifying plain text as table
                        flat_cells = [c for r in rows for c in r]
                        total_cells = max(len(flat_cells), 1)
                        numeric_pat = re.compile(r"^[\s\$\(\)\-\+\.,%\d]+$")
                        numeric_cells = sum(
                            1
                            for c in flat_cells
                            if numeric_pat.match(str(c or "").strip())
                        )
                        numeric_ratio = numeric_cells / total_cells

                        finance_keywords = {
                            "revenue",
                            "income",
                            "cost",
                            "expense",
                            "eps",
                            "capex",
                            "operating",
                            "margin",
                            "tac",
                            "guidance",
                        }
                        header = rows[0]
                        has_keyword = any(
                            any(k in str(cell).lower() for k in finance_keywords)
                            for cell in header
                        )

                        # Average cell length heuristic: extremely long cells across many columns often indicates non-table
                        avg_len = sum(len(str(c)) for c in flat_cells) / total_cells
                        if (numeric_ratio < 0.2 and not has_keyword) or (
                            avg_len > 60 and n_cols >= 4 and numeric_ratio < 0.4
                        ):
                            continue

                        # Per-row JSON documents
                        title = (
                            " | ".join(header)[:120]
                            if header
                            else f"pdfplumber_{strategy}_table_{t_idx}"
                        )
                        norm_headers = self._normalize_headers(header)
                        row_docs = self._row_documents_from_rows(
                            rows=rows,
                            original_headers=header,
                            normalized_headers=norm_headers,
                            doc_id=doc_id,
                            storage_path=storage_path,
                            extractor=f"pdfplumber_{strategy}",
                            page=page_idx,
                            table_index=t_idx,
                            table_confidence=numeric_ratio,
                            table_title=title,
                        )
                        if row_docs:
                            tables_count += 1
                            rows_count += len(row_docs)
                            docs.extend(row_docs)
                    except Exception as te:
                        logger.warning(
                            f"Failed to convert pdfplumber table on page {page_idx} index {t_idx}: {te}"
                        )
        return docs, tables_count, rows_count

    @staticmethod
    def _rows_to_markdown(rows: List[List[str]]) -> str:
        """Convert list-of-list table rows to GitHub-flavored Markdown table."""

        def escape(cell: str) -> str:
            return (cell or "").replace("|", "\\|").replace("\n", " ").strip()

        header = rows[0]
        body = rows[1:] if len(rows) > 1 else []
        md_lines = []
        md_lines.append("| " + " | ".join(escape(c) for c in header) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for r in body:
            md_lines.append("| " + " | ".join(escape(c) for c in r) + " |")
        return "\n".join(md_lines)

    @staticmethod
    def _normalize_headers(headers: List[str]) -> List[str]:
        """Normalize header names to lowercase snake_case unique keys."""
        norm: List[str] = []
        used = set()
        for i, h in enumerate(headers or []):
            key = re.sub(r"[^a-z0-9]+", "_", str(h or "").strip().lower())
            if not key or key == "_":
                key = f"col_{i+1}"
            base = key.strip("_") or f"col_{i+1}"
            key = base
            suffix = 1
            while key in used:
                suffix += 1
                key = f"{base}_{suffix}"
            norm.append(key)
            used.add(key)
        return norm

    def _row_documents_from_rows(
        self,
        *,
        rows: List[List[str]],
        original_headers: List[str],
        normalized_headers: List[str],
        doc_id: str,
        storage_path: str,
        extractor: str,
        page: int | None,
        table_index: int,
        table_confidence: float,
        table_title: str,
    ) -> List[Document]:
        """Create one JSON Document per table row using enhanced column name-value pairs.

        Enhanced JSON payload includes: row_number, table_info, column_value_pairs.
        Each row is embedded separately with proper document_id tracking.
        """
        docs: List[Document] = []
        headers_norm = normalized_headers or [
            f"col_{i+1}" for i in range(len(rows[0]) if rows else 0)
        ]
        headers_orig = original_headers or headers_norm
        body = rows[1:] if len(rows) > 1 else []
        cols_n = len(headers_norm)

        for r_idx, row in enumerate(body, start=1):
            values = [("" if v is None else str(v).strip()) for v in (row or [])]
            if len(values) < cols_n:
                values = values + [""] * (cols_n - len(values))
            elif len(values) > cols_n:
                values = values[:cols_n]

            # Create column name-value pairs for better semantic understanding
            column_value_pairs = []
            row_data = {}
            for i in range(cols_n):
                col_name = headers_orig[i] if i < len(headers_orig) else f"Column_{i+1}"
                col_key = headers_norm[i]
                col_value = values[i]

                # Add to structured data
                row_data[col_key] = col_value

                # Create readable column-value pairs
                if col_value and col_value.strip():
                    column_value_pairs.append(f"{col_name}: {col_value}")

            # Enhanced JSON payload with better structure
            payload = {
                "row_number": r_idx,
                "table_info": {
                    "title": table_title,
                    "page": page,
                    "table_index": table_index,
                    "extractor": extractor,
                },
                "columns": headers_orig,
                "data": row_data,
                "column_value_pairs": column_value_pairs,
                "readable_summary": f"Row {r_idx} from {table_title}: "
                + ", ".join(column_value_pairs),
            }

            # Use readable summary as main content for better embedding
            content = (
                payload["readable_summary"]
                + "\n\nStructured Data: "
                + json.dumps(payload, ensure_ascii=False)
            )

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "document_id": doc_id,  # Ensure document_id is included for langchain_embedding table
                        "doc_id": doc_id,
                        "content_type": "table_row",
                        "table_extractor": extractor,
                        "table_index": table_index,
                        "row_index": r_idx,
                        "row_number": r_idx,
                        "total_rows": max(len(rows) - 1, 0),
                        "total_cols": cols_n,
                        "page": page,
                        "source_path": storage_path,
                        "processing_pipeline": "langchain",
                        "table_confidence": round(float(table_confidence), 3),
                        "columns_original": headers_orig,
                        "columns_normalized": headers_norm,
                        "table_title": table_title,
                        "column_count": len(
                            [v for v in values if v and v.strip()]
                        ),  # Non-empty columns
                    },
                )
            )
        return docs

    async def _extract_and_analyze_charts(
        self, *, pdf_path: str, doc_id: str, storage_path: str
    ) -> tuple[List[Document], int]:
        """
        Extract charts/visualizations from PDF pages and analyze with OpenAI Vision API.

        Returns (chart_analysis_documents, charts_count).
        """
        if not _PYMUPDF_AVAILABLE or not _PIL_AVAILABLE:
            logger.warning("PyMuPDF or PIL not available, skipping chart extraction")
            return [], 0

        docs: List[Document] = []
        charts_count = 0

        try:
            # Open PDF with PyMuPDF
            pdf_doc = fitz.open(pdf_path)

            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]

                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")

                # Check if page likely contains charts/visualizations
                if await self._page_contains_charts(img_data):
                    # Analyze with OpenAI Vision API
                    chart_analysis = await self._analyze_chart_with_vision(
                        img_data, page_num + 1, doc_id
                    )

                    if chart_analysis:
                        # Create document with chart analysis
                        doc = Document(
                            page_content=chart_analysis["explanation"],
                            metadata={
                                "document_id": doc_id,  # Ensure document_id is included for langchain_embedding table
                                "doc_id": doc_id,
                                "content_type": "chart_analysis",
                                "chart_extractor": "openai_vision",
                                "page": page_num + 1,
                                "source_path": storage_path,
                                "processing_pipeline": "langchain",
                                "chart_type": chart_analysis.get(
                                    "chart_type", "unknown"
                                ),
                                "confidence": chart_analysis.get("confidence", 0.0),
                                "financial_metrics": chart_analysis.get(
                                    "financial_metrics", []
                                ),
                            },
                        )
                        docs.append(doc)
                        charts_count += 1

                        logger.info(
                            f"Chart analyzed on page {page_num + 1}: {chart_analysis.get('chart_type', 'unknown')}"
                        )

            pdf_doc.close()

        except Exception as e:
            logger.error(f"Chart extraction failed for {doc_id}: {e}")

        return docs, charts_count

    async def _page_contains_charts(self, img_data: bytes) -> bool:
        """
        Heuristic to determine if a page likely contains charts/visualizations.
        Uses basic image analysis to avoid unnecessary Vision API calls.
        """
        try:
            # Convert to PIL Image for analysis
            img = Image.open(BytesIO(img_data))

            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Simple heuristics:
            # 1. Check image dimensions (charts usually have reasonable aspect ratios)
            width, height = img.size
            aspect_ratio = width / height

            # 2. Check for color diversity (charts often have multiple colors)
            colors = img.getcolors(maxcolors=256 * 256 * 256)
            if colors and len(colors) > 10:  # More than 10 distinct colors
                return True

            # 3. If image is mostly white/text, likely not a chart
            # This is a simple check - could be enhanced
            return aspect_ratio > 0.5 and aspect_ratio < 3.0

        except Exception as e:
            logger.warning(f"Error in chart detection heuristic: {e}")
            return True  # Default to analyzing if uncertain

    async def _analyze_chart_with_vision(
        self, img_data: bytes, page_num: int, doc_id: str
    ) -> Dict[str, Any] | None:
        """
        Analyze chart/visualization using OpenAI Vision API.

        Returns detailed financial analysis of the chart.
        """
        try:
            # Encode image to base64
            img_base64 = base64.b64encode(img_data).decode("utf-8")

            # Create OpenAI client
            client = openai.AsyncOpenAI(
                api_key=SETTINGS.OPENAI.OPENAI_API_KEY.get_secret_value()
            )

            # Craft prompt for financial chart analysis
            prompt = """You are a financial analyst examining a chart or data visualization from an earnings presentation or financial document.

Please provide a detailed analysis including:

1. **Chart Type**: Identify the type of visualization (bar chart, line chart, pie chart, etc.)

2. **Financial Metrics**: List all financial metrics, KPIs, or data points visible in the chart

3. **Key Insights**: Provide 3-5 key financial insights from the data shown

4. **Trends & Patterns**: Describe any trends, growth patterns, or notable changes over time

5. **Quantitative Data**: Extract specific numbers, percentages, or values where visible

6. **Business Context**: Explain what this chart tells us about the company's financial performance

Format your response as a comprehensive analysis that would be useful for financial research and Q&A. Focus on being precise with numbers and clear about business implications.

If this is not a financial chart or data visualization, simply respond with "NOT_FINANCIAL_CHART"."""

            response = await client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000,
                temperature=0.1,  # Low temperature for consistent analysis
            )

            analysis_text = response.choices[0].message.content

            if analysis_text and "NOT_FINANCIAL_CHART" not in analysis_text:
                # Extract structured information from the analysis
                result = {
                    "explanation": analysis_text,
                    "chart_type": self._extract_chart_type(analysis_text),
                    "financial_metrics": self._extract_financial_metrics(analysis_text),
                    "confidence": 0.8,  # Could be enhanced with confidence scoring
                }

                logger.info(
                    f"Successfully analyzed chart on page {page_num} for doc {doc_id}"
                )
                return result
            else:
                logger.info(f"Page {page_num} does not contain financial chart")
                return None

        except Exception as e:
            logger.error(
                f"OpenAI Vision API error for doc {doc_id}, page {page_num}: {e}"
            )
            return None

    def _extract_chart_type(self, analysis_text: str) -> str:
        """Extract chart type from analysis text."""
        text_lower = analysis_text.lower()
        chart_types = {
            "bar chart": "bar_chart",
            "line chart": "line_chart",
            "pie chart": "pie_chart",
            "area chart": "area_chart",
            "scatter plot": "scatter_plot",
            "histogram": "histogram",
            "waterfall": "waterfall_chart",
            "funnel": "funnel_chart",
        }

        for chart_name, chart_type in chart_types.items():
            if chart_name in text_lower:
                return chart_type

        return "unknown"

    def _extract_financial_metrics(self, analysis_text: str) -> List[str]:
        """Extract financial metrics mentioned in the analysis."""
        text_lower = analysis_text.lower()
        metrics = []

        # Common financial metrics to look for
        financial_terms = [
            "revenue",
            "income",
            "profit",
            "loss",
            "margin",
            "ebitda",
            "eps",
            "capex",
            "opex",
            "cash flow",
            "roi",
            "roa",
            "roe",
            "debt",
            "equity",
            "growth",
            "market share",
            "users",
            "engagement",
            "retention",
            "traffic acquisition cost",
            "tac",
            "operating expenses",
            "r&d",
        ]

        for term in financial_terms:
            if term in text_lower:
                metrics.append(term)

        return list(set(metrics))  # Remove duplicates

    async def _extract_and_describe_images(
        self, *, pdf_path: str, doc_id: str, storage_path: str
    ) -> tuple[List[Document], int]:
        """
        Extract general images from PDF pages and describe them with OpenAI Vision API.

        This method focuses on general images (diagrams, photos, illustrations, etc.)
        that are not financial charts. It provides detailed descriptions for better
        semantic understanding and searchability.

        Returns (image_description_documents, images_count).
        """
        if not _PYMUPDF_AVAILABLE or not _PIL_AVAILABLE:
            logger.warning("PyMuPDF or PIL not available, skipping image extraction")
            return [], 0

        docs: List[Document] = []
        images_count = 0

        try:
            # Open PDF with PyMuPDF
            pdf_doc = fitz.open(pdf_path)

            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]

                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")

                # Check if page contains images (different heuristic than charts)
                if await self._page_contains_general_images(img_data):
                    # Describe with OpenAI Vision API
                    image_description = await self._describe_image_with_vision(
                        img_data, page_num + 1, doc_id
                    )

                    if image_description:
                        # Create document with image description
                        doc = Document(
                            page_content=image_description["description"],
                            metadata={
                                "document_id": doc_id,  # Ensure document_id is included for langchain_embedding table
                                "doc_id": doc_id,
                                "content_type": "image_description",
                                "image_extractor": "openai_vision",
                                "page": page_num + 1,
                                "source_path": storage_path,
                                "processing_pipeline": "langchain",
                                "image_type": image_description.get(
                                    "image_type", "unknown"
                                ),
                                "confidence": image_description.get("confidence", 0.0),
                                "visual_elements": image_description.get(
                                    "visual_elements", []
                                ),
                            },
                        )
                        docs.append(doc)
                        images_count += 1

                        logger.info(
                            f"Image described on page {page_num + 1}: {image_description.get('image_type', 'unknown')}"
                        )

            pdf_doc.close()

        except Exception as e:
            logger.error(f"Image extraction failed for {doc_id}: {e}")

        return docs, images_count

    async def _page_contains_general_images(self, img_data: bytes) -> bool:
        """
        Heuristic to determine if a page likely contains general images worth describing.
        Different from chart detection - focuses on diagrams, photos, illustrations, etc.
        """
        try:
            # Convert to PIL Image for analysis
            img = Image.open(BytesIO(img_data))

            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Heuristics for general images:
            width, height = img.size

            # 1. Check for sufficient image complexity (not just text)
            colors = img.getcolors(maxcolors=256 * 256 * 256)
            if colors:
                unique_colors = len(colors)
                # More colors suggest images/diagrams rather than plain text
                if unique_colors > 20:
                    return True

            # 2. Check image dimensions - very wide or very tall might be text
            aspect_ratio = width / height
            if aspect_ratio < 0.3 or aspect_ratio > 4.0:
                return False  # Likely text columns or headers

            # 3. Basic size threshold - too small images might not be worth describing
            if width < 200 or height < 200:
                return False

            return True

        except Exception as e:
            logger.warning(f"Error in image detection heuristic: {e}")
            return False  # Conservative approach for general images

    async def _describe_image_with_vision(
        self, img_data: bytes, page_num: int, doc_id: str
    ) -> Dict[str, Any] | None:
        """
        Describe general images using OpenAI Vision API.

        Provides detailed descriptions for diagrams, photos, illustrations, etc.
        """
        try:
            # Encode image to base64
            img_base64 = base64.b64encode(img_data).decode("utf-8")

            # Create OpenAI client
            client = openai.AsyncOpenAI(
                api_key=SETTINGS.OPENAI.OPENAI_API_KEY.get_secret_value()
            )

            # Craft prompt for general image description
            prompt = """You are an expert at analyzing and describing images, diagrams, and visual content. Please examine this image and provide a comprehensive description.

Please provide a detailed analysis including:

1. **Image Type**: Identify what type of visual content this is (diagram, photograph, illustration, flowchart, map, etc.)

2. **Visual Elements**: Describe the main visual elements, objects, people, or components visible

3. **Content Description**: Provide a detailed description of what the image shows, including:
   - Main subjects or focal points
   - Background elements
   - Text or labels visible in the image
   - Colors, layout, and composition

4. **Context & Purpose**: Explain what this image appears to be used for or what information it conveys

5. **Key Details**: Extract any specific information, names, numbers, or important details visible

Format your response as a comprehensive description that would help someone understand the image content without seeing it. Focus on being descriptive and informative.

If this appears to be just plain text or a mostly blank page, simply respond with "PLAIN_TEXT_PAGE"."""

            response = await client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=800,
                temperature=0.2,  # Low temperature for consistent descriptions
            )

            description_text = response.choices[0].message.content

            if description_text and "PLAIN_TEXT_PAGE" not in description_text:
                # Extract structured information from the description
                result = {
                    "description": description_text,
                    "image_type": self._extract_image_type(description_text),
                    "visual_elements": self._extract_visual_elements(description_text),
                    "confidence": 0.8,  # Could be enhanced with confidence scoring
                }

                logger.info(
                    f"Successfully described image on page {page_num} for doc {doc_id}"
                )
                return result
            else:
                logger.info(f"Page {page_num} does not contain meaningful images")
                return None

        except Exception as e:
            logger.error(
                f"OpenAI Vision API error for image description doc {doc_id}, page {page_num}: {e}"
            )
            return None

    def _extract_image_type(self, description_text: str) -> str:
        """Extract image type from description text."""
        text_lower = description_text.lower()
        image_types = {
            "diagram": "diagram",
            "flowchart": "flowchart",
            "photograph": "photograph",
            "illustration": "illustration",
            "map": "map",
            "screenshot": "screenshot",
            "logo": "logo",
            "infographic": "infographic",
            "schematic": "schematic",
            "blueprint": "blueprint",
        }

        for image_name, image_type in image_types.items():
            if image_name in text_lower:
                return image_type

        return "general_image"

    def _extract_visual_elements(self, description_text: str) -> List[str]:
        """Extract visual elements mentioned in the description."""
        text_lower = description_text.lower()
        elements = []

        # Common visual elements to look for
        visual_terms = [
            "person",
            "people",
            "building",
            "car",
            "tree",
            "logo",
            "text",
            "button",
            "arrow",
            "line",
            "box",
            "circle",
            "rectangle",
            "graph",
            "table",
            "image",
            "icon",
            "symbol",
            "diagram",
            "flowchart",
            "map",
            "photo",
            "illustration",
        ]

        for term in visual_terms:
            if term in text_lower:
                elements.append(term)

        return list(set(elements))  # Remove duplicates


async def process_document_with_langchain(doc_id: str) -> Dict[str, Any]:
    """
    Celery task function: Process document using pure LangChain pipeline.

    This replaces all custom processing with LangChain best practices:
    - DocumentLoader -> TextSplitter -> Embeddings -> VectorStore

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
                # Process based on document type
                if document.doc_type == DocumentType.EARNINGS_RELEASE:
                    result = await processor.process_earnings_release_from_minio(
                        doc_id=doc_id,
                        storage_path=document.raw_path,
                        minio_resource=minio_resource,
                    )
                elif document.doc_type == DocumentType.SLIDE_DECK:
                    result = await processor.process_slide_deck_from_minio(
                        doc_id=doc_id,
                        storage_path=document.raw_path,
                        minio_resource=minio_resource,
                    )
                elif document.doc_type == DocumentType.TRANSCRIPT:
                    result = await processor.process_earning_transcript_from_minio(
                        doc_id=doc_id,
                        storage_path=document.raw_path,
                        minio_resource=minio_resource,
                    )

                else:
                    # Default generic processing for other document types
                    result = await processor.process_general_pdf_from_minio(
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
