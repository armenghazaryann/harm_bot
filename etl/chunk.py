"""Document chunking with hierarchical and semantic strategies.

Implements intelligent chunking based on document type and content structure,
with token-based fallbacks and proper metadata tracking.
"""
from __future__ import annotations

import json
import time
import uuid
from io import BytesIO
from typing import Any, Dict, List

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.ext.asyncio import AsyncSession

from core.settings import SETTINGS

logger = structlog.get_logger("etl.chunk")

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
MAX_CHUNK_SIZE = 2000


async def chunk_document(
    *,
    document_id: str,
    parsed_content: Dict[str, Any],
    document_type: str,
    db_session: AsyncSession,
    minio_client,
    force: bool = False,
) -> Dict[str, Any]:
    """Chunk document content based on type and structure.

    Args:
        document_id: Document UUID
        parsed_content: Output from parse_pdf_by_strategy
        document_type: Document type for strategy selection
        db_session: Database session
        minio_client: MinIO client
        force: Skip idempotency check

    Returns:
        Chunking result with chunk metadata and manifest
    """
    start_time = time.time()

    # Check for existing chunks (idempotency) - using LangChain approach
    # For MVP, we'll skip the idempotency check since we're using LangChain vector store
    # In Growth phase, we can implement proper chunk tracking via manifest files
    if not force:
        logger.info(
            "Idempotency check skipped - using LangChain vector store approach",
            document_id=document_id,
        )

    # Route to appropriate chunking strategy (handle both enum values and strings)
    doc_type_str = (
        document_type.lower() if isinstance(document_type, str) else document_type
    )

    if doc_type_str in ("transcript", "earnings_transcript"):
        chunks = await _chunk_transcript(parsed_content, document_id)
    elif doc_type_str in ("release", "earnings_release"):
        chunks = await _chunk_earnings_release(parsed_content, document_id)
    elif doc_type_str in ("slides", "slide_deck"):
        chunks = await _chunk_slide_deck(parsed_content, document_id)
    else:  # general, press or default
        chunks = await _chunk_press_announcement(parsed_content, document_id)

    # Store chunks using LangChain approach (skip legacy database storage)
    # Chunks will be stored in vector store during embedding phase
    logger.info(
        "Chunks prepared for LangChain vector store",
        document_id=document_id,
        chunk_count=len(chunks),
    )

    # Store chunk manifest in MinIO
    manifest = {
        "document_id": document_id,
        "document_type": document_type,
        "chunk_count": len(chunks),
        "chunking_strategy": f"chunk_{document_type}",
        "chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "sequence": chunk["sequence"],
                "token_count": chunk.get("token_count", 0),
                "char_count": len(chunk["content"]),
                "metadata": chunk.get("metadata", {}),
            }
            for chunk in chunks
        ],
        "created_at": time.time(),
    }

    manifest_path = f"chunks/{document_id}_manifest.json"
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    minio_client.put_object(
        bucket_name=SETTINGS.MINIO.MINIO_BUCKET,
        object_name=manifest_path,
        data=BytesIO(manifest_bytes),
        length=len(manifest_bytes),
        content_type="application/json",
    )

    processing_time = time.time() - start_time
    logger.info(
        "Document chunking completed",
        document_id=document_id,
        chunk_count=len(chunks),
        processing_time=processing_time,
    )

    return {
        "status": "completed",
        "chunks": chunks,
        "chunk_count": len(chunks),
        "manifest_path": manifest_path,
        "processing_time": processing_time,
    }


async def _chunk_transcript(
    parsed_content: Dict[str, Any], document_id: str
) -> List[Dict[str, Any]]:
    """Chunk earnings transcript by sections and speakers."""

    # CRITICAL DEBUG: Log what we actually receive
    logger.info(
        "_chunk_transcript received data",
        document_id=document_id,
        parsed_content_type=type(parsed_content).__name__,
        parsed_content_keys=list(parsed_content.keys())
        if isinstance(parsed_content, dict)
        else "NOT_DICT",
        parsed_content_preview=str(parsed_content)[:200] if parsed_content else "NONE",
    )

    # parsed_content IS the content (pipeline passes parse_result["content"])
    sections = parsed_content.get("sections", [])
    full_text = parsed_content.get("full_text", "")

    # Debug logging to understand the parsed content structure
    logger.info(
        "Transcript chunking debug info",
        document_id=document_id,
        content_keys=list(parsed_content.keys()) if parsed_content else [],
        sections_count=len(sections),
        full_text_length=len(full_text) if full_text else 0,
        has_sections=bool(sections),
    )

    chunks = []
    sequence = 0

    # Create text splitter for fallback
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        add_start_index=True,
    )

    for section in sections:
        section_content = section.get("content", "")
        section_type = section.get("type", "unknown")

        # For Q&A sections, try to split by speaker/question
        if section_type == "qa_session":
            qa_chunks = _split_qa_by_speakers(section_content)
            for i, qa_chunk in enumerate(qa_chunks):
                sequence += 1
                chunks.append(
                    {
                        "chunk_id": str(uuid.uuid4()),
                        "sequence": sequence,
                        "content": qa_chunk["content"],
                        "content_normalized": qa_chunk["content"].strip(),
                        "metadata": {
                            "section_type": section_type,
                            "qa_index": i,
                            "speaker": qa_chunk.get("speaker"),
                            "page_range": section.get("page_range", []),
                        },
                        "token_count": len(qa_chunk["content"].split()),
                    }
                )
        else:
            # For prepared remarks, use semantic chunking
            docs = text_splitter.create_documents([section_content])
            for doc in docs:
                sequence += 1
                chunks.append(
                    {
                        "chunk_id": str(uuid.uuid4()),
                        "sequence": sequence,
                        "content": doc.page_content,
                        "content_normalized": doc.page_content.strip(),
                        "metadata": {
                            "section_type": section_type,
                            "page_range": section.get("page_range", []),
                            "start_index": doc.metadata.get("start_index", 0),
                        },
                        "token_count": len(doc.page_content.split()),
                    }
                )

    # Fallback: If no chunks were created from sections, try to chunk the full_text
    if not chunks and full_text and full_text.strip():
        logger.info(
            "No sections found, using full_text fallback chunking",
            document_id=document_id,
            full_text_length=len(full_text),
        )

        docs = text_splitter.create_documents([full_text])
        for doc in docs:
            if doc.page_content.strip():  # Only add non-empty chunks
                sequence += 1
                chunks.append(
                    {
                        "chunk_id": str(uuid.uuid4()),
                        "sequence": sequence,
                        "content": doc.page_content,
                        "content_normalized": doc.page_content.strip(),
                        "metadata": {
                            "section_type": "full_text_fallback",
                            "start_index": doc.metadata.get("start_index", 0),
                        },
                        "token_count": len(doc.page_content.split()),
                    }
                )

    logger.info(
        "Transcript chunking completed",
        document_id=document_id,
        chunk_count=len(chunks),
        sections_processed=len(sections),
    )

    return chunks


async def _chunk_earnings_release(
    parsed_content: Dict[str, Any], document_id: str
) -> List[Dict[str, Any]]:
    """Chunk earnings release with special handling for tables."""
    # parsed_content IS the content (pipeline passes parse_result["content"])
    full_text = parsed_content.get("full_text", "")
    tables = parsed_content.get("tables", [])

    chunks = []
    sequence = 0

    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        add_start_index=True,
    )

    # Chunk main text (only if we have content)
    if full_text and full_text.strip():
        docs = text_splitter.create_documents([full_text])
        for doc in docs:
            if doc.page_content.strip():  # Only add non-empty chunks
                sequence += 1
                chunks.append(
                    {
                        "chunk_id": str(uuid.uuid4()),
                        "sequence": sequence,
                        "content": doc.page_content,
                        "content_normalized": doc.page_content.strip(),
                        "metadata": {
                            "content_type": "text",
                            "start_index": doc.metadata.get("start_index", 0),
                        },
                        "token_count": len(doc.page_content.split()),
                    }
                )
    else:
        logger.warning(
            "No full_text found in earnings release content",
            document_id=document_id,
            content_keys=list(parsed_content.keys()) if parsed_content else [],
            full_text_length=len(full_text) if full_text else 0,
        )

    # Create separate chunks for each table
    for table in tables:
        table_text = _format_table_as_text(table)
        if table_text and table_text.strip():  # Only add non-empty table chunks
            sequence += 1
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "sequence": sequence,
                    "content": table_text,
                    "content_normalized": table_text.strip(),
                    "metadata": {
                        "content_type": "table",
                        "table_id": table.get("table_id"),
                        "table_method": table.get("method"),
                        "page": table.get("page"),
                    },
                    "token_count": len(table_text.split()),
                }
            )

    # Debug logging
    logger.info(
        "Earnings release chunking completed",
        document_id=document_id,
        chunk_count=len(chunks),
        table_count=len(tables),
        full_text_available=bool(full_text and full_text.strip()),
    )

    return chunks


async def _chunk_slide_deck(
    parsed_content: Dict[str, Any], document_id: str
) -> List[Dict[str, Any]]:
    """Chunk slide deck by individual slides."""
    # parsed_content IS the content (pipeline passes parse_result["content"])
    slides = parsed_content.get("slides", [])

    chunks = []

    for slide in slides:
        slide_content = slide.get("content", "")
        if slide_content.strip():  # Only create chunks for non-empty slides
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "sequence": slide.get("slide_number", 0),
                    "content": slide_content,
                    "content_normalized": slide_content.strip(),
                    "metadata": {
                        "content_type": "slide",
                        "slide_number": slide.get("slide_number"),
                        "has_potential_chart": slide.get("has_potential_chart", False),
                    },
                    "token_count": len(slide_content.split()),
                }
            )

    return chunks


async def _chunk_press_announcement(
    parsed_content: Dict[str, Any], document_id: str
) -> List[Dict[str, Any]]:
    """Chunk press announcement with standard text splitting."""
    # parsed_content IS the content (pipeline passes parse_result["content"])
    full_text = parsed_content.get("full_text", "")

    chunks = []
    sequence = 0

    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        add_start_index=True,
    )

    docs = text_splitter.create_documents([full_text])
    for doc in docs:
        sequence += 1
        chunks.append(
            {
                "chunk_id": str(uuid.uuid4()),
                "sequence": sequence,
                "content": doc.page_content,
                "content_normalized": doc.page_content.strip(),
                "metadata": {
                    "content_type": "text",
                    "start_index": doc.metadata.get("start_index", 0),
                    "dates_mentioned": parsed_content.get("dates_mentioned", []),
                },
                "token_count": len(doc.page_content.split()),
            }
        )

    return chunks


def _split_qa_by_speakers(qa_content: str) -> List[Dict[str, Any]]:
    """Split Q&A content by speakers/questions."""
    # Simple speaker detection - can be enhanced
    import re

    # Pattern to detect speaker names or Q: A: format
    speaker_pattern = r"^([A-Z][a-z]+ [A-Z][a-z]+|[A-Z]{2,}|Q:|A:)\s*[-:]?"

    chunks = []
    current_chunk = ""
    current_speaker = None

    for line in qa_content.split("\n"):
        match = re.match(speaker_pattern, line.strip())
        if match:
            # Save previous chunk if exists
            if current_chunk.strip():
                chunks.append(
                    {
                        "content": current_chunk.strip(),
                        "speaker": current_speaker,
                    }
                )
            # Start new chunk
            current_speaker = match.group(1)
            current_chunk = line
        else:
            current_chunk += "\n" + line

    # Add final chunk
    if current_chunk.strip():
        chunks.append(
            {
                "content": current_chunk.strip(),
                "speaker": current_speaker,
            }
        )

    return chunks if chunks else [{"content": qa_content, "speaker": None}]


def _format_table_as_text(table: Dict[str, Any]) -> str:
    """Format table data as readable text."""
    table_data = table.get("data", [])
    if not table_data:
        return f"Table {table.get('table_id', 'unknown')}: No data"

    # Simple table formatting
    lines = [f"Table {table.get('table_id', 'unknown')}:"]

    if isinstance(table_data, list) and table_data:
        if isinstance(table_data[0], dict):
            # Dictionary format
            for row in table_data[:10]:  # Limit to first 10 rows
                row_text = ", ".join([f"{k}: {v}" for k, v in row.items() if v])
                if row_text:
                    lines.append(row_text)
        else:
            # List format
            for row in table_data[:10]:
                if isinstance(row, (list, tuple)):
                    row_text = " | ".join([str(cell) for cell in row if cell])
                    if row_text:
                        lines.append(row_text)

    return "\n".join(lines)
