"""PDF parsing strategies for different document types.

Implements four specialized PDF processing strategies:
- Earnings Transcripts: speaker segmentation, Q&A extraction
- Earnings Releases: table extraction, financial metrics
- Slide Decks: per-slide processing, chart analysis with Vision
- Press/IR Announcements: straightforward text extraction
"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import structlog
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_core.documents import Document

from etl.ingest import download_from_minio

logger = structlog.get_logger("etl.parse_pdf")

# Optional dependencies for advanced parsing
try:
    import camelot

    _CAMELOT_AVAILABLE = True
except ImportError:
    camelot = None
    _CAMELOT_AVAILABLE = False

try:
    import pdfplumber

    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    pdfplumber = None
    _PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF

    _PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    _PYMUPDF_AVAILABLE = False


async def parse_pdf_by_strategy(
    *,
    document_id: str,
    storage_path: str,
    document_type: str,
    minio_client,
    force: bool = False,
) -> Dict[str, Any]:
    """Route PDF parsing based on document type strategy.

    Args:
        document_id: Document UUID
        storage_path: MinIO path to PDF
        document_type: One of: transcript, release, slides, press
        minio_client: MinIO client
        force: Skip idempotency check

    Returns:
        Parsed content with strategy-specific metadata
    """
    start_time = time.time()

    # Download PDF to temporary file
    temp_path = f"/tmp/{document_id}.pdf"
    try:
        await download_from_minio(
            storage_path=storage_path,
            minio_client=minio_client,
            local_path=temp_path,
        )

        # Route to appropriate strategy (handle both enum values and strings)
        doc_type_str = (
            document_type.lower() if isinstance(document_type, str) else document_type
        )

        if doc_type_str in ("transcript", "earnings_transcript"):
            result = await parse_earnings_transcript(temp_path, document_id)
        elif doc_type_str in ("release", "earnings_release"):
            result = await parse_earnings_release(temp_path, document_id)
        elif doc_type_str in ("slides", "slide_deck"):
            result = await parse_slide_deck(temp_path, document_id)
        else:  # general, press or default
            result = await parse_press_announcement(temp_path, document_id)

        result["processing_time"] = time.time() - start_time
        result["document_type"] = document_type
        result["strategy"] = f"parse_{document_type}"

        logger.info(
            "PDF parsing completed",
            document_id=document_id,
            document_type=document_type,
            processing_time=result["processing_time"],
            page_count=result.get("page_count", 0),
        )

        return result

    finally:
        # Clean up temporary file
        Path(temp_path).unlink(missing_ok=True)


async def parse_earnings_transcript(pdf_path: str, document_id: str) -> Dict[str, Any]:
    """Parse earnings transcript with speaker segmentation and Q&A extraction."""
    documents = await _load_pdf_with_fallback(pdf_path)

    # Debug logging to understand what's being extracted
    logger.info(
        "PDF loading debug info",
        document_id=document_id,
        document_count=len(documents),
        first_page_content_length=len(documents[0].page_content) if documents else 0,
        first_page_preview=documents[0].page_content[:200]
        if documents and documents[0].page_content
        else "EMPTY",
    )

    # Extract text and identify speakers/sections
    full_text = "\n\n".join([doc.page_content for doc in documents])

    # Debug logging for full text extraction
    logger.info(
        "Full text extraction debug",
        document_id=document_id,
        full_text_length=len(full_text),
        full_text_preview=full_text[:300] if full_text else "EMPTY_FULL_TEXT",
    )

    # Simple speaker detection (can be enhanced)
    speaker_pattern = r"^([A-Z][a-z]+ [A-Z][a-z]+|[A-Z]{2,})\s*[-:]"
    speakers = set(re.findall(speaker_pattern, full_text, re.MULTILINE))

    # Detect Q&A section
    qa_start = None
    qa_indicators = ["Q&A", "Question and Answer", "Questions and Answers", "Q-and-A"]
    for indicator in qa_indicators:
        match = re.search(rf"\b{re.escape(indicator)}\b", full_text, re.IGNORECASE)
        if match:
            qa_start = match.start()
            break

    sections = []
    if qa_start:
        sections.append(
            {
                "type": "prepared_remarks",
                "content": full_text[:qa_start].strip(),
                "page_range": [1, len(documents) // 2]
                if len(documents) > 1
                else [1, 1],
            }
        )
        sections.append(
            {
                "type": "qa_session",
                "content": full_text[qa_start:].strip(),
                "page_range": [len(documents) // 2 + 1, len(documents)]
                if len(documents) > 1
                else [1, 1],
            }
        )
    else:
        sections.append(
            {
                "type": "full_transcript",
                "content": full_text,
                "page_range": [1, len(documents)],
            }
        )

    return {
        "status": "completed",
        "content": {
            "full_text": full_text,
            "sections": sections,
            "speakers": list(speakers),
            "has_qa": qa_start is not None,
        },
        "page_count": len(documents),
        "extraction_method": "transcript_segmentation",
    }


async def parse_earnings_release(pdf_path: str, document_id: str) -> Dict[str, Any]:
    """Parse earnings release with table extraction and financial metrics."""
    documents = await _load_pdf_with_fallback(pdf_path)
    full_text = "\n\n".join([doc.page_content for doc in documents])

    tables = []

    # Extract tables using available libraries
    if _CAMELOT_AVAILABLE:
        try:
            camelot_tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
            for i, table in enumerate(camelot_tables):
                if table.df is not None and not table.df.empty:
                    tables.append(
                        {
                            "table_id": f"camelot_{i}",
                            "method": "camelot_lattice",
                            "data": table.df.to_dict("records"),
                            "accuracy": getattr(table, "accuracy", 0.0),
                        }
                    )
        except Exception as e:
            logger.warning("Camelot table extraction failed", error=str(e))

    if _PDFPLUMBER_AVAILABLE and not tables:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for i, table in enumerate(page_tables or []):
                        if table:
                            tables.append(
                                {
                                    "table_id": f"pdfplumber_p{page_num}_{i}",
                                    "method": "pdfplumber",
                                    "page": page_num + 1,
                                    "data": [
                                        dict(zip(table[0] or [], row))
                                        for row in table[1:]
                                        if row
                                    ],
                                }
                            )
        except Exception as e:
            logger.warning("PDFplumber table extraction failed", error=str(e))

    # Extract financial metrics patterns
    financial_patterns = {
        "revenue": r"\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B)?\s*(?:revenue|sales)",
        "profit": r"\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B)?\s*(?:profit|income|earnings)",
        "eps": r"(?:EPS|earnings per share).*?\$?(\d+\.\d+)",
    }

    metrics = {}
    for metric, pattern in financial_patterns.items():
        matches = re.findall(pattern, full_text, re.IGNORECASE)
        if matches:
            metrics[metric] = matches

    return {
        "status": "completed",
        "content": {
            "full_text": full_text,
            "tables": tables,
            "financial_metrics": metrics,
            "table_count": len(tables),
        },
        "page_count": len(documents),
        "extraction_method": "earnings_release_with_tables",
    }


async def parse_slide_deck(pdf_path: str, document_id: str) -> Dict[str, Any]:
    """Parse slide deck with per-slide processing and optional chart analysis."""
    documents = await _load_pdf_with_fallback(pdf_path)

    slides = []
    for i, doc in enumerate(documents):
        slide_data = {
            "slide_number": i + 1,
            "content": doc.page_content,
            "metadata": doc.metadata,
        }

        # TODO: Add chart detection and Vision API analysis
        # This would require OpenAI Vision API integration
        # For now, just detect if slide might contain charts
        chart_indicators = ["chart", "graph", "figure", "table", "%", "$"]
        has_potential_chart = any(
            indicator.lower() in doc.page_content.lower()
            for indicator in chart_indicators
        )
        slide_data["has_potential_chart"] = has_potential_chart

        slides.append(slide_data)

    return {
        "status": "completed",
        "content": {
            "slides": slides,
            "slide_count": len(slides),
            "charts_detected": sum(1 for s in slides if s.get("has_potential_chart")),
        },
        "page_count": len(documents),
        "extraction_method": "slide_deck_per_page",
    }


async def parse_press_announcement(pdf_path: str, document_id: str) -> Dict[str, Any]:
    """Parse press/IR announcement with straightforward text extraction."""
    documents = await _load_pdf_with_fallback(pdf_path)
    full_text = "\n\n".join([doc.page_content for doc in documents])

    # Extract basic metadata
    date_pattern = r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"
    dates = re.findall(date_pattern, full_text)

    return {
        "status": "completed",
        "content": {
            "full_text": full_text,
            "dates_mentioned": dates,
            "word_count": len(full_text.split()),
        },
        "page_count": len(documents),
        "extraction_method": "press_announcement_text",
    }


async def _load_pdf_with_fallback(pdf_path: str) -> List[Document]:
    """Load PDF with fallback between PyPDFLoader and UnstructuredPDFLoader."""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = await asyncio.get_event_loop().run_in_executor(None, loader.load)
        logger.debug("PDF loaded with PyPDFLoader", page_count=len(documents))
        return documents
    except Exception as e:
        logger.warning("PyPDFLoader failed, trying UnstructuredPDFLoader", error=str(e))
        try:
            loader = UnstructuredPDFLoader(pdf_path)
            documents = await asyncio.get_event_loop().run_in_executor(
                None, loader.load
            )
            logger.debug(
                "PDF loaded with UnstructuredPDFLoader", page_count=len(documents)
            )
            return documents
        except Exception as e2:
            logger.error(
                "Both PDF loaders failed",
                pypdf_error=str(e),
                unstructured_error=str(e2),
            )
            raise Exception(f"PDF loading failed: PyPDF: {e}, Unstructured: {e2}")
