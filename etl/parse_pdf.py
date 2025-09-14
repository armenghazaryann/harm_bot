"""PDF parsing strategies for different document types.

Implements four specialized PDF processing strategies:
- Earnings Transcripts: speaker segmentation, Q&A extraction
- Earnings Releases: table extraction, financial metrics
- Slide Decks: per-slide processing, chart analysis with Vision
- Press/IR Announcements: straightforward text extraction
"""
from __future__ import annotations

import asyncio
import base64
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from core.settings import SETTINGS
from etl.ingest import download_from_minio
from infra.costs.recorder import record_cost_event

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
    db_session: Optional[AsyncSession] = None,
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
            result = await parse_slide_deck(temp_path, document_id, db_session)
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


async def parse_slide_deck(
    pdf_path: str, document_id: str, db_session: Optional[AsyncSession] = None
) -> Dict[str, Any]:
    """Parse slide deck with comprehensive extraction: text, tables, and vision analysis."""
    start_time = time.time()

    # 1. Text extraction with fallback
    documents = await _load_pdf_with_fallback(pdf_path)
    full_text = "\n\n".join([doc.page_content for doc in documents])

    # 2. Table extraction (same approach as earnings releases)
    tables = []
    table_extractor = "none"

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
            table_extractor = "camelot" if tables else "none"
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
            table_extractor = "pdfplumber" if tables else "none"
        except Exception as e:
            logger.warning("PDFplumber table extraction failed", error=str(e))

    # 3. Vision analysis for slides with visual elements (concurrent processing)
    slides = []
    vision_descriptions = []
    total_vision_tokens = 0
    request_id = str(uuid.uuid4())

    # Initialize OpenAI client for Vision API
    openai_client = AsyncOpenAI(
        api_key=SETTINGS.OPENAI.OPENAI_API_KEY.get_secret_value()
    )

    # Create semaphore to limit concurrent Vision API requests
    vision_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests

    async def process_slide(i: int, doc: Document) -> Dict[str, Any]:
        """Process a single slide with potential vision analysis."""
        slide_data = {
            "slide_number": i + 1,
            "content": doc.page_content,
            "metadata": doc.metadata,
        }

        # Detect visual elements that need Vision API analysis
        chart_indicators = ["chart", "graph", "figure", "table", "%", "$"]
        has_potential_chart = any(
            indicator.lower() in doc.page_content.lower()
            for indicator in chart_indicators
        )
        slide_data["has_potential_chart"] = has_potential_chart

        # If slide has potential visual elements, use Vision API with semaphore
        if has_potential_chart and _PYMUPDF_AVAILABLE:
            async with vision_semaphore:  # Limit concurrent requests
                try:
                    vision_description, tokens_used = await _analyze_slide_with_vision(
                        pdf_path, i, openai_client, document_id
                    )
                    if vision_description:
                        slide_data["vision_description"] = vision_description
                        slide_data["tokens_used"] = tokens_used

                        logger.info(
                            "Vision analysis completed for slide",
                            document_id=document_id,
                            slide_number=i + 1,
                            tokens_used=tokens_used,
                        )
                except Exception as e:
                    logger.warning(
                        "Vision analysis failed for slide",
                        document_id=document_id,
                        slide_number=i + 1,
                        error=str(e),
                    )
                    slide_data["vision_error"] = str(e)

        return slide_data

    # Process all slides concurrently
    slide_tasks = [process_slide(i, doc) for i, doc in enumerate(documents)]

    logger.info(
        "Starting concurrent slide processing",
        document_id=document_id,
        total_slides=len(documents),
        max_concurrent_vision_requests=5,
    )

    slides = await asyncio.gather(*slide_tasks, return_exceptions=True)

    # Process results and handle any exceptions
    processed_slides = []
    for i, result in enumerate(slides):
        if isinstance(result, Exception):
            logger.error(
                "Slide processing failed",
                document_id=document_id,
                slide_number=i + 1,
                error=str(result),
            )
            # Create fallback slide data
            processed_slides.append(
                {
                    "slide_number": i + 1,
                    "content": documents[i].page_content if i < len(documents) else "",
                    "metadata": documents[i].metadata if i < len(documents) else {},
                    "has_potential_chart": False,
                    "processing_error": str(result),
                }
            )
        else:
            processed_slides.append(result)
            # Collect vision descriptions and tokens
            if "vision_description" in result:
                vision_descriptions.append(
                    {
                        "slide_number": result["slide_number"],
                        "description": result["vision_description"],
                        "content_type": "visual_analysis",
                    }
                )
                total_vision_tokens += result.get("tokens_used", 0)

    slides = processed_slides

    logger.info(
        "Concurrent slide processing completed",
        document_id=document_id,
        slides_processed=len(slides),
        vision_analyses_completed=len(vision_descriptions),
        total_vision_tokens=total_vision_tokens,
    )

    # 4. Generate embeddings for vision descriptions if any exist
    vision_embeddings = []
    if vision_descriptions and db_session:
        try:
            vision_embeddings = await _embed_vision_descriptions(
                vision_descriptions, document_id, db_session
            )
        except Exception as e:
            logger.warning(
                "Failed to generate embeddings for vision descriptions",
                document_id=document_id,
                error=str(e),
            )

    # 5. Extract financial metrics patterns (from old earnings release logic)
    financial_patterns = {
        "revenue": r"\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B)?\s*(?:revenue|sales)",
        "profit": r"\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|M|B)?\s*(?:profit|income|earnings)",
        "growth": r"(\d+(?:\.\d+)?)%\s*(?:growth|increase|decrease)",
        "margin": r"(\d+(?:\.\d+)?)%\s*(?:margin|rate)",
    }

    metrics = {}
    for metric, pattern in financial_patterns.items():
        matches = re.findall(pattern, full_text, re.IGNORECASE)
        if matches:
            metrics[metric] = matches

    processing_time = time.time() - start_time

    # 6. Record Vision API cost if used
    if total_vision_tokens > 0 and db_session:
        try:
            await record_cost_event(
                db_session,
                provider="openai",
                model="gpt-4o-mini",
                route="etl.parse_pdf.vision_analysis",
                request_id=request_id,
                prompt_tokens=total_vision_tokens,
                completion_tokens=0,  # Vision API doesn't separate these clearly
                total_tokens=total_vision_tokens,
                latency_ms=int(processing_time * 1000),
                status="completed",
                metadata={
                    "document_id": document_id,
                    "slides_analyzed": len(vision_descriptions),
                    "document_type": "slide_deck",
                },
            )
            await db_session.commit()
        except Exception as e:
            logger.warning("Failed to record vision API cost event", error=str(e))

    return {
        "status": "completed",
        "content": {
            "full_text": full_text,
            "slides": slides,
            "slide_count": len(slides),
            "tables": tables,
            "table_count": len(tables),
            "financial_metrics": metrics,
            "charts_detected": sum(1 for s in slides if s.get("has_potential_chart")),
            "vision_descriptions": vision_descriptions,
            "vision_embeddings": vision_embeddings,
        },
        "page_count": len(documents),
        "extraction_method": "slide_deck_comprehensive",
        "table_extractor": table_extractor,
        "processing_time": processing_time,
        "vision_tokens_used": total_vision_tokens,
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


async def _analyze_slide_with_vision(
    pdf_path: str, page_index: int, openai_client: AsyncOpenAI, document_id: str
) -> tuple[str, int]:
    """Analyze a specific slide using OpenAI Vision API.

    Args:
        pdf_path: Path to PDF file
        page_index: Zero-based page index
        openai_client: OpenAI async client
        document_id: Document UUID for logging

    Returns:
        Tuple of (description, tokens_used)
    """
    if not _PYMUPDF_AVAILABLE:
        raise Exception("PyMuPDF not available for slide image extraction")

    # Extract slide as image using PyMuPDF
    doc = fitz.open(pdf_path)
    page = doc[page_index]

    # Render page as image (high DPI for better Vision analysis)
    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    doc.close()

    # Encode image as base64
    img_base64 = base64.b64encode(img_data).decode("utf-8")

    # Prepare Vision API prompt focused on visual elements
    vision_prompt = """Analyze this slide image and describe any visual elements including:

1. **Charts & Graphs**: Describe any bar charts, line graphs, pie charts, scatter plots, etc. Include key data points, trends, and insights.

2. **Tables**: Describe table structure and key numerical data, especially percentages (%) and dollar amounts ($).

3. **Figures & Diagrams**: Describe any flowcharts, organizational charts, process diagrams, or illustrations.

4. **Financial Data**: Highlight any monetary values, percentages, growth rates, or financial metrics visible.

5. **Key Visual Insights**: What story do the visuals tell? What are the main takeaways?

Focus on extracting actionable business intelligence from visual elements. Be specific about numbers, trends, and relationships shown in the visuals."""

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
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
            max_tokens=500,
            temperature=0.1,  # Low temperature for consistent, factual descriptions
        )

        description = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else 0

        return description, tokens_used

    except Exception as e:
        logger.error(
            "Vision API analysis failed",
            document_id=document_id,
            page_index=page_index,
            error=str(e),
        )
        raise


async def _embed_vision_descriptions(
    vision_descriptions: List[Dict[str, Any]],
    document_id: str,
    db_session: AsyncSession,
) -> List[Dict[str, Any]]:
    """Generate embeddings for vision descriptions.

    Args:
        vision_descriptions: List of vision analysis results
        document_id: Document UUID
        db_session: Database session for cost tracking

    Returns:
        List of descriptions with embeddings
    """
    if not vision_descriptions:
        return []

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=SETTINGS.OPENAI.OPENAI_API_KEY.get_secret_value(),
    )

    # Extract texts for embedding
    texts = [desc["description"] for desc in vision_descriptions]

    try:
        # Generate embeddings
        vectors = await asyncio.get_event_loop().run_in_executor(
            None, embeddings.embed_documents, texts
        )

        # Combine descriptions with embeddings
        descriptions_with_embeddings = []
        for desc, vector in zip(vision_descriptions, vectors):
            desc_with_embedding = desc.copy()
            desc_with_embedding["embedding"] = vector
            descriptions_with_embeddings.append(desc_with_embedding)

        # Estimate token usage for cost tracking
        total_tokens = sum(len(text.split()) * 1.3 for text in texts)

        # Record embedding cost
        try:
            await record_cost_event(
                db_session,
                provider="openai",
                model="text-embedding-3-small",
                route="etl.parse_pdf.vision_embeddings",
                request_id=str(uuid.uuid4()),
                prompt_tokens=int(total_tokens),
                completion_tokens=0,
                total_tokens=int(total_tokens),
                latency_ms=0,  # Not tracking latency for this sub-operation
                status="completed",
                metadata={
                    "document_id": document_id,
                    "vision_descriptions_count": len(vision_descriptions),
                    "content_type": "vision_analysis",
                },
            )
        except Exception as e:
            logger.warning("Failed to record vision embedding cost", error=str(e))

        logger.info(
            "Vision descriptions embedded successfully",
            document_id=document_id,
            description_count=len(descriptions_with_embeddings),
            total_tokens=int(total_tokens),
        )

        return descriptions_with_embeddings

    except Exception as e:
        logger.error(
            "Failed to embed vision descriptions", document_id=document_id, error=str(e)
        )
        raise


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
