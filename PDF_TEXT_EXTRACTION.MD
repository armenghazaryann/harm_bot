# PDF Text Extraction (All Doc Types) — Library-First, Deterministic, Idempotent

This document defines a focused, text-only extraction plan for multiple PDF document types we handle: transcripts (meetings), earnings releases, slide decks, and press announcements. It standardizes inputs, outputs, storage keys, and Celery orchestration while reusing mature open-source libraries and optional cloud OCR services.

We do NOT build bespoke parsers where reliable libraries exist. We adopt library defaults, add light, deterministic normalization, and provide minimal heuristics only where necessary.

---

Supported Document Types
- Transcripts (earnings calls, meetings)
- Earnings releases (PDF releases with sections and tables)
- Slide decks (presentation PDFs)
- Press announcements (narrative PDFs)

Related docs
- `PDF_PROCESSING.MD` — general processing (pages, elements, tables, images)
- `TRANSCRIPT_TEXT_EXTRACTION.MD` — transcript-only flow (utterances, summary, graph)
- `test.py` — sliding-window summarization policies and prompts

---

High-Level Architecture (MVP)
- Trigger: Celery task after upload/registration.
- Input: MinIO `raw/{doc_id}/{filename}.pdf`.
- Pipeline (per doc_type):
  1) Extract text with layout-aware library (PyMuPDF or pdfminer) via Unstructured partition where appropriate.
  2) Normalize (de-hyphenate, whitespace, remove headers/footers when reliable).
  3) Structure into segments per doc type (see schemas below).
  4) Persist JSONL to MinIO under type-specific keys.
  5) Optionally trigger summarization (sliding-prefix or independent windows + reduce).
- Idempotency: Doc checksum + deterministic keys.

---

Libraries & Services (Reuse, Don’t Rebuild)
- Unstructured: `partition()` / `partition_pdf()` with `strategy="fast"` for rule-based parsing. Ref: docs.unstructured.io
- PyMuPDF (fitz): robust text extraction with positional info, reading order, per-page traversal.
- pdfminer.six / pdfplumber: alternative/fallback for text extraction or fine-grained layout control.
- Tables (for releases, optional in text-only flow): Camelot (lattice/stream) or tabula-py. Note: table extraction is defined more completely in `PDF_PROCESSING.MD`.
- Cloud OCR (for image-only pages, slides): AWS Textract, Google Document AI / Vision (PDF OCR), Azure Document Intelligence Read.

Recommended order of operations (per page)
1) Try PyMuPDF / pdfminer text; measure coverage.
2) If low coverage or images, call cloud OCR provider; tag `extraction_method`.
3) For releases with heavy tables, separate table extraction (Camelot/Tabula) handled by processing pipeline; text-flow here focuses on narrative text.

---

MinIO Storage Keys (Text-Only Artifacts)
- Transcripts: `transcripts/{doc_id}/utterances.jsonl` (See `TRANSCRIPT_TEXT_EXTRACTION.MD`)
- Earnings releases: `releases/{doc_id}/text.jsonl`
- Slide decks: `slides/{doc_id}/text.jsonl`
- Press announcements: `press/{doc_id}/text.jsonl`
- Reports: `{type}/{doc_id}/report.json` (coverage metrics and warnings)
- Summaries (optional): `{type}/{doc_id}/summary/`

All keys are deterministic; use versioned subpaths (e.g., `v1/`) if extraction rules change.

---

Common Normalization
- Remove repeated headers/footers (if detectable via consistent regex/bbox).
- Merge hyphenated line breaks; preserve paragraphs.
- Normalize whitespace and punctuation (em dashes, colon spacing).
- Keep `page_no` references to support debugging and alignment with tables/images.

---

Schemas by Document Type

1) Transcripts (Meetings)
- Output: `utterances.jsonl` lines of
  - `{ utterance_id, doc_id, turn_index, speaker, role, speech, section, page_spans, extraction_method, meta }`
- See `TRANSCRIPT_TEXT_EXTRACTION.MD` for detailed heuristics and Q&A detection.

2) Earnings Releases (Narrative Text Segments)
- Output: `text.jsonl` lines of
  - `{
      "segment_id": hash(doc_id|section|heading|text|index),
      "doc_id": UUID,
      "index": int,
      "section": "intro|highlights|financials|guidance|other",
      "heading": str|null,
      "text": str,   // paragraph or list item text
      "page_spans": [{"page_no": int}],
      "extraction_method": "text|ocr_cloud|ocr_tesseract",
      "meta": {"bullet": bool, "confidence?": float}
    }`
- Strategy:
  - Use Unstructured `partition_pdf(strategy="fast")` to detect `Title`, `NarrativeText`, `ListItem`.
  - Convert Titles to `heading`, group subsequent narrative/list items until next heading or section keyword.
  - Section inference via keyword rules: Highlights, Financial Results, Guidance, Non-GAAP Reconciliations.
  - Do not inline tables into text; reference tables separately via processing pipeline if needed.

3) Slide Decks (Per-Slide Text Segments)
- Output: `text.jsonl` lines of
  - `{
      "segment_id": hash(doc_id|slide_no|text|bbox_hash|index),
      "doc_id": UUID,
      "index": int,
      "slide_no": int,
      "text": str,
      "bbox": [x0,y0,x1,y1]|null,
      "extraction_method": "text|ocr_cloud|ocr_tesseract",
      "meta": {"font_size?": float, "is_title?": bool}
    }`
- Strategy:
  - For each page (slide):
    - Extract text blocks with PyMuPDF/pdfplumber.
    - If little/no text, run cloud OCR; aggregate line boxes into logical segments by proximity.
    - Optionally flag the largest text block as slide title.
  - Preserve slide numbers from page index; store bbox if available.

4) Press Announcements (Paragraph Segments)
- Output: `text.jsonl` lines of
  - `{
      "segment_id": hash(doc_id|heading|text|index),
      "doc_id": UUID,
      "index": int,
      "heading": str|null,
      "text": str,
      "page_spans": [{"page_no": int}],
      "extraction_method": "text|ocr_cloud|ocr_tesseract",
      "meta": {"bullet": bool}
    }`
- Strategy:
  - Use Unstructured to partition into `Title`, `NarrativeText`, `ListItem`.
  - Convert headings and collect following paragraphs / bullets as segments.
  - Keep paragraphs independent to support flexible chunking and summarization downstream.

---

Celery Tasks (Proposed)
- `workers.tasks.extract_pdf_text(doc_id: str)`
  - Load doc metadata (doc_type, raw_path) from Postgres.
  - Dispatch to type-specific extractor:
    - transcripts → writes `utterances.jsonl` (delegates to transcript pipeline)
    - releases → writes `releases/{doc_id}/text.jsonl`
    - slides → writes `slides/{doc_id}/text.jsonl`
    - press → writes `press/{doc_id}/text.jsonl`
  - Write `{type}/{doc_id}/report.json` with coverage, warnings, counts.
- `workers.tasks.summarize_text(doc_id: str)` (optional)
  - Generic summarization over segments list using `test.py` sliding policies.
  - Persist to `{type}/{doc_id}/summary/`.

Idempotency: Use Redis keying (already available) on `(doc_id, extraction_version)`.

---

Quality Metrics & Acceptance
- Coverage: ≥95% text captured for text-based docs; OCR used for remainder.
- Segment fidelity: Titles/sections are preserved for releases and press.
- Slides: Each slide emits at least one segment (title or text) or an empty placeholder with a warning.
- Deterministic order: stable sort by page/slide and reading order.

---

Configuration Knobs
- OCR provider: `aws|gcp|azure` (credentials via env).
- Extraction order: `pymupdf → pdfminer → ocr` per page.
- Unstructured partition strategy: default `fast`; avoid local ML dependencies.
- Heuristic toggles: header/footer removal, hyphen de-dup, minimum font-size delta for headings.
- Version tags: `EXTRACTION_VERSION` used to bucket outputs under `v{n}/`.

---

Error Handling
- MinIO read/write failures: retry with backoff; abort after max attempts.
- OCR timeouts: skip page with warning; continue.
- Unstructured failures on specific pages: fallback to PyMuPDF/pdfminer directly.
- Always produce a report with warnings and partial counts.

---

Integration with Downstream Steps
- Summarization: any `text.jsonl` can be summarized via sliding-window or reduce (see `test.py`).
- Embeddings & Retrieval: chunkers can consume `text.jsonl` uniformly across types.
- Graph-based RAG: 
  - Releases / Press: nodes for `Document`, `Section`, `Paragraph` with `CONTAINS` and `NEXT` relationships.
  - Slides: nodes for `Slide` and `SlideText` with `CONTAINS` and `NEXT`.
  - Transcripts: see `TRANSCRIPT_TEXT_EXTRACTION.MD`.

---

MVP → Growth → Scale
- MVP: single Celery worker, MinIO + Postgres only, cloud OCR optional; deterministic keys; idempotent reruns.
- Growth: add RabbitMQ fan-out for large batches; enrich with table linking for releases; company-specific heading heuristics.
- Scale: add async orchestration, shard by page, cache OCR results, and add observability (metrics, tracing).

---

References (Libraries / Docs)
- Unstructured partition PDF: https://docs.unstructured.io/open-source/core-functionality/partitioning
- PyMuPDF text recipes: https://pymupdf.readthedocs.io/en/latest/recipes-text.html
- pdfplumber: https://github.com/jsvine/pdfplumber
- Camelot (tables): https://camelot-py.readthedocs.io/en/master/
- tabula-py (tables): https://tabula-py.readthedocs.io/
- AWS Textract: https://docs.aws.amazon.com/textract/latest/dg/what-is.html
- Google Document AI (PDF/Forms): https://cloud.google.com/document-ai/
- Azure Document Intelligence Read OCR: https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/read
