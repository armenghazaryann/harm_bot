# PDF Processing (MVP) — Small Multimodal PDFs, No Custom Local Models

This document defines the processing step only: converting raw PDFs (~1 MB each) into structured, high-fidelity artifacts suitable for downstream chunking, embeddings, retrieval, and QA. It is optimized for multiple related PDFs with mixed content (transcripts, images, tables, links, conversations, cost reports) and avoids custom local ML models.

Scope (Processing Only)
- Input: PDFs in `kb/` and future uploads via the ingestion API.
- Output: page images, structured elements JSON, normalized tables JSON, extracted links and images, and per-document processing metadata.
- Excludes: chunking, embeddings, retrieval, KG build, Q/A, evaluation (covered elsewhere).
- Constraint: do not depend on custom local models; prefer library-based extraction and optional cloud OCR.

Key Objectives
- Preserve layout and semantics across text, tables, images, links, and conversational structure.
- Ensure deterministic, reproducible outputs; idempotent re-runs.
- Produce high-quality structured artifacts for accurate finance Q/A.

---

Inputs and Outputs
- Inputs
  - PDF bytes and basic metadata (filename, size, checksum).
- Outputs (stored in object storage and indexed in Postgres)
  - `pages/{doc_id}/{page:04}.png` — rendered page images (300–400 DPI)
  - `elements/{doc_id}/elements.jsonl` — one JSON object per element (text blocks, titles, list items, tables, figures, speaker turns, QA pairs)
  - `elements/{doc_id}/tables/{table_id}.json` — normalized, typed table data with units/currency
  - `images/{doc_id}/{image_id}.png` — extracted embedded images (figures, logos, charts)
  - `links/{doc_id}/links.jsonl` — outbound links (URL, page, bbox)
  - `processing/{doc_id}/report.json` — summary metrics and warnings (coverage, OCR rate, table counts)

---

Processing Pipeline
1) Register & Fingerprint
- Compute SHA256 checksum for idempotency.
- Insert into `documents` and `document_versions` with `doc_type`, `quarter`, `year`, `source_path` if known.
- Store the raw PDF at `raw/{doc_id}/{filename}.pdf`.

2) Validation & Normalization
- Validate PDF (readable, no encryption preventing text extraction).
- Normalize metadata (detect page count, dimensions, fonts).

3) Page Rendering (No custom models)
- Render each page to PNG at 300–400 DPI using a standard PDF renderer (e.g., PyMuPDF).
- Save to `pages/{doc_id}/{page:04}.png`.

4) Text Extraction (Library-based)
- Extract text with positional info (bbox, reading order, page numbers) using PDF libraries (e.g., PyMuPDF or pdfminer.six).
- Compute `text_coverage_ratio = extracted_text_chars / ocr_text_chars_if_any` as a quality metric.

5) OCR Fallback Policy (Cloud preferred; no custom local models)
- When a page has poor or no extractable text (e.g., scanned slides), run cloud OCR if configured:
  - Options: Google Cloud Vision, AWS Textract, Azure Read.
  - Store OCR text with bbox and confidence. Mark `extraction_method = "ocr_cloud"`.
- If cloud OCR is unavailable, allow standard non-ML OCR engine (e.g., Tesseract) as a fallback with explicit flag. Mark `extraction_method = "ocr_tesseract"`.

6) Partitioning into Elements (Layout-aware, library-first)
- Use Unstructured’s PDF partitioning in a non-ML mode (e.g., `strategy=fast`) to avoid local ML dependencies.
- Map outputs to canonical element types with geometry and page refs:
  - `Title`, `NarrativeText`, `ListItem`, `Table`, `Figure` (image), `Header/Footer`, `Footnote`.
- Finance-specific enrichment (rule-based):
  - `SpeakerSegment` (name, role) and `QA` detection for transcripts: identify participants, prepared remarks, and analyst Q&A via regex and formatting cues.
  - `SectionHeader` detection from font size/style and keywords (e.g., “Financial Results”, “Segment Results”, “Guidance”).

7) Table Extraction & Normalization (Deterministic, no ML)
- Attempt structured table extraction with Camelot/Tabula:
  - Try `lattice` mode first (when grid lines are present), fallback to `stream` for whitespace-aligned tables.
  - Validate with header detection and column type inference.
- If table is image-only, use cloud OCR table parsers (Textract/Document AI/Azure) if configured.
- Normalize tables to a typed JSON schema:
  - `{ table_id, title, page, columns: [name, type, unit, currency], rows: [ {col: value, ...} ], footnotes: [], source_bbox }`
  - Coerce numerics; extract units/currency; detect YoY/QoQ if present in headers.
- Store each table at `elements/{doc_id}/tables/{table_id}.json` and index in Postgres.

8) Image & Figure Extraction (No ML captioning)
- Extract embedded images (figures, charts, logos) and save as PNG under `images/{doc_id}/`.
- Capture metadata: `page`, `bbox`, `image_id`, approximate DPI.
- Do not run local captioning models; optionally store empty `alt_text` for future enrichment.

9) Link Extraction
- Parse link annotations from the PDF:
  - For each link: `{url, page, bbox, link_text?}`.
- Save to `links/{doc_id}/links.jsonl`.

10) Conversation & Transcript Structuring (Rule-based)
- Identify participants (name, role) from “Participants” sections in transcripts.
- Detect speaker turns with regex such as `^([A-Z][a-z]+\s[A-Z][a-z]+)\s\(([^)]+)\):` and known speaker/analyst lists.
- Group Q&A:
  - A Q block begins with an analyst intro or “Question” marker; answers follow by management speakers until next question.
  - Store as elements with `qa_id`, `questioner`, `answers[]`, and page spans.

11) Cost/Financial Section Detection (Rule-based)
- Detect sections with headers such as “Costs and Expenses”, “Operating Income”, “Capex/Capital Expenditures”, “Free Cash Flow”, “Reconciliation of GAAP and non-GAAP”.
- Tag related tables and narrative elements with section labels for easier downstream routing.

12) Element Assembly & Export
- Assemble a unified `elements.jsonl` where each line is one element object with:
  - `type` (Title/NarrativeText/ListItem/Table/Figure/Link/SpeakerSegment/QA/SectionHeader)
  - `page`, `bbox`, `text` (when applicable)
  - `metadata`: `{speaker, role, analyst_name, firm, section, units, currency, extraction_method, ocr_confidence}`
  - `refs`: `{table_id?, image_id?, link_id?}`
- Ensure deterministic ordering (by page, then reading order index).

13) Processing Report & Metrics
- Compute and store `processing/{doc_id}/report.json` with:
  - `pages_total`, `pages_ocr`, `text_coverage_ratio`, `tables_detected`, `tables_ocr`, `images_extracted`, `links_detected`.
  - Warnings: low OCR confidence pages, suspected misaligned tables, empty sections, link parsing errors.

14) Storage Layout & Keys
- MinIO key conventions (consistent with overall architecture):
  - `raw/{doc_id}/{filename}.pdf`
  - `pages/{doc_id}/{page:04}.png`
  - `elements/{doc_id}/elements.jsonl`
  - `elements/{doc_id}/tables/{table_id}.json`
  - `images/{doc_id}/{image_id}.png`
  - `links/{doc_id}/links.jsonl`
  - `processing/{doc_id}/report.json`

15) Idempotency & Reprocessing
- Idempotency keys:
  - document: `sha256(pdf_bytes)`
  - page image: `{doc_id, page_no, dpi}`
  - table: `sha256(norm_table_cells)`
  - element: `sha256(type|page|bbox|text_hash)`
- On reprocessing:
  - If parser/version changed, create a new `document_version`; preserve prior artifacts.
  - Soft-delete or mark superseded artifacts with lineage in Postgres.

16) Configuration Knobs (MVP)
- DPI for rendering (default: 300–400).
- Cloud OCR provider and credentials (optional but recommended for slides/images).
- Table extractor mode order: `lattice → stream → ocr_table`.
- Speaker/Q&A regex patterns and known participants list (per company).
- Section header keywords and minimum font-size delta to detect headers.

17) Quality Controls (Acceptance Criteria)
- 100% page coverage (rendered PNG per page).
- ≥95% text coverage for text-based docs; OCR needed for the remainder.
- All detected tables exported with typed JSON and page references.
- Q&A blocks identified for transcripts with at least analyst and primary responder.
- All outbound links enumerated with page/bbox.
- No dependency on custom local ML models; cloud OCR only when necessary.

18) Limitations & Future Enhancements
- Image understanding (charts) is deferred; no captioning without cloud VLMs.
- Complex table spanning across pages may need manual merge rules.
- Some transcripts may require company-specific heuristics for perfect speaker detection.
- Future: add confidence-driven reprocessing (e.g., switch to cloud OCR if coverage < threshold), and optional enrichment (NER, KPI tagging) via cloud services.