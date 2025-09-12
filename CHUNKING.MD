# Chunking (MVP) — Deterministic, Multimodal, Finance-Aware

This document defines the chunking step only: converting processed PDF elements into retrieval-ready chunks. It targets multiple small (~1 MB) but related PDFs with transcripts, images, tables, links, conversations, and cost reports. The design is deterministic, rule-based, and avoids custom local ML models.

Scope (Chunking Only)
- Input: structured artifacts from processing:
  - `elements/{doc_id}/elements.jsonl`
  - `elements/{doc_id}/tables/{table_id}.json`
  - `pages/{doc_id}/{page:04}.png` (for slide references only)
- Output:
  - `chunks/{doc_id}/chunk_manifest.jsonl` (one JSON per chunk)
- Excludes: embeddings, retrieval, knowledge graph, Q/A, evaluation.

Goals
- Preserve meaning: don’t split logical units (Q&A pairs, tables, slides).
- Provide rich metadata for precise filtering (quarter, speaker, section, table/slide refs).
- Produce consistent, idempotent chunk manifests across re-runs.

---

Chunk Types
- `prepared_remarks` (transcripts): speaker-attributed narrative blocks.
- `qna` (transcripts): one chunk per question with all corresponding answers.
- `narrative` (releases/press): section-based narrative text.
- `table` (releases/slides): atomic table with structured payload reference.
- `slide` (slides): per-slide text (title + bullets/OCR), with image reference.

Token Targets and Boundaries
- Default target: 500–1,200 tokens per text chunk.
- Hard boundaries: do not split across:
  - A single Q&A pair.
  - A single table.
  - A slide.
  - A speaker change (within prepared remarks).
- Soft boundaries: within long speaker sections or narrative sections, split at paragraph boundaries or bullet groups to stay near target range.

Normalization & Text Shaping
- Normalize whitespace, preserve bullet structure and numbering.
- Strip headers/footers and page numbers.
- Carry forward section headers and context path into metadata and as a short preamble line in `text` (e.g., `Section: Financial Results > Segment Results`).
- For transcripts, prepend speaker label (e.g., `Speaker: Ruth Porat (CFO)`).

---

Algorithms by Document Type

1) Transcripts
- Detect structural sections from elements: `Participants`, `Prepared Remarks`, `Q&A`.
- Prepared Remarks:
  - Group consecutive `SpeakerSegment` elements by same speaker.
  - Within a speaker block, split on paragraph boundaries to target 500–1,200 tokens, but never split a single paragraph if it’s short.
  - Metadata:
    - `speaker`, `role`, `section_path=[Prepared Remarks]`, `page_start/end`, `quarter`, `year`.
- Q&A:
  - Identify each question (analyst name, firm) and the sequence of answers by management until the next question.
  - Create exactly one chunk per Q&A pair with structure:
    - `text` formatted as: `Question (Analyst, Firm): ...\nAnswer(s): Speaker1: ...; Speaker2: ...`
  - Metadata:
    - `qa_id`, `analyst_name`, `firm`, `answer_speakers[]`, `section_path=[Q&A]`, `page_start/end`, `quarter`, `year`.

2) Earnings Releases
- Narrative:
  - Build section hierarchy from `SectionHeader` elements (e.g., `Financial Results`, `Segment Results`, `Guidance`, `Non-GAAP Reconciliation`).
  - Aggregate paragraphs under each section/subsection.
  - Split by paragraph groups to stay near token targets, keeping numeric-heavy sentences together.
  - Metadata: `section_path`, `page_start/end`, `quarter`, `year`.
- Tables:
  - Each normalized table becomes a single `table` chunk.
  - `text` should be a concise synopsis (e.g., `Q2 2025 Google Cloud revenue $X, +Y% YoY`).
  - `payload_ref`: key to `elements/{doc_id}/tables/{table_id}.json` (do not inline full table to avoid duplication).
  - Metadata: `table_id`, `title`, `units/currency`, `page`, `section_path`, `quarter`, `year`.

3) Slides / Decks
- One `slide` chunk per slide/page.
- `text`: `Slide: <title>\n- bullet 1\n- bullet 2 ...` using OCR output from processing.
- Metadata: `slide_no`, `page`, `image_ref=pages/{doc_id}/{page:04}.png`, `quarter`, `year`.
- If a slide contains a structured table already extracted, also create a separate `table` chunk referencing the table JSON; the `slide` chunk remains textual/visual summary.

4) Press/IR Announcements
- Treat as `narrative` with lightweight sections (title/date).
- Split into 400–800 token chunks by paragraphs or bullet groups.
- Metadata: `published_date` (if available), `section_path`, `page_start/end`.

---

Chunk Manifest Schema (JSONL)
Each line represents one chunk:

```
{
  "chunk_id": "uuid7-or-hash",
  "doc_id": "...",
  "doc_type": "transcript|release|slides|press",
  "quarter": "2025-Q2",
  "year": 2025,
  "chunk_type": "prepared_remarks|qna|narrative|table|slide",
  "text": "... retrieval text ...",
  "payload_ref": "elements/{doc_id}/tables/{table_id}.json" ,
  "metadata": {
    "section_path": ["Financial Results", "Segment Results"],
    "speaker": "Ruth Porat",
    "role": "CFO",
    "qa_id": "...",
    "analyst_name": "...",
    "firm": "...",
    "answer_speakers": ["Sundar Pichai", "Ruth Porat"],
    "table_id": "tbl_...",
    "table_title": "...",
    "units": "USD",
    "currency": "USD",
    "slide_no": 12,
    "page_start": 14,
    "page_end": 16,
    "source_path": "raw/{doc_id}/file.pdf",
    "extraction_method": "elements|ocr_cloud|ocr_tesseract"
  },
  "checksum": "sha256-of-stable-fields",
  "order": 23
}
```

Notes:
- `text` should be self-sufficient for embedding while staying concise.
- `payload_ref` is only populated for `table` chunks (and optionally for other chunk types if there’s a structured attachment).
- `order` is a deterministic sequence within the document (page then reading order), useful for UI navigation.

---

Idempotency & Determinism
- Chunk identity (`chunk_id`) is derived from a stable hash of: `doc_id | chunk_type | section_path | page_start | page_end | speaker/qa_id/table_id/slide_no | normalized(text)`.
- Re-running chunking with same inputs produces identical `chunk_id`s and `checksum`s.
- When parser rules change, bump a `chunker_version` (tracked outside this manifest) and write a new manifest; preserve lineage.

Acceptance Criteria
- No `qna` chunk splits a question from its answers.
- No `table` chunk mixes multiple tables; every table has a synopsis and `payload_ref`.
- `prepared_remarks` chunks do not cross speaker boundaries; sizes near target.
- `slide` chunks reference the page image; OCR text is included.
- All chunks include `quarter` and `year` (if derivable from filename or metadata).
- `chunk_manifest.jsonl` is sorted by `order` and fully covers the document’s retrievable content.

Quality Heuristics
- Minimum text size for a chunk: ~50 tokens unless it’s a standalone unit (e.g., short Q or single-slide bullet).
- Merge adjacent tiny paragraphs if the speakers/section are the same.
- Keep numeric sentences intact with nearby qualifiers (YoY/QoQ, currency).

Storage Layout
- Output path: `chunks/{doc_id}/chunk_manifest.jsonl`.
- Do not duplicate large JSON payloads; reference them via `payload_ref`.

Limitations & Future Enhancements
- Cross-page tables may require merge rules in processing to ensure atomic table chunks here.
- Slide figures without extracted text may need future captioning via cloud VLMs (out of scope for chunking).
- Company-specific transcript formats can be supported with pluggable rule profiles, but remain deterministic.
