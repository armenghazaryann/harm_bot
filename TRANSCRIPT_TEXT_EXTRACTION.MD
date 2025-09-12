# Transcript Text Extraction from PDFs (MVP) — MinIO → JSONL Utterances → Summary → Graph

This document specifies an MVP pipeline focused only on extracting conversational text from transcript-style PDFs and producing JSON-formatted chunks of the form:

{
  "speaker": "Operator",
  "speech": "Thank you. And that concludes our question-and-answer session. I would like to turn the conference back over to Jim Friedland for any closing remarks."
}

The outputs are persisted to MinIO and then used to trigger summarization and graph ingestion (Neo4j + Postgres) for graph-based RAG.

Scope
- Input: Already-uploaded files in MinIO with an event that enqueues a Celery task.
- Output: Transcript-only utterances JSONL, plus summary artifacts and graph updates.
- Constraints: No custom local ML diarization; use robust, deterministic rules + PDF text extraction libraries with optional cloud OCR fallback.
- Non-goals: Embeddings, retrieval, and non-transcript document processing are defined elsewhere; this doc covers transcript text extraction and its immediate downstream triggers only.

---

High-Level Flow
1) Celery task starts after upload and registry insert.
2) Fetch PDF bytes from MinIO given `doc_id` and `raw_path`.
3) Extract page text (library-based; PyMuPDF/pdfminer) with simple normalization.
4) Detect speaker turns using rule-based heuristics tuned to earnings call transcripts.
5) Produce `utterances.jsonl` with one JSON object per turn: `{speaker, speech, ...meta}`.
6) Save artifacts to MinIO under deterministic keys.
7) Trigger summarization (sliding-window policy akin to `test.py`), then reduce to a final summary.
8) Upsert nodes/edges into Neo4j (Call, Speaker, Utterance) and write reference rows to Postgres.

---

Storage Keys (MinIO)
- Raw PDF (already present): `raw/{doc_id}/{filename}.pdf`
- Utterances JSONL (this step): `transcripts/{doc_id}/utterances.jsonl`
- Extraction report/metrics: `transcripts/{doc_id}/report.json`
- Summarization artifacts (next step):
  - `transcripts/{doc_id}/summary/sliding_steps.json`
  - `transcripts/{doc_id}/summary/final_summary.txt`

All keys are deterministic and idempotent per `doc_id` + extraction version.

---

Celery Orchestration (MVP)
- Task names (proposed):
  - `workers.tasks.extract_transcript_utterances(doc_id)` → writes `utterances.jsonl`
  - `workers.tasks.summarize_transcript(doc_id)` → writes window steps and/or sliding-prefix steps
  - `workers.tasks.reduce_summary(doc_id)` → writes `final_summary.txt`
  - `workers.tasks.ingest_graph(doc_id)` → upserts Neo4j + Postgres

Suggested sequencing: chain the above tasks; use Redis-based idempotency keys (already in `workers/tasks.py`) to avoid duplicates. Retry on transient errors.

---

Input Assumptions
- The `documents` table has a row for `doc_id` with:
  - `doc_type = transcript`
  - `raw_path` pointing to MinIO key of the PDF
  - `checksum` for idempotency
- A MinIO client and credentials are configured (see `infra/resources.py -> MinIOResource`).

---

Text Extraction Strategy (Deterministic, Library-First)
- Primary extractors: PyMuPDF or pdfminer.six for text with layout order.
- OCR fallback (optional): Cloud providers (Google Vision, AWS Textract, Azure Read) for image-only pages (slides/scanned pages). Mark extraction method per page in the report.
- Normalization before speaker segmentation:
  - Remove page headers/footers when reliably detectable.
  - Collapse multiple spaces, preserve paragraph breaks.
  - Merge hyphenated line breaks where a word is split at end-of-line.
  - Track `page_no` to allow back-references in utterances and graph nodes.

---

Speaker Turn Detection (Rule-based; No Local ML)
Earnings call transcripts typically contain:
- “Participants” roster listing names and roles.
- Sections: “Prepared Remarks”, “Q&A”, sometimes “Operator Instructions”.
- Speaker labels appear as lines like:
  - `Operator:`
  - `Sundar Pichai (CEO):`
  - `Ruth Porat — President & Chief Investment Officer:` (em dash variants)
  - `Brian Nowak — Morgan Stanley:`

Heuristics (ordered):
- Segment document into sections by known headers: `Participants`, `Prepared Remarks`, `Question-and-Answer`, `Q&A`.
- Parse the Participants section to build a name/role roster.
- Detect speaker labels with robust patterns (examples, not implementation):
  - Title-case names with optional role in parentheses followed by colon.
  - Uppercase role or title tokens followed by colon (e.g., `OPERATOR:`), normalize to `Operator`.
  - Analyst lines often include firm names after an em dash; capture analyst display name and firm.
- A new turn starts when a speaker label is matched; the turn ends at the next speaker label or section boundary.
- Merge short lines from page wraps; maintain paragraph spacing within a single turn.
- Normalize em dashes and stray colon spacing.
- Filter boilerplate like safe-harbor disclaimers when reliably detectable.

Edge Cases and Resilience
- Pages with no text (images only) → OCR fallback.
- Line-initial timecodes or artifacts → strip via regex if consistent.
- Occasional unlabelled continuation paragraphs → attach to current speaker if within close proximity and no new label.
- Analysts with same first/last names → include firm in label to disambiguate.

References and Prior Art
- See community discussions on transcript regex strategies (e.g., StackOverflow on capturing speaker names before a colon). These inform the above rules while we keep the implementation deterministic and company-agnostic.

---

Utterances JSONL Schema
Write one line per utterance (JSON per line):
- `utterance_id`: stable hash of `{doc_id|speaker|speech|turn_index}`
- `doc_id`: UUID
- `turn_index`: 0-based order of appearance
- `speaker`: normalized display name (e.g., `Operator`, `Sundar Pichai`, `Brian Nowak (Morgan Stanley)`)
- `role`: `management|analyst|operator|unknown`
- `speech`: the full turn text (normalized)
- `section`: `prepared_remarks|qa|participants|other`
- `page_spans`: list of `{page_no, char_start, char_end}` if available
- `extraction_method`: `text|ocr_cloud|ocr_tesseract`
- `meta`: optional `{firm, title, roster_match_confidence, warnings[]}`

Example line:
{"utterance_id":"...","doc_id":"...","turn_index":42,"speaker":"Operator","role":"operator","speech":"Thank you...","section":"qa","page_spans":[{"page_no":17}],"extraction_method":"text"}

File: `transcripts/{doc_id}/utterances.jsonl`

Idempotency: Recompute and overwrite atomically (or versioned by `v=YYYYMMDD` if extraction rules change). Key stability ensures re-runs do not duplicate turns.

---

Extraction Report
Write `transcripts/{doc_id}/report.json` summarizing quality and counts:
- `pages_total`, `pages_ocr`, `text_coverage_ratio`
- `turns_total`, `speakers_distinct`, `qa_blocks_detected`
- `boilerplate_removed`, `unmatched_lines`
- warnings for low-confidence pages or atypical patterns

---

Summarization Trigger (Sliding Window + Reduce)
- Input: `utterances.jsonl`
- Policy: Use the sliding-prefix online policy and/or independent windows + reduce as in `test.py`.
- Environment: `OPENAI_API_KEY`, model default `gpt-4o-mini` unless overridden.
- Tunables: `window_size`, `stride`, `concurrency`, `timeout`, `max_retries`.
- Outputs:
  - Step log (for sliding-prefix): `transcripts/{doc_id}/summary/sliding_steps.json`
  - Final summary: `transcripts/{doc_id}/summary/final_summary.txt`

Operational notes
- Apply exponential backoff on rate limits and transient failures.
- Idempotent: The final summary content is a function of input utterances and prompts. If prompts change, bump a `summary_version` and write to `.../summary/v{n}/...`.

---

Graph Ingestion (Neo4j + Postgres)
Purpose: enable graph-based RAG by modeling the conversation structure and entities.

Neo4j Data Model (MVP)
- Nodes
  - `Call {doc_id, ticker?, quarter?, year?, source: raw_path}`
  - `Speaker {name, role, firm?}`
  - `Utterance {utterance_id, turn_index, section}`
- Relationships
  - `(Call)-[:CONTAINS]->(Utterance)`
  - `(Speaker)-[:SPOKE]->(Utterance)`
  - `(Utterance)-[:NEXT]->(Utterance)` for chronological linkage
  - `(Call)-[:HAS_SPEAKER]->(Speaker)`

Upsert/Idempotency
- Create uniqueness constraints on `Call.doc_id`, `Utterance.utterance_id`, and `Speaker (name, firm)` composite.
- Use MERGE for upserts; relationships MERGE’d with endpoints to avoid duplication.

Postgres (Reference + Retrieval Bridge)
- Persist a light-weight `utterances` table:
  - `doc_id UUID`, `utterance_id TEXT PK`, `turn_index INT`, `speaker TEXT`, `role TEXT`, `section TEXT`, `speech TEXT`, `created_at`, `version`
- Later (outside this doc): add embeddings in `utterance_embeddings` (pgvector), and join Neo4j ids if needed.

---

Failure Handling & Idempotency
- Use Redis-based idempotency keys per task payload (see `workers/tasks.py`).
- Common failure modes:
  - MinIO read errors → retry with backoff; fail if 5xx persists.
  - OCR provider timeouts → skip page with warning; mark `extraction_method` accordingly.
  - No speaker labels detected → fallback to `Unknown` speaker buckets; still emit text to avoid data loss.
- Partial writes: Write to temp keys, then move/rename to final keys to ensure atomic artifact visibility.

---

Metrics & Observability
- Log per-document lifecycle: started, extracted, utterances_count, summary_done, graph_done.
- Counters: pages_ocr, lines_unmatched, qa_detected.
- Timers: extraction_ms, summarization_ms, graph_ingest_ms.
- Persist metrics in `report.json` and emit to logs (structured logger).

---

Document Status Transitions (align with `api/features/documents/entities/document.py`)
- `PROCESSING` → upon task start
- `PROCESSED` → after generic PDF processing (see `PDF_PROCESSING.MD`)
- `CHUNKED` → after `utterances.jsonl` is written
- `INDEXED` (optional here) → after graph ingestion; embeddings indexing happens elsewhere

---

Acceptance Criteria (MVP)
- 100% of transcript text appears in `utterances.jsonl` as contiguous speaker turns.
- Operator, management, and analyst turns are distinguishable with ≥95% accurate labels on clean transcripts.
- Sliding-window summarization produces a single consolidated summary artifact.
- Neo4j contains `Call`, `Speaker`, `Utterance` nodes with `NEXT`/`SPOKE`/`CONTAINS` relationships for each document.
- Re-runs are idempotent for the same `doc_id` and extraction version.

---

Future Enhancements
- Company-specific heuristics and whitelists for speaker normalization.
- Entity/metric extraction for KPI nodes (products, regions, metrics) and relation edges (MENTIONS, QUOTES, QUANTIFIES).
- Automatic detection of Transcript vs. Release vs. Slides at ingestion time to route to the right pipeline.
- Streaming extraction for very large transcripts (shard by section).

---

Implementation Notes
- This document describes architecture and operational rules. See:
  - `PDF_PROCESSING.MD` for general PDF processing
  - `test.py` for sliding-window summarization patterns
  - `infra/resources.py` for MinIO resource
  - `workers/tasks.py` for Celery + idempotency pattern
