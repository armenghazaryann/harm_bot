# Application Architecture — Multimodal Graph RAG (MVP → Growth → Scale)

This document describes the end-to-end architecture for a highly accurate Retrieval-Augmented Generation (RAG) system over small but multimodal PDFs. It covers ingestion, processing, chunking, embeddings, graph construction, retrieval and Q/A, evaluation, and a scale plan. It is tailored to the current dataset (~25 PDFs, ~500 KB–1 MB each; total ~7 MB) and prioritizes accuracy on costs, images/charts, and textual statements.

Constraints and principles
- No custom local ML models. Use library-based extraction and cloud services (OCR/LLM/embeddings) only.
- Precision-first with strict citations and numeric verification.
- Deterministic, idempotent artifacts at every stage.

Core components
- Object storage: MinIO buckets for raw PDFs, page images, elements JSON, normalized tables, chunk manifests.
- Database: Postgres (+ pgvector) for metadata, vectors, and lightweight knowledge graph tables.
- API and workers: Single service (FastAPI) with background jobs for MVP.
- Cloud models:
  - Text embeddings: OpenAI `text-embedding-3-large`
  - Page/slide embeddings: VoyageAI `voyage-multimodal-3`
  - LLM for answers: OpenAI GPT-4.x family (e.g., `gpt-4.1` or `gpt-4o-mini`)
  - OCR (if needed): Google Vision, AWS Textract, or Azure Read

Related docs
- `PDF_PROCESSING.MD` — Processing step
- `CHUNKING.MD` — Chunking step
- `RETRIEVAL_AND_QA.MD` — Retrieval and Q/A step
- `EVALUATION.MD` — Evaluation plan

---

## 1) Data ingestion → processing → chunking → embeddings

High-level flow (MVP)
1. Upload/Register PDF → compute SHA256 checksum; store raw to `raw/{doc_id}/file.pdf`; insert rows in `documents` and `document_versions`.
2. Processing (`PDF_PROCESSING.MD`):
   - Render pages to PNG (300–400 DPI) → `pages/{doc_id}/{page:04}.png`.
   - Extract text (PyMuPDF/pdfminer). Cloud OCR only for pages with poor text extraction (Vision/Textract/Azure).
   - Partition to elements JSONL (Unstructured fast strategy). Finance heuristics: `SpeakerSegment`, `QA`, `SectionHeader`.
   - Tables: Camelot/Tabula; normalize to JSON with types/units/currency → `elements/{doc_id}/tables/{table_id}.json`.
   - Extract links and images; write processing report.
3. Chunking (`CHUNKING.MD`):
   - Transcripts: `prepared_remarks` by speaker and `qna` as one chunk per question with all answers.
   - Releases/Press: `narrative` by section hierarchy; `table` per normalized table with synopsis.
   - Slides: `slide` per slide with OCR text; separate `table` chunks if structured tables detected.
   - Output: `chunks/{doc_id}/chunk_manifest.jsonl` (deterministic order and IDs).
4. Embeddings:
   - `space=text`: OpenAI embeddings for narrative/transcript chunks.
   - `space=table`: OpenAI embeddings for table synopses; structured table JSON kept for exact numeric lookup.
   - `space=page`: VoyageAI multimodal embeddings for page/slide images.
   - Also build Postgres FTS (BM25) over `chunks.text`.

---

## 2) Document modalities supported
- Text (narratives, prepared remarks)
- Conversations (Q&A with analyst and management speakers)
- Tables (KPI/segment metrics, GAAP vs non-GAAP)
- Images/charts (slides, figures)
- Links (PDF annotations)
- Reports sections (e.g., Costs, OpEx, CapEx, reconciliation)

Processing and chunking preserve modality with rich metadata (speaker, section, table_id, slide_no, page ranges), enabling type-aware retrieval.

---

## 3) API contracts (MVP)

POST /upload/initiate
- Purpose: Prepare upload; return presigned URL (optional for small docs; inline upload also acceptable).
- Request:
```json
{
  "filename": "2025-q2-earnings-transcript.pdf",
  "content_type": "application/pdf",
  "doc_type": "transcript",
  "quarter": "2025-Q2",
  "year": 2025
}
```
- Response:
```json
{
  "doc_id": "...",
  "upload_url": "https://minio/...",
  "fields": {"key": "..."}
}
```

POST /ingest/complete
- Purpose: Register file at `raw/...` and enqueue processing job.
- Request:
```json
{ "doc_id": "...", "checksum": "sha256-..." }
```
- Response: `{ "status": "queued" }`

POST /query
- Purpose: Answer questions with strict citations and numeric verification.
- Request:
```json
{
  "question": "What was Q2 2025 Google Cloud revenue?",
  "filters": {"quarter": "2025-Q2", "doc_type": ["release","slides","transcript"]},
  "top_k": 8,
  "include_images": false
}
```
- Response:
```json
{
  "answer": "Google Cloud revenue in Q2 2025 was $X.XB (+Y% YoY).",
  "citations": [
    {"doc_id": "...", "doc_type": "release", "quarter": "2025-Q2", "page": 5, "table_id": "tbl_123"}
  ],
  "sources": {
    "chunks": [{"chunk_id": "...", "score": 0.81}],
    "tables": [{"table_id": "tbl_123", "payload_ref": "elements/.../tables/tbl_123.json"}],
    "slides": [{"page": 12, "image_ref": "pages/.../0012.png"}]
  },
  "verification": {
    "status": "verified",
    "details": [{"value": "1.2B", "from": "tbl_123", "tolerance": 0.001}]
  }
}
```

POST /eval/run
- Purpose: Execute evaluation suite over a dataset.
- Request: `{ "run_label": "nightly-2025-09-12" }`
- Response: `{ "run_id": "..." }`

GET /healthz
- Purpose: Liveness/readiness check.

---

## 4) Table structure (Postgres schema, conceptual)

Documents and versions
- `documents(id, source_path, doc_type, company, quarter, year, uploaded_at, checksum)`
- `document_versions(id, document_id, version, created_at, parser_version)`

Sections and chunks
- `sections(id, document_version_id, section_type, title, metadata_jsonb)`
- `chunks(id, section_id, chunk_type, text, payload_jsonb, page_start, page_end, metadata_jsonb, checksum)`
  - `chunk_type ∈ {prepared_remarks, qna, narrative, table, slide}`
  - `payload_jsonb` stores refs (e.g., `payload_ref` for tables)

Tables
- `tables(id, document_version_id, title, page, columns_jsonb, rows_jsonb, units, currency, footnotes_jsonb)`
  - Typed cells; numeric coercion; YoY/QoQ flags when present

Vectors and search
- `embeddings(id, chunk_id, space, model, model_version, vector)`
  - `space ∈ {text, table, page}`; pgvector HNSW per space
- FTS: GIN index over `chunks(text)`

Knowledge graph (lightweight in Postgres)
- `kg_nodes(id, type, properties_jsonb)`
  - Types: Company, Quarter, KPI, Segment, Person, Analyst, Firm, Document, Table, Slide, Chunk
- `kg_edges(id, src_id, dst_id, type, properties_jsonb)`
  - Types: part_of, mentions, has_kpi_value, asked_by, answered_by, supports, references, same_as
- `kg_summaries(id, scope, summary_text, citations_jsonb)`
  - Scope examples: {quarter: "2025-Q2"}, {segment: "Cloud", quarter: "2025-Q2"}

Jobs and eval
- `jobs(id, type, payload_jsonb, status, attempts, created_at, updated_at)`
- `eval_queries(id, question, gold_answer, metadata_jsonb)`
- `eval_runs(id, created_at, config_jsonb)`
- `eval_results(id, eval_run_id, query_id, result_jsonb)`

Indexes
- pgvector HNSW per `space`
- GIN FTS on `chunks(text)`
- GIN on JSONB filter fields (quarter, year, doc_type)

---

## 5) Graph-based dependencies (cross-doc context)

Purpose
- Capture how documents relate (e.g., quarterly continuity, segment metrics across releases, Q&A that references guidance) and enable explainable, multi-hop retrieval.

Node types
- Company, Quarter, KPI, Segment, Person (e.g., Sundar Pichai), Analyst, Firm, Document, Table, Slide, Chunk

Edge types
- `part_of(Document → Quarter)` — e.g., transcript/release belongs to a quarter
- `mentions(Chunk → Entity)` — chunk mentions KPI, Segment, Person, etc.
- `has_kpi_value(KPI ↔ Quarter/Segment, value, unit, source_table_id)`
- `asked_by(QA → Analyst/Firm)`, `answered_by(QA → Person)`
- `supports(Slide/Table → Claim/NarrativeSection)`
- `references(Document/Chunk → Document/Chunk)` — hyperlinks/explicit references
- `same_as(Entity ↔ Entity)` — entity resolution across docs

Dependency patterns
- Link transcript Q&A answers to the same quarter’s release tables via `has_kpi_value` and `supports` edges.
- Connect press/IR announcements to quarters/entities they reference.
- Maintain independence for unrelated docs by the absence of edges; retrieval can filter by connected subgraph.

Community summaries (GraphRAG-style)
- Generate per-quarter and per-segment/KPI summaries grounded in citations from their subgraphs; store in `kg_summaries`.
- Use for broad/comparative queries to reduce context size and improve faithfulness.

---

## 6) Evaluation plan (summary)

Dataset
- 50–100 queries spanning:
  - Numeric (KPI values; GAAP vs non-GAAP)
  - Guidance/attribution (speaker-aware)
  - Segment/quarter comparisons (multi-hop)
  - Visual/slide references
  - Press/IR facts

Metrics
- Retrieval: Recall@K per space (text/table/page), HybridHitRate@K
- Answering: NumericExact, NumericWithinTolerance, CitationCoverage, CitationCorrectness, AttributionAccuracy, Faithfulness (optional cloud judge)
- System: p50/p95 latency by stage; cost per query

Gating thresholds (MVP)
- NumericWithinTolerance ≥ 0.90
- CitationCorrectness ≥ 0.90
- AttributionAccuracy ≥ 0.85 (subset)
- HybridHitRate@10 ≥ 0.95

See `EVALUATION.MD` for full details.

---

## 7) Scale plan (ingest more, answer more)

MVP (current size: ~25 PDFs, ~7 MB total)
- Single FastAPI service; Postgres + pgvector; MinIO.
- Batch embeddings; deterministic processing; nightly eval runs.

Growth
- Orchestration: Prefect flows
- Fan-out: RabbitMQ for parallel chunk→embed→graph jobs
- Redis: idempotency keys, rate limits, lightweight locks
- Graph DB: Migrate KG to Neo4j if multi-hop queries become complex; sync from Postgres
- GPU workers for faster embeddings (batch OpenAI where possible; VoyageAI remains cloud)

Scale
- Temporal for durable workflows
- Partitioned pgvector; ANN tuning; caching for hot chunks
- Advanced graph summarization, community detection, and learned rerankers (cloud)

---

## 8) Graph-based answers with self-check (hallucination prevention)

Self-check mechanisms
- Numeric verification (deterministic):
  - Any numeric answer must be verified against structured table payloads (`payload_ref`).
  - Normalize units/currency; enforce tolerance. On mismatch, return both values with citations.
- Citation enforcement:
  - Answers must include citations (doc_type, quarter, page/slide, speaker/table_id). Reject uncited claims.
- Graph consistency:
  - For cross-quarter/segment claims, check against graph edges (`has_kpi_value`, `part_of`). Flag contradictions.
- Quote fidelity:
  - For attributions, include verbatim spans and the `speaker` from transcript chunks.
- Visual grounding (slides):
  - Attach `image_ref` for slide/page answers; optionally route to VLM for image-grounded verification (cloud only).

Fallback behaviors
- If insufficient evidence after retrieval, ask for clarification (e.g., quarter/segment) or return top-3 sources to refine.

---

## Operational considerations

Idempotency and lineage
- Keys: document SHA256, page render params, stable chunk hashes, embedding keys by (chunk_id, space, model_version), KG node/edge hashes.
- Versioning: new `document_version` and `chunker_version` when rules change; preserve lineage.

Observability
- Metrics per stage: pages rendered, OCR usage, elements extracted, tables normalized, chunks created, vectors written, KG nodes/edges added.
- Query metrics: recall@K by space, citation coverage, numeric verification pass rate, latency/cost.

Security
- API keys via env/secret manager; TLS for all cloud calls; PII-safe logging; GDPR-ready deletion by `doc_id` lineage.

---

## Directory and bucket conventions

MinIO
- `raw/{doc_id}/{filename}.pdf`
- `pages/{doc_id}/{page:04}.png`
- `elements/{doc_id}/elements.jsonl`
- `elements/{doc_id}/tables/{table_id}.json`
- `images/{doc_id}/{image_id}.png`
- `links/{doc_id}/links.jsonl`
- `chunks/{doc_id}/chunk_manifest.jsonl`
- `processing/{doc_id}/report.json`

Source tree (suggested)
- `api/` (FastAPI service)
- `workers/` (background jobs)
- `db/` (migrations, schema)
- `docs/` (this and related .MD files) or root as in this repo

---

## Implementation roadmap (MVP)
- Stand up schema and pgvector indexes; create buckets.
- Implement processing for one transcript and one release; produce `elements.jsonl`, tables JSON, and `chunk_manifest.jsonl`.
- Write embedding writers for text/table/page spaces; build FTS.
- Build minimal KG: Quarter, KPI, Segment, Person, Analyst, Table, Chunk with edges; generate a few community summaries.
- Implement multi-vector retrieval with numeric verification and strict citations.
- Create eval dataset (50–100 queries) and nightly eval job.

---

## References (best practices)
- Microsoft GraphRAG: https://microsoft.github.io/graphrag/ — graph-based retrieval with community summaries
- VoyageAI Multimodal-3: https://blog.voyageai.com/2024/11/12/voyage-multimodal-3/ — multimodal embeddings for slides/pages
- Unstructured partitioning: https://docs.unstructured.io/open-source/core-functionality/partitioning — layout-aware element extraction
- Camelot/Tabula: https://camelot-py.readthedocs.io/ / https://tabula-py.readthedocs.io/ — table extraction for PDFs
- TAPAS (tables QA, background): https://arxiv.org/abs/2004.02349 — we rely on deterministic lookup instead
