# PROJECT — Experiment RAG ETL (MVP → Growth → Scale)

This document is the single source of truth for how we will build, run, and evolve the RAG ETL system in this repo. It synthesizes and operationalizes the other design docs:

- `ARCHITECTURE.MD` — End-to-end architecture
- `PDF_PROCESSING.MD` — Processing step
- `CHUNKING.MD` — Chunking step
- `RETRIEVAL_AND_QA.MD` — Retrieval + Q/A step
- `EVALUATION.MD` — Evaluation plan

It adds concrete implementation details, tech stack, environment configuration, and an execution plan.


## 0) Goals, Scope, and Principles

- Accuracy-first for finance content (numbers, tables, prepared remarks, Q&A) with strict citations and numeric verification.
- Deterministic, idempotent artifacts at every stage (hash-addressable outputs, lineage/versioning).
- Lean MVP: single service, Postgres (+ pgvector), MinIO; background workers via Celery (Redis broker) with Flower.
- Cloud-first for AI: no custom local ML models. Use library-based PDF extraction, optional cloud OCR, and cloud embeddings/LLMs/VLMs.
- Dataset (current): ~25 Alphabet earnings documents (transcripts, releases, slides) in `kb/` (~500 KB–1 MB each).
- Must be shippable, testable, and secure by default.


## 1) End-to-End Architecture (MVP)

Flow
1. Ingest/Register → store raw to MinIO `raw/{doc_id}/...` + metadata rows in Postgres.
2. Process (PDF → pages/elements/tables/images/links) → `PDF_PROCESSING.MD`.
3. Chunk (deterministic, finance-aware types) → `CHUNKING.MD`.
4. Embed + Index (text/table/page spaces) → Postgres + pgvector + FTS.
5. Retrieve + Answer with citations + numeric verification → `RETRIEVAL_AND_QA.MD`.
6. Evaluate nightly over a curated dataset → `EVALUATION.MD`.

Runtime topology (MVP)
- Single FastAPI app hosting:
  - Public API: `/upload/*`, `/ingest/complete`, `/query`, `/eval/run`, `/healthz`.
  - Background jobs: Celery workers (Redis broker); Flower UI for monitoring. Optionally retain a `jobs` table for tracking/idempotency.
- Storage: MinIO (S3-compatible) for raw and derived artifacts.
- Database: Postgres 16 with pgvector extension. 


## 2) Tech Stack

Languages & Runtime
- Python ≥ 3.13 (per `pyproject.toml`)

Backend / API / Workers
- FastAPI (ASGI)
- Uvicorn (server)
- Pydantic v2 (settings + types)
- SQLAlchemy 2.0 (async) + `asyncpg`
- Alembic (migrations)
- Tenacity (backoff/retry)
- HTTPX (async HTTP client for cloud APIs)

Persistence
- Postgres 16
- pgvector extension (HNSW indexes by space)
- GIN FTS (BM25) on `chunks.text`

Object Storage
- MinIO (S3-compatible). Client: `minio` SDK or `boto3` (S3)

PDF Processing (library-first, no local ML)
- PyMuPDF (fitz) for rendering + text w/ positions
- Unstructured (partitioning, `strategy=fast`)
- Camelot/Tabula (tables; requires system deps) — recommended
- Tesseract for OCR

Embeddings & LLMs (cloud)
- Text embeddings: OpenAI `text-embedding-3-large`
- Page/slide embeddings: OpenAi gpt-4.1-mini
- LLM for answers: OpenAI GPT-4.x family (e.g., `gpt-4.1`, `gpt-4o-mini`)
- Optional VLM for visual reasoning: GPT-4.1o

Testing, Linting, CI
- Pytest, pytest-asyncio
- Ruff (lint), Black (format)
- Mypy (typing) — strict for core modules

Observability
- Structlog
- Prometheus client for metrics
- OpenTelemetry (optional later) for traces

Security
- Pydantic Settings for env config

## 3) API Surface (MVP)

- `POST /upload/initiate` → optional presigned URL (S3/MinIO) for large files
- `POST /ingest/complete` → register + enqueue processing job
- `POST /query` → returns answer, citations, sources, and numeric verification
- `POST /eval/run` → kicks off evaluation run, stores artifacts
- `GET /healthz` → liveness/readiness

Request/response examples are documented in `ARCHITECTURE.MD` and `RETRIEVAL_AND_QA.MD`.

[NOT MVP]
Admin/ops (internal only)
- `GET /jobs/{id}` → job status
- `POST /admin/reindex` → rebuild FTS / backfill embeddings for a space/model (guarded)


## 4) Storage Layout (MinIO)

Buckets/keys (see also `ARCHITECTURE.MD`)
- `raw/{doc_id}/{filename}.pdf`
- `pages/{doc_id}/{page:04}.png`
- `elements/{doc_id}/elements.jsonl`
- `elements/{doc_id}/tables/{table_id}.json`
- `images/{doc_id}/{image_id}.png`
- `links/{doc_id}/links.jsonl`
- `chunks/{doc_id}/chunk_manifest.jsonl`
- `processing/{doc_id}/report.json`

All outputs are deterministic and re-runnable. Idempotency keys are documented below.


## 5) Database Schema (Conceptual)

Tables (see `ARCHITECTURE.MD` for details)
- `documents` (registered files and metadata)
- `document_versions` (parser versions, lineage)
- `sections` (hierarchy context)
- `chunks` (prepared_remarks, qna, narrative, table, slide)
- `tables` (typed table JSON refs)
- `embeddings` (per-space vectors with model + version)
- `jobs` (background work items)
- `kg_nodes`, `kg_edges`, `kg_summaries` (lightweight KG)
- Eval: `eval_queries`, `eval_runs`, `eval_results`

Indexes
- pgvector HNSW per `space ∈ {text, table, page}` with `model`+`version`
- FTS GIN on `chunks(text)`
- JSONB GIN on fields like `quarter`, `year`, `doc_type`

Unique keys (idempotency)
- `documents(checksum)`
- `chunks(checksum)`
- `embeddings(chunk_id, space, model, model_version)`
- `tables(document_version_id, table_id)`


## 6) Processing Implementation Details (No local ML)

Reference: `PDF_PROCESSING.MD`

- Render: PyMuPDF at 300–400 DPI → `pages/…/####.png`
- Extract text w/ bbox: PyMuPDF or pdfminer.six
- OCR policy: Tesseract only if explicitly allowed
- Partition to elements: Unstructured `strategy=fast`; map to canonical types
- Tables: Camelot/Tabula (lattice → stream), normalize to typed JSON
- Figures/images: extract and store under `images/`
- Links: collect PDF annotations into `links/`
- Transcript structuring: rule-based `SpeakerSegment` + `QA`
- Cost/finance sections: rule-based header detection
- Deterministic `elements.jsonl` ordering (page, reading-order)
- Processing report: coverage metrics, warnings

Idempotency keys
- Document: `sha256(pdf_bytes)`
- Page image: `{doc_id, page_no, dpi}`
- Element: `sha256(type|page|bbox|text_hash)`
- Table: `sha256(norm_table_cells)`


## 7) Chunking Implementation Details (Deterministic)

Reference: `CHUNKING.MD`

- Chunk types: `prepared_remarks`, `qna`, `narrative`, `table`, `slide`
- Token targets: ~500–1,200 text tokens; do not split across Q&A pairs, tables, slides, or speaker boundaries
- Text shaping: normalize whitespace; prepend context preamble (speaker/section)
- Transcripts: group by speaker; Q&A → one chunk per Q with all answers
- Releases: section-based narrative chunks; structured tables → `table` chunks with synopsis + `payload_ref`
- Slides: one chunk per slide; OCR text + `image_ref`; separate `table` chunk if table detected
- Output: `chunks/{doc_id}/chunk_manifest.jsonl`

Idempotency key for `chunk_id`
- Hash of: `doc_id | chunk_type | section_path | page_start | page_end | speaker/qa_id/table_id/slide_no | normalized(text)`


## 8) Embeddings, Indexing, and Retrieval

Spaces & models
- `space=text` → OpenAI `text-embedding-3-large`
- `space=table` → OpenAI `text-embedding-3-large` over table synopses
- `space=page` → OpenAI
- FTS → Postgres GIN/BM25 on `chunks.text`

Indexing hygiene
- Record `model` and `model_version` with each vector
- Separate HNSW index per space; backfill on model change

Hybrid retrieval (MVP)
- Generate K per space (text/table/page) + K via FTS → union + dedupe
- Simple fusion with type-aware boosts (numeric → table priority; attribution → transcript priority; visual → page priority)
- Optional cloud reranker (Growth)


## 9) Answering and Verification

Reference: `RETRIEVAL_AND_QA.MD`

- Answer LLM: OpenAI GPT-4.x (choose model by latency/cost)
- Prompting rules: extractive, conservative, always cite `[doc_type, quarter, page/slide, speaker/table_id]`
- Numeric verification (deterministic):
  - Parse numbers from draft answer
  - Map to candidate table `payload_ref`
  - Normalize units/currency; compare to canonical table JSON with tolerance (e.g., 0.1%)
  - On mismatch: return discrepancy with citations

Failure modes
- If insufficient evidence: ask for clarification (e.g., quarter/segment) or return top-3 likely sources with snippets


## 10) Evaluation

Reference: `EVALUATION.MD`

- Dataset (50–100 queries): numeric, attribution, comparisons, slides, press
- Metrics: Retrieval Recall@K per space, HybridHitRate@K; NumericExact/WithinTolerance; CitationCoverage/Correctness; AttributionAccuracy; latency/cost
- Threshold gates (MVP): e.g., NumericWithinTolerance ≥ 0.90; CitationCorrectness ≥ 0.90
- Automation: `POST /eval/run` → stores JSON results + Markdown report


## 11) Background Jobs (Celery + Flower)

Jobs table (optional)
- `jobs(id, type, payload_jsonb, status ∈ {queued, running, done, failed}, attempts, created_at, updated_at)`

Celery workers
- Broker: Redis (`CELERY_BROKER_URL` or `REDIS_URL`), result backend can be Redis as well
- Queues: `ingest`, `chunk`, `embed`, `index`, `eval` (default routing configured in code)
- Start worker: `make worker` (runs `celery -A workers.celery_app.app worker -l info`)
- Start Flower: `make flower` (Celery monitoring UI at http://localhost:5555)

Job types (MVP)
- `process_document` → executes PDF processing pipeline
- `chunk_document` → generates chunk manifest
- `embed_chunks` → writes vectors for text/table/page spaces
- `index_build` → FTS build or reindex
- `eval_run` → executes evaluation suite

Observability per job
- Metrics: durations, counts, attempts
- Logs: structured JSON with `doc_id`, job id, stage


## 12) Configuration and Secrets

Environment variables (minimum)
- `OPENAI_API_KEY` (see `.env.example`)
- `USE_OPENAI=true|false` (toggle mock embeddings for dev)
- `DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname`
- `MINIO_ENDPOINT=http://localhost:9000`
- `MINIO_ACCESS_KEY=…`
- `MINIO_SECRET_KEY=…`
- `MINIO_BUCKET=rag`
- Optional OCR (choose one provider):
  - Google: `GOOGLE_APPLICATION_CREDENTIALS=…`
  - AWS Textract: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
  - Azure: `AZURE_FORM_RECOGNIZER_ENDPOINT`, `AZURE_FORM_RECOGNIZER_KEY`

Runtime knobs
- `PAGE_DPI=300|400`
- `EMBED_BATCH_SIZE=…` (e.g., 128)
- `MAX_CONCURRENCY=…` (job runner)
- `TOLERANCE_REL=0.001` (numeric verification)


## 13) Local Development

Recommended services (Docker Compose)
- Postgres 16 with pgvector
- Redis (Celery broker/result)
- MinIO

Typical workflow
1. Create `.env` from `.env.example` and fill keys
2. Start services: `make compose-up` (Postgres, Redis, MinIO + bucket init)
3. Install deps with uv: `make install` (or `make install-dev` for extras)
4. Run DB migrations: `make migrate`
5. Start API: `make api`
6. Start Celery worker: `make worker`
7. (Optional) Start Flower UI: `make flower` (http://localhost:5555)
8. Ingest a sample PDF from `kb/`
9. Run `/query` against known questions
10. Run `/eval/run` to produce baseline metrics

Testing & quality
- `pytest -q`
- `ruff check . && black --check .`
- `mypy` for core modules


## 14) Large-File Ingestion (Forward-Compatible MVP)

Even though our current dataset is many small PDFs, we support large-file ingestion via S3/MinIO presigned uploads:

- `POST /upload/initiate` returns a presigned URL (+ multipart parameters for very large files)
- Client uploads directly to MinIO (no proxying through API)
- `POST /ingest/complete` finalizes registration and enqueues `process_document`
- For CSVs: future Growth plan will shard into ~1M-row splits; For PDFs: split by pages or ranges
- Store shard/chunk manifests in Postgres for lineage


## 15) Security, Privacy, and Compliance

- Secrets only in environment/secret manager; never stored in repo
- TLS for all cloud API calls
- PII-safe structured logging; redaction option for queries
- GDPR-ready deletion: cascade by `doc_id` lineage (raw → derived → vectors → KG → eval artifacts)
- Optional S3 SSE or customer-managed keys for MinIO


## 16) Observability and SLOs

Metrics (Prometheus-ready)
- Ingest: docs processed, pages rendered, OCR pages, tables normalized
- Chunking: chunks produced per type, token-size distribution
- Index: vectors written per space, index sizes
- Query: latency breakdown (retrieval/LLM/verification), recall@K (sampled), citation coverage
- Costs: avg tokens, cost per query (estimated)

Logging
- JSON logs with request-id, doc_id, job_id, stage, timings

SLOs (MVP targets)
- Query p95 < 3s for small dataset
- NumericWithinTolerance ≥ 0.90; CitationCorrectness ≥ 0.90


## 17) Roadmap

MVP (this repo)
- Single FastAPI service; Postgres + pgvector; MinIO
- Deterministic processing → chunking → embeddings → retrieval + numeric verification
- Evaluation dataset and nightly eval job

Growth
- Orchestration: Prefect flows
- Fan-out: RabbitMQ for parallel chunk→embed jobs
- Redis: idempotency keys, rate limits, lightweight locks
- Optional: move KG to Neo4j if multi-hop queries expand

Scale
- Temporal for durable workflows
- Partitioned pgvector; ANN tuning; caching for hot chunks
- Advanced graph summarization and rerankers (cloud)


## 18) Dependencies to Add (Proposed)

Core
- fastapi, uvicorn[standard], pydantic, python-dotenv
- sqlalchemy[asyncio], asyncpg, alembic
- psycopg (for sync tooling / migrations where needed)
- pgvector (Python client helpers; DB extension installed separately)
- minio or boto3 (choose one; `minio` preferred for MinIO)
- httpx, tenacity
- prometheus-client, structlog (or stdlib JSON logger)

PDF Processing
- pymupdf, pdfminer.six, unstructured
- camelot-py (requires ghostscript) and/or tabula-py (requires Java) — optional
- pytesseract (optional, off by default)

Cloud Clients (optional, install only if used)
- google-cloud-vision
- boto3 (Textract)
- azure-ai-formrecognizer

Testing & Tooling
- pytest, pytest-asyncio, coverage
- ruff, black, mypy, types-* stubs as needed


## 19) Open Questions / Tradeoffs

- Table extraction: Camelot/Tabula system dependencies vs. relying more on cloud OCR for image tables?
- Page embeddings: cost/benefit of VoyageAI multimodal for our current queries; consider enabling only for slide/visual queries.
- Single `table` synopsis strategy: how rich should the synopsis be to improve retrieval while keeping it deterministic?
- When to introduce a graph DB for KG (if ever), given Postgres can sustain our current needs?


## 20) References

- See linked docs at top of this file for detailed per-stage guidance
- Microsoft GraphRAG — graph-based retrieval with community summaries
- VoyageAI Multimodal-3 — multimodal embeddings for slides/pages
- Unstructured — layout-aware element extraction
- Camelot/Tabula — PDF table extraction


---

Appendix A — Example Minimal .env

```
# Required
OPENAI_API_KEY=...
USE_OPENAI=true
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/rag
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=rag

# Optional OCR (choose one block)
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
# AWS_REGION=us-east-1
# AZURE_FORM_RECOGNIZER_ENDPOINT=...
# AZURE_FORM_RECOGNIZER_KEY=...
```

Appendix B — Directory Structure (proposed)

```
api/               # FastAPI app (routers, models, settings)
workers/           # background job runner and tasks
processing/        # PDF processing utilities
chunking/          # chunk building utilities
retrieval/         # retrieval + numeric verification
embeddings/        # embedding writers, indexers
kg/                # lightweight KG (optional modules)
db/                # migrations, schema SQL, Alembic
scripts/           # dev tools (seed, eval triggers)
docs/              # architecture docs (or keep in repo root)
```
