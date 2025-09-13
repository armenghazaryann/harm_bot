# RAG ETL Refactor Plan (KISS • DRY • GRASP • SOLID • Clean Code)

This plan rewrites the project architecture to be simple, modular, and testable while retaining production-grade capabilities. It consolidates duplicated logic, centralizes dependency injection, and cleanly separates the RAG stack (retrieval, rerank, self-check) from application concerns. The plan is phased (MVP → Growth → Scale) and focuses on minimal redundancy.

---

## 1) Executive Summary

- Unify the codebase under a layered architecture with a single dependency injection container.
- Create a dedicated `rag/` module for hybrid retrieval (FTS/BM25 + vector), reranking, and self-check.
- Consolidate ETL into a small, idempotent pipeline (ingest → parse → chunk → embed → index) with a manifest.
- Add cost analytics (per-call cost and latency) with DB schema and middleware/instrumentation.
- Eliminate large god-classes and repeated utilities; move to small, composable services.

Benefits: fewer files to reason about, clear boundaries, better testability, straightforward scaling.

---

## 2) Guiding Principles (Applied)

- KISS: Fewer moving parts; default to Postgres + pgvector + built-in FTS; avoid bespoke infra in MVP.
- DRY: Single source of truth for DI, settings, repositories, and RAG pipelines.
- GRASP: High cohesion per module; clear responsibilities (e.g., retriever vs reranker vs pipeline).
- SOLID: Interfaces for repositories and providers; dependency inversion through DI.
- Clean Code: Small files, explicit naming, predictable error handling, typed interfaces, tests.

---

## 3) Current State Findings & Redundancy Targets

- __Duplicated DI containers__: `api/di/container.py` and `workers/container.py` create parallel resource graphs. Action: replace with a single `di/container.py` shared by API and workers.
- __Overgrown modules__:
  - `api/features/query/service.py` (~35kB): mixed concerns (retrieval, ranking, prompting, answering, self-RAG). Action: split into `rag/pipeline/query_pipeline.py`, `rag/retrievers/*`, `rag/rankers/*`, `rag/self_check.py`, and a thin `QueryService` orchestrator.
  - `workers/langchain_processor.py` (~59kB): monolithic ingestion + embedding. Action: refactor to `etl/pipeline.py` with small, composable steps.
- __Split utilities__:
  - `infra/minio_utils.py` vs `infra/resources.py`: duplicate MinIO access patterns. Action: converge on `infra/resources.py` and thin helpers in `infrastructure/adapters`.
  - `infra/db_utils.py` provides DSN conversion only; incorporate into `infrastructure/vector/pgvector_store.py` (single source).
- __Retrieval scattered in feature code__: `api/features/query/retrievers.py` should move under `rag/retrievers/` and expose a consistent interface.
- __Controllers contain logic__: controllers under `api/features/*/controller.py` include orchestration that belongs to services/pipelines. Action: keep controllers thin.

---

## 4) Proposed Layered Architecture

- __Presentation (Interfaces)__: FastAPI routers/controllers, Streamlit UI adapters.
- __Application Layer__: Use cases and services orchestrating domain and RAG pipelines.
- __Domain Layer__: Entities, value objects, repository interfaces; business rules.
- __Infrastructure Layer__: SQLAlchemy models + repos, pgvector store, FTS adapter, MinIO/S3 adapter, LLM/reranker providers, cost analytics store.
- __RAG Module__: Retrieval (vector, FTS), fusion, reranking, self-check, prompts, evaluation.
- __DI Layer__: Single container composing resources and providers for all layers.

Tradeoffs: Clean separation increases indirection slightly; benefits outweigh costs via testability and replaceability.

---

## 5) New Folder Structure

```text
.
├─ app/                                  # presentation
│  ├─ api/
│  │  ├─ main.py                         # FastAPI app, lifespan, wiring
│  │  ├─ routers/
│  │  │  ├─ documents.py
│  │  │  └─ query.py
│  │  ├─ controllers/
│  │  │  ├─ documents_controller.py      # thin, delegates to services
│  │  │  └─ query_controller.py
│  │  └─ schemas/                        # pydantic DTOs
│  │     ├─ documents.py
│  │     └─ query.py
│  └─ ui/                                # streamlit adapters (optional)
│
├─ application/                          # use cases and orchestration
│  ├─ documents/
│  │  ├─ service.py                      # orchestrates ETL pipeline
│  │  └─ commands.py                     # register/upload commands
│  └─ query/
│     └─ service.py                      # orchestrates rag.pipeline
│
├─ domain/                               # business core
│  ├─ documents/
│  │  ├─ entities.py                     # Document, Chunk, Manifest
│  │  ├─ repository.py                   # interfaces (ports)
│  │  └─ events.py
│  ├─ query/
│  │  ├─ models.py                       # Query, RetrievalResult, Answer
│  │  └─ policies.py
│  └─ costs/
│     └─ entities.py                     # CostEvent, CostSummary
│
├─ infrastructure/                       # adapters + persistence (adapters)
│  ├─ resources/
│  │  ├─ db.py                           # async engine + sessions
│  │  ├─ minio.py                        # minio client
│  │  ├─ redis.py                        # redis client (optional)
│  │  └─ logging.py
│  ├─ persistence/
│  │  ├─ models/                         # SQLAlchemy tables
│  │  │  ├─ base.py
│  │  │  ├─ document.py
│  │  │  ├─ chunk.py
│  │  │  └─ cost.py
│  │  ├─ repositories/
│  │  │  ├─ document_repo.py             # implements domain.repository
│  │  │  └─ cost_repo.py
│  ├─ vector/
│  │  ├─ pgvector_store.py               # single source for PGVector
│  │  └─ embeddings.py                   # OpenAI/Voyage/etc. with cost hooks
│  ├─ search/
│  │  └─ fts.py                          # Postgres FTS (BM25-ish via ts_rank)
│  ├─ llm/
│  │  ├─ openai_client.py                # chat/complete with cost hooks
│  │  └─ reranker.py                     # Jina/Cohere/Voyage reranker adapter
│  └─ costs/
│     ├─ instrumentation.py              # wrapper + LangChain callbacks
│     └─ calculators.py                  # price sheets + normalization
│
├─ rag/                                  # RAG stack (single home)
│  ├─ retrievers/
│  │  ├─ vector_retriever.py
│  │  ├─ fts_retriever.py
│  │  └─ hybrid_retriever.py             # fusion (RRF), dedup, scoring
│  ├─ rankers/
│  │  ├─ reranker.py                     # call infrastructure.llm.reranker
│  │  └─ rrf.py                          # reciprocal rank fusion helper
│  ├─ pipeline/
│  │  ├─ query_pipeline.py               # end-to-end retrieval → rerank → answer
│  │  └─ self_check.py                   # Self-RAG Lite assertions + rewrite
│  ├─ prompts/
│  │  └─ answer_prompt.py
│  └─ evaluation/
│     └─ metrics.py
│
├─ etl/                                  # ingestion pipeline
│  ├─ pipeline.py                        # ingest → parse → chunk → embed → index
│  ├─ ingest.py                          # presigned/multipart + registration
│  ├─ parse.py                           # Unstructured/PyMuPDF/Tika
│  ├─ chunk.py                           # hierarchical/semantic splitters
│  ├─ embed.py                           # batch embeds + rate limiting
│  └─ index.py                           # persist embeddings + FTS index
│
├─ di/
│  └─ container.py                       # single DI container for API & workers
│
├─ core/
│  ├─ settings.py
│  ├─ errors.py
│  └─ utils.py
│
├─ workers/
│  ├─ celery_app.py                      # or simple background runners
│  └─ tasks.py                           # thin wrappers calling etl.pipeline
│
├─ tests/
│  ├─ unit/
│  └─ integration/
│
└─ docs/                                 # design docs and runbooks
```

Notes:
- The current `api/features/*` code will be mapped into `app/` and `application/` with logic moved into `rag/` and `etl/`.
- SQLAlchemy models move under `infrastructure/persistence/models/` with repositories in `infrastructure/persistence/repositories/`.

---

## 6) Centralized Dependency Injection

- Single container `di/container.py` wires:
  - Resources: DB, MinIO, Redis (opt), logger
  - Providers: PGVector store, embeddings, FTS, reranker, LLM
  - Services: `application.documents.service.DocumentService`, `application.query.service.QueryService`, `infrastructure.costs.instrumentation.CostTracker`
  - Pipelines: `etl.pipeline.Pipeline`, `rag.pipeline.QueryPipeline`
- Remove `api/di/container.py` and `workers/container.py`; they become imports of the single container.
- DI is the single place to swap providers (e.g., embeddings model, reranker vendor).

---

## 7) RAG: Hybrid Retrieval, Reranking, Self-Check

- __Retrievers__:
  - `vector_retriever.py`: wraps PGVector similarity search with unified return type.
  - `fts_retriever.py`: wraps Postgres FTS. MVP uses `ts_rank_cd` (BM25-like). Optionally adopt `pg_bm25` or `pgroonga` later.
  - `hybrid_retriever.py`: combine via Reciprocal Rank Fusion (RRF), deduplicate by chunk_id/doc_id, and normalize scores.
- __Reranking__:
  - `rankers/reranker.py`: adapter to Jina/Cohere/Voyage rerankers (configurable). Input: top M; Output: re-scored ranked list.
- __Self-Check (Self-RAG Lite)__:
  - `pipeline/self_check.py`: LLM-based grading and minimal query rewrite (HyDE/RAG-Fusion optional) with guardrails. Toggle via settings.
- __Answering__:
  - `pipeline/query_pipeline.py`: Orchestrates retrieval → rerank → context assembly → answer prompt → citations → optional chain-of-verification pass. Exposes structured metadata (scores, sources, verification).

Clean split ensures `application/query/service.py` is thin, delegating to the pipeline with config.

---

## 8) ETL: Idempotent and Small

- __Pipeline__ (`etl/pipeline.py`):
  - Steps: register → parse → chunk → embed → index
  - All steps are idempotent keyed by `document_id` and content checksum; step outputs are recorded in a manifest stored in Postgres and MinIO (`raw/`, `parsed/`, `chunks/`, `embeddings/`).
  - Chunk manifest contains shard references and per-chunk metadata (sequence, page range, checksum, sizes).
- __Parsing__: Use Unstructured/PyMuPDF; keep deterministic extraction with optional cloud OCR fallback (config).
- __Chunking__: Hierarchical/semantic splitters; token-based fallbacks (`CHUNK_TOKENS`, `CHUNK_OVERLAP_TOKENS`).
- __Embedding__: Batch with rate limiting; retries; exponential backoff; store vectors via PGVector in the same DB.
- __Index__: Ensure GIN index for FTS fields and HNSW/IVFFLAT for pgvector if configured.
- __Workers__: Keep Celery if already running; otherwise an async background runner suffices for MVP.

This replaces `workers/langchain_processor.py` with ~200–300 lines across `etl/*`.

---

## 9) Cost Analytics (Per-call Cost & Latency)

- __Goals__: Track input/output tokens, unit pricing, total USD, latency, provider/model, endpoint, correlation id.
- __Schema__ (Alemic migration):
  - `cost_event` (UUID id, ts timestamptz, provider text, model text, route text, user_id text null, request_id text, correlation_id text, tokens_prompt int, tokens_completion int, tokens_total int, input_units numeric, output_units numeric, cost_input_usd numeric(12,6), cost_output_usd numeric(12,6), cost_total_usd numeric(12,6), latency_ms int, status text, metadata jsonb)
  - `cost_summary_daily` (date, provider, model, route, totals…)
  - Indexes: by ts, provider+model, route, request_id; summary partitioned by day.
- __Instrumentation__:
  - Wrap OpenAI/LLM clients and rerankers in `infrastructure/costs/instrumentation.py`.
  - Provide LangChain callback handler to capture token usage automatically where applicable.
  - Middleware in FastAPI to attach `request_id`/`correlation_id`.
- __Reporting__: Simple queries and a Grafana/Metabase dashboard later.

---

## 10) Database & Indexing

- Postgres hosts both metadata and vectors (pgvector).
- Ensure:
  - `CREATE EXTENSION IF NOT EXISTS vector;`
  - HNSW index (or IVFFLAT) on vector column.
  - GIN index on `to_tsvector('english', content_normalized)` for FTS.
- For true BM25, evaluate `pg_bm25` extension; otherwise use `ts_rank_cd` with weights as BM25-ish.

---

## 11) Settings & Configuration

- Single `core/settings.py` (already present) remains; add sections for reranker provider, LLM models, and cost logging toggle.
- Remove scattered env logic; DI container consumes `SETTINGS` exclusively.

---

## 12) Testing Strategy

- Unit tests: retrievers, fusion (RRF), reranker adapter, self-check guards, cost calculators.
- Integration tests: query pipeline end-to-end (FTS + vector + rerank), ETL pipeline with small PDFs/CSVs.
- Contract tests: repository interfaces against an ephemeral Postgres.
- Add pytest, hypothesis for property tests where useful, and test containers.

---

## 13) Observability & Reliability

- Structured logging via `structlog` with correlation ids.
- Timing metrics per step (retrieve, rerank, answer) and ETL stages.
- Error taxonomy with retries and dead-letter queues (if Celery used); otherwise store failed jobs with reasons.

---

## 14) Migration Plan (Phased)

- __MVP (1–3 days)__
  1. Create `di/container.py` and rewire API/Workers to use it.
  2. Introduce `rag/` with retrievers and hybrid fusion; move `api/features/query/retrievers.py` here.
  3. Implement `rag/pipeline/query_pipeline.py`; slim down `application/query/service.py`.
  4. Replace `workers/langchain_processor.py` with `etl/pipeline.py` + small tasks in `workers/tasks.py`.
  5. Add cost tables and instrumentation; wrap OpenAI and reranker clients.

- __Growth (next 1–2 weeks)__
  - Add reranker vendor options, configurable ranks.
  - Self-RAG Lite with verification + rewrite.
  - Batch embeddings with circuit breakers and rate limits; manifest checkpoints.

- __Scale (later)__
  - Partitioning (pg_partman), UUID7 keys, read replicas for queries.
  - Orchestrator (Temporal) if/when needed; RabbitMQ for fan-out.
  - Advanced search (semantic reranking with cross-encoders), feedback loops, telemetry dashboards.

---

## 15) Mapping Old → New (Delete/Shrink List)

- Delete or shrink substantially:
  - `workers/langchain_processor.py` → replaced by `etl/pipeline.py` + `etl/*` (−80–90%).
  - `api/features/query/service.py` → move to `rag/pipeline/query_pipeline.py` + thin service (−60–80%).
  - `api/features/query/controller.py` → keep thin controller only (−50%).
  - `infra/minio_utils.py` → fold into `infrastructure/resources/minio.py` (+ small helpers).
  - `infra/db_utils.py` → move DSN conversion into `infrastructure/vector/pgvector_store.py`.
  - `api/di/container.py`, `workers/container.py` → replaced by `di/container.py`.
  - `workers/document_processor.py`, `workers/indexing.py`, `workers/initialization.py` → functionality covered by `etl/*`.

---

## 16) Acceptance Criteria

- Single DI container used by both API and workers.
- Query path uses hybrid retrieval (FTS + vector) + optional reranker + optional self-check with clean toggles.
- ETL path is idempotent and emits a manifest per document.
- Cost events are persisted per call with model/provider costs and latency.
- Controllers are thin; services are small; tests cover core paths.

---

## 17) Implementation Notes & Tradeoffs

- Postgres FTS ranking via `ts_rank_cd` approximates BM25; acceptable for MVP. Swap for a true BM25 extension when needed.
- Keeping vectors in Postgres reduces system complexity; scaling options exist (partitioning, replicas).
- DI centralization removes implicit coupling and eases testing by swapping providers.

---

## 18) Immediate Next Steps (Actionable)

1. Create `di/container.py` with resources and providers; wire `app/api/main.py` and `workers/celery_app.py` to it.
2. Move `api/features/query/retrievers.py` → `rag/retrievers/` and implement `hybrid_retriever.py` with RRF.
3. Implement `rag/pipeline/query_pipeline.py` (retrieval → rerank → answer → citations) and slim `application/query/service.py`.
4. Build `etl/pipeline.py` with manifest tracking; remove `workers/langchain_processor.py`.
5. Add Alembic migration for `cost_event` and `cost_summary_daily`; implement instrumentation wrappers.
6. Add unit/integration tests for hybrid retrieval and ETL.

This plan keeps the system lean today and paves clean paths to grow and scale later without re-architecting again.
