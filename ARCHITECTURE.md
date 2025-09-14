# Architecture Overview

## System Components

- **FastAPI Service** – HTTP API exposing document upload, query, and conversation endpoints. Runs as a stateless container.
- **PostgreSQL + pgvector** – Primary metadata store (documents, versions, chunk manifests, cost ledger, conversation history) and vector storage for embeddings in the MVP. In Growth/Scale we may off‑load vectors to Milvus.
- **MinIO** – S3‑compatible object storage for raw files, intermediate page shards, chunk JSON manifests, and final artifacts.
- **Celery Workers** – Background workers that process the ETL pipeline stages (parse, chunk, embed, index). Uses Redis as a broker and result backend.
- **Redis** – Fast in‑memory store for task queues, rate‑limit tokens, and idempotency keys.
- **OpenAI (Chat/Embeddings/Vision) via LangChain** – LLM provider for text generation, embedding creation, and optional OCR on images.
- **Jina AI Reranker** – Optional post‑retrieval reranking service to improve relevance.
- **Self‑RAG Lite (optional)** – LLM‑based self‑grading and query‑rewrite flow integrated in the query service.

## Data Flow (MVP)

1. **Upload** – Client requests a presigned multipart URL from the API, uploads directly to MinIO, then notifies the API (or the upload endpoint records the manifest).
2. **Job Queue** – A DB `job` row is created; a FastAPI background task enqueues a Celery job.
3. **Parse** – Worker downloads the object from MinIO, extracts text/pages using `unstructured` / `PyMuPDF` / `tika`, writes intermediate JSON to MinIO.
4. **Chunk** – Page‑range sharding (25‑50 pages) → semantic chunking via LangChain → chunk manifests stored in MinIO and metadata rows in Postgres.
5. **Embed** – Batches of chunk texts are sent to OpenAI embeddings; vectors stored in `pgvector` column.
6. **Index** – FTS `tsvector` column is populated; optional RRF hybrid ranking configuration.
7. **Query** – API receives a user query, performs optional HyDE generation, retrieves top‑k vectors, runs optional Jina rerank, and returns answer with citations.

## Growth / Scale Extensions

- **RabbitMQ** for fan‑out of shard jobs and DLQ handling.
- **Milvus** as a dedicated vector store for >10 M vectors.
- **Temporal** for durable, observable workflows.
- **Redis** for caching query results and rate‑limiting per tenant.
- **Kubernetes** with HPA for autoscaling API and workers.
- **Partitioning** of Postgres tables (`pg_partman`) for multi‑tenant scaling.

The architecture is deliberately modular: each component can be swapped (e.g., replace Celery with Prefect) without touching the core business logic.
