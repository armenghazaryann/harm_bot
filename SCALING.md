# Scaling Plan

This document outlines how to evolve the current MVP into a system that reliably handles thousands of concurrent uploads and multi‑TB corpora. It follows a pragmatic MVP → Growth → Scale path aligned with this codebase.

## Current State (MVP)

- Single Postgres (+ `pgvector`) for metadata and embeddings.
- Object storage MinIO for PDFs and ETL manifests.
- Celery + Redis for background jobs (ETL); Flower for visibility.
- FastAPI service exposing Documents, Query, Conversation endpoints.
- LangChain `PGVector` + Postgres FTS hybrid retrieval with RRF.
- Optional JinaAI reranker (API fallback if unset/failed).
- Minimal cost ledger (`cost_event`) and conversation history.

Key files:
- API: `api/main.py`, DI: `di/container.py`.
- ETL: `etl/pipeline.py`, `etl/parse_pdf.py`, `etl/chunk.py`, `etl/embed.py`, `etl/index.py`.
- Workers: `workers/celery_app.py`, `workers/tasks.py`.
- Retrieval: `rag/retrievers/*.py`, reranker: `rag/rankers/reranker.py`.

## Objectives & SLOs

- Throughput: ≥ 5,000 parallel uploads of ~1 GB each; support a single file up to 100 GB.
- Query P50 latency: < 1.5 s for top‑k=8 with optional rerank; < 800 ms without rerank.
- ETL reliability: ≥ 99.9 % step success across parse → chunk → embed → index with automatic retries.
- Cost guardrails: enforce per‑tenant monthly budget and per‑minute OpenAI/Reranker QPS limits.

## Bottlenecks to Address

- Upload path: API proxying large files. **Solution**: presigned multipart directly to MinIO.
- Parsing: memory/CPU spikes for giant PDFs. **Solution**: page/page‑range sharding and streaming.
- Embeddings: OpenAI rate limits. **Solution**: batch + concurrency control + rate limiting.
- Indexing: single table write hotspots. **Solution**: partitioning and write batching.
- Queries: reranker + FTS under load. **Solution**: caching, tuned indexes, optional rerank.

## Growth Plan

### 1) Direct‑to‑MinIO Presigned Multipart Uploads
- Add endpoints to create/complete multipart uploads.
- Store upload manifest in Postgres and MinIO.
- Benefits: avoids API bottleneck; supports 100 GB files.

### 2) Sharded Parsing & Chunking for Large Files
- Split PDFs by page ranges (25‑50 pages) and publish shard jobs.
- Each shard parses and chunks independently, writes manifest to MinIO.
- Consolidation step merges manifests before embedding.

### 3) Queue Fan‑out & Idempotency
- Introduce RabbitMQ for fan‑out of shard jobs; keep Redis for idempotency keys.
- Define DLQs for parse and embedding failures.
- Idempotency via Redis `SET NX` keys per job payload.

### 4) Embedding Throughput Controls
- Centralize embedding batcher with token budget management, exponential backoff, and worker concurrency caps.

### 5) Postgres Hardening
- Create FTS GIN indexes early; add missing indexes on `document` and `cost_event`.
- Introduce PgBouncer for connection pooling.

### 6) Retrieval & Rerank Efficiency
- Cache recent vector/FTS results in Redis.
- Make Jina reranker optional per request; add budget control.

### 7) Observability & DX (Lightweight)
- Add Prometheus counters (API RPS, worker throughput, queue depths).
- Add request IDs/correlation IDs across API → worker → DB.
- Keep logs JSON‑structured; add sampling for verbose ETL logs.

## Scale Plan

When Growth limits are reached, move to durable, highly parallel orchestration and partitioned data at scale.

### 1) Durable Orchestration (Temporal)
- Model ETL as a Temporal workflow with Activities and Child Workflows.
- Built‑in retries, backoff, saga pattern for compensations.

### 2) Data Partitioning & Lifecycle
- Partition `langchain_pg_embedding` and related tables by `collection_id`/`tenant_id` and/or time (monthly) using `pg_partman`.
- Add `tenant_id` to metadata, enforce RLS.
- Implement TTL on `cost_event`, conversation `message` pruning.

### 3) Query Path at Scale
- Add pgBouncer + read replicas; route reads to replicas.
- Separate FTS materialization table with `tsvector` column and triggers.
- Tune PG `work_mem`, GIN `fastupdate`, autovacuum.
- Implement in‑process LRU cache for reranked results; per‑tenant QPS limits.

### 4) Uploads & Storage
- Standardize presigned multipart; enforce server‑side encryption in MinIO.
- Cross‑region replication; lifecycle policies for stale artifacts.

### 5) Cost & Quotas
- Expand `cost_event` with tenant, route groups, budgets.
- Enforce per‑tenant, per‑model daily/monthly caps.
- Alert on budget burn via webhook/email.

### 6) Security & Compliance
- Secrets in a vault (HashiCorp Vault or cloud secret manager). Rotate keys.
- TLS everywhere (API, MinIO, Postgres). VPC isolation.
- GDPR: delete/retention, audit logs for conversation/message, avoid storing PII.

## Scale Plan with Milvus, Kafka, and High‑Load Architecture

**Goal**: Extend the existing scaling roadmap to support multi‑tenant, high‑throughput workloads (tens of thousands of daily active users, millions of monthly active users) by introducing a dedicated vector store (Milvus) and an event‑streaming backbone (Kafka) while retaining the proven MinIO + PostgreSQL metadata layer.

### Load Estimates (Target MVP‑Growth‑Scale)
- **DAU**: 10 k – 20 k
- **MAU**: 100 k – 200 k
- **Concurrent Uploads**: 5 k × ~1 GB (peak), occasional 100 GB bulk ingest.
- **Query Throughput**: 2 k QPS (average), spikes to 5 k QPS during batch analytics.
- **Embedding Rate**: 1 M tokens/min across all tenants (OpenAI budget‑controlled).

### Core Components
| Component | Role | Reasoning |
|-----------|------|-----------|
| **Milvus (v2.x) Cluster** | Distributed vector store for embeddings | Provides sub‑ms ANN search, horizontal sharding, replication, supports IVF‑PQ / HNSW indexes. Replaces pgvector for scale. |
| **Kafka (Confluent‑compatible)** | Event streaming for ETL stages, job dispatch, audit logs | Guarantees ordered, durable streams; supports high‑throughput fan‑out to multiple consumer groups (parsers, chunkers, embedder). |
| **MinIO** | Object storage for raw PDFs, page shards, chunk manifests | Retains existing S3‑compatible storage, easy to scale with erasure coding. |
| **PostgreSQL + pgvector (metadata only)** | Stores document metadata, cost ledger, conversation history, tenant config | Relational guarantees, ACID for business data; vector data moved to Milvus. |
| **Redis** | Caching, rate‑limit tokens, idempotency keys | Low‑latency lookups for hot query results and per‑tenant QPS caps. |
| **FastAPI Stateless Services** | API gateway, query endpoint, upload orchestration | Deploy behind an L7 load balancer; horizontal scaling via container replicas. |
| **Celery (optional) → Kafka Consumers** | Background workers for parsing/chunking/embedding | Transition plan: replace Celery queues with Kafka consumer groups for better throughput. |

### Data Flow (High‑Level)
1. **Upload** – Client obtains presigned MinIO multipart URLs (unchanged). After completion, a *"document_uploaded"* event is published to Kafka `documents.upload` topic.
2. **Ingestion Workers** – Kafka consumers read the event, fetch the object, and emit *"parse"* events per shard.
3. **Parsing & Chunking** – Workers process page‑range shards, produce *"chunk_ready"* events with JSON manifest stored in MinIO.
4. **Embedding** – Embedding service consumes `chunk_ready`, calls OpenAI, writes embeddings to Milvus, and stores chunk metadata in PostgreSQL.
5. **Index Refresh** – Milvus automatically updates its ANN indexes; optional manual `flush` for bulk loads.
6. **Query** – API queries Milvus for nearest vectors, falls back to PostgreSQL FTS, optionally reranks via Jina. Results are cached in Redis.

### Trade‑offs & Considerations
- **Milvus vs pgvector**: Milvus offers orders‑of‑magnitude faster ANN at scale and native sharding, but introduces operational complexity (cluster management, backup strategy). pgvector is simple and sufficient for < 10 M vectors.
- **Kafka vs RabbitMQ**: Kafka provides higher throughput and durable log‑based replay, ideal for replaying failed ETL stages. RabbitMQ is easier to set up for low‑volume workloads.
- **Consistency**: Vector search is eventually consistent after Milvus flush; critical queries may require a short read‑after‑write delay.
- **Cost**: Running Milvus nodes and Kafka brokers increases infrastructure cost; however, reduced query latency can lower compute spend on API servers.
- **Migration Path**: Dual‑write embeddings to both pgvector and Milvus during rollout; once Milvus is validated, deprecate pgvector usage for search.

### How‑to‑Migrate
1. **Provision Milvus Cluster** (3‑node replica set, enable Raft consensus).
2. **Deploy Kafka** (3‑broker cluster, enable topic replication factor 3).
3. **Update ETL Pipeline** to publish/consume Kafka events (use `confluent‑kafka` Python client).
4. **Backfill Existing Embeddings**: Stream existing `langchain_pg_embedding` rows into Milvus via a one‑off job.
5. **Feature Flag**: Switch query service to Milvus after health checks; monitor latency and result quality.
6. **Retire pgvector** once confidence is high.

### Operational Practices
- **Monitoring**: Prometheus exporters for Milvus (`milvus_exporter`), Kafka (`kafka_exporter`), and Redis. Alert on consumer lag > 5 s, Milvus query latency > 200 ms.
- **Backup**: Periodic snapshots of Milvus collections (via `milvusctl backup`) and MinIO bucket versioning.
- **Security**: TLS for all inter‑service traffic, IAM policies for MinIO, SASL/SCRAM for Kafka, and secret management via Vault.
- **Scaling**: Add Milvus shards when vector count > 50 M; increase Kafka partitions per topic to match consumer parallelism.

## Executive Summary
The scaling plan provides a clear, phased roadmap to evolve the RAG ETL system from a lean MVP to a high‑throughput, multi‑tenant production service. By progressively introducing presigned multipart uploads, sharded processing, RabbitMQ/Kafka fan‑out, Milvus vector storage, and robust operational practices, the architecture can support tens of thousands of daily active users and multi‑TB corpora while maintaining low latency, reliability, and cost control.

## Architectural Principles
- **Modularity** – Each component (storage, queue, compute, vector store) is independent and replaceable.
- **Stateless Services** – FastAPI and workers are stateless, enabling horizontal scaling behind a load balancer.
- **Observability‑First** – Metrics, logs, and traces are emitted from the start to facilitate debugging and capacity planning.
- **Security by Design** – TLS, IAM, and secret management are applied to all inter‑service communication.
- **Cost‑Effective Scaling** – Start with a single Postgres + pgvector; introduce Milvus/Kafka only when load justifies the added operational overhead.

## Risk Management & Mitigation
| Risk | Impact | Mitigation |
|------|--------|------------|
| **Data loss during migration** | Loss of embeddings or metadata | Perform incremental dual‑write migration, validate checksums, and keep backups until cut‑over.
| **Service outage due to queue backlog** | Increased latency, failed uploads | Auto‑scale RabbitMQ/Kafka consumers, implement back‑pressure via Redis token bucket.
| **OpenAI rate‑limit exhaustion** | Embedding pipeline stalls | Token‑budget throttling, exponential backoff, and fallback to cached embeddings.
| **Security breach** | Unauthorized data access | Enforce TLS, rotate secrets via Vault, enable audit logging.
| **Operational complexity with Milvus/Kafka** | Higher maintenance overhead | Use managed services where possible, automate health checks, and maintain clear runbooks.

## Migration Strategy
The migration follows a **blue‑green** approach with feature flags:
1. **Deploy Milvus & Kafka** alongside existing stack.
2. **Dual‑write**: Write new embeddings to both pgvector and Milvus; keep both indexes in sync.
3. **Validate**: Run side‑by‑side queries, compare latency and relevance.
4. **Cut‑over**: Switch the query service to Milvus via a config flag.
5. **Decommission**: After a monitoring period, retire pgvector‑based retrieval and remove related code.

## Operational Excellence
- **Monitoring**: Prometheus exporters for Milvus, Kafka, Redis, PostgreSQL; Grafana dashboards for latency, queue lag, CPU/memory.
- **Alerting**: CPU > 80 % for >5 min, Kafka consumer lag > 30 s, Milvus query latency > 200 ms, upload error rate > 1 %.
- **Backup & DR**: Daily snapshots of Milvus collections (`milvusctl backup`), MinIO versioning, PostgreSQL pg_dump; restore procedures documented.
- **Security**: TLS for all traffic, IAM policies on MinIO, SASL/SCRAM for Kafka, Vault‑managed secrets for DB credentials and OpenAI keys.
- **Incident Response**: Runbooks for upload failures, embedding bottlenecks, and query spikes; include run‑book checklists and escalation paths.

## Cost Estimation
- **Milvus Cluster**: 3 nodes (CPU‑optimized, 8 vCPU, 32 GB RAM each) – approx. $0.10 /CPU‑hour → ~$720 /month.
- **Kafka Cluster**: 3 brokers (4 vCPU, 16 GB RAM) – $0.08 /CPU‑hour → ~$460 /month.
- **MinIO**: 5 TB storage (erasure‑coded) – $0.02 /GB‑month → $100 /month.
- **PostgreSQL** (primary + 1 replica): 2 vCPU, 8 GB RAM each – $0.05 /CPU‑hour → ~$150 /month.
- **Redis** (cache): 2 vCPU, 4 GB RAM – $0.04 /CPU‑hour → ~$80 /month.
- **Compute (FastAPI workers)**: 8 vCPU, 16 GB RAM (autoscaling) – average $0.07 /CPU‑hour → ~$400 /month.
- **OpenAI Embedding Costs**: $0.0004 / 1 K tokens → ~ $200 /month for 1 M tokens/min.
- **Total Approximate Monthly Cost**: **~$2,110** (excluding network egress and backup storage).

## High‑Load Methods
- **Autoscaling**: Use Kubernetes Horizontal Pod Autoscaler (HPA) on API and worker deployments based on CPU and custom QPS metrics.
- **Load Testing**: Generate realistic traffic with Locust or k6 targeting 5 k concurrent uploads and 2 k QPS query load; monitor latency, error rates, and queue depth.
- **Rate Limiting**: Enforce per‑tenant QPS caps via Redis token bucket; burst protection for upload endpoints.
- **Capacity Planning**: DAU 20 k → estimate 2 M daily queries; provision Milvus shards accordingly (≈50 M vectors per shard).
- **Back‑pressure**: Kafka consumer lag thresholds trigger temporary upload throttling or query degradation (disable rerank).

## Trade‑offs Summary
| Aspect | Milvus vs pgvector | Kafka vs RabbitMQ | Operational Complexity |
|--------|-------------------|------------------|------------------------|
| **Performance** | Sub‑ms ANN search, horizontal scaling | Higher throughput, durable log replay | Higher (cluster ops, monitoring) |
| **Cost** | Higher node cost, storage for indexes | Slightly higher broker cost | Moderate (needs Zookeeper/KRaft) |
| **Maturity** | Actively developed, cloud‑native | Mature, simpler for small scale | Kafka requires more expertise |
| **Flexibility** | Supports multiple index types, dynamic scaling | Topic partitioning for fan‑out | More moving parts, need schema evolution |

## Migration Steps
1. **Provision Milvus** (3‑node Raft cluster) and deploy via Helm.
2. **Deploy Kafka** (3‑broker) with replication factor 3.
3. **Update ETL**: replace pgvector write calls with Milvus `insert` API; keep dual‑write during rollout.
4. **Backfill**: Stream existing `langchain_pg_embedding` rows into Milvus using a one‑off job.
5. **Switch Retrieval**: Feature‑flag to query Milvus; validate latency and relevance.
6. **Deprecate pgvector**: After stability, remove pgvector usage and related migrations.
7. **Migrate Event Handling**: Publish new `document_uploaded` events to Kafka; gradually shift consumers from RabbitMQ.
8. **Cleanup**: Decommission RabbitMQ brokers and related queues.

## Operational Runbooks
- **Monitoring**: Prometheus exporters for Milvus, Kafka, Redis, PostgreSQL; Grafana dashboards for latency, queue lag, CPU/memory.
- **Alerting**: CPU > 80 % for >5 min, Kafka consumer lag > 30 s, Milvus query latency > 200 ms, upload error rate > 1 %.
- **Backup**: Daily snapshots of Milvus collections (`milvusctl backup`), MinIO versioning, PostgreSQL pg_dump.
- **Disaster Recovery**: Restore Milvus from latest snapshot, replay Kafka logs to recover missed events, re‑hydrate PostgreSQL from backup.
- **Security**: TLS for all inter‑service traffic, IAM policies on MinIO buckets, SASL/SCRAM for Kafka, Vault‑managed secrets for DB credentials and OpenAI keys.
- **Incident Response**: Runbooks for upload failures (check MinIO health), embedding bottlenecks (rate‑limit OpenAI), and query spikes (scale workers, enable cache warm‑up).

## Summary
The scaling roadmap provides a comprehensive, phased approach to evolve the RAG ETL system from a lean MVP to a high‑throughput, multi‑tenant production service. By progressively introducing presigned multipart uploads, sharded parsing, RabbitMQ/Kafka fan‑out, Milvus vector storage, and robust operational practices, the architecture can handle tens of thousands of daily active users and multi‑TB corpora while maintaining low latency, reliability, and cost control. The plan balances performance gains with operational complexity, offering clear migration steps, trade‑off analysis, and runbooks to ensure smooth transitions and ongoing stability.

## References
- Upload → `api/features/documents/router.py`, controller/service, and `workers/tasks.py`.
- ETL orchestrator → `etl/pipeline.py`.
- Embeddings → `etl/embed.py`.
- Indexing → `etl/index.py`.
- Hybrid retrieval → `rag/retrievers/hybrid_retriever.py`.
- Reranker → `rag/rankers/reranker.py`.
- Self‑RAG Lite → `rag/pipeline/self_check.py`.
- Cost ledger → `infra/costs/recorder.py`, `infra/costs/pricing.py`.
