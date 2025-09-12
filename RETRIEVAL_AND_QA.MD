# Retrieval and Q/A (MVP) — Multimodal, Cloud-Only Models, Finance-Grade

This document defines the Retrieval and Q/A step only, assuming processing and chunking have produced structured artifacts. It is optimized for many small (~1 MB) but related PDFs with transcripts, releases, slides, and press/IR. The design avoids custom local models and uses cloud services for embeddings and LLM/VLM.

Scope (Retrieval + Q/A Only)
- Input:
  - Chunk manifests: `chunks/{doc_id}/chunk_manifest.jsonl`
  - Structured tables: `elements/{doc_id}/tables/{table_id}.json`
  - Page images (for slide/visual queries): `pages/{doc_id}/{page:04}.png`
- Output:
  - Final answer text with strict citations
  - Referenced sources (chunks, tables, slides)
  - Numeric verification report when applicable
- Excludes: processing, chunking, evaluation (see separate docs)

Constraints
- No custom local ML models. Use cloud APIs for:
  - Text embeddings (OpenAI)
  - Multimodal/page/slide embeddings (VoyageAI Multimodal-3)
  - LLM for answer generation (OpenAI GPT-4.x family) and optional VLM for visual reasoning

Goals
- Precision-first answers for financial content (KPIs, guidance, attributions, quarter/segment comparisons).
- Strict citations (doc_type, quarter, page/slide, speaker/table_id) with deterministic grounding.
- Numeric verification using structured table payloads.

---

Embedding Spaces & Indexing (Cloud)
- Text space (for narrative and transcript chunks)
  - Model: OpenAI `text-embedding-3-large`
  - Storage: Postgres + pgvector (HNSW index) — index name `idx_vec_text`
- Table space (for structured financial tables via synopsis text)
  - Model: OpenAI `text-embedding-3-large`
  - Storage: pgvector — index name `idx_vec_table`
  - Note: Each `table` chunk includes `payload_ref` to the full structured table JSON for exact numeric lookup.
- Page/Slide space (for visual slide/pages)
  - Model: VoyageAI `voyage-multimodal-3` (cloud multimodal embeddings)
  - Input: `pages/{doc_id}/{page:04}.png`
  - Storage: pgvector — index name `idx_vec_page`
- Keyword/BM25
  - Postgres FTS over `chunks.text` with GIN index — used for hybrid retrieval and exact keyword matches (e.g., analyst name, speaker, KPI terms).

Index hygiene
- Maintain separate HNSW indexes per space with explicit `model_name` and `model_version` recorded with each vector.
- Re-embed and rebuild affected indexes when changing models; backfill with versioned embeddings in background.

---

Query Router (Heuristics)
- Inspect the query to choose retrieval strategy and filters.
- Signals and routing:
  - Numeric/KPI queries (regex for numbers, currency, YoY/QoQ, EPS, revenue, margin):
    - Prioritize `space=table` candidates (table synopsis) → exact lookup from structured JSON.
  - Attribution/speaker queries (names/roles like “Sundar”, “Ruth Porat”, “CFO”):
    - Prioritize transcript chunks (`prepared_remarks`, `qna`) and filter by `speaker` if present.
  - Slide/visual queries (keywords: “slide”, “figure”, “chart”, “deck”):
    - Use `space=page` candidates; return slide image refs and OCR text.
  - Broad/comparative queries across quarters/segments:
    - Pull more candidates from text + tables; optionally include lightweight per-quarter summaries if available.
- Filters (if provided by user or inferred): `quarter`, `year`, `doc_type`, `speaker`, `segment`, `KPI`.

---

Candidate Generation (Hybrid, Multi-Vector)
- Compute top-K candidates independently:
  - `K_text` from `idx_vec_text`
  - `K_table` from `idx_vec_table`
  - `K_page` from `idx_vec_page` (for visual queries or as fallback)
  - `K_fts` from BM25 FTS
- Union and dedupe by `chunk_id`.
- Score fusion (MVP): linear combination with type-aware boosts:
  - Table priority for numeric queries
  - Transcript priority for speaker queries
  - Page/slide priority for visual queries
- Optional rerank (Growth): cloud reranker (e.g., Cohere Rerank) over the union set.

---

Context Assembly
- Build a compact, typed context set:
  - Narrative/Transcript chunks: include `speaker`/`section_path` preamble and paragraph text.
  - Tables: include the concise synopsis in context; keep a pointer to structured JSON for exact numeric lookup.
  - Slides: include slide title and top bullets; attach `image_ref` for VLM-capable queries.
- Token budget policy (MVP):
  - Reserve caps per type: e.g., 60% text, 25% tables, 15% slides.
  - Prefer candidates with exact quarter matches and entity mentions.

---

Answer Generation (Cloud)
- Primary LLM (text): OpenAI GPT-4.x (e.g., `gpt-4.1` or `gpt-4o-mini` depending on latency/cost)
- Visual Qs (optional): GPT-4o or Gemini 1.5 via image inputs (cloud VLM); only triggered when the router marks a visual query or the user requests slide inspection.
- Prompting guidelines:
  - Always cite sources as `[doc_type, quarter, page/slide, speaker/table_id]`.
  - Use conservative, extractive style; avoid speculation.
  - For numeric answers, report values with units/currency and specify GAAP vs non-GAAP when both are present.

Numeric Verification (Deterministic)
- If the draft answer includes numbers:
  - Parse numeric spans and map them to candidate tables via `payload_ref`.
  - Perform exact lookup with unit/currency normalization; allow configured tolerance (e.g., 0.1% for rounding).
  - On mismatch, return both values and flag a discrepancy with citations.

Failure Handling
- If no candidates pass confidence/filters:
  - Ask for clarification (e.g., missing quarter or segment) or widen the time window.
  - Return top-3 likely sources with brief snippets to help the user refine the query.

---

API Contract (MVP)
- `POST /query`
  - Request:
    ```json
    {
      "question": "What was Q2 2025 Google Cloud revenue?",
      "filters": {"quarter": "2025-Q2", "doc_type": ["release", "slides", "transcript"]},
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
        "chunks": [ {"chunk_id": "...", "score": 0.81} ],
        "tables": [ {"table_id": "tbl_123", "payload_ref": "elements/.../tables/tbl_123.json"} ],
        "slides": [ {"page": 12, "image_ref": "pages/.../0012.png"} ]
      },
      "verification": {
        "status": "verified",
        "details": [{"value": "1.2B", "from": "tbl_123", "tolerance": 0.001}]
      }
    }
    ```

Observability & Metrics
- Retrieval: per-space recall@K, hit rate by query type.
- Q/A: citation coverage, numeric verification pass rate, latency breakdown (embed lookup, rerank, LLM).
- Costs: tokens per query, embedding/LLM/VLM costs.

Security & Compliance
- Never store API keys in code; use env/secret manager.
- PII-safe logging; redact query if requested; TLS in transit.

Limitations & Future Enhancements
- Visual reasoning limited to cloud VLM; no local captioning.
- Reranking optional in MVP; consider cloud rerankers for improved precision.
- Multi-hop reasoning can benefit from a knowledge graph; integrate in a separate KG module when needed.
