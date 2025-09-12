# Evaluation (MVP) — Finance-Grade, Cloud-Only Models

This document defines the evaluation step only: how we measure the quality, reliability, and cost/latency of the Retrieval + Q/A system. It assumes processing and chunking are already complete and avoids custom local ML models. Cloud models may be used for optional semantic judgments.

Scope (Evaluation Only)
- Inputs:
  - Chunk manifests: `chunks/{doc_id}/chunk_manifest.jsonl`
  - Structured tables: `elements/{doc_id}/tables/{table_id}.json`
  - Retrieval + Q/A API (`POST /query`)
  - Evaluation dataset of queries with gold answers and/or references
- Outputs:
  - Per-query and aggregate metrics (JSON + Markdown report)
  - Error analysis artifacts
- Excludes: processing, chunking, retrieval/Q&A implementation (see separate docs)

Constraints
- No custom local ML models.
- Use cloud APIs when semantic judgments are required (e.g., OpenAI GPT-4.x as judge). Prefer deterministic checks when possible (numeric/table verification).

Goals
- Ensure precise, auditable answers for financial content with strict citations.
- Track performance regressions and gate deployments via thresholds.
- Keep evaluation reproducible, versioned, and explainable.

---

Evaluation Dataset
- Composition targets (50–100 queries initially):
  - KPIs & numeric (e.g., Revenue, Operating Income, EPS, margin, FX impacts)
  - Guidance & qualitative attributions (what CFO/CEO said)
  - Segment/quarter comparisons (QoQ, YoY, multi-hop)
  - Slides/visual references (identify correct slide/page and content)
  - Press/IR factual statements (dates, claims)
- Balance across quarters and document types in `kb/`.
- Ground truth creation:
  - Numeric: pull from normalized table JSON; store canonical values with units/currency; include tolerance rules.
  - Quotes/attributions: store verbatim spans from transcripts with page and speaker.
  - Slide references: store slide/page numbers and expected title/bullets.

Dataset format (JSONL)
Each line defines one evaluation query.

```json
{
  "query_id": "eval_0001",
  "question": "What was Q2 2025 Google Cloud revenue?",
  "metadata": {
    "category": "numeric",
    "quarter": "2025-Q2",
    "doc_types": ["release", "slides", "transcript"]
  },
  "gold": {
    "expected_value": 1200000000.0,
    "unit": "USD",
    "currency": "USD",
    "tolerance_rel": 0.001,
    "source_table_ids": ["tbl_123"],
    "citations": [
      {"doc_type": "release", "quarter": "2025-Q2", "page": 5, "table_id": "tbl_123"}
    ]
  }
}
```

---

Metrics
- Retrieval
  - `Recall@K_text`, `Recall@K_table`, `Recall@K_page` — whether at least one relevant chunk appeared in the top-K for each space (uses ground-truth mapping to chunks/tables/slides when available).
  - `HybridHitRate@K` — whether any relevant candidate appeared after union.
- Answering
  - `NumericExact` — exact numeric match after normalization (currency/units) and rounding.
  - `NumericWithinTolerance` — within `tolerance_rel` or `tolerance_abs` when provided.
  - `CitationCoverage` — answer includes at least one citation.
  - `CitationCorrectness` — cited locations (doc_type, quarter, page/slide, table_id) match ground truth.
  - `AttributionAccuracy` — correct speaker/role for attribution questions.
  - `Faithfulness` (optional, cloud judge) — answer is supported by provided context (LLM-as-judge rubric).
- System
  - Latency (p50/p95) per stage: retrieval, rerank (if any), LLM, verification.
  - Cost per query: embeddings, LLM/VLM tokens.

Rubrics
- Faithfulness (cloud judge): ask a constrained LLM prompt to rate if every factual statement is supported by the provided context; return pass/fail + rationale.
- Attribution: require exact match on speaker name and role OR membership in the answer_speakers set for Q&A chunks.

---

Scoring Pipeline (Per Query)
1) Run `POST /query` with the question and any filters specified in the eval record.
2) Capture:
   - Final answer, citations, sources, verification details.
   - Candidate lists per space (text/table/page) for recall calculations.
3) Numeric checks (deterministic):
   - If `expected_value` present, normalize units/currency and compare with answer-extracted numbers; compute exact/tolerance metrics.
4) Citation checks:
   - Compare provided citations to gold citations; allow multiple acceptable sources when listed.
5) Attribution checks:
   - Compare detected speaker(s) against gold attributions.
6) Optional semantic checks (cloud judge):
   - Send context + answer to a rubric-based judge prompt; parse pass/fail.
7) Record per-query metrics and aggregate.

---

Reporting
- JSON output (machine-readable): `eval/results/{run_id}.json`
  - Contains per-query scores, system metrics, and aggregates.
- Markdown summary: `eval/reports/{run_id}.md`
  - KPI table with pass rates and deltas vs baseline.
  - Top error categories and examples with citations.

Example aggregate report (fields)
```json
{
  "run_id": "2025-09-12T12:00:00Z",
  "config": {"models": {"text": "text-embedding-3-large", "page": "voyage-mm-3", "llm": "gpt-4.1"}},
  "totals": {"queries": 100},
  "metrics": {
    "NumericExact": 0.74,
    "NumericWithinTolerance": 0.92,
    "CitationCoverage": 0.98,
    "CitationCorrectness": 0.94,
    "AttributionAccuracy": 0.88,
    "Recall@10_text": 0.93,
    "Recall@10_table": 0.96,
    "Recall@10_page": 0.85,
    "Latency_p95_ms": 2800,
    "Cost_per_query_usd": 0.014
  }
}
```

---

Thresholds & Gating (MVP)
- Gate deploy if any of the following are below threshold:
  - `NumericWithinTolerance` < 0.90
  - `CitationCorrectness` < 0.90
  - `AttributionAccuracy` < 0.85 (for attribution subset)
  - `HybridHitRate@10` < 0.95
- Track latency and cost regressions (>20% increase triggers review).

Versioning & Reproducibility
- Include model names and versions in `config`.
- Store commit hash of the service at the time of the run.
- Version the evaluation dataset; maintain changelog.

---

Automation
- API: `POST /eval/run` kicks off an evaluation run over the dataset; returns `run_id`.
- Scheduler: nightly runs; store artifacts under `eval/results/` and `eval/reports/`.
- Alerting: notify on threshold violations with top failing examples.

Security & Compliance
- Do not log sensitive content; redact if necessary.
- Keep API keys in env/secret manager; TLS for all cloud calls.

Limitations & Future Enhancements
- Faithfulness uses a cloud LLM judge; explore deterministic entailment checks where possible.
- Visual/slide correctness can be improved with image-grounded judges (cloud VLMs) as budget allows.
- Add multi-hop graph-aware metrics after KG integration.
