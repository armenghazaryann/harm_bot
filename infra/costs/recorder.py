"""Minimal cost event recorder.

Inserts one row per provider call (OpenAI, Jina) with usage and latency.
No summaries; dashboards deferred to UI layer.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def record_cost_event(
    session: AsyncSession,
    *,
    provider: str,
    model: str,
    route: str,
    request_id: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    unit_price_in_usd: Optional[float] = None,
    unit_price_out_usd: Optional[float] = None,
    cost_in_usd: Optional[float] = None,
    cost_out_usd: Optional[float] = None,
    cost_total_usd: Optional[float] = None,
    latency_ms: Optional[int] = None,
    status: str = "ok",
    correlation_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Insert a cost_event row using raw SQL (works without ORM models).

    Assumes an Alembic migration has created the `cost_event` table with the schema
    described in REFACTOR_PLAN.MD.
    """
    ev_id = str(uuid.uuid7()) if hasattr(uuid, "uuid7") else str(uuid.uuid4())
    ts = int(time.time())

    meta_json = json.dumps(metadata or {}, ensure_ascii=False)

    sql = text(
        """
        INSERT INTO cost_event (
            id, ts, provider, model, route, request_id, correlation_id,
            prompt_tokens, completion_tokens, total_tokens,
            unit_price_in_usd, unit_price_out_usd,
            cost_in_usd, cost_out_usd, cost_total_usd,
            latency_ms, status, metadata
        ) VALUES (
            :id, to_timestamp(:ts), :provider, :model, :route, :request_id, :correlation_id,
            :prompt_tokens, :completion_tokens, :total_tokens,
            :unit_price_in_usd, :unit_price_out_usd,
            :cost_in_usd, :cost_out_usd, :cost_total_usd,
            :latency_ms, :status, :metadata
        )
        """
    )

    await session.execute(
        sql,
        {
            "id": ev_id,
            "ts": ts,
            "provider": provider,
            "model": model,
            "route": route,
            "request_id": request_id,
            "correlation_id": correlation_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "unit_price_in_usd": unit_price_in_usd,
            "unit_price_out_usd": unit_price_out_usd,
            "cost_in_usd": cost_in_usd,
            "cost_out_usd": cost_out_usd,
            "cost_total_usd": cost_total_usd,
            "latency_ms": latency_ms,
            "status": status,
            "metadata": meta_json,
        },
    )
    # Caller is responsible for committing the transaction.
