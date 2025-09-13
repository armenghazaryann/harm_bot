# Pricing utilities for cost recording
"""Helper functions to calculate monetary cost of LLM usage with 4‑decimal precision
and to invoke :func:`infra.costs.recorder.record_cost_event` with the computed values.

The MVP uses a simple per‑token pricing model. Prices are expressed in USD per token
for prompt (input) and completion (output) tokens. The helper rounds all monetary
values to **four decimal places** to avoid floating‑point noise.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Tuple

from infra.costs.recorder import record_cost_event

# Default per‑token prices (USD). These are typical OpenAI rates for gpt‑4o‑mini.
DEFAULT_UNIT_PRICE_IN_USD: float = 0.000015  # $ per input token
DEFAULT_UNIT_PRICE_OUT_USD: float = 0.00006  # $ per output token


def _round_usd(value: float) -> float:
    """Round a float to 4 decimal places using Decimal for exactness.

    Args:
        value: The raw monetary amount.
    Returns:
        The amount rounded to 4 decimal places.
    """
    d = Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    return float(d)


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    unit_price_in_usd: Optional[float] = None,
    unit_price_out_usd: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Calculate input, output and total cost for an LLM call.

    The function uses the provided per‑token prices or falls back to the defaults.
    All returned values are rounded to four decimal places.
    """
    price_in = (
        unit_price_in_usd
        if unit_price_in_usd is not None
        else DEFAULT_UNIT_PRICE_IN_USD
    )
    price_out = (
        unit_price_out_usd
        if unit_price_out_usd is not None
        else DEFAULT_UNIT_PRICE_OUT_USD
    )

    cost_in = prompt_tokens * price_in
    cost_out = completion_tokens * price_out
    total = cost_in + cost_out

    return _round_usd(cost_in), _round_usd(cost_out), _round_usd(total)


async def record_cost_event_with_pricing(
    session,
    *,
    provider: str,
    model: str,
    route: str,
    request_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: Optional[int] = None,
    status: str = "ok",
    correlation_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Convenience wrapper that calculates costs and records the event.

    This function centralises the pricing logic so callers only need to supply the
    token counts. It computes the monetary values with four‑decimal precision and
    forwards everything to :func:`infra.costs.recorder.record_cost_event`.
    """
    cost_in, cost_out, total = calculate_cost(prompt_tokens, completion_tokens)
    await record_cost_event(
        session,
        provider=provider,
        model=model,
        route=route,
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        unit_price_in_usd=DEFAULT_UNIT_PRICE_IN_USD,
        unit_price_out_usd=DEFAULT_UNIT_PRICE_OUT_USD,
        cost_in_usd=cost_in,
        cost_out_usd=cost_out,
        cost_total_usd=total,
        latency_ms=latency_ms,
        status=status,
        correlation_id=correlation_id,
        metadata=metadata,
    )
