#!/usr/bin/env python3
import os
import json
import asyncio
import argparse
import time
from typing import List, Dict, Any, Tuple

from openai import AsyncOpenAI
from openai import APIConnectionError, APIStatusError, RateLimitError
from dotenv import load_dotenv

load_dotenv(".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

DEFAULT_WINDOW_SIZE = 4
DEFAULT_STRIDE = 3
DEFAULT_CONCURRENCY = 5
DEFAULT_REQUEST_TIMEOUT = 60.0
MAX_RETRIES = 5


def load_out_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("out.json must contain a list of objects")
    for i, item in enumerate(data):
        if "speech" not in item:
            raise ValueError(f"Item {i} missing 'speech' field")
    return data


def build_windows(
    items: List[Dict[str, Any]],
    window_size: int,
    stride: int
) -> List[Tuple[int, int, List[Dict[str, Any]]]]:
    windows: List[Tuple[int, int, List[Dict[str, Any]]]] = []
    n = len(items)
    i = 0
    while i + window_size <= n:
        j = i + window_size
        windows.append((i, j, items[i:j]))
        i += stride
    return windows


def build_window_prompt(segment: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, item in enumerate(segment, 1):
        spk = item.get("speaker", "Unknown")
        speech = (item.get("speech") or "").strip()
        lines.append(f"{idx}. {spk}: {speech}")
    conversation = "\n\n".join(lines)
    prompt = (
        "You are a precise financial earnings call summarizer. "
        "Summarize the following excerpts into a concise, factual, and neutral summary with:\n"
        "- Key business themes and product highlights\n"
        "- Notable metrics or quantified statements\n"
        "- Guidance/outlook or macro commentary\n"
        "- Any risks or headwinds mentioned\n\n"
        "Avoid speculation. Keep to 6-10 bullet points. Use short bullets. No marketing fluff.\n\n"
        f"Excerpts:\n{conversation}\n\n"
        "Return only the bullet list."
    )
    return prompt


def build_incremental_prompt(prefix: str, segment: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, item in enumerate(segment, 1):
        spk = item.get("speaker", "Unknown")
        speech = (item.get("speech") or "").strip()
        lines.append(f"{idx}. {spk}: {speech}")
    conversation = "\n\n".join(lines)
    prefix_block = prefix.strip()
    if prefix_block:
        prefix_text = prefix_block
    else:
        prefix_text = "(none)"
    prompt = (
        "You are an online meeting summarizer following an incremental policy. "
        "Given the current transcript window and the EXISTING SUMMARY PREFIX, you must either: \n"
        "- Append ONLY the genuinely new bullet points that should be added beyond the prefix, OR\n"
        "- If nothing new should be added, respond EXACTLY with: NO-UPDATE\n\n"
        "Rules:\n"
        "- Do NOT repeat or rewrite the prefix.\n"
        "- Keep output to short, neutral bullets.\n"
        "- No speculation; only content supported by the window.\n"
        "- If adding bullets, return ONLY the new bullets (no header, no prefix echo).\n"
        "- If no new content is warranted, return ONLY: NO-UPDATE\n\n"
        f"Existing Summary Prefix:\n{prefix_text}\n\n"
        f"Current Transcript Window:\n{conversation}\n\n"
        "Your response:" 
    )
    return prompt


def build_reduce_prompt(window_summaries: List[str]) -> str:
    joined = "\n\n---\n\n".join(window_summaries)
    prompt = (
        "You are a precise financial earnings call summarizer. "
        "You will receive multiple partial summaries of adjacent transcript windows. "
        "Produce a single, deduplicated, logically ordered executive summary with:\n"
        "- Headline themes (3-5 bullets)\n"
        "- Business highlights (5-8 bullets)\n"
        "- Quant/metrics (3-6 bullets)\n"
        "- Outlook/risks (3-6 bullets)\n\n"
        "Rules:\n"
        "- No contradictions; if conflicts exist, pick the most conservative consistent statements.\n"
        "- Remove redundancy; keep each bullet atomic and crisp.\n"
        "- Maintain neutral tone; no speculation.\n"
        "- Target ~200-350 words total.\n\n"
        f"Partial Summaries:\n{joined}\n\n"
        "Return only the final structured bullet list."
    )
    return prompt


async def call_openai_chat(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")

    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()},
                ],
                timeout=timeout,
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except (RateLimitError, APIConnectionError, APIStatusError):
            if attempt == MAX_RETRIES:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 20.0)

    raise RuntimeError("Failed to get response from OpenAI after retries")


async def summarize_window(
    client: AsyncOpenAI,
    model: str,
    segment: List[Dict[str, Any]],
) -> str:
    system_prompt = "You are a rigorous assistant for summarizing earnings call transcripts accurately."
    user_prompt = build_window_prompt(segment)
    return await call_openai_chat(client, model, system_prompt, user_prompt)


async def run_sliding_prefix_policy(
    client: AsyncOpenAI,
    model: str,
    items: List[Dict[str, Any]],
    window_size: int,
    save_steps_path: str,
) -> str:
    """
    Implements the Sliding Window policy with prefix forcing, approximating
    the paper's approach. Maintains a fixed-size window over turns; at each
    step, asks the model to append ONLY new bullets beyond the prefix, or
    return NO-UPDATE. If new bullets appear, append to prefix.

    Saves an audit log of steps to save_steps_path (JSONL-like list).
    """
    n = len(items)
    if n == 0:
        return ""

    log_steps: List[Dict[str, Any]] = []
    prefix = ""

    # Initialize prefix with first window
    start = 0
    end = min(window_size, n)
    init_window = items[start:end]
    init_summary = await summarize_window(client, model, init_window)
    prefix = init_summary.strip()
    log_steps.append({
        "type": "init",
        "window_start": start,
        "window_end": end,
        "added": init_summary.strip(),
        "prefix_len": len(prefix),
    })

    # Slide by one turn at a time
    for i in range(end, n):
        window_start = max(0, i - window_size + 1)
        window_end = i + 1
        segment = items[window_start:window_end]
        user_prompt = build_incremental_prompt(prefix, segment)
        system_prompt = (
            "You are a rigorous assistant for incremental meeting summarization."
        )
        resp = await call_openai_chat(client, model, system_prompt, user_prompt)
        normalized = resp.strip()
        if normalized.upper() == "NO-UPDATE" or normalized == "":
            log_steps.append({
                "type": "no_update",
                "window_start": window_start,
                "window_end": window_end,
            })
            continue
        # Append new bullets to prefix
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        prefix += normalized
        log_steps.append({
            "type": "append",
            "window_start": window_start,
            "window_end": window_end,
            "added": normalized,
            "prefix_len": len(prefix),
        })

    # Persist step log
    try:
        with open(save_steps_path, "w", encoding="utf-8") as f:
            json.dump(log_steps, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return prefix.strip()


async def reduce_summaries(
    client: AsyncOpenAI,
    model: str,
    summaries: List[str],
) -> str:
    system_prompt = "You are a rigorous assistant for synthesizing multiple summaries into one coherent executive summary."
    user_prompt = build_reduce_prompt(summaries)
    return await call_openai_chat(client, model, system_prompt, user_prompt)


async def bounded_gather(tasks, limit: int):
    semaphore = asyncio.Semaphore(limit)

    async def sem_task(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_task(t) for t in tasks))


async def main():
    parser = argparse.ArgumentParser(description="Sliding-window transcript summarizer then reducer.")
    parser.add_argument("--input", default="out.json", help="Path to input JSON file.")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, help="Window size (e.g., 3).")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Stride (e.g., 1).")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Max concurrent OpenAI calls.")
    parser.add_argument("--model", default=OPENAI_MODEL, help="OpenAI chat model (default from env OPENAI_MODEL).")
    parser.add_argument("--save-intermediate", default="summaries.json", help="Where to save window summaries.")
    parser.add_argument("--save-final", default="final_summary.txt", help="Where to save reduced final summary.")
    parser.add_argument(
        "--policy",
        choices=["sliding_prefix", "independent_reduce"],
        default="sliding_prefix",
        help="Summarization policy to use.",
    )
    parser.add_argument(
        "--save-steps",
        default="sliding_steps.json",
        help="For sliding_prefix policy: path to save step log.",
    )
    args = parser.parse_args()

    items = load_out_json(args.input)
    print(f"Loaded {len(items)} transcript entries.")
    t0 = time.time()

    client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    if args.policy == "sliding_prefix":
        final_summary = await run_sliding_prefix_policy(
            client=client,
            model=args.model,
            items=items,
            window_size=args.window_size,
            save_steps_path=args.save_steps,
        )
        with open(args.save_final, "w", encoding="utf-8") as f:
            f.write(final_summary.strip() + "\n")
        print(f"[sliding_prefix] Final summary written to {args.save_final}")
        print(f"Step log written to {args.save_steps}")
    else:
        # independent_reduce policy (original behavior)
        windows = build_windows(items, args.window_size, args.stride)
        if not windows:
            print("No windows generated; check window size and stride.")
            return
        print(f"Generated {len(windows)} windows (size={args.window_size}, stride={args.stride}).")
        tasks = [summarize_window(client, args.model, segment) for _, _, segment in windows]
        window_summaries = await bounded_gather(tasks, args.concurrency)

        inter_payload = []
        for (start, end, _), summary in zip(windows, window_summaries):
            inter_payload.append({
                "window_start_index": start,
                "window_end_index": end,  # exclusive
                "summary": summary,
            })
        with open(args.save_intermediate, "w", encoding="utf-8") as f:
            json.dump(inter_payload, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(inter_payload)} window summaries -> {args.save_intermediate}")

        final_summary = await reduce_summaries(client, args.model, [w["summary"] for w in inter_payload])
        with open(args.save_final, "w", encoding="utf-8") as f:
            f.write(final_summary.strip() + "\n")
        print(f"[independent_reduce] Final summary written to {args.save_final}")

    print(f"Done in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    asyncio.run(main())