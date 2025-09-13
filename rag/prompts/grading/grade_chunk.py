"""Grading prompt for Self-RAG Lite (fast mode).

A compact JSON-only instruction to score relevance, supportiveness, usefulness.
"""
from __future__ import annotations


def build_grade_prompt(*, question: str, chunk: str) -> str:
    return (
        "You are grading a document chunk for answering a question.\n"
        'Respond ONLY with compact JSON: {"relevance": n, "supportiveness": n, "usefulness": n}.\n'
        "Each score in [0..1].\n"
        f"Question: {question}\nChunk: {chunk}"
    )
