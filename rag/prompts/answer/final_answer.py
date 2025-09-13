"""Final answer prompt builder.

Constructs a concise instruction to answer strictly from evidence with citations
in the format [document_name:chunk_id].
"""
from __future__ import annotations

from typing import List


def build_answer_prompt(*, evidence_lines: List[str], question: str) -> str:
    evidence_block = "\n".join(evidence_lines or [])
    # Updated citation format to use human‑readable document names.
    prompt = (
        "Use ONLY the evidence to answer the question. "
        "Every factual sentence MUST include citations like [document_name:chunk_id].\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"Question: {question}\n\n"
        "Final answer with citations:"
    )
    return prompt
