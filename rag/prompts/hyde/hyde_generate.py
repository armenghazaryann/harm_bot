"""HyDE (Hypothetical Document Embeddings) prompt for query expansion.

Generates a hypothetical document that would answer the query, which is then
used for better vector similarity search.
"""

HYDE_PROMPT_TEMPLATE = """You are an expert financial analyst. Given a question about a company's earnings,
write a hypothetical passage that would contain the answer to this question.

The passage should be written in the style of an earnings transcript or financial report.
Focus on specific financial metrics, business segments, and strategic initiatives.

Question: {question}

Hypothetical passage:"""


def build_hyde_prompt(question: str) -> str:
    """Build HyDE prompt for query expansion."""
    return HYDE_PROMPT_TEMPLATE.format(question=question)
