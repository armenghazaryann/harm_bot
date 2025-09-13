"""Chain-of-Verification prompts for Self-RAG Lite answer validation.

Implements a lightweight verification process to check answer quality and
provide confidence scoring.
"""

VERIFICATION_PROMPT_TEMPLATE = """You are a verification assistant. Your task is to check if the given answer
is well-supported by the provided evidence and identify any potential issues.

Evidence:
{evidence}

Question: {question}

Answer to verify: {answer}

Please verify this answer by:
1. Checking if the answer is supported by the evidence
2. Identifying any claims that lack evidence support
3. Noting any contradictions or inconsistencies
4. Providing a confidence score (0-100)

Respond in JSON format:
{
  "is_supported": true/false,
  "unsupported_claims": ["claim1", "claim2"],
  "contradictions": ["contradiction1"],
  "confidence_score": 85,
  "verification_notes": "Brief explanation of the verification"
}"""

REVISION_PROMPT_TEMPLATE = """Based on the verification feedback, please revise the answer to be more accurate
and better supported by the evidence.

Original Answer: {original_answer}

Verification Feedback: {verification_feedback}

Evidence:
{evidence}

Question: {question}

Provide a revised answer that addresses the verification concerns:"""


def build_verification_prompt(evidence: str, question: str, answer: str) -> str:
    """Build verification prompt for answer validation."""
    return VERIFICATION_PROMPT_TEMPLATE.format(
        evidence=evidence, question=question, answer=answer
    )


def build_revision_prompt(
    original_answer: str, verification_feedback: str, evidence: str, question: str
) -> str:
    """Build revision prompt for answer improvement."""
    return REVISION_PROMPT_TEMPLATE.format(
        original_answer=original_answer,
        verification_feedback=verification_feedback,
        evidence=evidence,
        question=question,
    )
