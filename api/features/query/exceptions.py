"""Exceptions for the Query feature."""
from typing import Any, Dict, List, Optional

from api.shared.exceptions import RAGETLException


class QueryException(RAGETLException):
    """Base exception for query operations."""

    pass


class QueryValidationError(QueryException):
    """Raised when query validation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "QUERY_VALIDATION_ERROR", details)


class SearchError(QueryException):
    """Raised when search operations fail."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "SEARCH_ERROR", details)


class EmbeddingError(QueryException):
    """Raised when embedding operations fail."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "EMBEDDING_ERROR", details)


class AnswerGenerationError(QueryException):
    """Raised when answer generation fails."""

    def __init__(
        self, message: str, model: str, details: Optional[Dict[str, Any]] = None
    ):
        full_message = f"Answer generation failed with model '{model}': {message}"
        error_details = {"model": model}
        if details:
            error_details.update(details)
        super().__init__(full_message, "ANSWER_GENERATION_ERROR", error_details)


class NoResultsError(QueryException):
    """Raised when no search results are found."""

    def __init__(self, query: str):
        message = f"No results found for query: '{query}'"
        super().__init__(message, "NO_RESULTS_FOUND", {"query": query})


class InsufficientContextError(QueryException):
    """Raised when insufficient context is available for answer generation."""

    def __init__(self, required_chunks: int, available_chunks: int):
        message = f"Insufficient context: required {required_chunks} chunks, but only {available_chunks} available"
        super().__init__(
            message,
            "INSUFFICIENT_CONTEXT",
            {"required_chunks": required_chunks, "available_chunks": available_chunks},
        )


class ModelNotAvailableError(QueryException):
    """Raised when the requested model is not available."""

    def __init__(self, model_name: str, available_models: List[str]):
        message = f"Model '{model_name}' is not available. Available models: {available_models}"
        super().__init__(
            message,
            "MODEL_NOT_AVAILABLE",
            {"requested_model": model_name, "available_models": available_models},
        )


class RateLimitError(QueryException):
    """Raised when rate limits are exceeded."""

    def __init__(self, service: str, retry_after: Optional[int] = None):
        message = f"Rate limit exceeded for {service}"
        if retry_after:
            message += f". Retry after {retry_after} seconds"

        details: Dict[str, Any] = {"service": service}
        if retry_after:
            details["retry_after"] = str(retry_after)

        super().__init__(message, "RATE_LIMIT_EXCEEDED", details)
