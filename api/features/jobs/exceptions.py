"""Exceptions for the Jobs feature."""
from typing import Any, Dict, Optional

from api.shared.exceptions import RAGETLException


class JobException(RAGETLException):
    """Base exception for job operations."""
    pass


class JobNotFoundError(JobException):
    """Raised when a job is not found."""
    
    def __init__(self, job_id: str):
        message = f"Job with ID '{job_id}' not found"
        super().__init__(message, "JOB_NOT_FOUND", {"job_id": job_id})


class JobExecutionError(JobException):
    """Raised when job execution fails."""
    
    def __init__(self, job_id: str, message: str, details: Optional[Dict[str, Any]] = None):
        full_message = f"Job '{job_id}' execution failed: {message}"
        error_details = {"job_id": job_id}
        if details:
            error_details.update(details)
        super().__init__(full_message, "JOB_EXECUTION_ERROR", error_details)


class JobValidationError(JobException):
    """Raised when job validation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "JOB_VALIDATION_ERROR", details)


class JobInvalidStateError(JobException):
    """Raised when job is in an invalid state for the requested operation."""
    
    def __init__(self, job_id: str, current_state: str, required_state: str):
        message = f"Job '{job_id}' is in state '{current_state}', but '{required_state}' is required"
        super().__init__(
            message, 
            "JOB_INVALID_STATE", 
            {
                "job_id": job_id,
                "current_state": current_state,
                "required_state": required_state
            }
        )


class JobCancellationError(JobException):
    """Raised when job cancellation fails."""
    
    def __init__(self, job_id: str, reason: str):
        message = f"Failed to cancel job '{job_id}': {reason}"
        super().__init__(message, "JOB_CANCELLATION_ERROR", {"job_id": job_id, "reason": reason})


class JobQueueError(JobException):
    """Raised when job queue operations fail."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "JOB_QUEUE_ERROR", details)


class EvaluationError(JobException):
    """Raised when evaluation operations fail."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "EVALUATION_ERROR", details)


class JobTimeoutError(JobException):
    """Raised when job execution times out."""
    
    def __init__(self, job_id: str, timeout_seconds: int):
        message = f"Job '{job_id}' timed out after {timeout_seconds} seconds"
        super().__init__(
            message, 
            "JOB_TIMEOUT", 
            {
                "job_id": job_id,
                "timeout_seconds": timeout_seconds
            }
        )


class JobResourceError(JobException):
    """Raised when job cannot acquire required resources."""
    
    def __init__(self, job_id: str, resource: str, message: str):
        full_message = f"Job '{job_id}' failed to acquire resource '{resource}': {message}"
        super().__init__(
            full_message, 
            "JOB_RESOURCE_ERROR", 
            {
                "job_id": job_id,
                "resource": resource,
                "reason": message
            }
        )
