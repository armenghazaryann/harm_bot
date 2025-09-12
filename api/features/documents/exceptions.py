"""Exceptions for the Documents feature."""
from typing import Any, Dict, Optional

from api.shared.exceptions import RAGETLException


class DocumentException(RAGETLException):
    """Base exception for document operations."""
    pass


class DocumentNotFoundError(DocumentException):
    """Raised when a document is not found."""
    
    def __init__(self, doc_id: str):
        message = f"Document with ID '{doc_id}' not found"
        super().__init__(message, "DOCUMENT_NOT_FOUND", {"doc_id": doc_id})


class DocumentUploadError(DocumentException):
    """Raised when document upload fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DOCUMENT_UPLOAD_ERROR", details)


class DocumentProcessingError(DocumentException):
    """Raised when document processing fails."""
    
    def __init__(self, doc_id: str, message: str, details: Optional[Dict[str, Any]] = None):
        full_message = f"Processing failed for document '{doc_id}': {message}"
        error_details = {"doc_id": doc_id}
        if details:
            error_details.update(details)
        super().__init__(full_message, "DOCUMENT_PROCESSING_ERROR", error_details)


class DocumentValidationError(DocumentException):
    """Raised when document validation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DOCUMENT_VALIDATION_ERROR", details)


class DocumentStorageError(DocumentException):
    """Raised when document storage operations fail."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DOCUMENT_STORAGE_ERROR", details)


class DocumentAlreadyExistsError(DocumentException):
    """Raised when trying to create a document that already exists."""
    
    def __init__(self, identifier: str, field: str = "checksum"):
        message = f"Document with {field} '{identifier}' already exists"
        super().__init__(message, "DOCUMENT_ALREADY_EXISTS", {field: identifier})


class DocumentInvalidStateError(DocumentException):
    """Raised when document is in an invalid state for the requested operation."""
    
    def __init__(self, doc_id: str, current_state: str, required_state: str):
        message = f"Document '{doc_id}' is in state '{current_state}', but '{required_state}' is required"
        super().__init__(
            message, 
            "DOCUMENT_INVALID_STATE", 
            {
                "doc_id": doc_id,
                "current_state": current_state,
                "required_state": required_state
            }
        )
