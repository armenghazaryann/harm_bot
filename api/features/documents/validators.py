"""Validators for document operations."""
import logging
from typing import Optional

from fastapi import UploadFile, HTTPException

from api.features.documents.entities import DocumentType

logger = logging.getLogger(__name__)


class DocumentValidator:
    """Validates uploaded documents for processing."""

    # Supported MIME types
    SUPPORTED_MIME_TYPES = {
        "application/pdf",
        "text/csv",
        "text/plain",
        "text/markdown",
        "application/json",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    }

    # Maximum file size (100MB for MVP)
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB in bytes

    @classmethod
    async def validate_upload(cls, file: UploadFile) -> bytes:
        """
        Validate an uploaded file and return its content.

        Args:
            file: The uploaded file

        Returns:
            The file content as bytes

        Raises:
            HTTPException: If validation fails
        """
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Validate file extension and MIME type
        cls._validate_file_type(file)

        # Read file content
        content = await file.read()

        # Validate file size
        cls._validate_file_size(content)

        # Reset file pointer for potential downstream use
        await file.seek(0)

        return content

    @classmethod
    def _validate_file_type(cls, file: UploadFile) -> None:
        """Validate file type based on extension and MIME type."""
        filename = file.filename.lower() if file.filename else ""

        # Check file extension
        if not any(
            filename.endswith(ext)
            for ext in [".pdf", ".csv", ".txt", ".md", ".json", ".docx", ".doc"]
        ):
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {filename}"
            )

        # Check MIME type
        if file.content_type and file.content_type not in cls.SUPPORTED_MIME_TYPES:
            # Allow some common variations
            if file.content_type.startswith("text/") and any(
                filename.endswith(ext) for ext in [".txt", ".csv", ".md"]
            ):
                pass  # Allow text files with various subtypes
            else:
                logger.warning(
                    f"Potentially unsupported MIME type: {file.content_type} for file: {filename}"
                )

    @classmethod
    def _validate_file_size(cls, content: bytes) -> None:
        """Validate file size."""
        if len(content) > cls.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {cls.MAX_FILE_SIZE // (1024 * 1024)}MB",
            )

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file provided")

    @classmethod
    def determine_document_type(
        cls, filename: str, content_type: Optional[str]
    ) -> DocumentType:
        """
        Determine document type based on filename containing 'transcript'.

        Args:
            filename: The filename
            content_type: The MIME content type (ignored for MVP)

        Returns:
            'transcript' if filename contains 'transcript', otherwise 'unknown'
        """
        filename_lower = filename.lower()

        # For MVP: only support TRANSCRIPT type
        if "transcript" in filename_lower:
            return DocumentType.TRANSCRIPT

        return DocumentType.TRANSCRIPT
