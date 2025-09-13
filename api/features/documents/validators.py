"""Validators for document operations."""
import logging
from typing import Optional

from fastapi import UploadFile, HTTPException

from api.features.documents.entities import DocumentType

logger = logging.getLogger(__name__)


class DocumentValidator:
    # refactored

    # Supported MIME types
    SUPPORTED_MIME_TYPES = {
        "application/pdf",
        "text/plain",
        "text/markdown",
    }

    # Maximum file size (100MB for MVP)
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB in bytes

    @classmethod
    async def validate_upload(cls, file: UploadFile) -> bytes:
        # refactored
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
        # refactored
        filename = file.filename.lower() if file.filename else ""

        # Check file extension
        if not any(
            filename.endswith(ext) for ext in [".pdf", ".txt", ".md", ".docx", ".doc"]
        ):
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {filename}"
            )

        # Check MIME type
        if file.content_type and file.content_type not in cls.SUPPORTED_MIME_TYPES:
            # Allow some common variations
            if file.content_type.startswith("text/") and any(
                filename.endswith(ext) for ext in [".txt", ".md", ".docx", ".doc"]
            ):
                pass  # Allow text files with various subtypes
            else:
                logger.warning(
                    f"Potentially unsupported MIME type: {file.content_type} for file: {filename}"
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported MIME type: {file.content_type} for file: {filename}",
                )

    @classmethod
    def _validate_file_size(cls, content: bytes) -> None:
        # refactored
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
        Determine document type based on filename and simple heuristics.

        Args:
            filename: The filename
            content_type: The MIME content type (ignored for MVP)

        Returns:
            A best-effort classification into DocumentType. Defaults to TRANSCRIPT
            to preserve MVP behavior if unknown.
        """
        # Normalize separators for robust matching
        filename_lower = (filename or "").lower()
        normalized = (
            filename_lower.replace("-", "_").replace(" ", "_").replace("__", "_")
        )

        # Transcript
        if "transcript" in normalized:
            return DocumentType.TRANSCRIPT

        # Earnings Release (handle earnings_release, earnings-release, earnings release)
        if "earnings_release" in normalized or (
            "earnings" in normalized and "release" in normalized
        ):
            return DocumentType.EARNINGS_RELEASE

        # Slide Decks
        if any(
            token in normalized for token in ["slide", "slides", "deck", "slide_deck"]
        ):
            return DocumentType.SLIDE_DECK

        # Press Announcements
        # if any(token in normalized for token in ["press", "announcement", "news_release", "press_release"]):
        #     return DocumentType.GENERAL

        # Default to transcript for MVP compatibility
        return DocumentType.GENERAL
