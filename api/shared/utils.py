"""Common utility functions following DRY and KISS principles."""
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, Any
from uuid import UUID


def calculate_checksum(content: bytes, algorithm: str = "sha256") -> str:
    """Calculate checksum for content using specified algorithm."""
    hasher = hashlib.new(algorithm)
    hasher.update(content)
    return hasher.hexdigest()


def get_file_extension(filename: str) -> str:
    """Get file extension from filename."""
    return Path(filename).suffix.lower()


def get_mime_type(filename: str) -> str:
    """Get MIME type from filename."""
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"


def generate_storage_path(
    filename: str, entity_id: UUID, prefix: str = "documents"
) -> str:
    """Generate storage path for file."""
    return f"{prefix}/{entity_id}/{filename}"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, "_")
    return filename


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to specified length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def split_text_on_boundary(text: str, boundary: str = " ") -> list[str]:
    """Split text on specified boundary."""
    return [part.strip() for part in text.split(boundary) if part.strip()]


def is_valid_uuid(uuid_string: str) -> bool:
    """Check if string is valid UUID."""
    try:
        UUID(uuid_string)
        return True
    except ValueError:
        return False


def format_bytes(size: int) -> str:
    """Format bytes to human readable format."""
    size_float = float(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_float < 1024.0:
            return f"{size_float:.1f} {unit}"
        size_float /= 1024.0
    return f"{size_float:.1f} PB"


def extract_file_metadata(filename: str, content: bytes) -> Dict[str, Any]:
    """Extract metadata from file."""
    return {
        "filename": filename,
        "extension": get_file_extension(filename),
        "mime_type": get_mime_type(filename),
        "size": len(content),
        "size_human": format_bytes(len(content)),
        "checksum": calculate_checksum(content),
    }
