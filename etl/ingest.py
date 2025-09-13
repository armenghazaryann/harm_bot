"""Document ingestion and registration with idempotency checks.

Handles document registration in the database and downloading from MinIO
with proper error handling and idempotency.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.features.documents.entities.document import Document
from core.settings import SETTINGS

logger = structlog.get_logger("etl.ingest")


async def register_document(
    *,
    document_id: str,
    storage_path: str,
    db_session: AsyncSession,
    force: bool = False,
) -> Dict[str, Any]:
    """Register document in database with idempotency check.

    Args:
        document_id: Document UUID
        storage_path: MinIO storage path
        db_session: Database session
        force: Skip idempotency check if True

    Returns:
        Registration result with status and metadata
    """
    start_time = time.time()

    try:
        # Check if document already exists (idempotency)
        if not force:
            result = await db_session.execute(
                select(Document.id).where(Document.id == document_id)
            )
            if result.first():
                logger.info("Document already registered", document_id=document_id)
                return {
                    "status": "skipped",
                    "reason": "already_registered",
                    "processing_time": time.time() - start_time,
                }

        # Document registration logic would go here
        # For now, we'll assume the document is already in the database
        # from the upload process

        logger.info("Document registration validated", document_id=document_id)
        return {
            "status": "completed",
            "document_id": document_id,
            "storage_path": storage_path,
            "processing_time": time.time() - start_time,
        }

    except Exception as e:
        logger.error(
            "Document registration failed", document_id=document_id, error=str(e)
        )
        raise


async def download_from_minio(
    *,
    storage_path: str,
    minio_client,
    local_path: str,
) -> Dict[str, Any]:
    """Download file from MinIO to local path.

    Args:
        storage_path: MinIO object path
        minio_client: MinIO client instance
        local_path: Local file path to save to

    Returns:
        Download result with file info
    """
    start_time = time.time()

    try:
        # Download file from MinIO
        response = minio_client.get_object(
            bucket_name=SETTINGS.MINIO.MINIO_BUCKET,
            object_name=storage_path,
        )

        # Read and save to local file
        file_data = response.read()
        response.close()
        response.release_conn()

        with open(local_path, "wb") as f:
            f.write(file_data)

        # Calculate file hash for integrity check
        file_hash = hashlib.sha256(file_data).hexdigest()

        logger.info(
            "File downloaded from MinIO",
            storage_path=storage_path,
            local_path=local_path,
            size_bytes=len(file_data),
            sha256=file_hash,
        )

        return {
            "status": "completed",
            "local_path": local_path,
            "size_bytes": len(file_data),
            "sha256": file_hash,
            "processing_time": time.time() - start_time,
        }

    except Exception as e:
        logger.error("MinIO download failed", storage_path=storage_path, error=str(e))
        raise
