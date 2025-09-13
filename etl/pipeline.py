"""ETL Pipeline: Idempotent PDF processing with manifest tracking.

This replaces the monolithic workers/langchain_processor.py with a modular,
idempotent pipeline that supports four PDF strategies:
- Earnings Transcripts: speaker segmentation, Q&A extraction
- Earnings Releases: table extraction, financial metrics
- Slide Decks: per-slide processing, chart analysis with Vision
- Press/IR Announcements: straightforward text extraction

All steps are idempotent and tracked in a manifest stored in both Postgres and MinIO.
"""
from __future__ import annotations

import json
import time
import uuid
from io import BytesIO
from typing import Any, Dict

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from core.settings import SETTINGS
from etl.ingest import register_document
from etl.parse_pdf import parse_pdf_by_strategy
from etl.chunk import chunk_document
from etl.embed import embed_chunks
from etl.index import index_chunks
from infra.costs.recorder import record_cost_event

logger = structlog.get_logger("etl.pipeline")


class ETLPipeline:
    """Main ETL pipeline orchestrator with idempotent step execution."""

    def __init__(self):
        self.pipeline_id = str(uuid.uuid4())

    async def process_document(
        self,
        *,
        document_id: str,
        storage_path: str,
        document_type: str,
        db_session: AsyncSession,
        minio_client,
        force_reprocess: bool = False,
    ) -> Dict[str, Any]:
        """Execute the full ETL pipeline: ingest → parse → chunk → embed → index.

        Args:
            document_id: UUID of the document
            storage_path: MinIO path to the PDF file
            document_type: One of: transcript, release, slides, press
            db_session: Database session
            minio_client: MinIO client
            force_reprocess: Skip idempotency checks if True

        Returns:
            Processing results with manifest and metrics
        """
        start_time = time.time()
        manifest = {
            "document_id": document_id,
            "pipeline_id": self.pipeline_id,
            "storage_path": storage_path,
            "document_type": document_type,
            "steps": {},
            "created_at": time.time(),
        }

        try:
            # Step 1: Register/validate document
            logger.info("Step 1: Register document", document_id=document_id)
            register_result = await register_document(
                document_id=document_id,
                storage_path=storage_path,
                db_session=db_session,
                force=force_reprocess,
            )
            manifest["steps"]["register"] = register_result

            # Step 2: Parse PDF by strategy
            logger.info("Step 2: Parse PDF", document_type=document_type)
            parse_result = await parse_pdf_by_strategy(
                document_id=document_id,
                storage_path=storage_path,
                document_type=document_type,
                minio_client=minio_client,
                force=force_reprocess,
            )
            manifest["steps"]["parse"] = parse_result

            # Step 3: Chunk document
            logger.info("Step 3: Chunk document")

            # Debug logging to see what's being passed from parsing to chunking
            logger.info(
                "Pipeline debug: parse_result structure",
                document_id=document_id,
                parse_result_keys=list(parse_result.keys()) if parse_result else [],
                content_keys=list(parse_result["content"].keys())
                if parse_result and "content" in parse_result
                else [],
                content_preview=str(parse_result["content"])[:200]
                if parse_result and "content" in parse_result
                else "NO_CONTENT",
            )

            chunk_result = await chunk_document(
                document_id=document_id,
                parsed_content=parse_result["content"],
                document_type=document_type,
                db_session=db_session,
                minio_client=minio_client,
                force=force_reprocess,
            )
            manifest["steps"]["chunk"] = chunk_result

            # Step 4: Embed chunks
            logger.info("Step 4: Embed chunks")
            embed_result = await embed_chunks(
                document_id=document_id,
                chunks=chunk_result["chunks"],
                db_session=db_session,
                force=force_reprocess,
            )
            manifest["steps"]["embed"] = embed_result

            # Step 5: Index chunks
            logger.info("Step 5: Index chunks")
            index_result = await index_chunks(
                document_id=document_id,
                chunks_with_embeddings=embed_result["chunks_with_embeddings"],
                db_session=db_session,
                force=force_reprocess,
            )
            manifest["steps"]["index"] = index_result

            # Store manifest in MinIO
            manifest_path = f"manifests/{document_id}.json"
            manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
            minio_client.put_object(
                bucket_name=SETTINGS.MINIO.MINIO_BUCKET,
                object_name=manifest_path,
                data=BytesIO(manifest_bytes),
                length=len(manifest_bytes),
                content_type="application/json",
            )

            processing_time = time.time() - start_time
            logger.info(
                "ETL pipeline completed",
                document_id=document_id,
                processing_time=processing_time,
                total_chunks=chunk_result.get("chunk_count", 0),
            )

            # Record cost event for the entire pipeline
            try:
                await record_cost_event(
                    db_session,
                    provider="etl",
                    model="pipeline",
                    route="etl.process_document",
                    request_id=self.pipeline_id,
                    latency_ms=int(processing_time * 1000),
                    status="completed",
                    metadata={
                        "document_type": document_type,
                        "chunk_count": chunk_result.get("chunk_count", 0),
                        "steps_completed": len(manifest["steps"]),
                    },
                )
                await db_session.commit()
            except Exception as e:
                logger.warning("Failed to record pipeline cost event", error=str(e))

            return {
                "status": "completed",
                "document_id": document_id,
                "pipeline_id": self.pipeline_id,
                "manifest": manifest,
                "processing_time": processing_time,
                "chunk_count": chunk_result.get("chunk_count", 0),
            }

        except Exception as e:
            logger.error(
                "ETL pipeline failed",
                document_id=document_id,
                error=str(e),
                pipeline_id=self.pipeline_id,
            )
            manifest["error"] = str(e)
            manifest["status"] = "failed"

            # Record failure cost event
            try:
                await record_cost_event(
                    db_session,
                    provider="etl",
                    model="pipeline",
                    route="etl.process_document",
                    request_id=self.pipeline_id,
                    latency_ms=int((time.time() - start_time) * 1000),
                    status="failed",
                    metadata={"error": str(e), "document_type": document_type},
                )
                await db_session.commit()
            except Exception:
                pass

            raise


# Factory function for easy integration
async def process_document_etl(
    document_id: str,
    storage_path: str,
    document_type: str,
    db_session: AsyncSession,
    minio_client,
    force_reprocess: bool = False,
) -> Dict[str, Any]:
    """Factory function to create and run ETL pipeline."""
    pipeline = ETLPipeline()
    return await pipeline.process_document(
        document_id=document_id,
        storage_path=storage_path,
        document_type=document_type,
        db_session=db_session,
        minio_client=minio_client,
        force_reprocess=force_reprocess,
    )
