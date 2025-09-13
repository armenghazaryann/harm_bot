from typing import Optional
from uuid import UUID

import logging
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, File, UploadFile, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.features.documents.controller import DocumentController
from api.features.documents.exceptions import DocumentNotFoundError
from api.features.documents.dtos import (
    DocumentUploadResponse,
    DocumentStatusResponse,
    DocumentListResponse,
    DocumentDeleteRequest,
)
from api.shared.dtos import PaginationRequest, HealthCheckResponse
from api.di.container import ApplicationContainer as DependencyContainer
from api.shared.response import ResponseModel
from api.shared.db import get_db_session

router = APIRouter()
logger = logging.getLogger("rag.documents.router")


@router.get("/health", response_model=ResponseModel[HealthCheckResponse])
async def health_check():
    """Health check endpoint for documents service."""
    return ResponseModel.success(
        data=HealthCheckResponse(
            status="healthy", dependencies={"storage": "ok", "database": "ok"}
        ),
        message="Documents service is healthy",
    )


@router.post("/upload", response_model=ResponseModel[DocumentUploadResponse])
@inject
async def upload_document(
    file: UploadFile = File(...),
    controller: DocumentController = Depends(
        Provide[DependencyContainer.controllers.document_controller]
    ),
    db_session: AsyncSession = Depends(get_db_session),
):
    """Upload a document for processing."""
    try:
        result = await controller.upload_document(file, db_session=db_session)
        return ResponseModel.success(
            data=result, message=f"Document {file.filename} uploaded successfully"
        )
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{doc_id}/status", response_model=ResponseModel[DocumentStatusResponse])
@inject
async def get_document_status(
    doc_id: UUID,
    controller: DocumentController = Depends(
        Provide[DependencyContainer.controllers.document_controller]
    ),
    db_session: AsyncSession = Depends(get_db_session),
):
    """Get document processing status."""
    try:
        result = await controller.get_document_status(doc_id, db_session=db_session)
        return ResponseModel.success(
            data=result, message="Document status retrieved successfully"
        )
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to get document status")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=ResponseModel[DocumentListResponse])
@inject
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None, description="Filter by status"),
    controller: DocumentController = Depends(
        Provide[DependencyContainer.controllers.document_controller]
    ),
    db_session: AsyncSession = Depends(get_db_session),
):
    """List documents with optional filtering."""
    try:
        pagination = PaginationRequest(skip=skip, limit=limit)
        result = await controller.list_documents(pagination, db_session=db_session)
        return ResponseModel.success(
            data=result, message="Documents retrieved successfully"
        )
    except Exception as e:
        logger.exception("Failed to list documents")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{doc_id}", response_model=ResponseModel[str])
@inject
async def delete_document(
    doc_id: UUID,
    delete_chunks: bool = Query(True, description="Delete associated chunks"),
    delete_embeddings: bool = Query(True, description="Delete embeddings"),
    controller: DocumentController = Depends(
        Provide[DependencyContainer.controllers.document_controller]
    ),
    db_session: AsyncSession = Depends(get_db_session),
):
    """Delete a document and optionally its associated data."""
    try:
        delete_request = DocumentDeleteRequest(
            doc_id=doc_id,
            delete_chunks=delete_chunks,
            delete_embeddings=delete_embeddings,
        )
        result = await controller.delete_document(doc_id, delete_request, db_session)
        return ResponseModel.success(
            data=result, message="Document deleted successfully"
        )
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to delete document")
        raise HTTPException(status_code=500, detail=str(e))
