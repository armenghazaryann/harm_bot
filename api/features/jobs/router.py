"""Router for the Jobs feature."""
from typing import Optional
from uuid import UUID

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Query

from api.features.jobs.controller import JobController
from api.features.jobs.dtos import (
    JobStatusResponse,
    JobListResponse,
    EvalRunRequest,
    EvalRunResponse,
    JobCancelRequest,
    JobRetryRequest,
    JobStatsResponse
)
from api.shared.dtos import PaginationRequest, HealthCheckResponse
from api.di.container import DependencyContainer
from api.shared.response import ResponseModel

router = APIRouter()


@router.get("/health", response_model=ResponseModel[HealthCheckResponse])
async def health_check():
    """Health check endpoint for jobs service."""
    return ResponseModel.success(
        data=HealthCheckResponse(
            status="healthy",
            dependencies={"celery": "ok", "redis": "ok", "database": "ok"}
        ),
        message="Jobs service is healthy"
    )


@router.get("/{job_id}", response_model=ResponseModel[JobStatusResponse])
@inject
async def get_job_status(
    job_id: UUID,
    controller: JobController = Depends(Provide[DependencyContainer.job_controller])
):
    """Get the status of a specific job."""
    return await controller.get_job_status(job_id)


@router.get("/", response_model=ResponseModel[JobListResponse])
@inject
async def list_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    controller: JobController = Depends(Provide[DependencyContainer.job_controller])
):
    """List jobs with optional filtering."""
    pagination = PaginationRequest(skip=skip, limit=limit)
    return await controller.list_jobs(pagination, job_type, status)


@router.post("/eval/run", response_model=ResponseModel[EvalRunResponse])
@inject
async def run_evaluation(
    request: EvalRunRequest,
    controller: JobController = Depends(Provide[DependencyContainer.job_controller])
):
    """Run an evaluation suite."""
    return await controller.run_evaluation(request)


@router.post("/{job_id}/cancel", response_model=ResponseModel[str])
@inject
async def cancel_job(
    job_id: UUID, 
    request: JobCancelRequest,
    controller: JobController = Depends(Provide[DependencyContainer.job_controller])
):
    """Cancel a running job."""
    return await controller.cancel_job(job_id, request)


@router.post("/{job_id}/retry", response_model=ResponseModel[JobStatusResponse])
@inject
async def retry_job(
    job_id: UUID, 
    request: JobRetryRequest,
    controller: JobController = Depends(Provide[DependencyContainer.job_controller])
):
    """Retry a failed job."""
    return await controller.retry_job(job_id, request)


@router.get("/stats/overview", response_model=ResponseModel[JobStatsResponse])
@inject
async def get_job_stats(
    controller: JobController = Depends(Provide[DependencyContainer.job_controller])
):
    """Get job statistics."""
    return await controller.get_job_stats()
