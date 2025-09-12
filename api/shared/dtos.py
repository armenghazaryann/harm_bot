"""Shared DTOs for the RAG ETL API."""
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field

T = TypeVar("T")


class BaseDTO(BaseModel):
    """Base DTO with common configuration."""
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class PaginationRequest(BaseDTO):
    """Request DTO for pagination."""
    skip: int = Field(default=0, ge=0, description="Number of items to skip")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of items to return")


class PaginationResponse(BaseDTO, Generic[T]):
    """Response DTO for paginated results."""
    items: List[T] = Field(description="List of items")
    total: int = Field(description="Total number of items")
    skip: int = Field(description="Number of items skipped")
    limit: int = Field(description="Maximum number of items returned")
    has_next: bool = Field(description="Whether there are more items")


class HealthCheckResponse(BaseDTO):
    """Health check response DTO."""
    status: str = Field(description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="0.1.0")
    dependencies: Dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseDTO):
    """Error response DTO."""
    error_code: str = Field(description="Error code")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TaskResponse(BaseDTO):
    """Response DTO for async tasks."""
    task_id: str = Field(description="Task identifier")
    status: str = Field(description="Task status")
    message: str = Field(description="Status message")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MetadataDTO(BaseDTO):
    """Generic metadata DTO."""
    key: str = Field(description="Metadata key")
    value: Any = Field(description="Metadata value")
    type: str = Field(description="Value type")


class TimestampMixin:
    """Mixin for timestamp fields.
    
    This mixin can be used with any Pydantic model to add created_at and updated_at fields.
    """
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")
