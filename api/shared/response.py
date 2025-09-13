from typing import Generic, Optional, TypeVar, Any, Dict

from pydantic import BaseModel, Field

T = TypeVar("T")


class ResponseModel(BaseModel, Generic[T]):
    data: Optional[T] = Field(description="Response data", default=None)
    message: Optional[str] = Field(description="Response message", examples=["Success"])
    status: str = Field(default="ok", description="Response status")

    class Config:
        json_encoders: Dict[Any, Any] = {
            # Add custom encoders if needed
        }

    @classmethod
    def success(
        cls, data: Optional[T] = None, message: str = "Success"
    ) -> "ResponseModel[T]":
        """Create a successful response."""
        return cls(data=data, message=message, status="ok")

    @classmethod
    def error(cls, message: str, data: Optional[T] = None) -> "ResponseModel[T]":
        """Create an error response."""
        return cls(data=data, message=message, status="error")
