from typing import Generic, Optional, TypeVar, Any

from pydantic import BaseModel, Field

T = TypeVar("T")


class ResponseModel(BaseModel, Generic[T]):
    data: T = Field(description="Response data", default=None)
    message: Optional[str] = Field(description="Response message", examples=["Success"])
    status: str = Field(default="ok", description="Response status")
    
    class Config:
        json_encoders = {
            # Add custom encoders if needed
        }
    
    @classmethod
    def success(cls, data: T = None, message: str = "Success") -> "ResponseModel[T]":
        """Create a successful response."""
        return cls(data=data, message=message, status="ok")
    
    @classmethod
    def error(cls, message: str, data: T = None) -> "ResponseModel[T]":
        """Create an error response."""
        return cls(data=data, message=message, status="error")
