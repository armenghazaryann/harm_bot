"""Class-based view helper for FastAPI routers."""
from typing import Any, Callable, Type, TypeVar

from fastapi import APIRouter

T = TypeVar("T")


def cbv(router: APIRouter) -> Callable[[Type[T]], Type[T]]:
    """
    Class-based view decorator for FastAPI routers.
    
    This decorator allows you to use class-based views with FastAPI routers,
    similar to Django's class-based views.
    
    Usage:
        @cbv(router)
        class MyController:
            @router.get("/endpoint")
            async def my_endpoint(self):
                return {"message": "Hello"}
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Get all methods from the class
        for method_name in dir(cls):
            method = getattr(cls, method_name)
            
            # Skip private methods and non-callables
            if method_name.startswith("_") or not callable(method):
                continue
                
            # Check if the method has route decorations
            if hasattr(method, "__annotations__") and hasattr(method, "__wrapped__"):
                # Update the method to be a class method
                setattr(cls, method_name, method)
        
        return cls
    
    return decorator
