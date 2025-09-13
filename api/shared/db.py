"""Shared database utilities and dependencies for FastAPI routers."""
from typing import Any, AsyncGenerator

from dependency_injector.wiring import Provide, inject
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from di.container import ApplicationContainer
from infra.resources import DatabaseResource


@inject
async def get_db_session(
    db: DatabaseResource = Depends(
        Provide[ApplicationContainer.infrastructure.database]
    ),
) -> AsyncGenerator[AsyncSession, Any]:
    """Yield an AsyncSession per-request and ensure proper close.

    This dependency can be reused across all routers that need database access.
    """
    session = db.get_session()
    try:
        yield session
    finally:
        try:
            await session.close()
        except Exception:
            pass
