"""Base classes and common patterns for the application with enhanced repository pattern."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Tuple
from uuid import UUID

from sqlalchemy import select, func, delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

T = TypeVar("T", bound=DeclarativeBase)


class BaseEntity(DeclarativeBase):
    """Base entity class for SQLAlchemy models."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create entity from dictionary."""
        entity = cls()
        for key, value in data.items():
            if hasattr(entity, key):
                setattr(entity, key, value)
        return entity


class BaseRepository(ABC, Generic[T]):
    """Enhanced base repository with common CRUD operations."""

    model: Type[T]

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, entity: T) -> T:
        """Create new entity."""
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def create_many(self, entities: List[T]) -> List[T]:
        """Create multiple entities."""
        self.session.add_all(entities)
        await self.session.flush()
        for entity in entities:
            await self.session.refresh(entity)
        return entities

    async def get_by_id(self, entity_id: UUID) -> Optional[T]:
        """Get entity by ID."""
        stmt = select(self.model).where(self.model.id == entity_id)  # type: ignore[attr-defined]
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_field(
        self, field_name: str, value: Any, limit: Optional[int] = None
    ) -> List[T]:
        """Get entities by field value."""
        field = getattr(self.model, field_name)
        stmt = select(self.model).where(field == value)  # type: ignore[attr-defined]

        if limit:
            stmt = stmt.limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_fields(self, **filters: Any) -> List[T]:
        """Get entities by multiple field values."""
        stmt = select(self.model)

        for field_name, value in filters.items():
            if hasattr(self.model, field_name):
                field = getattr(self.model, field_name)
                stmt = stmt.where(field == value)  # type: ignore[attr-defined]

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update(self, entity: T) -> T:
        """Update existing entity."""
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def update_by_id(self, entity_id: UUID, **kwargs: Any) -> Optional[T]:
        """Update entity by ID with field values."""
        stmt = (
            update(self.model)
            .where(self.model.id == entity_id)  # type: ignore[attr-defined]
            .values(**kwargs)
            .returning(self.model)
        )
        result = await self.session.execute(stmt)
        await self.session.flush()
        entity = result.scalar_one_or_none()
        if entity:
            await self.session.refresh(entity)
        return entity

    async def delete(self, entity_id: UUID) -> bool:
        """Delete entity by ID."""
        stmt = delete(self.model).where(self.model.id == entity_id)  # type: ignore[attr-defined]
        result = await self.session.execute(stmt)
        await self.session.flush()
        return result.rowcount > 0

    async def delete_by_field(self, field_name: str, value: Any) -> int:
        """Delete entities by field value."""
        field = getattr(self.model, field_name)
        stmt = delete(self.model).where(field == value)
        result = await self.session.execute(stmt)
        await self.session.flush()
        return result.rowcount

    async def list(
        self,
        offset: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None,
        **filters: Any,
    ) -> Tuple[List[T], int]:
        """List entities with pagination and filters."""
        # Build count query
        count_stmt = select(func.count(self.model.id))  # type: ignore[attr-defined]

        # Build select query
        stmt = select(self.model)

        # Apply filters
        for field_name, value in filters.items():
            if hasattr(self.model, field_name):
                field = getattr(self.model, field_name)
                if value is not None:
                    if isinstance(value, (list, tuple)):
                        stmt = stmt.where(field.in_(value))
                        count_stmt = count_stmt.where(field.in_(value))
                    else:
                        stmt = stmt.where(field == value)
                        count_stmt = count_stmt.where(field == value)

        # Apply ordering
        if order_by:
            if order_by.startswith("-"):
                field_name = order_by[1:]
                if hasattr(self.model, field_name):
                    field = getattr(self.model, field_name)
                    stmt = stmt.order_by(field.desc())
            else:
                if hasattr(self.model, order_by):
                    field = getattr(self.model, order_by)
                    stmt = stmt.order_by(field.asc())

        # Apply pagination
        stmt = stmt.offset(offset).limit(limit)

        # Execute queries
        result = await self.session.execute(stmt)
        count_result = await self.session.execute(count_stmt)

        # count_result.scalar() may be None, default to 0
        total = count_result.scalar() or 0
        return list(result.scalars().all()), int(total)

    async def exists(self, entity_id: UUID) -> bool:
        """Check if entity exists."""
        stmt = select(self.model.id).where(self.model.id == entity_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def count(self, **filters: Any) -> int:
        """Count entities with filters."""
        stmt = select(func.count(self.model.id))  # type: ignore[attr-defined]

        for field_name, value in filters.items():
            if hasattr(self.model, field_name) and value is not None:
                field = getattr(self.model, field_name)
                if isinstance(value, (list, tuple)):
                    stmt = stmt.where(field.in_(value))
                else:
                    stmt = stmt.where(field == value)

        result = await self.session.execute(stmt)
        return int(result.scalar() or 0)


class BaseService(ABC, Generic[T]):
    """Base service class with repository integration."""

    def __init__(self, repository: BaseRepository[T]):
        self.repository = repository

    async def get_by_id(self, entity_id: UUID) -> Optional[T]:
        """Get entity by ID."""
        return await self.repository.get_by_id(entity_id)

    async def list(
        self, offset: int = 0, limit: int = 100, **filters: Any
    ) -> Tuple[List[T], int]:
        """List entities with pagination."""
        return await self.repository.list(offset, limit, **filters)

    async def count(self, **filters: Any) -> int:
        """Count entities."""
        return await self.repository.count(**filters)


class BaseValidator(ABC):
    """Base validator for input validation."""

    @abstractmethod
    async def validate(self, data: Any) -> bool:
        """Validate input data."""
        pass


class ValidationError(Exception):
    """Raised when validation fails."""

    pass
