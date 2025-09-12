"""Entity registry to ensure SQLAlchemy loads all table metadata.

Import all entity modules here so Alembic autogenerate can discover them.
"""
# Import base first to expose BaseEntity.metadata
from api.shared.entities.base import BaseEntity  # noqa: F401

# Feature: Documents
from api.features.documents.entities.document import Document  # noqa: F401

# Feature: Query (Chunks and Embeddings)
from api.features.query.entities.chunk import Chunk  # noqa: F401
from api.features.query.entities.embedding import Embedding  # noqa: F401

# Feature: Jobs
from api.features.jobs.entities.job import Job  # noqa: F401
