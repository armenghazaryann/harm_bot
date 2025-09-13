"""Indexing utilities for vector search - refactored for new architecture."""
from typing import Dict, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from infra.db_utils import DatabaseManager


class VectorIndexManager:
    """Manages vector indexes with proper separation of concerns."""

    def __init__(self):
        pass

    async def create_pgvector_hnsw_index(
        self,
        table_name: str = "embeddings",
        column_name: str = "embedding",
        dimensions: int = 1536,
        m: int = 16,
        ef_construction: int = 64,
    ) -> Dict[str, Any]:
        """Create HNSW index for pgvector."""
        db = await DatabaseManager.get_resource()

        async with db.get_session() as session:
            index_name = f"{table_name}_{column_name}_hnsw_idx"

            # Create index SQL
            sql = f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {table_name} USING hnsw ({column_name} vector_cosine_ops)
            WITH (m = {m}, ef_construction = {ef_construction});
            """

            await session.execute(text(sql))
            await session.commit()

            return {
                "index_name": index_name,
                "table": table_name,
                "column": column_name,
                "type": "hnsw",
                "dimensions": dimensions,
                "m": m,
                "ef_construction": ef_construction,
            }

    async def create_fts_index(
        self, table_name: str = "chunks", column_name: str = "content"
    ) -> Dict[str, Any]:
        """Create full-text search index."""
        db = await DatabaseManager.get_resource()

        async with db.get_session() as session:
            index_name = f"{table_name}_{column_name}_fts_idx"

            # Create FTS index
            sql = f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {table_name} USING gin (to_tsvector('english', {column_name}));
            """

            await session.execute(text(sql))
            await session.commit()

            return {
                "index_name": index_name,
                "table": table_name,
                "column": column_name,
                "type": "fts",
            }

    async def check_index_exists(self, index_name: str, session: AsyncSession) -> bool:
        """Check if index exists."""
        sql = """
        SELECT EXISTS (
            SELECT 1 FROM pg_indexes
            WHERE indexname = :index_name
        );
        """

        result = await session.execute(text(sql), {"index_name": index_name})
        return result.scalar()

    async def drop_index(self, index_name: str) -> Dict[str, Any]:
        """Drop index if exists."""
        db = await DatabaseManager.get_resource()

        async with db.get_session() as session:
            sql = f"DROP INDEX IF EXISTS {index_name};"
            await session.execute(text(sql))
            await session.commit()

            return {"index_name": index_name, "action": "dropped"}


async def index_chunks_pgvector(doc_id: str) -> Dict[str, Any]:
    """Index chunks for a document in pgvector with HNSW index.

    This function creates vector indexes for embeddings of chunks belonging to
    the specified document. It's designed to be called by Celery workers.

    Args:
        doc_id: The document ID to index chunks for

    Returns:
        Dictionary with indexing results including index names and status
    """
    index_manager = VectorIndexManager()

    # Create HNSW index for embeddings table
    hnsw_result = await index_manager.create_pgvector_hnsw_index(
        table_name="embeddings",
        column_name="embedding",
        dimensions=1536,
        m=16,
        ef_construction=64,
    )

    # Create FTS index for chunks table
    fts_result = await index_manager.create_fts_index(
        table_name="chunks", column_name="content"
    )

    return {
        "doc_id": doc_id,
        "hnsw_index": hnsw_result,
        "fts_index": fts_result,
        "status": "completed",
    }
