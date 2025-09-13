"""Minimal conversation service: create, append, fetch messages.

Uses raw SQL via SQLAlchemy AsyncSession to avoid ORM overhead. Matches the
simple schema created by the Alembic migration for `conversation` and `message`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def create_conversation(
    session: AsyncSession,
    *,
    title: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    conv_id = str(uuid.uuid4())
    sql = text(
        """
        INSERT INTO conversation (id, user_id, title)
        VALUES (:id, :user_id, :title)
        """
    )
    await session.execute(sql, {"id": conv_id, "user_id": user_id, "title": title})
    # Ensure immediate visibility for subsequent reads
    await session.commit()
    return conv_id


async def append_message(
    session: AsyncSession,
    *,
    conversation_id: str,
    role: str,
    content: str,
    tokens: Optional[int] = None,
) -> str:
    msg_id = str(uuid.uuid4())
    sql = text(
        """
        INSERT INTO message (id, conversation_id, role, content, tokens)
        VALUES (:id, :conversation_id, :role, :content, :tokens)
        """
    )
    await session.execute(
        sql,
        {
            "id": msg_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "tokens": tokens,
        },
    )
    # Bump conversation updated_at for recency ordering
    await session.execute(
        text("UPDATE conversation SET updated_at = NOW() WHERE id = :id"),
        {"id": conversation_id},
    )
    await session.commit()
    return msg_id


async def fetch_recent_messages(
    session: AsyncSession,
    *,
    conversation_id: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    sql = text(
        """
        SELECT id::text as id, role, content, tokens, created_at
        FROM message
        WHERE conversation_id = :conversation_id
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )
    res = await session.execute(
        sql, {"conversation_id": conversation_id, "limit": limit}
    )
    rows = res.mappings().all()
    # Return chronological order
    return list(reversed([dict(r) for r in rows]))


async def list_conversations(
    session: AsyncSession,
    *,
    user_id: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """List conversations, optionally filtered by user_id, most recent first."""
    if user_id:
        sql = text(
            """
            SELECT id::text as id, user_id, title, created_at, updated_at
            FROM conversation
            WHERE user_id = :user_id
            ORDER BY updated_at DESC
            LIMIT :limit
            """
        )
        params = {"user_id": user_id, "limit": limit}
    else:
        sql = text(
            """
            SELECT id::text as id, user_id, title, created_at, updated_at
            FROM conversation
            ORDER BY updated_at DESC
            LIMIT :limit
            """
        )
        params = {"limit": limit}
    res = await session.execute(sql, params)
    return [dict(r) for r in res.mappings().all()]


async def get_conversation(
    session: AsyncSession,
    *,
    conversation_id: str,
) -> Optional[Dict[str, Any]]:
    """Fetch a single conversation by id."""
    sql = text(
        """
        SELECT id::text as id, user_id, title, created_at, updated_at
        FROM conversation
        WHERE id = :id
        """
    )
    res = await session.execute(sql, {"id": conversation_id})
    row = res.mappings().first()
    return dict(row) if row else None
