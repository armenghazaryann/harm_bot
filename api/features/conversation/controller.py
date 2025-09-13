"""Controller for the Conversation feature."""
from typing import Optional
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.features.conversation.dtos import (
    AppendMessageRequest,
    ConversationDTO,
    ConversationListResponse,
    MessageDTO,
)
from api.features.conversation.service import (
    create_conversation as svc_create,
    append_message as svc_append,
    fetch_recent_messages as svc_fetch,
    list_conversations as svc_list,
    get_conversation as svc_get,
)


class ConversationController:
    """Controller handling conversation CRUD and message operations."""

    def __init__(self) -> None:
        # No service dependencies required; using simple SQL functions
        pass

    async def create_conversation(
        self,
        *,
        title: Optional[str],
        user_id: Optional[str],
        db_session: AsyncSession,
    ) -> ConversationDTO:
        conv_id = await svc_create(db_session, title=title, user_id=user_id)
        conv = await svc_get(db_session, conversation_id=conv_id)
        if not conv:
            raise HTTPException(status_code=500, detail="Failed to create conversation")
        return ConversationDTO(
            id=UUID(conv["id"]), title=conv.get("title"), user_id=conv.get("user_id")
        )

    async def list_conversations(
        self,
        *,
        user_id: Optional[str],
        limit: int,
        db_session: AsyncSession,
    ) -> ConversationListResponse:
        items = await svc_list(db_session, user_id=user_id, limit=limit)
        dtos = [
            ConversationDTO(
                id=UUID(i["id"]), title=i.get("title"), user_id=i.get("user_id")
            )
            for i in items
        ]
        return ConversationListResponse(items=dtos, total=len(dtos))

    async def get_messages(
        self,
        *,
        conversation_id: UUID,
        limit: int,
        db_session: AsyncSession,
    ) -> list[MessageDTO]:
        msgs = await svc_fetch(
            db_session, conversation_id=str(conversation_id), limit=limit
        )
        return [
            MessageDTO(
                id=UUID(m["id"]),
                role=str(m.get("role", "user")),
                content=str(m.get("content", "")),
                tokens=m.get("tokens"),
            )
            for m in msgs
        ]

    async def append_message(
        self,
        *,
        conversation_id: UUID,
        request: AppendMessageRequest,
        db_session: AsyncSession,
    ) -> MessageDTO:
        _ = await svc_append(
            db_session,
            conversation_id=str(conversation_id),
            role=request.role,
            content=request.content,
            tokens=request.tokens,
        )
        # Return the last message appended
        msgs = await svc_fetch(
            db_session, conversation_id=str(conversation_id), limit=1
        )
        if not msgs:
            raise HTTPException(status_code=500, detail="Failed to append message")
        m = msgs[-1]
        return MessageDTO(
            id=UUID(m["id"]),
            role=str(m.get("role", "user")),
            content=str(m.get("content", "")),
            tokens=m.get("tokens"),
        )
