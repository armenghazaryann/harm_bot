"""Router for the Conversation feature."""
from uuid import UUID
from typing import Optional

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from di.container import ApplicationContainer as DependencyContainer
from api.features.conversation.controller import ConversationController
from api.features.conversation.dtos import (
    AppendMessageRequest,
    ConversationDTO,
    ConversationListResponse,
    CreateConversationRequest,
    MessageDTO,
    MessagesResponse,
)
from api.shared.db import get_db_session
from api.shared.dtos import HealthCheckResponse
from api.shared.response import ResponseModel

router = APIRouter()


@router.get("/health", response_model=ResponseModel[HealthCheckResponse])
async def health_check():
    return ResponseModel.success(
        data=HealthCheckResponse(status="healthy", dependencies={"database": "ok"}),
        message="Conversation service is healthy",
    )


@router.post("/", response_model=ResponseModel[ConversationDTO])
@inject
async def create_conversation(
    request: CreateConversationRequest,
    controller: ConversationController = Depends(
        Provide[DependencyContainer.controllers.conversation_controller]
    ),
    db_session: AsyncSession = Depends(get_db_session),
):
    try:
        conv = await controller.create_conversation(
            title=request.title, user_id=request.user_id, db_session=db_session
        )
        return ResponseModel.success(data=conv, message="Conversation created")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=ResponseModel[ConversationListResponse])
@inject
async def list_conversations(
    user_id: Optional[str] = Query(None, description="Filter by user id"),
    limit: int = Query(50, ge=1, le=200),
    controller: ConversationController = Depends(
        Provide[DependencyContainer.controllers.conversation_controller]
    ),
    db_session: AsyncSession = Depends(get_db_session),
):
    try:
        result = await controller.list_conversations(
            user_id=user_id, limit=limit, db_session=db_session
        )
        return ResponseModel.success(data=result, message="Conversations listed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}", response_model=ResponseModel[ConversationDTO])
@inject
async def get_conversation(
    conversation_id: UUID,
    controller: ConversationController = Depends(
        Provide[DependencyContainer.controllers.conversation_controller]
    ),
    db_session: AsyncSession = Depends(get_db_session),
):
    try:
        # Delegate to controller via messages fetch helper
        from api.features.conversation.service import get_conversation as svc_get

        conv = await svc_get(db_session, conversation_id=str(conversation_id))
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        dto = ConversationDTO(
            id=conversation_id,
            title=conv.get("title"),
            user_id=conv.get("user_id"),
        )
        return ResponseModel.success(data=dto, message="Conversation fetched")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{conversation_id}/messages", response_model=ResponseModel[MessagesResponse]
)
@inject
async def get_messages(
    conversation_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    controller: ConversationController = Depends(
        Provide[DependencyContainer.controllers.conversation_controller]
    ),
    db_session: AsyncSession = Depends(get_db_session),
):
    try:
        items = await controller.get_messages(
            conversation_id=conversation_id, limit=limit, db_session=db_session
        )
        return ResponseModel.success(
            data=MessagesResponse(items=items, total=len(items)),
            message="Messages fetched",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{conversation_id}/messages", response_model=ResponseModel[MessageDTO])
@inject
async def append_message(
    conversation_id: UUID,
    request: AppendMessageRequest,
    controller: ConversationController = Depends(
        Provide[DependencyContainer.controllers.conversation_controller]
    ),
    db_session: AsyncSession = Depends(get_db_session),
):
    try:
        msg = await controller.append_message(
            conversation_id=conversation_id, request=request, db_session=db_session
        )
        return ResponseModel.success(data=msg, message="Message appended")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
