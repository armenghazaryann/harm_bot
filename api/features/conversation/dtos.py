"""DTOs for the Conversation feature."""
from typing import List, Optional
from uuid import UUID

from pydantic import Field

from api.shared.dtos import BaseDTO


class CreateConversationRequest(BaseDTO):
    """Request to create a conversation."""

    title: Optional[str] = Field(default=None, description="Conversation title")
    user_id: Optional[str] = Field(default="1", description="User identifier")


class ConversationDTO(BaseDTO):
    """Conversation DTO."""

    id: UUID = Field(description="Conversation identifier")
    title: Optional[str] = Field(default=None, description="Conversation title")
    user_id: Optional[str] = Field(default="1", description="User identifier")


class CreateConversationResponse(BaseDTO):
    """Response when creating a conversation."""

    conversation: ConversationDTO = Field(description="Created conversation")


class MessageDTO(BaseDTO):
    """Conversation message DTO."""

    id: UUID = Field(description="Message identifier")
    role: str = Field(description="Message role: user or assistant")
    content: str = Field(description="Message content")
    tokens: Optional[int] = Field(default=None, description="Token count if available")


class AppendMessageRequest(BaseDTO):
    """Append a message to a conversation."""

    role: str = Field(description="Message role: user or assistant")
    content: str = Field(description="Message content")
    tokens: Optional[int] = Field(default=None, description="Token count if available")


class ConversationListResponse(BaseDTO):
    """List conversations response."""

    items: List[ConversationDTO] = Field(description="Conversations")
    total: int = Field(description="Total conversations (approx)")


class MessagesResponse(BaseDTO):
    """Messages list response."""

    items: List[MessageDTO] = Field(description="Messages in chronological order")
    total: int = Field(description="Total messages returned")
