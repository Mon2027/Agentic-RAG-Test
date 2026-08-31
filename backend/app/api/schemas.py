"""Pydantic schemas for API request/response models."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Message type enum."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """Chat message model."""

    role: MessageType
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """Chat request model."""

    messages: list[Message]
    session_id: str | None = None
    context: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    """Chat response model."""

    message: Message
    session_id: str
    sources: list[dict[str, Any]] | None = None
    charts: list[str] | None = None


class UploadResponse(BaseModel):
    """File upload response model."""

    success: bool
    file_id: str | None = None
    file_name: str
    file_type: str
    message: str
    metadata: dict[str, Any] | None = None


class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentInfo(BaseModel):
    """Document information model."""

    file_id: str
    file_name: str
    file_type: str
    file_size: int
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    chunk_count: int | None = None
    error_message: str | None = None


class AnalysisResult(BaseModel):
    """Data analysis result model."""

    summary: str
    statistics: dict[str, Any] | None = None
    trends: list[dict[str, Any]] | None = None
    chart_paths: list[str] | None = None


class SearchResult(BaseModel):
    """RAG search result model."""

    content: str
    source: str
    page: int | None = None
    score: float
    metadata: dict[str, Any] | None = None


class RAGResponse(BaseModel):
    """RAG query response model."""

    answer: str
    sources: list[SearchResult]
    confidence: float


class ChatStreamResponse(BaseModel):
    """Streaming chat response event model."""

    type: str  # "start", "token", "tool_start", "tool_end", "end", "error"
    content: str | None = None
    session_id: str | None = None
    tool: str | None = None
    message: str | None = None
