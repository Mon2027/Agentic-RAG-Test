"""API 的请求、响应数据模型。

Pydantic 根据字段类型校验请求数据，并将响应模型转换为 JSON；这些模型也用于
生成 OpenAPI 文档。字段名和枚举值属于接口契约，修改时需要同步考虑客户端。
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """消息角色；继承 str 使枚举值可直接序列化为 JSON 字符串。"""

    # 这些值必须与客户端传入的 role 字段一致。
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """单条对话消息，包括角色、正文和创建时间。"""

    role: MessageType
    content: str
    # 每次创建实例时调用 datetime.now，避免多个消息共享同一个默认时间。
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """普通聊天与流式聊天共用的请求体。"""

    # 消息列表由客户端提供；session_id 缺省时，路由会创建新会话。
    messages: list[Message]
    session_id: str | None = None
    # 预留给客户端传递附加上下文；当前路由未读取此字段。
    context: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    """普通聊天接口的响应体。"""

    message: Message
    session_id: str
    # 可选字段用于携带检索来源和生成图表；未提供时序列化结果为 null。
    sources: list[dict[str, Any]] | None = None
    charts: list[str] | None = None


class UploadResponse(BaseModel):
    """报告上传、数据上传及重建索引接口共用的响应体。"""

    success: bool
    # file_id 用于后续查询、删除或重建索引；字段允许为空以兼容其他响应场景。
    file_id: str | None = None
    file_name: str
    file_type: str
    message: str
    # 当前上传路由会在 metadata 中返回文件字节数。
    metadata: dict[str, Any] | None = None


class DocumentStatus(str, Enum):
    """PDF 文档从等待处理到完成或失败的状态。"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentInfo(BaseModel):
    """文档信息的数据模型。

    当前文档路由直接返回字典，未声明此模型为 response_model；这里定义了可复用的
    文档字段结构，不能据此假定现有路由会强制校验这些字段。
    """

    file_id: str
    file_name: str
    file_type: str
    file_size: int
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    # 分块数量主要适用于已建立向量索引的 PDF。
    chunk_count: int | None = None
    error_message: str | None = None


class AnalysisResult(BaseModel):
    """数据分析结果：摘要为必填，其余为按需生成的结构化内容。"""

    summary: str
    statistics: dict[str, Any] | None = None
    trends: list[dict[str, Any]] | None = None
    chart_paths: list[str] | None = None


class SearchResult(BaseModel):
    """单条 RAG 检索结果及其来源、页码和相关度分数。"""

    content: str
    source: str
    page: int | None = None
    score: float
    metadata: dict[str, Any] | None = None


class RAGResponse(BaseModel):
    """RAG 问答结果，包含回答、引用来源及置信度。"""

    answer: str
    sources: list[SearchResult]
    confidence: float


class ChatStreamResponse(BaseModel):
    """流式聊天单个 SSE 事件的数据结构。

    流式路由目前直接生成事件字典并写入 SSE，没有使用此模型做运行时校验。
    不同 type 对应不同可选字段，客户端应按事件类型读取。
    """

    type: str  # 事件类型：start、token、tool_start、tool_end、end、error。
    content: str | None = None
    session_id: str | None = None
    tool: str | None = None
    message: str | None = None
