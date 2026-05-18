import uuid
from datetime import UTC, datetime

from enums import DocumentStatus, Role
from pydantic import BaseModel, Field, PositiveInt, model_validator


class Tenant(BaseModel):
    id: str
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_active: bool = True


class Document(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    tenant_id: str
    filename: str
    file_type: str
    status: DocumentStatus
    size_bytes: PositiveInt


class Chunk(BaseModel):
    id: str
    document_id: str
    text: str
    chunk_index: int


class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    tenant_id: str
    messages: list[ChatMessage]
    stream: bool

    @model_validator(mode="after")
    def validate_messages(self) -> "ChatRequest":
        if not self.messages:
            raise ValueError("Messages can't be empty. Atleast 1 message is required.")
        return self


class ChatResponse(BaseModel):
    message: str
    trace_id: str
    tokens_used: int


class IngestionRequest(BaseModel):
    tenant_id: str
    document_id: str
    s3_key: str


class IngestionStatus(BaseModel):
    document_id: str
    status: DocumentStatus
    chunks_created: int
    error: str | None = None


class QueryResult(BaseModel):
    chunk: Chunk
    score: float
    document_id: str


class ErrorResponse(BaseModel):
    code: str
    message: str
    detail: str | None = None
