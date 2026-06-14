from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict
from uuid import UUID

from docmind_core.models import Chunk
from pydantic import BaseModel


class LlmPromptContext(TypedDict):
    prompt: str
    chunks: list[Chunk]


@dataclass
class Tenant:
    id: UUID
    name: str
    slug: str
    created_at: datetime


class RegisterRequest(BaseModel):
    name: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str
