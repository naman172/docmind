from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict
from uuid import UUID

from docmind_core.models import Chunk


class LlmPromptContext(TypedDict):
    prompt: str
    chunks: list[Chunk]


@dataclass
class Tenant:
    id: UUID
    name: str
    slug: str
    created_at: datetime
