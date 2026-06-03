from typing import TypedDict

from docmind_core.models import Chunk


class LlmPromptContext(TypedDict):
    prompt: str
    chunks: list[Chunk]
