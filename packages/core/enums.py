from enum import StrEnum


class DocumentStatus(StrEnum):
    pending = "PENDING"
    processing = "PROCESSING"
    completed = "COMPLETED"
    failed = "FAILED"


class Role(StrEnum):
    user = "USER"
    assistant = "ASSISTANT"
    system = "SYSTEM"
