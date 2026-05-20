from enum import StrEnum


class DocumentStatus(StrEnum):
    pending = "PENDING"
    processing = "PROCESSING"
    completed = "COMPLETED"
    failed = "FAILED"


class Role(StrEnum):
    USER = "user"
    ASSISSTANT = "assisstant"
    SYSTEM = "system"
