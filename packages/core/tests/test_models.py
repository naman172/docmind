from typing import Any

import pytest
from docmind_core.enums import Role
from docmind_core.models import ChatMessage, ChatRequest
from pydantic import ValidationError


def make_valid_request(**overrides: Any) -> dict[str, Any]:
    base = {
        "collection_name": "test",
        "tenant_id": "tenant-123",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    return {**base, **overrides}


def test_chat_request_valid_construction() -> None:
    request = ChatRequest(**make_valid_request())
    assert request.tenant_id == "tenant-123"
    assert request.stream is False
    assert len(request.messages) == 1


def test_chat_request_rejects_empty_messages() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ChatRequest(**make_valid_request(messages=[]))

    errors = exc_info.value.errors()
    print(errors)
    assert len(errors) == 1
    assert errors[0]["type"] == "value_error"  # what kind of error


def test_chat_request_rejects_invalid_role() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            **make_valid_request(messages=[{"role": "admin", "content": "hello"}])
        )


def test_chat_request_rejects_uppercase_role_value() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            **make_valid_request(messages=[{"role": "USER", "content": "hello"}])
        )


def test_chat_request_rejects_missing_tenant_id() -> None:
    with pytest.raises(ValidationError):
        data = make_valid_request()
        del data["tenant_id"]
        ChatRequest(**data)


def test_chat_message_role_coercion() -> None:
    msg = ChatMessage(role=Role.USER, content="hello")
    assert msg.role == Role.USER
    assert msg.role.value == "user"
