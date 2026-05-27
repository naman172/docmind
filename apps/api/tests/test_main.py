from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from docmind_api.main import app
from httpx import ASGITransport, AsyncClient


def make_chat_payload(**overrides: Any) -> dict[str, Any]:
    base = {
        "collection_name": "test",
        "tenant_id": "tenant-123",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    return {**base, **overrides}


@pytest.fixture
def async_client() -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app), base_url="http://localhost:8000"
    )


@pytest.mark.asyncio
async def test_health(async_client: AsyncClient) -> None:
    async with async_client as ac:
        response = await ac.get("/health")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_non_streaming(async_client: AsyncClient) -> None:
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "hello back"

    with (
        patch(
            "docmind_api.main.build_llm_prompt",
            new_callable=AsyncMock,
            return_value="mock prompt",
        ),
        patch(
            "docmind_api.main.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
    ):
        async with async_client as ac:
            request = make_chat_payload()
            response = await ac.post("/chat", json=request)

    assert response.status_code == 200
    assert response.json() == {"response": "hello back"}


async def fake_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[MagicMock, None]:
    chunk = MagicMock()
    chunk.choices[0].delta.content = "hello back"
    yield chunk


async def mock_acompletion(
    *args: Any, **kwargs: Any
) -> AsyncGenerator[MagicMock, None]:
    return fake_stream()


@pytest.mark.asyncio
async def test_chat_streaming(async_client: AsyncClient) -> None:
    with (
        patch(
            "docmind_api.main.build_llm_prompt",
            new_callable=AsyncMock,
            return_value="mock prompt",
        ),
        patch(
            "docmind_api.main.litellm.acompletion",
            new=AsyncMock(side_effect=mock_acompletion),
        ),
    ):
        async with async_client as ac:
            request = make_chat_payload(stream=True)
            response = await ac.post("/chat", json=request)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "hello back" in response.text


@pytest.mark.asyncio
async def test_chat_missing_messages_returns_422(async_client: AsyncClient) -> None:
    async with async_client as ac:
        response = await ac.post(
            "/chat", json={"tenant_id": "tenant-123", "stream": False}
        )

    assert response.status_code == 422
