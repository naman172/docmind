import uuid
from collections.abc import AsyncGenerator, Iterator
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from docmind_api.main import app
from docmind_api.models import Tenant
from docmind_core.models import Chunk
from httpx import ASGITransport, AsyncClient


def make_chat_payload(**overrides: Any) -> dict[str, Any]:
    base = {
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    return {**base, **overrides}


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    with (
        patch("docmind_api.main.init_pool", new_callable=AsyncMock),
        patch("docmind_api.main.close_pool", new_callable=AsyncMock),
        patch("docmind_api.main.get_pool"),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://localhost:8000"
        ) as ac:
            yield ac


@pytest.fixture
def mock_sparse_index() -> Iterator[None]:
    from docmind_api.main import _sparse_indexes

    _sparse_indexes["test"] = (MagicMock(), {})  # fake bm25, empty vocab
    yield
    _sparse_indexes.clear()  # clean up after test


@pytest.mark.asyncio
async def test_health(async_client: AsyncClient) -> None:
    response = await async_client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_non_streaming(
    async_client: AsyncClient, mock_sparse_index: None
) -> None:
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "hello back"

    with (
        patch(
            "docmind_api.main.build_llm_prompt",
            new_callable=AsyncMock,
            return_value={
                "prompt": "mock prompt",
                "chunks": [
                    Chunk(
                        document_id=uuid.uuid4(),
                        source_file="test",
                        text="mock prompt",
                        chunk_index=0,
                    )
                ],
            },
        ),
        patch(
            "docmind_api.main.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
        patch(
            "docmind_api.main.get_current_tenant",
            return_value=Tenant(
                id=uuid.uuid4(), name="test", slug="test", created_at=datetime.now()
            ),
        ),
    ):
        request = make_chat_payload()
        response = await async_client.post("/chat", json=request)

    assert response.status_code == 200
    assert response.json()["response"] == "hello back"
    assert len(response.json()["retrieved_chunks"]) == 1


async def fake_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[MagicMock, None]:
    chunk = MagicMock()
    chunk.choices[0].delta.content = "hello back"
    yield chunk


async def mock_acompletion(
    *args: Any, **kwargs: Any
) -> AsyncGenerator[MagicMock, None]:
    return fake_stream()


@pytest.mark.asyncio
async def test_chat_streaming(
    async_client: AsyncClient, mock_sparse_index: None
) -> None:
    with (
        patch(
            "docmind_api.main.build_llm_prompt",
            new_callable=AsyncMock,
            return_value={
                "prompt": "mock prompt",
                "chunks": [
                    Chunk(
                        document_id=uuid.uuid4(),
                        source_file="test",
                        text="mock prompt",
                        chunk_index=0,
                    )
                ],
            },
        ),
        patch(
            "docmind_api.main.litellm.acompletion",
            new=AsyncMock(side_effect=mock_acompletion),
        ),
        patch(
            "docmind_api.main.get_current_tenant",
            new_callable=AsyncMock,
            return_value=Tenant(
                id=uuid.uuid4(), name="test", slug="test", created_at=datetime.now()
            ),
        ),
    ):
        request = make_chat_payload(stream=True)
        response = await async_client.post("/chat", json=request)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "hello back" in response.text

    assert "event: retrieved_chunks" in response.text
    assert "event: response" in response.text
    assert "hello back" in response.text
    assert "event: done" in response.text


@pytest.mark.asyncio
async def test_chat_missing_messages_returns_422(async_client: AsyncClient) -> None:
    with patch(
        "docmind_api.main.get_current_tenant",
        return_value=Tenant(
            id=uuid.uuid4(), name="test", slug="test", created_at=datetime.now()
        ),
    ):
        response = await async_client.post(
            "/chat", json={"tenant_id": "tenant-123", "stream": False}
        )

    assert response.status_code == 422
