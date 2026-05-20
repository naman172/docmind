from collections.abc import AsyncGenerator, Mapping

import litellm
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from packages.core.models import ChatRequest

app = FastAPI()


@app.get("/health")
async def health() -> Mapping[str, str]:
    return {"status": "ok"}


async def stream_chat(request: ChatRequest) -> AsyncGenerator[str, None]:
    response = await litellm.acompletion(
        model="ollama/llama3.2",
        messages=[
            {"role": m.role.value, "content": m.content} for m in request.messages
        ],
        stream=True,
    )

    async for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content


@app.post("/chat", response_model=None)
async def chat(request: ChatRequest) -> Mapping[str, str] | StreamingResponse:
    if request.stream:
        return StreamingResponse(stream_chat(request), media_type="text/event-stream")

    response = await litellm.acompletion(
        model="ollama/llama3.2",
        messages=[
            {"role": m.role.value, "content": m.content} for m in request.messages
        ],
    )

    content = response.choices[0].message.content
    return {"response": content}
