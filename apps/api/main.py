from collections.abc import Mapping

import litellm
from fastapi import FastAPI

from packages.core.models import ChatRequest

app = FastAPI()


@app.get("/health")
async def health() -> Mapping[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest) -> Mapping[str, str]:
    response = await litellm.acompletion(
        model="ollama/llama3.2",
        messages=[
            {"role": m.role.value, "content": m.content} for m in request.messages
        ],
    )

    return {"response": response.choices[0].message.content}
