import time
import uuid
from collections.abc import AsyncGenerator

import litellm
from docmind_core.chunking import chunk_by_tokens
from docmind_core.embeddings import embed_texts
from docmind_core.models import ChatRequest, SyncIngestRequest
from docmind_core.prompts import build_rag_prompt
from docmind_core.vector_store import create_collection, search, upsert_chunks
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def build_llm_prompt(request: ChatRequest) -> str:
    query = request.messages[-1]
    embedding = await embed_texts([query.content])
    context_points = await search(request.collection_name, embedding[0], 5)
    context_chunks = [point.chunk.text for point in context_points]
    return build_rag_prompt(context_chunks)


async def stream_chat(request: ChatRequest) -> AsyncGenerator[str, None]:
    query = request.messages[-1]
    prompt = await build_llm_prompt(request)

    response = await litellm.acompletion(
        model="ollama/llama3.2",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query.content},
        ],
        stream=True,
    )

    async for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content


# returns StreamingResponse or JSONResponse depending on request.stream
@app.post("/chat", response_model=None)
async def chat(request: ChatRequest) -> Response:
    if request.stream:
        return StreamingResponse(stream_chat(request), media_type="text/event-stream")

    query = request.messages[-1]
    prompt = await build_llm_prompt(request)
    print(prompt)
    response = await litellm.acompletion(
        model="ollama/llama3.2",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query.content},
        ],
    )

    content = response.choices[0].message.content
    return JSONResponse({"response": content})


@app.post("/ingest")
async def ingest(request: SyncIngestRequest) -> dict[str, str]:
    text = request.text
    collection_name = request.collection_name

    start_time = time.perf_counter()
    doc_id = uuid.uuid4()
    chunks = chunk_by_tokens(text, doc_id)
    embeddings = await embed_texts([chunk.text for chunk in chunks])

    await create_collection(collection_name)
    await upsert_chunks(collection_name, chunks, embeddings)
    end_time = time.perf_counter()

    latency_in_ms = (end_time - start_time) * 1000

    return {"chunks_indexed": str(len(chunks)), "latency_ms": str(latency_in_ms)}
