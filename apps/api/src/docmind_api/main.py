import json
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import litellm
from docmind_core.chunking import chunk_fixed
from docmind_core.embeddings import embed_texts
from docmind_core.models import ChatRequest, SyncIngestRequest
from docmind_core.prompts import build_rag_prompt
from docmind_core.sparse import (
    build_sparse_index,
    chunk_to_sparse_vector,
    load_sparse_index,
)
from docmind_core.vector_store import (
    create_collection,
    search_hybrid,
    upsert_chunks,
)
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
from langsmith import traceable
from rank_bm25 import BM25Okapi

from docmind_api.models import LlmPromptContext

_sparse_indexes: dict[str, tuple[BM25Okapi, dict[str, int]]] = {}

load_dotenv()
SPARSE_INDEX_DIR = os.environ.get("SPARSE_INDEX_DIR", "data/sparse_indexes")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("starting up")
    for file in Path(SPARSE_INDEX_DIR).rglob("*.pkl"):
        _sparse_indexes[file.stem] = load_sparse_index(file.absolute())
    yield
    print("shutting down")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def build_llm_prompt(request: ChatRequest) -> LlmPromptContext:
    query = request.messages[-1]
    embedding = await embed_texts([query.content])
    bm25, vocab = _sparse_indexes[request.collection_name]
    indices, weights = chunk_to_sparse_vector(query.content, bm25, vocab)
    context_points = await search_hybrid(
        request.collection_name, embedding[0], indices, weights
    )
    context_chunks = [point.chunk.text for point in context_points]
    return {
        "prompt": build_rag_prompt(context_chunks),
        "chunks": [point.chunk for point in context_points],
    }


async def stream_chat(request: ChatRequest) -> AsyncGenerator[str, None]:
    query = request.messages[-1]
    promptContext = await build_llm_prompt(request)

    yield (
        f"event: retrieved_chunks\n"
        f"data: {
            json.dumps([c.model_dump_json() for c in promptContext['chunks']])
        }\n\n"
    )

    response = await litellm.acompletion(
        model="ollama/llama3.2",
        messages=[
            {"role": "system", "content": promptContext["prompt"]},
            {"role": "user", "content": query.content},
        ],
        stream=True,
    )

    async for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield (f"event: response\ndata: {content}\n\n")

    yield "event: done\ndata: {}\n\n"


@traceable(name="llm-generation")
async def generate_response(prompt: str, query: str) -> str:
    response = await litellm.acompletion(
        model="ollama/llama3.2",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
    )

    content: str = response.choices[0].message.content
    return content


async def non_stream_chat(request: ChatRequest) -> JSONResponse:
    query = request.messages[-1]
    promptContext = await build_llm_prompt(request)

    content = await generate_response(promptContext["prompt"], query.content)

    return JSONResponse(
        {
            "response": content,
            "retrieved_chunks": [
                chunk.model_dump(mode="json") for chunk in promptContext["chunks"]
            ],
        }
    )


# returns StreamingResponse or JSONResponse depending on request.stream
@app.post("/chat", response_model=None)
async def chat(request: ChatRequest) -> Response:
    if request.collection_name not in _sparse_indexes:
        raise HTTPException(
            status_code=400, detail="Collection not found. Ingest documents first."
        )

    if request.stream:
        return StreamingResponse(stream_chat(request), media_type="text/event-stream")

    return await non_stream_chat(request)


async def embed_in_batches(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings.extend(await embed_texts(batch))
    return embeddings


@app.post("/ingest")
async def ingest(request: SyncIngestRequest) -> dict[str, str]:
    text = request.text
    collection_name = request.collection_name

    start_time = time.perf_counter()
    doc_id = uuid.uuid4()
    chunks = chunk_fixed(text, doc_id, "user_input")
    embeddings = await embed_in_batches([chunk.text for chunk in chunks])

    # BM25 index is built only from the current ingest request, not the full collection.
    bm25, vocab = build_sparse_index(chunks)
    _sparse_indexes[collection_name] = (bm25, vocab)

    await create_collection(collection_name)
    await upsert_chunks(collection_name, chunks, embeddings, bm25, vocab)
    end_time = time.perf_counter()

    latency_in_ms = (end_time - start_time) * 1000

    return {"chunks_indexed": str(len(chunks)), "latency_ms": str(latency_in_ms)}
