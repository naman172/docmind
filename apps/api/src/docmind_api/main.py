import json
import os
import re
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
    scroll_chunks,
    search_hybrid,
    upsert_chunks,
)
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from langsmith import traceable
from rank_bm25 import BM25Okapi

from docmind_api.auth import (
    create_access_token,
    get_current_tenant,
    hash_password,
    verify_password,
)
from docmind_api.db import (
    close_pool,
    create_tenant,
    get_pool,
    get_tenant_by_slug,
    init_pool,
)
from docmind_api.models import LlmPromptContext, LoginRequest, RegisterRequest, Tenant

_sparse_indexes: dict[str, tuple[BM25Okapi, dict[str, int]]] = {}

load_dotenv()
SPARSE_INDEX_DIR = os.environ.get("SPARSE_INDEX_DIR", "data/sparse_indexes")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("starting up")
    for file in Path(SPARSE_INDEX_DIR).rglob("*.pkl"):
        _sparse_indexes[file.stem] = load_sparse_index(file.absolute())
    await init_pool()
    yield
    await close_pool()
    print("shutting down")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    pool = get_pool()
    await pool.fetchval("SELECT 1")
    return {"status": "ok", "db": "ok"}


async def build_llm_prompt(
    request: ChatRequest, collection_name: str
) -> LlmPromptContext:
    query = request.messages[-1]
    embedding = await embed_texts([query.content])
    bm25, vocab = _sparse_indexes[collection_name]
    indices, weights = chunk_to_sparse_vector(query.content, bm25, vocab)
    context_points = await search_hybrid(
        collection_name, embedding[0], indices, weights
    )
    context_chunks = [point.chunk.text for point in context_points]
    return {
        "prompt": build_rag_prompt(context_chunks),
        "chunks": [point.chunk for point in context_points],
    }


async def stream_chat(
    request: ChatRequest, collection_name: str
) -> AsyncGenerator[str, None]:
    query = request.messages[-1]
    promptContext = await build_llm_prompt(request, collection_name)

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


async def non_stream_chat(request: ChatRequest, collection_name: str) -> JSONResponse:
    query = request.messages[-1]
    promptContext = await build_llm_prompt(request, collection_name)

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
async def chat(
    request: ChatRequest, tenant: Tenant = Depends(get_current_tenant)
) -> Response:
    collection_name = tenant.slug
    if collection_name not in _sparse_indexes:
        raise HTTPException(
            status_code=400, detail="Collection not found. Ingest documents first."
        )

    if request.stream:
        return StreamingResponse(
            stream_chat(request, collection_name), media_type="text/event-stream"
        )

    return await non_stream_chat(request, collection_name)


async def embed_in_batches(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings.extend(await embed_texts(batch))
    return embeddings


@app.post("/ingest")
async def ingest(
    request: SyncIngestRequest, tenant: Tenant = Depends(get_current_tenant)
) -> dict[str, str]:
    text = request.text
    collection_name = tenant.slug

    start_time = time.perf_counter()
    doc_id = uuid.uuid4()
    chunks = chunk_fixed(text, doc_id, "user_input")
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Text too short to index. Please provide more content.",
        )

    embeddings = await embed_in_batches([chunk.text for chunk in chunks])

    all_chunks = await scroll_chunks(collection_name)
    bm25, vocab = build_sparse_index(all_chunks + chunks)
    _sparse_indexes[collection_name] = (bm25, vocab)

    await create_collection(collection_name)
    await upsert_chunks(collection_name, chunks, embeddings, bm25, vocab)
    end_time = time.perf_counter()

    latency_in_ms = (end_time - start_time) * 1000

    return {"chunks_indexed": str(len(chunks)), "latency_ms": str(latency_in_ms)}


@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register(registerRequest: RegisterRequest) -> dict[str, str]:
    name = registerRequest.name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", name)
    hashed_password = hash_password(registerRequest.password)

    try:
        await create_tenant(name, slug, hashed_password)
    except ValueError:
        raise HTTPException(
            409, "This name is already registered please use a different name"
        )

    return {"message": "Created", "data": slug}


@app.post("/auth/token", status_code=status.HTTP_200_OK)
async def login(loginRequest: LoginRequest) -> dict[str, str]:
    result = await get_tenant_by_slug(loginRequest.username)
    if result is None:
        raise HTTPException(401, "Invalid credentials")

    tenant, hashed_password = result

    if not verify_password(loginRequest.password, hashed_password):
        raise HTTPException(401, "Invalid credentials")

    token = create_access_token(tenant.id)
    return {"message": "Success", "data": token}
