import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from docmind_core.models import Chunk, QueryResult

load_dotenv()

QDRANT_BASE_URL = os.environ.get("QDRANT_BASE_URL", "http://localhost:6333")
MODEL_VECTOR_DIMENSION = int(os.environ.get("MODEL_VECTOR_DIMENSION", 768))
client = QdrantClient(url=QDRANT_BASE_URL)


async def create_collection(
    collection_name: str, vector_size: int = MODEL_VECTOR_DIMENSION
) -> None:
    if client.collection_exists(collection_name=collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size, distance=models.Distance.COSINE
        ),
    )


async def upsert_chunks(
    collection_name: str, chunks: list[Chunk], embeddings: list[list[float]]
) -> None:
    # TODO: validate embedding_model matches collection before insert

    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=str(chunk.id),
                payload={
                    "text": chunk.text,
                    "document_id": str(chunk.document_id),
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                },
                vector=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ],
    )


async def search(
    collection_name: str, query_embedding: list[float], top_k: int = 10
) -> list[QueryResult]:
    api_response = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    response: list[QueryResult] = []

    for point in api_response.points:
        payload = point.payload

        if payload is None:
            continue

        response.append(
            QueryResult(
                score=point.score,
                chunk=Chunk(
                    id=uuid.UUID(str(point.id)),
                    text=payload["text"],
                    document_id=payload["document_id"],
                    chunk_index=payload["chunk_index"],
                    token_count=payload["token_count"],
                ),
            )
        )

    return response
