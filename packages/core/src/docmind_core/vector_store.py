import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from docmind_core.models import Chunk, QueryResult
from docmind_core.sparse import chunk_to_sparse_vector

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
        vectors_config={
            "dense": models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )


async def upsert_chunks(
    collection_name: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    bm25: BM25Okapi,
    vocab: dict[str, int],
) -> tuple[BM25Okapi, dict[str, int]]:
    # TODO: validate embedding_model matches collection before insert

    points = []

    for chunk, embedding in zip(chunks, embeddings):
        indices, weights = chunk_to_sparse_vector(chunk.text, bm25, vocab)
        sparse_vector = models.SparseVector(
            indices=indices,
            values=weights,
        )

        points.append(
            models.PointStruct(
                id=str(chunk.id),
                payload={
                    "text": chunk.text,
                    "document_id": str(chunk.document_id),
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "source_file": chunk.source_file,
                },
                vector={
                    "dense": embedding,
                    "sparse": sparse_vector,
                },
            )
        )

    client.upsert(collection_name=collection_name, points=points)

    return (bm25, vocab)


async def search(
    collection_name: str, query_embedding: list[float], top_k: int = 10
) -> list[QueryResult]:
    api_response = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        using="dense",
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
                    source_file=payload.get("source_file"),
                ),
            )
        )

    return response


async def search_hybrid(
    collection_name: str,
    query_embedding: list[float],
    indices: list[int],
    weights: list[float],
    top_k: int = 10,
) -> list[QueryResult]:
    api_response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=query_embedding,
                using="dense",
                limit=top_k,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=indices,
                    values=weights,
                ),
                using="sparse",
                limit=top_k,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
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
                    source_file=payload.get("source_file"),
                ),
            )
        )

    return response
