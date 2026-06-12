import asyncio
import json
import os
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path

from docmind_core.chunking import chunk_fixed
from docmind_core.embeddings import embed_texts
from docmind_core.models import Chunk
from docmind_core.sparse import (
    build_sparse_index,
    load_sparse_index,
    save_sparse_index,
)
from docmind_core.vector_store import (
    create_collection,
    upsert_chunks,
)
from docmind_evals.constants import RAGAS_COLLECTION_NAME
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()
SPARSE_INDEX_DIR = os.environ.get("SPARSE_INDEX_DIR", "packages/evals/cache")


def load_corpus(corpus_dir: Path) -> list[tuple[str, str]]:
    return [
        (str(item.relative_to(corpus_dir)), item.read_text())
        for item in corpus_dir.rglob("*.md")
    ]


def build_chunks(
    corpus: list[tuple[str, str]],
    chunker: Callable[[str, uuid.UUID, str], Iterable[Chunk]],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source_file, text in corpus:
        chunks.extend(chunker(text, uuid.uuid4(), source_file))

    return chunks


async def embed_in_batches(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = await embed_texts(batch)
        embeddings.extend(batch_embeddings)
    return embeddings


async def upsert_in_batches(
    collection_name: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    bm25: BM25Okapi,
    vocab: dict[str, int],
    batch_size: int = 100,
) -> None:
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]
        await upsert_chunks(
            collection_name, batch_chunks, batch_embeddings, bm25, vocab
        )


async def index_chunks(
    chunks: list[Chunk], collection_name: str, cache_path: Path
) -> tuple[BM25Okapi, dict[str, int]]:
    sparse_path = cache_path.with_suffix(".pkl")

    if cache_path.exists():
        return load_sparse_index(sparse_path)

    raw_chunks = [chunk.text for chunk in chunks]
    embeddings = await embed_in_batches(raw_chunks)
    cache_path.write_text(json.dumps({"chunks": raw_chunks, "embeddings": embeddings}))

    bm25, vocab = build_sparse_index(chunks)
    save_sparse_index(bm25, vocab, sparse_path)

    print(f"embeddings & vocab for {collection_name} succcessfully created")

    await create_collection(collection_name)
    await upsert_in_batches(collection_name, chunks, embeddings, bm25, vocab)

    return (bm25, vocab)


async def main() -> None:
    SCRIPT_DIR = Path(__file__).parent
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent

    corpus_dir = REPO_ROOT / "packages/evals/corpus"
    cache_dir = REPO_ROOT / SPARSE_INDEX_DIR
    cache_dir.mkdir(exist_ok=True)

    corpus = load_corpus(corpus_dir)

    chunks = build_chunks(corpus, chunk_fixed)
    await index_chunks(
        chunks, RAGAS_COLLECTION_NAME, cache_dir / f"{RAGAS_COLLECTION_NAME}.json"
    )

    print("upsert succcessfully completed")


if __name__ == "__main__":
    asyncio.run(main())
