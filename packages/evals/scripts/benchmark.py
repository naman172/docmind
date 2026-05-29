import asyncio
import json
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path

from docmind_core.chunking import chunk_fixed, chunk_recursive
from docmind_core.embeddings import embed_texts
from docmind_core.models import Chunk
from docmind_core.vector_store import create_collection, search, upsert_chunks
from tabulate import tabulate  # type: ignore[import-untyped]


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
    batch_size: int = 100,
) -> None:
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]
        await upsert_chunks(collection_name, batch_chunks, batch_embeddings)


async def index_chunks(
    chunks: list[Chunk], collection_name: str, cache_path: Path
) -> None:
    if cache_path.exists():
        return

    raw_chunks = [chunk.text for chunk in chunks]
    embeddings = await embed_in_batches(raw_chunks)
    cache_path.write_text(json.dumps({"chunks": raw_chunks, "embeddings": embeddings}))
    print(f"embeddings for {collection_name} succcessfully created")

    await create_collection(collection_name)
    await upsert_in_batches(collection_name, chunks, embeddings)


async def run_query(
    question: str,
    expected_source: str,
    grounding_truth: str,
    collection_name: str,
    top_k: int = 5,
) -> dict[str, bool | float]:
    query_embedding = (await embed_texts([question]))[0]
    response = await search(collection_name, query_embedding, top_k)

    return {
        "source_match": any(
            expected_source == point.chunk.source_file for point in response
        ),
        "answer_present": any(
            grounding_truth in point.chunk.text for point in response
        ),
        "top_score": response[0].score if response else 0.0,
    }


async def main() -> None:
    SCRIPT_DIR = Path(__file__).parent
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent

    corpus_dir = REPO_ROOT / "packages/evals/corpus"
    queries_path = REPO_ROOT / "packages/evals/queries.json"
    cache_dir = REPO_ROOT / "packages/evals/cache"
    cache_dir.mkdir(exist_ok=True)

    queries = json.loads(queries_path.read_text())
    corpus = load_corpus(corpus_dir)

    chunkers = {
        "chunk_fixed": lambda text, doc_id, source: chunk_fixed(text, doc_id, source),
        "chunk_recursive": lambda text, doc_id, source: chunk_recursive(
            text, doc_id, source
        ),
    }

    headers = ["Chunker", "Source Match", "Answer Present", "Avg Top Score"]
    results = []

    for chunker_name, chunker_fn in chunkers.items():
        chunks = build_chunks(corpus, chunker_fn)
        await index_chunks(
            chunks, chunker_name, cache_dir / f"cache_{chunker_name}_512_50_25.json"
        )
        print(f"upsert for {chunker_name} succcessfully completed")

        query_responses = []
        for query in queries:
            query_responses.append(
                await run_query(
                    query["question"],
                    query["source_file"],
                    query["grounding_truth"],
                    chunker_name,
                )
            )

        chunker_result = [
            chunker_name,
            sum(1 for query_result in query_responses if query_result["source_match"]),
            sum(
                1 for query_result in query_responses if query_result["answer_present"]
            ),
            sum(query_result["top_score"] for query_result in query_responses)
            / len(queries),
        ]

        results.append(chunker_result)

    print(tabulate(results, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    asyncio.run(main())
