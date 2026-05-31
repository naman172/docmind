## Pre-implementation investigation

Queried top-20 for 3 chunk_recursive failures (high top score, Answer Present = False).

Query 1: [Where can await be used in FastAPI functions?] — grounding_truth not found in top 20
Query 2: [How are query parameters interpreted in FastAPI?] — grounding_truth not found in top 20
Query 3: [What does FastAPI provide from Python type declarations for request bodies?] — grounding_truth found at rank 8

## Benchmark Results (Hybrid Search: BM25 + Dense + RRF)

Corpus: FastAPI documentation (152 files)
Chunk size: 512 characters, overlap: 50, min_new_chars: 25
Queries: 20 verified questions with grounding_truth substrings
Embedding model: nomic-embed-text (Ollama, local)
Retrieval: Qdrant hybrid search (BM25 sparse + dense vectors, RRF fusion)
Top-K: 5

| Chunker         | Source Match | Answer Present | Avg Top Score |
|-----------------|--------------|----------------|---------------|
| chunk_fixed     | 15           | 14             | 0.643333      |
| chunk_recursive | 15           | 9              | 0.588333      |
| chunk_markdown  | 16           | 12             | 0.604167      |

Note: Avg Top Score is not comparable to dense-only results — RRF produces
rank-based fusion scores, not cosine similarities.

## Key Finding
Hybrid search improved chunk_fixed Answer Present from 11/20 (dense-only) to
14/20 (+27%). BM25 recovered exact-term failures as hypothesised — queries
containing exact technical terms (function names, syntax) were failing dense
retrieval but are now retrieved correctly via sparse matching.
chunk_markdown achieved highest Source Match (16/20); structure-aware chunking
improves source attribution even when Answer Present is not highest.

## Next Steps
- Cohere reranking deferred — 14/20 may not justify added latency and cost
- Integrate hybrid retrieval into /ingest and /chat endpoints
