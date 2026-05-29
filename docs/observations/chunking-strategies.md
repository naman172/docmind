# Observations

## Corpus
Flattening the corpus directory caused silent overwrites on name collisions, which negatively affected source file identification in `queries.json`. Re-downloaded with directory tree preserved.

## Tail Handling in chunk_fixed
The last chunk almost never divides evenly. When the window runs past the end of the text, we take what remains. But how short is too short? A 12-character final chunk may be noise rather than meaningful content. We made this threshold a tunable function parameter (`min_new_chars`) with a default value of 25.

## Chunker Design
Although the primary function of a chunker is to split text according to a defined strategy, all chunkers carry a dependency on document identity (`document_id` and `source_file`). This coupling is worth revisiting when the ingestion pipeline matures.

`chunk_recursive` does not apply overlap between separator-split chunks — overlap only applies at the character-split base case. This is acceptable because separators preserve semantic boundaries by definition. As a consequence, `chunk_recursive` performs better on well-structured documents where paragraph boundaries coincide with topic boundaries, and may underperform on dense prose where context bleeds across paragraphs.

## Embedding Cache
Four parameters together define a unique version of the index:
- `chunker_name`
- `chunk_size`
- `overlap`
- `min_new_chars`

If any one of them changes, the cache is stale. The cache filename encodes all four parameters — if anything changes, the filename changes, the old cache is ignored, and a new one is created automatically.

On every run:
- Build the cache filename from current parameters
- If the file exists → skip embedding and upsert, Qdrant is already populated
- If the file doesn't exist → embed, upsert, then save the cache file

**Known limitation:** cache and Qdrant state can diverge if the Qdrant volume is wiped. If Qdrant loses its data (container restart, volume deletion), the cache file still exists but the collection is empty. The benchmark will skip indexing and query an empty collection, producing silently wrong results. Mitigation: delete cache files manually before re-running.

## Batching: Embedding and Upsert
Both the embedding client and the vector store require batching for any corpus of real size. Sending all chunks in a single request causes timeouts (embedding) and payload size rejections (Qdrant's 32MB limit).

Current batch sizes:
- Embedding: 32 chunks per request (conservative for local Ollama)
- Upsert: 100 chunks per request

Both values are tunable. Too large causes failures; too small causes unnecessary latency. Revisit when moving to OpenAI embeddings in production.

## source_file Payload Drift
The `Chunk` model gained a `source_file` field that was never added to the Qdrant payload on write or read. No error was raised — the field was silently stored as `None` and reconstructed as `None` at query time. The bug was only detected when the first benchmark run showed `Source Match = 0` across all 20 queries. Fix: added `source_file` to the upsert payload and read it back defensively using `.get()` to handle points indexed before the schema change.

**Principle:** be strict on write, lenient on read. Payload schema and model schema must be kept in sync — there is no automatic enforcement.

## Benchmark Results

Corpus: FastAPI documentation (152 files)
Chunk size: 512 characters, overlap: 50, min_new_chars: 25
Queries: 20 verified questions with grounding_truth substrings
Embedding model: nomic-embed-text (Ollama, local)
Top-K: 5

| Chunker         | Source Match | Answer Present | Avg Top Score |
|-----------------|--------------|----------------|---------------|
| chunk_fixed     | 14/20        | 11/20          | 0.751         |
| chunk_recursive | 15/20        | 8/20           | 0.842         |

## Findings

chunk_recursive achieves higher average similarity scores but lower answer
presence than chunk_fixed. Splitting aggressively on semantic boundaries
can separate grounding_truth content from its surrounding context, reducing
the chance it appears in top-K results.

chunk_fixed's larger, consistent chunks are more likely to contain complete
sentences including the answer. The tradeoff is lower semantic coherence
per chunk.

Neither strategy is clearly superior. The right choice depends on corpus
structure — chunk_recursive suits well-structured documents with clear
paragraph boundaries; chunk_fixed suits dense prose where answers span
multiple sentences.

## Next Steps
- Re-run with chunk_size=256 to test whether smaller chunks improve answer presence
- Implement chunk_semantic when embedding client is available in benchmark context
- Run RAGAS faithfulness scores for a more rigorous quality signal
