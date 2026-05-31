# ADR-002: Hybrid Retrieval

## Status
Accepted

## Context
Benchmark showed 11/20 answer presence with chunk_fixed dense-only; 2 of 3 investigated failures had grounding_truth absent from top 20 despite high similarity scores; hypothesis is exact technical terms are being matched semantically rather than literally

## Decision
Add BM25 sparse retrieval alongside dense, combine with RRF, add Cohere rerank on top-20 candidates


## Consequences
- BM25 index must be rebuilt when new documents are ingested; persisted as
  a pickle file alongside the embedding cache
- Cohere Rerank adds latency and API cost per query (deferred — current
  hybrid results at 14/20 may not justify the additional complexity)
- Measured improvement: dense-only 11/20 → hybrid 14/20 Answer Present
  on chunk_fixed (+27%)
- Avg Top Score not comparable across dense-only and hybrid runs due to
  RRF rank-based scoring vs cosine similarity
- Collection schema change required: existing Qdrant collections must be
  dropped and re-indexed when adding sparse vectors

## Deferred
Cohere reranking deferred for the following reasons:
- 6 remaining failures after hybrid search are likely indexing problems,
  not ranking problems — reranking top-20 candidates cannot recover chunks
  absent from the index
- Adds ~200-400ms latency and per-query API cost without measured benefit
- Decision criteria for adding reranking: confirm that failing queries
  have grounding_truth present in top-20 results but absent from top-5;
  if grounding_truth is not in top-20 at all, reranking cannot help and
  the problem is in the index, not the ranking
