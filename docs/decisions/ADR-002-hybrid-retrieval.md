# ADR-002: Hybrid Retrieval

## Status
Accepted

## Context
Benchmark showed 11/20 answer presence with chunk_fixed dense-only; 2 of 3 investigated failures had grounding_truth absent from top 20 despite high similarity scores; hypothesis is exact technical terms are being matched semantically rather than literally

## Decision
Add BM25 sparse retrieval alongside dense, combine with RRF, add Cohere rerank on top-20 candidates


## Consequences
BM25 index must be rebuilt when new documents are ingested; Cohere Rerank adds latency and API cost per query; expected improvement on exact-term queries
