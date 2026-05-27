# Week 3 Observations

## Retrieval quality
- First end-to-end test: query "what is FastAPI?" against a 3-sentence corpus
- Cosine similarity score: 0.654 — lower than expected for a direct match
- Hypothesis: short corpus with overlapping chunks dilutes signal. Retest with a longer, varied document.

## Idempotency gap
- Chunk IDs are currently random (uuid4) — re-running ingestion creates duplicate points in Qdrant instead of overwriting
- Fix: chunk IDs must be deterministic, derived from document_id + chunk_index
- Will address when building async ingestion pipeline
- Will become ADR when implemented

## Open questions
- What similarity score should we expect for a direct semantic match? Need a larger corpus to establish a baseline.
