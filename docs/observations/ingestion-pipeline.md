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

## Prompt Failure
In the initial version of the prompt template, we included both the retrieved context and the user query inside a system message before sending the actual user message to the LLM. We observed that this structure confused the model and produced invalid or inconsistent responses.

To resolve this, we simplified the prompt by removing the user query from the system message and limiting it to only the retrieved context. The user query was then sent separately as the actual user message, which resulted in more accurate responses.

## Open questions
- What similarity score should we expect for a direct semantic match? Need a larger corpus to establish a baseline.
