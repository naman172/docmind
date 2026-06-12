# Faithfulness Evaluation

A faithfulness score of **0.8778** was obtained during a 20-sample RAGAS evaluation run. This indicates that, for the samples successfully evaluated, approximately **88% of the generated claims were supported by the retrieved context**, suggesting strong grounding performance.

However, the evaluation encountered multiple **Groq free-tier token-per-day (TPD) rate limit errors** after approximately half of the samples had been processed. Several evaluation jobs failed with HTTP 429 responses, and one job timed out. RAGAS still produced a final score by aggregating the results from the samples that completed successfully.

As a result, the reported faithfulness score should be interpreted as a **partial-run result rather than a complete 20-sample evaluation**. An earlier 5-sample evaluation produced a score of **0.7633**, while this run produced **0.8778**. The difference between these results suggests that additional samples and a rate-limit-free evaluation environment are needed to obtain a more stable and representative faithfulness measurement.


## Configuration

- Corpus: FastAPI docs (152 files)
- Chunker: chunk_fixed (512 chars, 50 overlap)
- Retriever: Hybrid BM25 + dense, RRF fusion, top-10
- Answer model: llama3.2 (Ollama local)
- Judge model: llama-3.3-70b-versatile (Groq)
- Dataset: 20 questions, fastapi_eval.json
