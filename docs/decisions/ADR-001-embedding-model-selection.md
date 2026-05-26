# ADR-001: Embedding Model Selection

## Status
Accepted

## Context
docmind requires an embedding model for converting document chunks into vectors for similarity search. The choice affects retrieval quality, storage cost, and operational complexity when changing models.

For production, we evaluated OpenAI's embedding models as the primary candidates due to their strong benchmark performance, wide adoption, and LiteLLM compatibility.

## Decision
| Model | Dimensions (default) | Quality | Cost | Typical Use |
|---------|---------:|---------|---------|---------|
| `text-embedding-3-small` | 1536 | Very good | Much cheaper | Most RAG systems, semantic search, recommendations |
| `text-embedding-3-large` | 3072 | Best available | ~6.5× more expensive | High-accuracy retrieval, legal/medical/financial search, large-scale production systems |

OpenAI's benchmark results show:
| Benchmark                          | Small | Large |
| ---------------------------------- | ----- | ----- |
| MIRACL (multilingual retrieval)    | 44.0  | 54.9  |
| MTEB (general embedding benchmark) | 62.3  | 64.6  |

So the large model is better, but not dramatically better for every use case.

We decided to go forward with text-embedding-3-small because it provides strong retrieval quality at much lower cost and storage requirements. We'd switch to text-embedding-3-large only if it is observed that retrieval accuracy is limiting the application.

Note: For the purpose of development we'll be using the free embedding model from ollama, which we'll later replace with the selected text-embedding-3-small when hosting the application

## Consequences
Switching embedding models mid-project requires the following considerations:
- Vectors in Qdrant are tied to the embedding model that produced them
- Changing models requires full collection rebuild and re-ingestion
- Collection metadata will store the embedding model name at creation time. The ingestion worker will assert model name matches before writing vectors to prevent silent mixed-vector corruption.
