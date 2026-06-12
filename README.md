# docmind

> Multi-tenant RAG-as-a-service — upload documents, get answers, measure everything.

**[Live Demo](https://docmind.yourdomain.com)** · **[Architecture](docs/architecture.md)** · **[5-min Loom walkthrough](#demo)**

![CI](https://github.com/naman172/docmind/actions/workflows/ci.yml/badge.svg)

---

## Current State
Synchronous RAG pipeline with hybrid retrieval (BM25 sparse + dense vectors, RRF fusion).
Supports document ingestion and query answering via streaming and non-streaming endpoints.
RAGAS faithfulness baseline: 0.8778 (partial run — Groq free tier rate limit).
Known limitation: BM25 index is in-memory, rebuilt from pickle on startup — stale if documents are ingested separately.

---

## What it is

docmind is a production-grade, multi-tenant platform that lets teams upload their own document corpora and query them through a streaming chat interface backed by a retrieval-augmented generation pipeline.

It is not a prototype. It runs on Kubernetes, processes documents asynchronously through Kafka, isolates tenant data at the database and vector layer, and blocks retrieval regressions in CI before they reach production.

---

## Metrics

| Metric | Value |
|---|---|
| P95 end-to-end query latency | TBD |
| P95 retrieval latency | TBD |
| RAGAS faithfulness (eval set) | 0.8778 (partial — 9/20 samples, Groq rate limit) |
| RAGAS answer relevancy (eval set) | TBD |
| Ingestion throughput | TBD docs/min |
| Cost per query | TBD |
| Test coverage (core packages) | TBD % |

> Metrics are measured against a 30-question evaluation set and a k6 load test at 50 concurrent users. See [docs/evals.md](docs/evals.md) for methodology.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           Tenant boundary                            │
│                                                                      │
│   React dashboard ──► FastAPI (api) ──► Qdrant (tenant namespace)   │
│                             │                    │                   │
│                             └──► Postgres (RLS)  │                   │
│                                                  │                   │
│   Upload ──► S3 ──► Kafka topic ──► Worker ──────┘                   │
│                  (ingestion.requested)                               │
└──────────────────────────────────────────────────────────────────────┘
```

**Full architecture diagram:** [docs/architecture.md](docs/architecture.md)

### Services

| Service | Role |
|---|---|
| `apps/api` | FastAPI — serves chat, ingestion status, tenant management |
| `apps/worker` | Kafka consumer — chunks, embeds, indexes documents |
| `apps/web` | React — tenant dashboard (documents, queries, cost) |
| Postgres | Metadata store with row-level security per tenant |
| Qdrant | Vector store with per-tenant collection isolation |
| Kafka | Async ingestion backbone |
| S3 | Raw document storage |
| LangSmith | LLM trace observability |
| OpenTelemetry | Distributed tracing across all services |

### Design decisions

See [docs/decisions/](docs/decisions/) for Architecture Decision Records. Key decisions:

- **Kafka for ingestion, not a task queue.** Document processing is a streaming problem — Kafka gives us replay, consumer lag metrics, and horizontal scaling without coupling the API to the worker.
- **Per-tenant Qdrant collections, not a shared index.** Namespace isolation in a shared index is a footgun; separate collections are operationally clean and the performance overhead is acceptable at this scale.
- **Postgres RLS over application-layer filtering.** Application-layer tenant filtering fails silently when a query is missing a WHERE clause. RLS fails loudly.
- **LiteLLM as the LLM router.** Model-agnostic from the start. Lets us swap providers per tenant or per cost tier without touching application code.
- **RAGAS in CI, not just monitoring.** Retrieval quality regressions are invisible in application metrics. Catching them pre-merge is worth the CI runtime cost.

---

## Quickstart

```bash
git clone https://github.com/you/docmind
cd docmind
cp .env.example .env        # add your LLM API key
docker compose up
```

Open `http://localhost:3000`. Sign up, upload a PDF, start asking questions.

The compose stack runs: api, worker, postgres, qdrant, kafka (via Redpanda), redpanda-console.

---

## Demo

[![Loom walkthrough thumbnail](docs/assets/loom-thumbnail.png)](https://loom.com/share/YOUR_LOOM_ID)

The walkthrough shows:
1. Tenant sign-up and document upload
2. Async ingestion progress (Kafka consumer lag)
3. Hybrid retrieval + reranking on a real corpus
4. Evaluation dashboard with faithfulness scores
5. LangSmith trace for a single query (retrieval → rerank → generation)
6. K8s worker pods scaling under load

---

## Tenant isolation model

Each tenant gets:
- A dedicated Qdrant collection (`tenant_{id}_documents`)
- A Postgres schema with RLS policies enforced at the database level
- An S3 prefix (`tenants/{id}/`)
- A per-tenant cost budget enforced at the LiteLLM routing layer

**Isolation test:** `docs/isolation-test.md` documents the pen-test procedure used to verify cross-tenant data cannot be accessed, including the SQL and API requests used and their expected (and observed) failure modes.

---

## Evaluation

The evaluation suite lives in `packages/evals/` and runs on every PR that touches:
- Any prompt template
- Any retrieval parameter (chunk size, top-k, rerank threshold)
- The embedding model

A merge is blocked if faithfulness drops below **0.82** or answer relevancy drops below **0.78** relative to the baseline dataset.

```bash
# Run evals locally
uv run pytest packages/evals/ -v
```

See [docs/evals.md](docs/evals.md) for dataset construction methodology and the choice of thresholds.

---

## Deployment

### Local (Docker Compose)
```bash
docker compose up
```

### Kubernetes (kind — local cluster)
```bash
kind create cluster --name docmind
helm upgrade --install docmind-api infra/helm/api -f infra/helm/api/values.dev.yaml
helm upgrade --install docmind-worker infra/helm/worker -f infra/helm/worker/values.dev.yaml
```

### Production (AWS EKS)
```bash
# One-time cluster setup
eksctl create cluster -f infra/aws/cluster.yaml

# Deploy
helm upgrade --install docmind-api infra/helm/api -f infra/helm/api/values.prod.yaml
helm upgrade --install docmind-worker infra/helm/worker -f infra/helm/worker/values.prod.yaml
```

Workers scale via KEDA on Kafka consumer lag — not CPU. At zero pending documents, workers scale to zero. See [docs/k8s.md](docs/k8s.md).

---

## Cost model

**Target cost at idle:** <$60/month (EKS cluster, RDS t3.micro, Qdrant on-cluster)
**Target cost under demo load:** <$150/month

Per-query LLM cost is bounded by the LiteLLM routing config in `packages/core/llm.py`. The default routes to `claude-haiku-4-5` for retrieval-augmented answers and `claude-sonnet-4-6` only when explicitly requested by the tenant configuration.

Per-tenant cost is tracked in Postgres and surfaced in the dashboard. Tenants can set a monthly token budget; requests above the budget return a 402 with a clear error.

See [docs/cost.md](docs/cost.md) for the full breakdown.

---

## Scaling

See [docs/scaling.md](docs/scaling.md). The short version:

| Bottleneck | Mitigation |
|---|---|
| Qdrant read throughput | Replicas per collection, then sharding by tenant_id |
| Embedding throughput | Batched embedding calls, horizontal worker scaling |
| LLM cost | Semantic caching (Redis), per-tenant budget caps |
| Postgres connections | PgBouncer in transaction mode |

The current architecture is designed to handle ~100 tenants and ~10K queries/day comfortably within the cost target. What changes at 10× that scale is documented in the scaling doc.

---

## Project structure

```
docmind/
├── apps/
│   ├── api/              # FastAPI service
│   ├── worker/           # Kafka consumer / ingestion worker
│   └── web/              # React tenant dashboard
├── packages/
│   ├── core/             # Shared models, LLM client, retriever
│   └── evals/            # RAGAS eval suite, LLM-as-judge pipeline
├── infra/
│   ├── helm/             # Per-service Helm charts
│   ├── docker/           # Dockerfiles + compose
│   └── aws/              # eksctl cluster config, CDK stacks
├── docs/
│   ├── architecture.md   # Full architecture diagram + narrative
│   ├── decisions/        # ADRs
│   ├── evals.md          # Eval methodology
│   ├── k8s.md            # Kubernetes design notes
│   ├── scaling.md        # Scaling analysis
│   ├── cost.md           # Cost breakdown
│   ├── runbook.md        # Ops runbook
│   └── isolation-test.md # Tenant isolation test procedure
├── .github/
│   └── workflows/        # CI (lint, test, eval, deploy)
├── .env.example
├── docker-compose.yml
├── pyproject.toml
├── CHANGELOG.md
└── README.md
```

---

## Development

```bash
# Install dependencies (Python)
uv sync

# Lint and type-check
uv run ruff check .
uv run mypy .

# Run tests
uv run pytest apps/api/tests/ packages/core/tests/ -v

# Run integration tests (requires Docker)
uv run pytest tests/integration/ -v --timeout=60

# Run evals
uv run pytest packages/evals/ -v
```

Pre-commit hooks run ruff and mypy on every commit. Install with:
```bash
pre-commit install
```

---

## Observability

- **LangSmith:** LLM traces, per-call cost, prompt version, token counts. Every query has a `trace_id` propagated from the React frontend through the API through the LLM call.
- **OpenTelemetry:** Distributed traces exported to Honeycomb (free tier). Span IDs tied to `trace_id`.
- **Structured logs:** Every log line includes `trace_id`, `tenant_id`, `service`, `level`. JSON format in production.

The five numbers that matter are on the ops dashboard: P95 retrieval latency, P95 generation latency, Kafka consumer lag, error rate, and cost-per-query.

---

## Tech stack

**Backend:** Python 3.12, FastAPI, Pydantic v2, LiteLLM, LlamaIndex (retrieval patterns), aiokafka
**Storage:** Postgres 16, Qdrant, S3
**Messaging:** Kafka (Redpanda in local dev, MSK or self-hosted in prod)
**Evals:** RAGAS, LLM-as-judge (custom), LangSmith
**Frontend:** React 18, TypeScript
**Infra:** Docker, Kubernetes (EKS), Helm, KEDA, GitHub Actions
**Observability:** OpenTelemetry, Honeycomb, LangSmith, structlog

---

## What I'd do differently

*(This section gets filled in at the end of the project — honest post-mortems are more credible than polished narratives.)*

---

## License

MIT
