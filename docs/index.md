# Documentation Index

| Page | Contents |
|---|---|
| [Architecture](architecture.md) | Layers, ports & adapters, request flow, DI composition |
| [Configuration](configuration.md) | Every environment variable, grouped and explained |
| [Retrieval](retrieval.md) | Indexing pipeline, chunking, hybrid search, generation guards |
| [Agentic Retrieval](agentic-retrieval.md) | Corrective RAG loop: grader, planner, sub-query fusion |
| [GraphRAG](graphrag.md) | Entity graph, extraction, query expansion, optional backends |
| [Reliability](reliability.md) | Retry/DLQ, idempotency, locks, circuit breaker, blue-green reindex |
| [Security](security.md) | JWT lifecycle & rotation, ACL enforcement, PII redaction |
| [Operations](operations.md) | Deployment, worker roles, maintenance scripts, troubleshooting |
| [Evaluation](evaluation.md) | Golden dataset, corpus, metrics, LLM-as-a-judge, A/B runs |
| [Development](development.md) | Test markers, linting/typing, how to add a feature end-to-end |

## Where to start

- **Run it** — [Quick Start](../README.md#quick-start) in the README.
- **Understand the design** — [Architecture](architecture.md).
- **Tune retrieval quality** — [Retrieval](retrieval.md), then measure with
  [Evaluation](evaluation.md).
- **Operate in production** — [Operations](operations.md) and
  [Reliability](reliability.md).

## Conventions used across these pages

- Setting names are written as environment variables (`SEARCH_HYBRID`); every
  one of them is a field of `src/config.py::Settings`.
- Queue names: `ingest_queue`, `query_queue`, `delete_queue`, `reindex_queue`.
- "Active collection" means the Qdrant collection the public alias currently
  points to (see [Reliability](reliability.md#blue-green-reindex)).
