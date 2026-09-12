# Operations

## Deployment

```bash
docker compose up -d --build
```

Services: `rabbitmq`, `qdrant`, `postgres`, `redis`, `minio`, `worker`.
Scale the query and background roles independently:

```bash
docker compose up -d --scale worker-query=4
```

(with role-specific compose overrides, or run the image twice with different
`WORKER_QUEUES`).

## Worker roles

| Role | Queues | Notes |
|---|---|---|
| `query` | `query_queue` | latency-sensitive; keep away from OCR spikes |
| `background` | `ingest_queue`, `delete_queue`, `reindex_queue` | heavy parsing/OCR |
| `all` | everything | default for development |

## Startup sequence

1. `configure_logging` (+ tracing if `OTEL_ENDPOINT`).
2. Health server binds `/healthz`, `/readyz`, `/metrics`.
3. Embedding-version warning + **dimension fail-fast** (exits on mismatch).
4. Consumers start; RabbitMQ topology (DLX, retry queues) is declared.

## Health and monitoring

- `GET /healthz` — liveness.
- `GET /readyz` — per-dependency checks (RabbitMQ, Postgres, Qdrant, MinIO)
  with `HEALTH_CHECK_TIMEOUT` budget; 503 with per-check errors.
- `GET /metrics` — Prometheus: `rag_messages_total`,
  `rag_message_processing_seconds`, `rag_vector_search_seconds`,
  `rag_llm_generation_seconds`, `rag_llm_tokens_total`, `rag_llm_cache_total`,
  `rag_llm_truncations_total`, `rag_circuit_breaker_total`.

## Reindexing

```json
{"token": "<admin jwt>", "vector_dimension": null}
```

to `reindex_queue` (admin group required). Blue-green by default — the live
collection is never dropped; the alias flips atomically when the shadow is
complete (see [Reliability](reliability.md#blue-green-reindex)).

## Maintenance

| Task | Command |
|---|---|
| ACL groups | `uv run python scripts/manage_groups.py set <user> g1,g2` |
| Purge soft-deleted docs | `uv run python scripts/purge_deleted_documents.py --days 30 --apply` |
| Index eval corpus | `PYTHONPATH=src uv run python scripts/index_eval_corpus.py` |
| Offline quality report | `PYTHONPATH=src uv run python scripts/run_eval.py --llm-judge` |

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `embedding_dimension_mismatch_fail_fast` at startup | model changed without reindex | restore `EMBEDDING_MODEL` or reindex |
| `physical collection ... blue-green promotion` | legacy deployment with a real collection named like the alias | move/drop it once, then re-run reindex |
| Messages stuck in `<queue>.dlq` | permanent failures (bad input) or exhausted retries | inspect `reason`/`error` in logs, fix and republish |
| `Parser selection failed` in logs | MIME mismatch (renamed binary) | sender must upload the real file type |
| Redis down | caches fail open, locks become in-process, blacklist disabled | restore Redis; no data loss, but duplicate work possible |
