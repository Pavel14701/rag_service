# Reliability

How the service keeps data correct and stays available under failures.

## Error taxonomy

| Exception | Meaning | Transport behavior |
|---|---|---|
| `TransientError` | infrastructure hiccup, concurrent work | retry with backoff (5s / 30s / 120s) |
| `PermanentError` | invalid input, poison message | straight to the DLQ, no retry |
| `IndexingError` | indexing failed (transient) | document → `FAILED`, message retried |
| `PermanentIndexingError` | invalid file, parse impossible | document → `FAILED_INVALID`, message to DLQ |
| `IndexLockedError` | another worker is indexing the doc | retried later, document untouched |

## Retry, DLQ and idempotency

Failed messages are republished to per-delay TTL queues
(`<queue>.retry.5s/30s/120s`) that dead-letter back to the main queue; after
the last attempt the message lands in `<queue>.dlq`.

- Broker **redeliveries** (worker crashed mid-processing) are checked against
  an idempotency cache in Redis (`message_id` based): processed messages are
  skipped instead of applied twice.
- `PermanentError` skips the retry ladder entirely.
- The main queue optionally enforces `x-max-length` with `reject-publish`
  (`QUEUE_MAX_LENGTH`) so a dead consumer cannot grow the broker without
  bound.

## Distributed ingest lock

`IndexerService` wraps every `index_document` in `lock:ingest:<doc_id>`
(`SET NX EX`, TTL `INDEX_LOCK_TTL`); release is a compare-and-delete Lua
script so a stale owner can never drop a lock re-acquired by another worker.
A busy lock raises `IndexLockedError` — the document is **not** marked failed.
Without `REDIS_URL` an in-process fallback keeps the same contract
(single-worker exclusivity).

## Circuit breaker

Every LLM provider call is guarded: after
`LLM_CIRCUIT_FAILURE_THRESHOLD` consecutive failures the circuit opens and
calls fail fast (`CircuitOpenError`) until `LLM_CIRCUIT_RESET_TIMEOUT`
elapses; a half-open trial call closes it again. Cache hits count as
successes, keeping the breaker closed on healthy traffic.

## Blue-Green reindex

`REINDEX_BLUE_GREEN=true` (default) replaces the old "drop and recreate":

1. `create_shadow_collection` builds `documents__green_<ts>` with the same
   HNSW/full-text config; **search keeps serving the live collection**.
2. Writes are dual-routed (active + green) so concurrent ingest is not lost.
3. Documents are streamed in keyset batches and re-indexed.
4. `optimize_collection` rebuilds HNSW on the green collection.
5. `promote_shadow` flips the public alias atomically
   (`DeleteAliasOperation` + `CreateAliasOperation` in one call) and drops
   the superseded collection.

Edge cases: promoting twice is a no-op; promoting when the configured name is
still a *physical* collection (pre-blue-green deployment) raises a descriptive
`RuntimeError` — migrate manually once, the code never deletes live data on
its own.

## Fail-fast startup checks

- **Embedding dimension vs collection** — mismatch exits the process
  (`embedding_dimension_mismatch_fail_fast`) instead of failing every query
  at runtime. Remediation: restore `EMBEDDING_MODEL` or reindex.
- **Model-version marker** — a warning when stored points were built by a
  different model (`reindex` resolves it).

## MIME validation

Before a parser is selected, libmagic verifies the real content type against
the extension for known formats (`.pdf`, `.docx`, `.md`, `.txt`, `.html`).
A mismatch is a permanent failure. Availability probe of the native binding
runs in a subprocess once; if libmagic is unusable the check fails open.

## Maintenance

- `scripts/purge_deleted_documents.py --days 30 --apply` — hard-delete
  soft-deleted rows after the retention window (dry-run without `--apply`).
- Every write to the vector store is mirrored by `delete_by_doc_id` before
  re-upsert, so no ghost chunks survive a shrinking document.
