# Async RAG Service

A production-oriented, fully asynchronous **Retrieval-Augmented Generation (RAG)**
service on Python 3.12. Documents and user questions travel through RabbitMQ;
documents are parsed, chunked and embedded into Qdrant; questions are answered
strictly from the retrieved context — with **access control enforced inside the
vector search itself**.

Built as a Clean Architecture / Ports & Adapters backend: `domain` has zero
knowledge of infrastructure, `application` services depend only on Protocol
ports, `infrastructure` implements them, and a dishka DI container wires it all
in a single composition root.

---

## Features

**Ingestion & indexing**
- Markdown, PDF, DOCX and a universal `unstructured` fallback; OCR for scans
  (`PDF_OCR_STRATEGY` / `PDF_OCR_LANGUAGES`) with a hard parse timeout
  (`PARSE_TIMEOUT`) against hung tesseract processes
- MIME validation via libmagic before parsing: an `.exe` renamed to `.pdf` is a
  *permanent* failure, never an OCR hang
- Deterministic chunk IDs (`uuid5`) + explicit per-document purge before upsert
  → **no ghost chunks** after a document shrinks
- Quality guards: tiny-chunk merging (`CHUNK_MIN_CHARS`), row-wise table
  chunking with the header repeated in every chunk, HTML tables preserved
  (`table_html`), element typing (`type`: table / image / title)
- **Parent-Child retrieval** (opt-in): children are embedded, the LLM sees the
  parent text; sibling hits are auto-merged

**Retrieval & generation**
- Hybrid search: dense vectors + lexical BM25 fused with RRF; graceful
  degradation to pure vector search
- BM25 noise guards: stop-word stripping and a score threshold for lexical
  candidates; a similarity threshold and a character budget for the packed
  context
- Prompt-injection isolation: retrieved context is wrapped in `<context>` tags
  and explicitly marked as untrusted data
- Pluggable providers: DeepSeek (default), any OpenAI-compatible endpoint
  (vLLM, Ollama, OpenAI, ...), Anthropic; per-request provider/model routing
  with server-side allowlists
- Optional **query rewriting** through the LLM (graceful fallback to the
  original query)
- Optional **semantic answer cache**: near-duplicate questions reuse a previous
  answer without an LLM call (opt-in, see the ACL caveat in the settings)
- Truncated answers (`finish_reason=length` / `stop_reason=max_tokens`) are
  **never cached**; optional continuation requests merge the full answer
- **Agentic retrieval** (opt-in): an LLM grader validates the hits and
  triggers corrective rewrites; an LLM planner splits complex questions into
  sub-queries searched in parallel and fused with RRF
- **GraphRAG** (opt-in): relation triples are extracted at indexing time and
  chunks of graph-neighboring documents are merged into the context at query
  time; pseudo-graph in Qdrant, optional Neo4j backend via an extra

**Reliability**
- **Blue-Green reindex**: a shadow collection is built while search keeps
  serving the live one; the collection alias flips atomically at the end
- Distributed Redis lock (`SET NX EX` + compare-and-delete) serializes
  concurrent indexing of the same document
- Fail-fast startup check: embedding dimension vs collection config
- Error taxonomy: `TransientError` → retry with exponential backoff;
  `PermanentError` → Dead Letter Queue immediately; soft-deleted-invalid status
  (`failed_invalid`) for bad input files
- Idempotency on broker redeliveries (processed message ids remembered in
  Redis)
- Circuit breaker on every LLM provider; per-call pooled HTTP client with
  keepalive (no socket exhaustion under burst)

**Security**
- JWT per message: algorithm allowlisting (anti algorithm-confusion), key
  rotation without downtime (`*_PREVIOUS` keys), mandatory `exp`, revocation by
  `jti` via in-memory or Redis blacklist
- ACL enforced **inside Qdrant**: `owner_id == user OR access_group IN groups`
  — unauthorized chunks never even reach the LLM
- Optional PII redaction (`ANONYMIZE_CONVERSATIONS`) of questions/answers
  before persistence (GDPR / ФЗ-152 friendly)

**Observability & operations**
- Prometheus metrics (queues, vector search, LLM latency/tokens/cache/truncations,
  circuit breaker), OpenTelemetry tracing, structlog JSON logging
- Health server: `/healthz`, `/readyz` (per-dependency checks), `/metrics`
- Worker role split: `WORKER_QUEUES=query|background|all` for independent scaling
- RabbitMQ reliability: per-queue DLX/DLQ, TTL retry queues with exponential
  backoff, optional `x-max-length` + `reject-publish`

**Quality**
- Offline evaluation: precision@k, recall@k, MRR, NDCG, hit rate, faithfulness
  (lexical heuristic or **LLM-as-a-judge**), refusal rate; A/B diffing of
  configurations
- 300+ tests with in-memory doubles (no DB/broker/Redis needed), mutation
  testing via mutmut on the pure-logic modules, strict mypy on `src` **and**
  `tests`, ruff (pydocstyle, pyflakes, naming, quotes, isort)

---

## Tech Stack

| Component | Technology |
|---|---|
| Runtime | Python 3.12+, asyncio, [uv](https://docs.astral.sh/uv/) |
| Message broker | RabbitMQ (aio-pika) |
| Vector DB | Qdrant (REST + optional gRPC with message-size caps) |
| Metadata | PostgreSQL 16 + SQLAlchemy 2 (asyncpg), Alembic migrations, partial index for live rows |
| File storage | MinIO (S3 API) |
| Embeddings | sentence-transformers, `intfloat/multilingual-e5-small` (384 dim), query/passage prefixes, batch + query cache |
| LLM | DeepSeek (`deepseek-chat`), any OpenAI-compatible API, Anthropic Messages — via `LLM_PROVIDER` |
| Document parsing | unstructured (+ markdown / beautifulsoup4), OCR via tesseract, libmagic MIME checks |
| Caches | Redis (optional, L2) / in-process TTL + semantic answer cache |
| DI | dishka |
| Configuration | pydantic-settings (`.env`) |
| Observability | Prometheus, structlog, OpenTelemetry (OTLP/HTTP) |
| Testing | pytest + pytest-asyncio, mutmut; mypy `strict` (src + tests), ruff |

---

## Documentation

Full documentation lives in [docs/](docs/index.md):

| Page | Contents |
|---|---|
| [Architecture](docs/architecture.md) | layers, ports & adapters, request flows |
| [Configuration](docs/configuration.md) | every environment variable, explained |
| [Retrieval](docs/retrieval.md) | chunking, hybrid search, caching, prompt hardening |
| [Agentic Retrieval](docs/agentic-retrieval.md) | corrective RAG loop: grader, planner, sub-query fusion |
| [GraphRAG](docs/graphrag.md) | entity graph, extraction, expansion, optional backends |
| [Reliability](docs/reliability.md) | retry/DLQ, idempotency, locks, blue-green reindex |
| [Security](docs/security.md) | JWT lifecycle, ACL enforcement, PII redaction |
| [Operations](docs/operations.md) | deployment, roles, maintenance, troubleshooting |
| [Evaluation](docs/evaluation.md) | golden dataset, corpus, metrics, LLM-as-a-judge |
| [Development](docs/development.md) | markers, lint/typing policy, adding features |

---

## Architecture

A detailed walkthrough (ports & adapters table, request flows, DI) lives in
[docs/architecture.md](docs/architecture.md). Overview:

```
                 ┌──────────┐
  API / producer ─► RabbitMQ │
                 └──────────┘
       ┌──────────────┬───────────────┬────────────────┐
       ▼              ▼               ▼                ▼
 ingest_queue   query_queue    delete_queue    reindex_queue
       │              │               │                │
       ▼              ▼               ▼                ▼
 IndexerService  RetrieverService   DocumentManager (delete / reindex)
       │              │               │                │
       ▼              ▼               ▼                ▼
 parse+chunk    vector search    vectors purge      shadow build +
 embed+upsert   + LLM (ACL       + file delete      alias flip
 (Qdrant/MinIO)  filtered)                          (blue-green)
```

### Layers

```
src/
├── domain/                     # entities, error taxonomy, PII policy (zero deps)
│   ├── model.py                #   Document, Conversation, DocStatus, exceptions
│   └── pii.py                  #   redact_pii (pure regex policies)
├── application/
│   ├── interfaces.py           # ALL ports (Protocols): repository, vector store,
│   │                           #   embeddings, LLM, parser, selector, lock, caches
│   └── services.py             # IndexerService, RetrieverService, DocumentManager
├── infrastructure/             # adapters implementing the ports
│   ├── caching.py              #   TTLCache / RedisCache (LLM answers, embeddings)
│   ├── resilience.py           #   circuit breaker + distributed locks (Redis)
│   ├── observability.py        #   structlog config, Prometheus metrics, OTel tracing
│   ├── embedding.py            #   sentence-transformers (prefixes, batching, cache)
│   ├── llm.py                  #   OpenAI-compatible / Anthropic / DeepSeek + router
│   │                           #   + query rewriter (+ shared truncation-safe pipeline)
│   ├── vector_store.py         #   Qdrant adapter + BM25/RRF hybrid + blue-green
│   ├── parsing.py              #   markdown / pdf(OCR) / docx / unstructured + selector
│   ├── repositories.py         #   PostgreSQL (read replica support) + ORM models
│   ├── file_storage.py         #   MinIO + hashing
│   ├── security.py             #   JWT signer / validator / blacklists
│   └── observability.py        #   logging + metrics + tracing
├── entrypoints/
│   ├── consumers.py            # base consumer (retry/DLX/idempotency) + 4 handlers
│   ├── health.py               # /healthz /readyz /metrics
│   └── main.py                 # bootstrap: logging, tracing, fail-fast checks
├── config.py                   # pydantic-settings
├── container.py                # dishka composition root
├── alembic.ini
├── migrations/                 # async env.py + revisions (partial index, …)
├── eval/
│   ├── corpus/*.md             # eval document corpus (deterministic ids)
│   └── golden_dataset.json     # golden eval dataset (15 cases)
├── scripts/
│   ├── run_eval.py             # offline evaluation CLI (--llm-judge)
│   └── index_eval_corpus.py    # indexes eval/corpus with stable doc ids
│   ├── manage_groups.py        # ACL group management CLI
│   └── purge_deleted_documents.py  # hard-delete soft-deleted rows (cron)
└── tests/                      # pytest suite, in-memory doubles in conftest
```

---

## Configuration Highlights

Full reference: [docs/configuration.md](docs/configuration.md).
The most important variables:

| Variable | Default | Purpose |
|---|---|---|
| `WORKER_QUEUES` | `all` | `query` / `background` / `all` role split |
| `LLM_PROVIDER` | `deepseek` | `deepseek` \| `openai` \| `anthropic` |
| `LLM_ENABLED_PROVIDERS` | `LLM_PROVIDER` | per-request routing allowlist |
| `LLM_ALLOWED_MODELS` | *(empty = any)* | model allowlist (cost control) |
| `LLM_MAX_TOKENS` / `LLM_TEMPERATURE` | `500` / `0.1` | generation limits |
| `LLM_CONTINUE_ON_TRUNCATION` | `false` | merge follow-ups when cut by `max_tokens` |
| `SEARCH_HYBRID` | `false` | BM25 + RRF fusion |
| `SEARCH_HYBRID_STOPWORDS` | *(empty)* | stop words stripped from the lexical query |
| `SEARCH_BM25_SCORE_THRESHOLD` | `0.0` | lexical noise gate |
| `SEARCH_SCORE_THRESHOLD` | `0.0` | vector similarity gate |
| `SEARCH_CONTEXT_MAX_CHARS` | `12000` | LLM context character budget |
| `PARENT_CHILD_CHILD_CHARS` | `0` | Parent-Child retrieval (children size) |
| `QUERY_REWRITE_ENABLED` | `false` | LLM query rewriting before search |
| `SEMANTIC_CACHE_ENABLED` | `false` | near-duplicate answer reuse (see ACL note) |
| `REINDEX_BLUE_GREEN` | `true` | shadow collection + atomic alias flip |
| `CHUNK_MIN_CHARS` | `0` | merge tiny text chunks |
| `PARSE_TIMEOUT` | `300.0` | hard parse/OCR time budget (seconds) |
| `PARSE_VALIDATE_MIME` | `true` | magic-bytes vs extension check |
| `INDEX_LOCK_TTL` | `300.0` | distributed ingest lock TTL (seconds) |
| `REDIS_URL` | *(empty)* | enables L2 caches, locks, idempotency, JWT blacklist |
| `QUEUE_MAX_LENGTH` | `0` | RabbitMQ `x-max-length` cap (0 = unlimited) |
| `ANONYMIZE_CONVERSATIONS` | `false` | PII redaction before persistence |
| `JWT_ALGORITHM` / `JWT_SECRET` / `JWT_PUBLIC_KEY` | `HS256` | signing / verification |
| `OTEL_ENDPOINT` | *(empty)* | OTLP/HTTP tracing export |

---

## Evaluation

The golden dataset pairs each question with the corpus files that must be
retrieved. The corpus lives in `eval/corpus/*.md` (a small but coherent
fictional company handbook); document ids are deterministic, so the
dataset references real ids out of the box.

Index the corpus once (MinIO/Qdrant/Postgres must be up):

```bash
PYTHONPATH=src uv run python scripts/run_eval.py \
    --dataset eval/golden_dataset.json --json-out report.json

# semantic faithfulness judged by the LLM (falls back to the lexical
# heuristic when the judge call fails):
PYTHONPATH=src uv run python scripts/run_eval.py \
    --dataset eval/golden_dataset.json --llm-judge
```

`compare_reports(baseline, variant)` produces per-metric deltas for A/B runs
(e.g. hybrid on/off, different chunk sizes, providers).

---

## Development

Tests are marked by type; run subsets with `-m`:

```bash
uv run pytest                                 # full suite (in-memory doubles)
uv run pytest -m retrieval                    # search quality only
uv run pytest -m "llm"                        # provider clients & routing
uv run pytest -m "not consumers"              # everything except consumers
```

Markers: `indexing`, `retrieval`, `llm`, `consumers`, `parsing`, `security`,
`evaluation`, `observability`, `infra` (registered in `pyproject.toml`,
enforced via `--strict-markers`).

```bash
uv run ruff check src tests scripts           # lint (E,F,Q,D,N)
uv run ruff format --check src tests scripts  # formatting (79 cols, single quotes)
uv run mypy src tests                         # strict typing, src + tests
uv run mutmut run                             # mutation testing (WSL on Windows)
```

Maintenance scripts:

```bash
uv run python scripts/manage_groups.py set <user_id> group1,group2
uv run python scripts/purge_deleted_documents.py --days 30 --apply
```

---

## Known Limitations

- Real OCR (`ocr_only` / `hi_res`) requires `tesseract` + `poppler` in the
  worker image. `PARSE_TIMEOUT` cancels the *await*; the executor thread itself
  needs a process-level cap in hardened deployments.
- The semantic answer cache is in-process and opt-in: in multi-tenant setups a
  cached answer was generated under a specific user's ACL — enable only when
  that is acceptable.
- LLM client instances are cached lazily per (provider, model); the model
  allowlist keeps that set bounded.
- The golden dataset covers the bundled `eval/corpus` handbook; extend it with
  cases from your own documents as they accumulate.
- mutmut on Windows requires WSL.

## License

MIT — see [LICENSE](LICENSE).
