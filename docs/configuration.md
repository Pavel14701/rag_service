# Configuration

All settings are environment variables (optionally via a `.env` file next to
`pyproject.toml`), defined in `src/config.py::Settings`. Defaults are chosen
to match the development docker-compose.

## Core

| Variable | Default | Purpose |
|---|---|---|
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/` | broker DSN; use `amqps://` for TLS |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | vector DB endpoint |
| `QDRANT_HTTPS` / `QDRANT_API_KEY` | `false` / — | Qdrant TLS / auth |
| `QDRANT_PREFER_GRPC` | `true` | gRPC transport with 32 MB message caps |
| `POSTGRES_DSN` | `postgresql+asyncpg://rag:rag@localhost:5432/rag` | primary DB |
| `POSTGRES_READ_DSN` | — | optional read replica for SELECTs |
| `MINIO_ENDPOINT` / `MINIO_SECURE` | `localhost:9000` / `false` | object storage |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | `minioadmin` | object storage auth |
| `MINIO_CONNECTION_POOL_SIZE` | `10` | MinIO client pool |
| `REDIS_URL` | — | enables L2 caches, locks, idempotency, JWT blacklist |
| `REDIS_CACHE_PREFIX` | `rag` | key namespace for all Redis usage |

## Workers

| Variable | Default | Purpose |
|---|---|---|
| `WORKER_QUEUES` | `all` | `query` / `background` / `all` |
| `QUEUE_MAX_LENGTH` | `0` | main-queue cap + `reject-publish` (0 = off) |
| `HEALTH_CHECK_TIMEOUT` | `2.0` | readiness probe budget (seconds) |
| `METRICS_PORT` | `8000` | health/metrics HTTP port |

## Embeddings

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | HF model id; changing it requires a reindex (dimension is fail-fast checked) |
| `EMBEDDING_QUERY_PREFIX` / `EMBEDDING_PASSAGE_PREFIX` | `query: ` / `passage: ` | e5 instruction prefixes |
| `EMBEDDING_DEVICE` | auto | `cpu` / `cuda` |
| `EMBEDDING_BATCH_SIZE` | model default | encoder batch size |
| `EMBEDDING_QUERY_CACHE_TTL` / `..._SIZE` | `3600` / `512` | in-process query-embedding cache |

## LLM

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `deepseek` | `deepseek` \| `openai` \| `anthropic` |
| `LLM_MODEL` | provider default | overrides the provider default model |
| `LLM_ENABLED_PROVIDERS` | `LLM_PROVIDER` | comma list; enables per-request routing |
| `LLM_ALLOWED_MODELS` | *(empty = any)* | model allowlist for requests |
| `LLM_MAX_TOKENS` / `LLM_TEMPERATURE` | `500` / `0.1` | generation parameters |
| `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — | required when the provider is enabled |
| `DEEPSEEK_BASE_URL` / `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` | vendor defaults | endpoints |
| `LLM_CONTINUE_ON_TRUNCATION` | `false` | continuation requests for cut answers |
| `LLM_CIRCUIT_FAILURE_THRESHOLD` / `LLM_CIRCUIT_RESET_TIMEOUT` | `5` / `60.0` | circuit breaker tuning |
| `HTTP_MAX_CONNECTIONS` / `HTTP_KEEPALIVE_CONNECTIONS` | `200` / `50` | shared httpx pool |
| `QUERY_REWRITE_ENABLED` | `false` | LLM query rewriting before search |

## Search & chunking

| Variable | Default | Purpose |
|---|---|---|
| `SEARCH_HYBRID` | `false` | enable BM25 + RRF fusion |
| `SEARCH_HYBRID_RRF_K` / `SEARCH_HYBRID_CANDIDATES` | `60` / `50` | RRF tuning / lexical candidate cap |
| `SEARCH_HYBRID_STOPWORDS` | *(empty)* | stop words removed from the lexical query |
| `SEARCH_BM25_SCORE_THRESHOLD` | `0.0` | lexical candidate gate |
| `SEARCH_SCORE_THRESHOLD` | `0.0` | vector similarity gate (0 = off) |
| `SEARCH_CONTEXT_MAX_CHARS` | `12000` | LLM context character budget (0 = off) |
| `SEMANTIC_CACHE_ENABLED` | `false` | near-duplicate answer reuse (multi-tenant ACL caveat) |
| `SEMANTIC_CACHE_MAXSIZE` / `..._TTL` / `..._THRESHOLD` | `256` / `3600` / `0.95` | cache tuning |
| `PARENT_CHILD_CHILD_CHARS` | `0` | children size for Parent-Child retrieval (0 = off) |
| `CHUNK_MIN_CHARS` | `0` | merge tiny text chunks (0 = off) |
| `PARSE_TIMEOUT` | `300.0` | hard parse/OCR budget (seconds, 0 = off) |
| `PARSE_VALIDATE_MIME` | `true` | magic-bytes vs extension check |
| `PDF_OCR_STRATEGY` / `PDF_OCR_LANGUAGES` | `auto` / `eng` | unstructured OCR handling |

### Agentic retrieval

| Variable | Default | Purpose |
|---|---|---|
| `AGENTIC_GRADER_ENABLED` | `false` | LLM grades hits; negative verdict triggers a corrective rewrite ([details](agentic-retrieval.md)) |
| `AGENTIC_PLANNER_ENABLED` | `false` | LLM splits the question into sub-queries searched in parallel |
| `AGENTIC_MAX_ROUNDS` | `0` | max search rounds with a grader wired (0/no grader = single pass) |
| `AGENTIC_SUBQUERY_LIMIT` | `3` | max sub-queries per round |

### GraphRAG

| Variable | Default | Purpose |
|---|---|---|
| `GRAPH_EXTRACT_ENABLED` | `false` | extract relation triples while indexing ([details](graphrag.md)) |
| `GRAPH_EXPAND_ENABLED` | `false` | merge chunks of graph-neighboring documents at query time |
| `GRAPH_MAX_HOPS` | `1` | graph traversal depth |
| `GRAPH_BACKEND` | `qdrant` | `qdrant` (pseudo-graph) or `neo4j` (requires the `graph-neo4j` extra) |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | *(empty)* | Neo4j connection when `GRAPH_BACKEND=neo4j` |

## Reindex & locks

| Variable | Default | Purpose |
|---|---|---|
| `REINDEX_BLUE_GREEN` | `true` | shadow collection + atomic alias flip |
| `INDEX_LOCK_TTL` | `300.0` | `lock:ingest:<doc_id>` TTL (seconds) |
| `QDRANT_HNSW_M` / `QDRANT_HNSW_EF_CONSTRUCT` / `QDRANT_HNSW_EF` | — | HNSW tuning at create/query time |

## Security

| Variable | Default | Purpose |
|---|---|---|
| `JWT_ALGORITHM` | `HS256` | `HS*` or `RS*/ES256` |
| `JWT_SECRET` / `JWT_SECRET_PREVIOUS` | — | HMAC keys (current, rotation) |
| `JWT_PUBLIC_KEY` / `JWT_PUBLIC_KEY_PREVIOUS` / `JWT_PRIVATE_KEY` | — | asymmetric keys |
| `JWT_ISSUER` / `JWT_AUDIENCE` | — | enforced when set |
| `ANONYMIZE_CONVERSATIONS` | `false` | PII redaction before persistence |

## Observability

| Variable | Default | Purpose |
|---|---|---|
| `LOG_LEVEL` / `LOG_JSON` | `INFO` / `true` | structlog output |
| `OTEL_ENDPOINT` | — | OTLP/HTTP exporter (tracing off when unset) |
| `METRICS_PORT` | `8000` | health/metrics server |
