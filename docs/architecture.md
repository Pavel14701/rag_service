# Architecture

## Layers

```
entrypoints          RabbitMQ consumers, health server, bootstrap
      │  depends on ▼
application          use-case services + ports (Protocols)
      │  depends on ▼
domain               entities, error taxonomy, pure policies (zero deps)

infrastructure       adapters implementing the ports (Qdrant, Postgres,
      │              MinIO, LLM APIs, parsing, JWT, locks, caches)
      ▼
container (dishka)   composition root: the only place that knows concretes
```

The dependency rule, enforced by convention and checked in CI:

- `domain` imports nothing but the standard library;
- `application` imports `domain` + its own `interfaces.py` (ports);
- `infrastructure` imports ports + `domain` and implements them;
- `entrypoints` import `application` only;
- `container.py` may import everything.

## Ports and adapters

| Port (`application/interfaces.py`) | Adapter (`infrastructure/`) | Notes |
|---|---|---|
| `DocumentRepository` | `repositories.PostgresDocumentRepository` | read-replica aware, keyset iteration |
| `VectorStore` | `vector_store.QdrantStore` | hybrid BM25/RRF, blue-green, MIME-free |
| `EmbeddingModel` | `embedding.SentenceTransformerEmbedding` | e5 prefixes, query cache |
| `FileStorage` | `file_storage.MinioStorage` | S3 API |
| `LLMGenerator` | `llm.OpenAIChatClient` / `AnthropicClient` / `DeepSeekClient` / `LLMRouter` | shared truncation-safe pipeline |
| `TokenValidator` | `security.JWTValidator` | rotation + blacklist |
| `DocumentParser` / `ParserSelector` | `parsing.*` + `DocumentParserSelector` | MIME-checked selection |
| `DistributedLock` | `resilience.RedisDistributedLock` / `InProcessDistributedLock` | fallback without Redis |
| `SemanticCache` | `caching.InMemorySemanticCache` | opt-in, no-op when disabled |
| `QueryRewriter` | `llm.LLMQueryRewriter` / `NoOpQueryRewriter` | graceful fallback |

## Request flow (query path)

1. Producer publishes to `query_queue` (JWT + query + optional routing).
2. `QueryConsumer.handle` validates the token and the routing choices
   (provider/model allowlists) **before** any expensive work.
3. `RetrieverService.answer_query`:
   optional rewrite → embed the (rewritten) query → Qdrant search with the
   ACL filter → score/budget guards → auto-merge parent hits → semantic
   cache lookup → LLM generation (cache → breaker → HTTP) → persist the
   conversation (optionally PII-redacted) → reply.
4. The reply is published to `reply_to` (or `query_reply_queue`).

## Index flow (ingest path)

1. `IngestConsumer` checks ownership.
2. `IndexerService.index_document` acquires the distributed lock, downloads
   the file from MinIO, selects the parser (MIME-validated), parses with a
   hard timeout, chunks (tiny-merge, tables, parent-child), purges old
   chunks of the document, embeds children, upserts — all under the lock.
3. Status transitions: `PENDING → INDEXED`, or `FAILED` (transient) /
   `FAILED_INVALID` (permanent — no retry, straight to DLQ).

## DI composition

`container.py` wires every port with a `@provide` factory. Providers are
APP-scoped singletons; the composition root is the only module allowed to
import both `application` and `infrastructure`. Adding a feature means:

1. define/extend a Protocol in `application/interfaces.py`;
2. implement it in `infrastructure/`;
3. inject it into the service through the constructor;
4. register the provider in `container.py`.
