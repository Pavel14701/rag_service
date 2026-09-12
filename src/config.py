"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    rabbitmq_url: str = 'amqp://guest:guest@localhost:5672/'
    qdrant_host: str = 'localhost'
    qdrant_port: int = 6333
    postgres_dsn: str = 'postgresql+asyncpg://rag:rag@localhost:5432/rag'
    postgres_read_dsn: str | None = None  # optional read replica DSN
    minio_endpoint: str = 'localhost:9000'
    minio_access_key: str = 'minioadmin'
    minio_secret_key: str = 'minioadmin'
    minio_bucket: str = 'documents'
    # LLM provider selection: "deepseek" (default), "openai" (any OpenAI
    # chat-completions compatible API: OpenAI, vLLM, Ollama, Together, ...)
    # or "anthropic". Each provider has its own key/base URL; ``llm_model``
    # overrides the provider default model when set.
    llm_provider: str = 'deepseek'
    llm_model: str | None = None
    # Per-request routing: comma-separated providers a request may choose
    # (empty = only the default provider), and an optional comma-separated
    # allowlist of models tenants may request (empty = any enabled provider
    # model). Restrict both in multi-tenant deployments (cost control).
    llm_enabled_providers: str = ''
    llm_allowed_models: str = ''
    llm_max_tokens: int = 500
    deepseek_api_key: str | None = None
    deepseek_base_url: str = 'https://api.deepseek.com/v1'
    openai_api_key: str | None = None
    openai_base_url: str = 'https://api.openai.com/v1'
    anthropic_api_key: str | None = None
    anthropic_base_url: str = 'https://api.anthropic.com'
    embedding_model: str = 'intfloat/multilingual-e5-small'
    embedding_query_prefix: str = 'query: '
    embedding_passage_prefix: str = 'passage: '
    collection_name: str = 'documents'
    jwt_secret: str | None = None
    # JWT hardening: asymmetric algorithms (RS256/ES256) verify with the
    # public key; HS* verify with jwt_secret. "previous" keys accept
    # tokens signed before a key rotation (zero-downtime rotation:
    # add new key, stop issuing with old, drop old after max TTL).
    jwt_algorithm: str = 'HS256'
    jwt_public_key: str | None = None
    jwt_private_key: str | None = None
    jwt_secret_previous: str | None = None
    jwt_public_key_previous: str | None = None
    jwt_issuer: str | None = None
    jwt_audience: str | None = None

    # Redact well-known PII (emails, phones, cards, IBANs, API keys)
    # before conversations are persisted to Postgres.
    anonymize_conversations: bool = False

    # TLS for backing services: MinIO HTTPS, Qdrant HTTPS (+ API key).
    # RabbitMQ TLS is enabled by using an amqps:// RABBITMQ_URL.
    minio_secure: bool = False
    qdrant_https: bool = False
    qdrant_api_key: str | None = None
    # gRPC transport: faster serialization and ~32 MB message limits
    # protect from huge batch responses on high-load reindexes.
    qdrant_prefer_grpc: bool = True
    log_level: str = 'INFO'
    log_json: bool = True
    metrics_port: int = 8000
    health_check_timeout: float = 2.0
    otel_endpoint: str | None = None

    # Worker role split: "query" (real-time), "background"
    # (ingest/delete/reindex), or "all" (single process runs
    # everything; default for dev).
    worker_queues: str = 'all'

    # Embedding model performance.
    embedding_device: str | None = None  # e.g. "cuda", "cpu"; None = auto
    embedding_batch_size: int | None = None
    embedding_query_cache_ttl: float = 3600.0
    embedding_query_cache_size: int = 512

    # LLM answer cache (in-process TTL cache).
    llm_cache_ttl: float = 3600.0
    llm_cache_size: int = 256

    # Distributed indexing lock (Redis): TTL for lock:ingest:{doc_id}.
    index_lock_ttl: float = 300.0

    # Qdrant HNSW tuning (None = server defaults).
    qdrant_hnsw_m: int | None = None
    qdrant_hnsw_ef_construct: int | None = None
    qdrant_hnsw_ef: int | None = None

    # LLM generation
    llm_temperature: float = 0.1

    # Circuit breaker guarding the LLM API: after
    # ``llm_circuit_failure_threshold`` consecutive failures, generation
    # fails fast with CircuitOpenError for ``llm_circuit_reset_timeout``
    # seconds (then one trial call is allowed).
    llm_circuit_failure_threshold: int = 5
    llm_circuit_reset_timeout: float = 60.0

    # Truncated answers (finish_reason=length / stop_reason=max_tokens)
    # are never cached; when True, a follow-up continuation request
    # merges the cut-off generation into one full answer.
    llm_continue_on_truncation: bool = False

    # Optional Redis for distributed caches (LLM answers, query embeddings).
    # When unset, in-process TTL caches are used.
    redis_url: str | None = None
    redis_cache_prefix: str = 'rag'

    # Hybrid search: fuse vector hits with lexical BM25 ranking (RRF).
    search_hybrid: bool = False
    search_hybrid_rrf_k: int = 60
    search_hybrid_candidates: int = 50

    # Retrieval quality guards.
    # Min vector similarity score to keep a hit (0 = disabled;
    # hybrid RRF scores have another scale - use the BM25 one).
    search_score_threshold: float = 0.0
    # Character budget for the LLM context (0 = unlimited); keeps
    # headroom for the answer inside the LLM max_tokens window.
    search_context_max_chars: int = 12000
    # Comma-separated stop words dropped from the BM25 keyword query
    # (popular terms flooding lexical candidates with noise).
    search_hybrid_stopwords: str = ''
    # Min BM25 score for a lexical candidate to enter RRF fusion.
    search_bm25_score_threshold: float = 0.0

    # Chunking quality: text chunks shorter than this (chars) are
    # merged into the neighbouring text chunk (0 = disabled).
    chunk_min_chars: int = 0

    # Hard parsing time budget in seconds (0 = unlimited). Guards
    # against OCR/tesseract hangs on binary garbage renamed to .pdf.
    parse_timeout: float = 300.0
    # Verify real MIME type (magic bytes) against the file extension
    # before parsing; mismatch = permanent failure (no retry).
    parse_validate_mime: bool = True

    # Shared HTTP client pool for outbound LLM API calls (sockets are
    # the scarcest resource on a high-load host: a per-call client
    # exhausts them under burst).
    http_max_connections: int = 200
    http_keepalive_connections: int = 50

    # Queue length cap with reject-publish (0 = unlimited, current
    # behavior). Protects RabbitMQ memory when consumers fall behind.
    queue_max_length: int = 0

    # Semantic answer cache: reuse an LLM answer when a previously
    # seen question is near-identical in embedding space (cosine >=
    # threshold). Opt-in: in multi-tenant deployments cached answers
    # were generated under a specific user's ACL - enable only when
    # that is acceptable (e.g. uniform access across tenants).
    semantic_cache_enabled: bool = False
    semantic_cache_maxsize: int = 256
    semantic_cache_ttl: float = 3600.0
    semantic_cache_threshold: float = 0.95

    # Query rewriting: an auxiliary LLM call normalizes/ expands the
    # question before search (better recall, +1 LLM round-trip).
    query_rewrite_enabled: bool = False

    # Blue-Green reindex: build a shadow collection and flip the
    # collection alias atomically instead of dropping the live one
    # (search keeps serving during the whole reindex).
    reindex_blue_green: bool = True

    # Parent-Child retrieval: text elements are split into parent
    # chunks (max_tokens) and child chunks (this many chars); children
    # are embedded, but the LLM receives the parent context and hits
    # on children of the same parent are auto-merged (0 = disabled).
    parent_child_child_chars: int = 0

    # PDF OCR (unstructured partition strategy) for scanned documents.
    pdf_ocr_strategy: str = 'auto'  # auto | hi_res | ocr_only | fast
    pdf_ocr_languages: str = 'eng'  # comma-separated tesseract languages

    minio_connection_pool_size: int = 10

    @property
    def llm_enabled_providers_list(self) -> list[str]:
        """Providers available for per-request routing (incl. default)."""
        raw = [
            p.strip()
            for p in self.llm_enabled_providers.split(',')
            if p.strip()
        ]
        return raw or [self.llm_provider]

    @property
    def llm_allowed_models_list(self) -> list[str]:
        """Models a request may explicitly ask for (empty = unrestricted)."""
        return [
            m.strip() for m in self.llm_allowed_models.split(',') if m.strip()
        ]

    @field_validator('llm_provider')
    @classmethod
    def _validate_llm_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if (
            normalized == 'openai-compatible'
        ):  # alias for self-hosted endpoints
            return 'openai'
        if normalized not in {'deepseek', 'openai', 'anthropic'}:
            raise ValueError(
                'llm_provider must be one of: deepseek, openai, anthropic '
                f'(got {value!r})'
            )
        return normalized

    @field_validator('jwt_algorithm')
    @classmethod
    def _validate_jwt_algorithm(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in (
            'HS256',
            'HS384',
            'HS512',
            'RS256',
            'RS384',
            'RS512',
            'ES256',
        ):
            raise ValueError(f'jwt_algorithm {value!r} is not supported')
        return normalized

    def jwt_verification_keys(self) -> list[str]:
        """Verification keys: current first, then rotation predecessor."""
        keys: list[str] = []
        if self.jwt_algorithm.startswith('HS'):
            if self.jwt_secret:
                keys.append(self.jwt_secret)
            if self.jwt_secret_previous:
                keys.append(self.jwt_secret_previous)
        else:
            if self.jwt_public_key:
                keys.append(self.jwt_public_key)
            if self.jwt_public_key_previous:
                keys.append(self.jwt_public_key_previous)
        if not keys:
            raise ValueError(
                f'{self.jwt_algorithm} requires the corresponding JWT key '
                '(JWT_SECRET for HS*, JWT_PUBLIC_KEY for RS*/ES*)'
            )
        return keys

    model_config = {'env_file': '.env', 'env_file_encoding': 'utf-8'}
