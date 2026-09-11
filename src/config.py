"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    postgres_dsn: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"
    postgres_read_dsn: str | None = None  # optional read replica DSN
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "documents"
    # LLM provider selection: "deepseek" (default), "openai" (any OpenAI
    # chat-completions compatible API: OpenAI, vLLM, Ollama, Together, ...)
    # or "anthropic". Each provider has its own key/base URL; ``llm_model``
    # overrides the provider default model when set.
    llm_provider: str = "deepseek"
    llm_model: str | None = None
    llm_max_tokens: int = 500
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str | None = None
    anthropic_base_url: str = "https://api.anthropic.com"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_query_prefix: str = "query: "
    embedding_passage_prefix: str = "passage: "
    collection_name: str = "documents"
    jwt_secret: str | None = None
    # JWT hardening: asymmetric algorithms (RS256/ES256) verify with the
    # public key; HS* verify with jwt_secret. "previous" keys accept
    # tokens signed before a key rotation (zero-downtime rotation:
    # add new key, stop issuing with old, drop old after max TTL).
    jwt_algorithm: str = "HS256"
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
    log_level: str = "INFO"
    log_json: bool = True
    metrics_port: int = 8000
    health_check_timeout: float = 2.0
    otel_endpoint: str | None = None

    # Worker role split: "query" (real-time), "background" (ingest/delete/reindex),
    # or "all" (single process runs everything; default for dev).
    worker_queues: str = "all"

    # Embedding model performance.
    embedding_device: str | None = None  # e.g. "cuda", "cpu"; None = auto
    embedding_batch_size: int | None = None
    embedding_query_cache_ttl: float = 3600.0
    embedding_query_cache_size: int = 512

    # LLM answer cache (in-process TTL cache).
    llm_cache_ttl: float = 3600.0
    llm_cache_size: int = 256

    # Qdrant HNSW tuning (None = server defaults).
    qdrant_hnsw_m: int | None = None
    qdrant_hnsw_ef_construct: int | None = None
    qdrant_hnsw_ef: int | None = None

    # LLM generation
    llm_temperature: float = 0.1

    # Circuit breaker guarding the LLM API: after ``llm_circuit_failure_threshold``
    # consecutive failures, generation fails fast with CircuitOpenError for
    # ``llm_circuit_reset_timeout`` seconds (then one trial call is allowed).
    llm_circuit_failure_threshold: int = 5
    llm_circuit_reset_timeout: float = 60.0

    # Optional Redis for distributed caches (LLM answers, query embeddings).
    # When unset, in-process TTL caches are used.
    redis_url: str | None = None
    redis_cache_prefix: str = "rag"

    # Hybrid search: fuse vector hits with lexical BM25 ranking (RRF).
    search_hybrid: bool = False
    search_hybrid_rrf_k: int = 60
    search_hybrid_candidates: int = 50

    # PDF OCR (unstructured partition strategy) for scanned documents.
    pdf_ocr_strategy: str = "auto"  # auto | hi_res | ocr_only | fast
    pdf_ocr_languages: str = "eng"  # comma-separated tesseract languages

    minio_connection_pool_size: int = 10

    @field_validator("llm_provider")
    @classmethod
    def _validate_llm_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized == "openai-compatible":  # alias for self-hosted endpoints
            return "openai"
        if normalized not in {"deepseek", "openai", "anthropic"}:
            raise ValueError(
                "llm_provider must be one of: deepseek, openai, anthropic "
                f"(got {value!r})"
            )
        return normalized

    @field_validator("jwt_algorithm")
    @classmethod
    def _validate_jwt_algorithm(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in (
            "HS256",
            "HS384",
            "HS512",
            "RS256",
            "RS384",
            "RS512",
            "ES256",
        ):
            raise ValueError(f"jwt_algorithm {value!r} is not supported")
        return normalized

    def jwt_verification_keys(self) -> list[str]:
        """Verification keys: current first, then rotation predecessor."""
        if self.jwt_algorithm.startswith("HS"):
            keys = [self.jwt_secret]
            if self.jwt_secret_previous:
                keys.append(self.jwt_secret_previous)
        else:
            keys = [self.jwt_public_key]
            if self.jwt_public_key_previous:
                keys.append(self.jwt_public_key_previous)
        if not keys or not all(keys):
            raise ValueError(
                f"{self.jwt_algorithm} requires the corresponding JWT key "
                "(JWT_SECRET for HS*, JWT_PUBLIC_KEY for RS*/ES*)"
            )
        return keys

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}