"""Dishka container configuration."""

from dishka import make_async_container, Provider, Scope, provide
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from qdrant_client import QdrantClient
from minio import Minio

from config import Settings
from shared.caching import TTLCache, RedisCache, SharedCaches
from shared.circuit_breaker import CircuitBreaker
from application.services.indexer import IndexerService
from application.services.retriever import RetrieverService
from application.services.document_manager import DocumentManager
from application.interfaces import (
    DocumentRepository,
    EmbeddingModel,
    FileStorage,
    LLMGenerator,
    TokenValidator,
    VectorStore,
)
from infrastructure.file_storage.minio_storage import MinioStorage
from infrastructure.parsing.factory import ParserFactory
from infrastructure.vector_store.qdrant_store import QdrantStore
from infrastructure.repositories.postgres_repo import PostgresDocumentRepository
from infrastructure.embedding.sentence_transformer import SentenceTransformerEmbedding
from infrastructure.llm.deepseek_client import DeepSeekClient
from infrastructure.llm.openai_client import OpenAIChatClient
from infrastructure.llm.anthropic_client import AnthropicClient
from infrastructure.security.jwt_validator import JWTValidator


class AppProvider(Provider):
    scope = Scope.APP

    @provide
    def settings(self) -> Settings:
        return Settings()

    @provide
    def minio_client(self, settings: Settings) -> Minio:
        return Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
            connection_pool_size=settings.minio_connection_pool_size,
        )

    @provide
    def file_storage(self, minio: Minio, settings: Settings) -> FileStorage:
        return MinioStorage(minio, settings.minio_bucket)

    @provide
    def qdrant_client(self, settings: Settings) -> QdrantClient:
        return QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            https=settings.qdrant_https,
            api_key=settings.qdrant_api_key,
        )

    @provide
    def vector_store(
        self, qdrant: QdrantClient, settings: Settings
    ) -> VectorStore:
        return QdrantStore(
            qdrant,
            settings.collection_name,
            hnsw_m=settings.qdrant_hnsw_m,
            hnsw_ef_construct=settings.qdrant_hnsw_ef_construct,
            hnsw_ef=settings.qdrant_hnsw_ef,
            fulltext_enabled=settings.search_hybrid,
            hybrid_candidates=settings.search_hybrid_candidates,
            hybrid_rrf_k=settings.search_hybrid_rrf_k,
        )

    @provide
    def engine(self, settings: Settings) -> AsyncEngine:
        return create_async_engine(settings.postgres_dsn, echo=False)

    @provide
    def session_factory(
        self, engine: AsyncEngine
    ) -> async_sessionmaker[AsyncSession]:
        return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    @provide
    def document_repo(
        self, session_factory: async_sessionmaker[AsyncSession], settings: Settings
    ) -> DocumentRepository:
        # Optional read replica: read queries go to the replica when
        # POSTGRES_READ_DSN is configured; writes always hit the primary.
        read_session_factory = None
        if settings.postgres_read_dsn:
            read_engine = create_async_engine(settings.postgres_read_dsn, echo=False)
            read_session_factory = async_sessionmaker(
                read_engine, expire_on_commit=False, class_=AsyncSession
            )
        return PostgresDocumentRepository(session_factory, read_session_factory)

    @provide
    def shared_caches(self, settings: Settings) -> SharedCaches:
        """Optional shared Redis cache (redis=None -> in-process TTL caches)."""
        redis_cache = None
        if settings.redis_url:
            redis_cache = RedisCache(
                url=settings.redis_url,
                ttl=settings.llm_cache_ttl,
                prefix=settings.redis_cache_prefix,
            )
        return SharedCaches(redis=redis_cache)

    @provide
    def embedding_model(
        self, settings: Settings, caches: SharedCaches
    ) -> EmbeddingModel:
        return SentenceTransformerEmbedding(
            settings.embedding_model,
            query_prefix=settings.embedding_query_prefix,
            passage_prefix=settings.embedding_passage_prefix,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
            query_cache=caches.pick(
                maxsize=settings.embedding_query_cache_size,
                ttl=settings.embedding_query_cache_ttl,
            ),
        )

    @provide
    def llm_circuit_breaker(self, settings: Settings) -> CircuitBreaker:
        return CircuitBreaker(
            name="llm",
            failure_threshold=settings.llm_circuit_failure_threshold,
            reset_timeout=settings.llm_circuit_reset_timeout,
        )

    @provide
    def llm_client(
        self, settings: Settings, breaker: CircuitBreaker, caches: SharedCaches
    ) -> LLMGenerator:
        """Build the LLM client for the configured provider.

        Supported providers (``LLM_PROVIDER``): ``deepseek`` (default),
        ``openai`` (any OpenAI chat-completions compatible endpoint —
        OpenAI, vLLM, Ollama, Together, ...) and ``anthropic``.
        ``LLM_MODEL`` overrides the provider default model.
        """
        cache = caches.pick(maxsize=settings.llm_cache_size, ttl=settings.llm_cache_ttl)
        model = settings.llm_model

        if settings.llm_provider == "deepseek":
            if not settings.deepseek_api_key:
                raise ValueError(
                    "DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek"
                )
            return DeepSeekClient(
                settings.deepseek_api_key,
                settings.deepseek_base_url,
                model=model or "deepseek-chat",
                max_tokens=settings.llm_max_tokens,
                cache=cache,
                circuit_breaker=breaker,
            )

        if settings.llm_provider == "openai":
            if not settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when LLM_PROVIDER=openai"
                )
            return OpenAIChatClient(
                settings.openai_api_key,
                settings.openai_base_url,
                model=model or "gpt-4o-mini",
                max_tokens=settings.llm_max_tokens,
                cache=cache,
                circuit_breaker=breaker,
            )

        # settings.llm_provider == "anthropic" (validated in Settings)
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic"
            )
        return AnthropicClient(
            settings.anthropic_api_key,
            settings.anthropic_base_url,
            model=model or "claude-3-5-haiku-latest",
            max_tokens=settings.llm_max_tokens,
            cache=cache,
            circuit_breaker=breaker,
        )

    @provide
    def token_validator(self, settings: Settings) -> TokenValidator:
        from infrastructure.security.token_blacklist import RedisTokenBlacklist

        blacklist = None
        if settings.redis_url:
            import redis as redis_lib

            blacklist = RedisTokenBlacklist(
                redis_lib.Redis.from_url(settings.redis_url),
                prefix=f"{settings.redis_cache_prefix}:revoked-jti",
            )
        return JWTValidator(
            settings.jwt_verification_keys(),
            algorithm=settings.jwt_algorithm,
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            blacklist=blacklist,
        )

    @provide
    def indexer_service(
        self,
        file_storage: FileStorage,
        vector_store: VectorStore,
        repo: DocumentRepository,
        embedding: EmbeddingModel,
        settings: Settings,
    ) -> IndexerService:
        return IndexerService(
            file_storage,
            vector_store,
            repo,
            embedding,
            embedding_version=settings.embedding_model,
        )

    @provide
    def retriever_service(
        self,
        vector_store: VectorStore,
        repo: DocumentRepository,
        embedding: EmbeddingModel,
        llm: LLMGenerator,
        settings: Settings,
    ) -> RetrieverService:
        pii_redactor = None
        if settings.anonymize_conversations:
            from shared.pii import redact_pii

            pii_redactor = redact_pii
        return RetrieverService(
            vector_store,
            repo,
            embedding,
            llm,
            default_temperature=settings.llm_temperature,
            hybrid_enabled=settings.search_hybrid,
            hybrid_rrf_k=settings.search_hybrid_rrf_k,
            pii_redactor=pii_redactor,
        )

    @provide
    def document_manager(
        self,
        file_storage: FileStorage,
        vector_store: VectorStore,
        repo: DocumentRepository,
        indexer: IndexerService,
        embedding: EmbeddingModel,
        settings: Settings,
    ) -> DocumentManager:
        return DocumentManager(
            file_storage,
            vector_store,
            repo,
            indexer,
            embedding,
            embedding_model_name=settings.embedding_model,
        )


def create_container():
    return make_async_container(AppProvider())