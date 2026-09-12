"""Dishka container configuration."""

from dishka import (
    AsyncContainer,
    make_async_container,
    Provider,
    Scope,
    provide,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from qdrant_client import QdrantClient
import httpx
from minio import Minio

from config import Settings
from infrastructure.caching import RedisCache, SharedCaches
from infrastructure.resilience import CircuitBreaker
from application.services import IndexerService
from application.services import RetrieverService
from application.services import DocumentManager
from application.interfaces import (
    DistributedLock,
    DocumentGrader,
    EntityExtractor,
    GraphStore,
    DocumentRepository,
    EmbeddingModel,
    FileStorage,
    LLMGenerator,
    ParserSelector,
    QueryPlanner,
    QueryRewriter,
    SemanticCache,
    TokenValidator,
    VectorStore,
)
from infrastructure.file_storage import MinioStorage
from infrastructure.vector_store import QdrantStore
from infrastructure.repositories import (
    PostgresDocumentRepository,
)
from infrastructure.embedding import (
    SentenceTransformerEmbedding,
)
from infrastructure.llm import DeepSeekClient
from infrastructure.llm import OpenAIChatClient
from infrastructure.llm import AnthropicClient
from infrastructure.llm import LLMProviderSpec, LLMRouter
from infrastructure.security import JWTValidator


class AppProvider(Provider):
    """Dependency-inversion wiring for application components."""

    scope = Scope.APP

    @provide
    def settings(self) -> Settings:
        """Build application settings from environment variables."""
        return Settings()

    @provide
    def minio_client(self, settings: Settings) -> Minio:
        """Build the MinIO client from settings."""
        return Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )

    @provide
    def file_storage(self, minio: Minio, settings: Settings) -> FileStorage:
        """Expose MinIO storage behind the FileStorage port."""
        return MinioStorage(minio, settings.minio_bucket)

    @provide
    def qdrant_client(self, settings: Settings) -> QdrantClient:
        """Build the Qdrant client (HTTPS/API-key/gRPC aware).

        gRPC message-size caps prevent a huge batch response from
        exhausting worker memory during bulk (re)indexing.
        """
        grpc_options = {
            'grpc.max_receive_message_length': 32 * 1024 * 1024,
            'grpc.max_send_message_length': 32 * 1024 * 1024,
        }
        return QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            https=settings.qdrant_https,
            api_key=settings.qdrant_api_key,
            prefer_grpc=settings.qdrant_prefer_grpc,
            grpc_options=grpc_options,
        )

    @provide
    def vector_store(
        self, qdrant: QdrantClient, settings: Settings
    ) -> VectorStore:
        """Build the Qdrant-backed vector store adapter."""
        return QdrantStore(
            qdrant,
            settings.collection_name,
            hnsw_m=settings.qdrant_hnsw_m,
            hnsw_ef_construct=settings.qdrant_hnsw_ef_construct,
            hnsw_ef=settings.qdrant_hnsw_ef,
            fulltext_enabled=settings.search_hybrid,
            hybrid_candidates=settings.search_hybrid_candidates,
            hybrid_rrf_k=settings.search_hybrid_rrf_k,
            bm25_stopwords=frozenset(
                w.strip()
                for w in settings.search_hybrid_stopwords.split(',')
                if w.strip()
            ),
            bm25_score_threshold=settings.search_bm25_score_threshold,
        )

    @provide
    def http_client(self, settings: Settings) -> httpx.AsyncClient:
        """Shared pooled HTTP client (singleton socket budget).

        LLM adapters must reuse this client instead of building a
        per-call one: connection storms exhaust host sockets and
        TLS handshakes dominate latency under high load.
        """
        return httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=settings.http_keepalive_connections,
                max_connections=settings.http_max_connections,
                keepalive_expiry=30.0,
            ),
            # Guards against hung AI providers.
            timeout=httpx.Timeout(20.0, connect=5.0),
        )

    @provide
    def engine(self, settings: Settings) -> AsyncEngine:
        """Create the async SQLAlchemy engine."""
        return create_async_engine(settings.postgres_dsn, echo=False)

    @provide
    def session_factory(
        self, engine: AsyncEngine
    ) -> async_sessionmaker[AsyncSession]:
        """Create the async session factory for repositories."""
        return async_sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )

    @provide
    def document_repo(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> DocumentRepository:
        """Build the Postgres-backed document repository."""
        # Optional read replica: read queries go to the replica when
        # POSTGRES_READ_DSN is configured; writes always hit the primary.
        read_session_factory = None
        if settings.postgres_read_dsn:
            read_engine = create_async_engine(
                settings.postgres_read_dsn, echo=False
            )
            read_session_factory = async_sessionmaker(
                read_engine, expire_on_commit=False, class_=AsyncSession
            )
        return PostgresDocumentRepository(
            session_factory, read_session_factory
        )

    @provide
    def shared_caches(self, settings: Settings) -> SharedCaches:
        """Optional shared Redis cache (None -> in-process TTL caches)."""
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
        """Build the sentence-transformers embedding adapter."""
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
        """Build the circuit breaker guarding LLM API calls."""
        return CircuitBreaker(
            name='llm',
            failure_threshold=settings.llm_circuit_failure_threshold,
            reset_timeout=settings.llm_circuit_reset_timeout,
        )

    @provide
    def llm_client(
        self,
        settings: Settings,
        breaker: CircuitBreaker,
        caches: SharedCaches,
        http_client: httpx.AsyncClient,
    ) -> LLMGenerator:
        """Build the LLM router.

        ``LLM_PROVIDER`` selects the default provider;
        ``LLM_ENABLED_PROVIDERS`` (comma-separated) enables per-request
        routing among several providers (request fields ``llm_provider`` /
        ``llm_model`` in the query message). ``LLM_ALLOWED_MODELS``
        restricts explicit model overrides. Each enabled provider requires
        its API key at startup. Supported providers: ``deepseek``,
        ``openai`` (any OpenAI chat-completions compatible endpoint) and
        ``anthropic``.
        """
        cache = caches.pick(
            maxsize=settings.llm_cache_size, ttl=settings.llm_cache_ttl
        )

        deepseek_key = settings.deepseek_api_key or ''
        openai_key = settings.openai_api_key or ''
        anthropic_key = settings.anthropic_api_key or ''

        def deepseek_factory(model: str) -> LLMGenerator:
            return DeepSeekClient(
                deepseek_key,
                settings.deepseek_base_url,
                model=model,
                max_tokens=settings.llm_max_tokens,
                cache=cache,
                circuit_breaker=breaker,
                continue_on_truncation=settings.llm_continue_on_truncation,
                http_client=http_client,
            )

        def openai_factory(model: str) -> LLMGenerator:
            return OpenAIChatClient(
                openai_key,
                settings.openai_base_url,
                model=model,
                max_tokens=settings.llm_max_tokens,
                cache=cache,
                circuit_breaker=breaker,
                continue_on_truncation=settings.llm_continue_on_truncation,
                http_client=http_client,
            )

        def anthropic_factory(model: str) -> LLMGenerator:
            return AnthropicClient(
                anthropic_key,
                settings.anthropic_base_url,
                model=model,
                max_tokens=settings.llm_max_tokens,
                cache=cache,
                circuit_breaker=breaker,
                continue_on_truncation=settings.llm_continue_on_truncation,
                http_client=http_client,
            )

        factories = {
            'deepseek': deepseek_factory,
            'openai': openai_factory,
            'anthropic': anthropic_factory,
        }
        defaults = {
            'deepseek': 'deepseek-chat',
            'openai': 'gpt-4o-mini',
            'anthropic': 'claude-3-5-haiku-latest',
        }
        keys = {
            'deepseek': (deepseek_key, 'DEEPSEEK_API_KEY'),
            'openai': (openai_key, 'OPENAI_API_KEY'),
            'anthropic': (anthropic_key, 'ANTHROPIC_API_KEY'),
        }

        enabled = settings.llm_enabled_providers_list
        if settings.llm_provider not in enabled:
            raise ValueError(
                f'LLM_PROVIDER={settings.llm_provider!r} must be among '
                f'LLM_ENABLED_PROVIDERS={enabled}'
            )
        specs: dict[str, LLMProviderSpec] = {}
        for name in enabled:
            api_key, env_name = keys[name]
            if not api_key:
                raise ValueError(
                    f'{env_name} is required when provider {name!r} is enabled'
                )
            specs[name] = LLMProviderSpec(
                factory=factories[name],
                default_model=settings.llm_model or defaults[name],
            )
        return LLMRouter(
            specs,
            settings.llm_provider,
            allowed_models=settings.llm_allowed_models_list or None,
        )

    @provide
    def token_validator(self, settings: Settings) -> TokenValidator:
        """Build the JWT validator with rotation keys and blacklist."""
        from infrastructure.security import RedisTokenBlacklist

        blacklist = None
        if settings.redis_url:
            import redis as redis_lib

            blacklist = RedisTokenBlacklist(
                redis_lib.Redis.from_url(settings.redis_url),
                prefix=f'{settings.redis_cache_prefix}:revoked-jti',
            )
        return JWTValidator(
            settings.jwt_verification_keys(),
            algorithm=settings.jwt_algorithm,
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            blacklist=blacklist,
        )

    @provide
    def parser_selector(self, settings: Settings) -> ParserSelector:
        """Build the parser selector with OCR settings from config."""
        from infrastructure.parsing import DocumentParserSelector

        return DocumentParserSelector(
            pdf_ocr_strategy=settings.pdf_ocr_strategy,
            pdf_ocr_languages=settings.pdf_ocr_languages,
        )

    @provide
    def semantic_cache(self, settings: Settings) -> SemanticCache:
        """Near-duplicate answer cache (no-op while disabled)."""
        from infrastructure.caching import InMemorySemanticCache

        return InMemorySemanticCache(
            maxsize=settings.semantic_cache_maxsize,
            ttl=settings.semantic_cache_ttl,
            threshold=settings.semantic_cache_threshold,
            enabled=settings.semantic_cache_enabled,
        )

    @provide
    def query_rewriter(
        self, settings: Settings, llm: LLMGenerator
    ) -> QueryRewriter:
        """LLM query rewriter; pass-through when disabled."""
        from infrastructure.llm import LLMQueryRewriter, NoOpQueryRewriter

        if settings.query_rewrite_enabled:
            return LLMQueryRewriter(llm)
        return NoOpQueryRewriter()

    @provide
    def document_grader(
        self, settings: Settings, llm: LLMGenerator
    ) -> DocumentGrader:
        """LLM hit grader for agentic retrieval; pass-through when off."""
        from infrastructure.agentic import (
            LLMDocumentGrader,
            NoOpDocumentGrader,
        )

        if settings.agentic_grader_enabled:
            return LLMDocumentGrader(llm)
        return NoOpDocumentGrader()

    @provide
    def query_planner(
        self, settings: Settings, llm: LLMGenerator
    ) -> QueryPlanner:
        """LLM sub-question planner; pass-through when disabled."""
        from infrastructure.agentic import LLMQueryPlanner, NoOpQueryPlanner

        if settings.agentic_planner_enabled:
            return LLMQueryPlanner(llm)
        return NoOpQueryPlanner()

    @provide
    def entity_extractor(
        self, settings: Settings, llm: LLMGenerator
    ) -> EntityExtractor:
        """LLM relation extractor for GraphRAG; no-op when disabled."""
        from infrastructure.graph_extract import (
            LLMEntityExtractor,
            NoOpEntityExtractor,
        )

        if settings.graph_extract_enabled or settings.graph_expand_enabled:
            return LLMEntityExtractor(llm)
        return NoOpEntityExtractor()

    @provide
    def graph_store(
        self,
        settings: Settings,
        qdrant: QdrantClient,
        embedding: EmbeddingModel,
    ) -> GraphStore:
        """Entity pseudo-graph in Qdrant; optional Neo4j backend."""
        if settings.graph_backend == 'neo4j':
            from infrastructure.graph_store_neo4j import Neo4jGraphStore

            if not (settings.neo4j_uri and settings.neo4j_user):
                msg = 'GRAPH_BACKEND=neo4j requires NEO4J_URI/NEO4J_USER'
                raise ValueError(msg)
            return Neo4jGraphStore(
                uri=settings.neo4j_uri,
                user=settings.neo4j_user,
                password=settings.neo4j_password,
            )
        if settings.graph_backend != 'qdrant':
            msg = f'Unknown GRAPH_BACKEND: {settings.graph_backend}'
            raise ValueError(msg)
        from infrastructure.graph_store import QdrantGraphStore

        return QdrantGraphStore(
            client=qdrant,
            entities_collection=f'{settings.collection_name}__entities',
            embedding=embedding,
        )

    @provide
    def distributed_lock(self, settings: Settings) -> DistributedLock:
        """Redis lock across workers; in-process fallback without Redis."""
        if settings.redis_url:
            from redis.asyncio import Redis

            from infrastructure.resilience import RedisDistributedLock

            return RedisDistributedLock(Redis.from_url(settings.redis_url))
        from infrastructure.resilience import InProcessDistributedLock

        return InProcessDistributedLock()

    @provide
    def indexer_service(
        self,
        file_storage: FileStorage,
        vector_store: VectorStore,
        repo: DocumentRepository,
        embedding: EmbeddingModel,
        parser_selector: ParserSelector,
        lock: DistributedLock,
        entity_extractor: EntityExtractor,
        graph_store: GraphStore,
        settings: Settings,
    ) -> IndexerService:
        """Build the document indexing service."""
        return IndexerService(
            file_storage,
            vector_store,
            repo,
            embedding,
            parser_selector,
            embedding_version=settings.embedding_model,
            lock=lock,
            lock_ttl=settings.index_lock_ttl,
            chunk_min_chars=settings.chunk_min_chars,
            parse_timeout=settings.parse_timeout,
            child_chars=settings.parent_child_child_chars,
            entity_extractor=entity_extractor,
            graph_store=graph_store,
        )

    @provide
    def retriever_service(
        self,
        vector_store: VectorStore,
        repo: DocumentRepository,
        embedding: EmbeddingModel,
        llm: LLMGenerator,
        semantic_cache: SemanticCache,
        query_rewriter: QueryRewriter,
        document_grader: DocumentGrader,
        query_planner: QueryPlanner,
        entity_extractor: EntityExtractor,
        graph_store: GraphStore,
        settings: Settings,
    ) -> RetrieverService:
        """Build the retrieval and answer service."""
        pii_redactor = None
        if settings.anonymize_conversations:
            from domain.pii import redact_pii

            pii_redactor = redact_pii
        return RetrieverService(
            vector_store,
            repo,
            embedding,
            llm,
            default_temperature=settings.llm_temperature,
            hybrid_enabled=settings.search_hybrid,
            hybrid_rrf_k=settings.search_hybrid_rrf_k,
            score_threshold=settings.search_score_threshold,
            context_max_chars=settings.search_context_max_chars,
            semantic_cache=semantic_cache,
            query_rewriter=query_rewriter,
            document_grader=document_grader,
            query_planner=query_planner,
            agentic_max_rounds=settings.agentic_max_rounds,
            agentic_subquery_limit=settings.agentic_subquery_limit,
            entity_extractor=entity_extractor,
            graph_store=graph_store,
            graph_max_hops=settings.graph_max_hops,
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
        """Build the document management service."""
        return DocumentManager(
            file_storage,
            vector_store,
            repo,
            indexer,
            embedding,
            embedding_model_name=settings.embedding_model,
            blue_green=settings.reindex_blue_green,
        )


def create_container() -> AsyncContainer:
    """Assemble the async DI container."""
    return make_async_container(AppProvider())
