"""Dishka container configuration."""

from dishka import make_async_container, Provider, Scope, provide
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
import qdrant_client
from minio import Minio

from config import Settings
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
from infrastructure.vector_store.qdrant_store import QdrantStore
from infrastructure.repositories.postgres_repo import PostgresDocumentRepository
from infrastructure.embedding.sentence_transformer import SentenceTransformerEmbedding
from infrastructure.llm.deepseek_client import DeepSeekClient
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
            secure=False,
        )

    @provide
    def file_storage(self, minio: Minio, settings: Settings) -> FileStorage:
        return MinioStorage(minio, settings.minio_bucket)

    @provide
    def qdrant_client(self, settings: Settings) -> qdrant_client.QdrantClient:
        return qdrant_client.QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

    @provide
    def vector_store(
        self, qdrant: qdrant_client.QdrantClient, settings: Settings
    ) -> VectorStore:
        return QdrantStore(qdrant, settings.collection_name)

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
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> DocumentRepository:
        return PostgresDocumentRepository(session_factory)

    @provide
    def embedding_model(self, settings: Settings) -> EmbeddingModel:
        return SentenceTransformerEmbedding(settings.embedding_model)

    @provide
    def llm_client(self, settings: Settings) -> LLMGenerator:
        return DeepSeekClient(settings.deepseek_api_key, settings.deepseek_base_url)

    @provide
    def token_validator(self, settings: Settings) -> TokenValidator:
        return JWTValidator(settings.jwt_secret)

    @provide
    def indexer_service(
        self,
        file_storage: FileStorage,
        vector_store: VectorStore,
        repo: DocumentRepository,
        embedding: EmbeddingModel,
    ) -> IndexerService:
        return IndexerService(file_storage, vector_store, repo, embedding)

    @provide
    def retriever_service(
        self,
        vector_store: VectorStore,
        repo: DocumentRepository,
        embedding: EmbeddingModel,
        llm: LLMGenerator,
    ) -> RetrieverService:
        return RetrieverService(vector_store, repo, embedding, llm)

    @provide
    def document_manager(
        self,
        file_storage: FileStorage,
        vector_store: VectorStore,
        repo: DocumentRepository,
        indexer: IndexerService,
        embedding: EmbeddingModel,
    ) -> DocumentManager:
        return DocumentManager(file_storage, vector_store, repo, indexer, embedding)


def create_container():
    return make_async_container(AppProvider())