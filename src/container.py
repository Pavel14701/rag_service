"""Dishka container configuration."""

from dishka import make_async_container, Provider, Scope, provide
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
import asyncpg
import qdrant_client
from minio import Minio

from config import Settings
from application.services.indexer import IndexerService
from application.services.retriever import RetrieverService
from application.services.document_manager import DocumentManager
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
    def file_storage(self, minio: Minio, settings: Settings) -> MinioStorage:
        return MinioStorage(minio, settings.minio_bucket)

    @provide
    def qdrant_client(self, settings: Settings) -> qdrant_client.QdrantClient:
        return qdrant_client.QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

    @provide
    def vector_store(self, qdrant: qdrant_client.QdrantClient, settings: Settings) -> QdrantStore:
        return QdrantStore(qdrant, settings.collection_name)

    @provide
    def engine(self, settings: Settings):
        return create_async_engine(settings.postgres_dsn, echo=False)

    @provide
    def session_factory(self, engine):
        return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    @provide
    def document_repo(self, session_factory) -> PostgresDocumentRepository:
        return PostgresDocumentRepository(session_factory)

    @provide
    def embedding_model(self, settings: Settings) -> SentenceTransformerEmbedding:
        return SentenceTransformerEmbedding(settings.embedding_model)

    @provide
    def llm_client(self, settings: Settings) -> DeepSeekClient:
        return DeepSeekClient(settings.deepseek_api_key, settings.deepseek_base_url)

    @provide
    def token_validator(self, settings: Settings) -> JWTValidator:
        return JWTValidator(settings.jwt_secret)

    @provide
    def indexer_service(
        self,
        file_storage: MinioStorage,
        vector_store: QdrantStore,
        repo: PostgresDocumentRepository,
        embedding: SentenceTransformerEmbedding,
    ) -> IndexerService:
        return IndexerService(file_storage, vector_store, repo, embedding)

    @provide
    def retriever_service(
        self,
        vector_store: QdrantStore,
        repo: PostgresDocumentRepository,
        embedding: SentenceTransformerEmbedding,
        llm: DeepSeekClient,
    ) -> RetrieverService:
        return RetrieverService(vector_store, repo, embedding, llm)

    @provide
    def document_manager(
        self,
        file_storage: MinioStorage,
        vector_store: QdrantStore,
        repo: PostgresDocumentRepository,
        indexer: IndexerService,
        embedding: SentenceTransformerEmbedding,
    ) -> DocumentManager:
        return DocumentManager(file_storage, vector_store, repo, indexer, embedding)


def create_container():
    return make_async_container(AppProvider())