"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    postgres_dsn: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "documents"
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    embedding_model: str = "intfloat/multilingual-e5-small"
    collection_name: str = "documents"
    jwt_secret: str
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}