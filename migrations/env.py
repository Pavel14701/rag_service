"""Alembic environment: async engine, URL from POSTGRES_DSN (env / .env)."""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# ORM metadata for autogenerate.
from infrastructure.repositories import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """POSTGRES_DSN from the environment, falling back to Settings (.env)."""
    url = os.environ.get("POSTGRES_DSN")
    if url:
        return url
    from config import Settings  # requires DEEPSEEK_API_KEY / JWT_SECRET in .env

    return Settings().postgres_dsn


def run_migrations_offline() -> None:
    """Offline mode: emit SQL to stdout without a DB connection."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Online mode: run migrations over an async (asyncpg) engine."""
    config.set_main_option("sqlalchemy.url", _database_url())
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
