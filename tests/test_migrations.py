"""Migration parity checks: Alembic DDL must match the ORM metadata.

Runs Alembic in offline mode (SQL generated without a live DB) and
compares the emitted DDL against the ORM metadata, so a drifted
migration fails the test suite.
"""

import io
import re

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.schema import CreateIndex, CreateColumn

from infrastructure.repositories.postgres_repo import Base


@pytest.fixture(autouse=True)
def _postgres_dsn(monkeypatch):
    # Offline mode never connects; Settings() in env.py needs POSTGRES_DSN
    # so it doesn't demand the app-only DEEPSEEK_API_KEY / JWT_SECRET.
    monkeypatch.setenv("POSTGRES_DSN", "postgresql+asyncpg://u:p@localhost/db")


def _alembic_config(buffer: io.StringIO) -> Config:
    cfg = Config("alembic.ini", output_buffer=buffer)
    # Offline mode never connects; the URL only selects the dialect.
    cfg.set_main_option("sqlalchemy.url", "postgresql+asyncpg://u:p@localhost/db")
    return cfg


def _offline_sql(direction: str, revision: str) -> str:
    buffer = io.StringIO()
    cfg = _alembic_config(buffer)
    if direction == "upgrade":
        command.upgrade(cfg, revision, sql=True)
    else:
        command.downgrade(cfg, revision, sql=True)
    return buffer.getvalue()


def _normalize(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().rstrip(";").strip()


@pytest.fixture()
def migration_sql() -> str:
    return _offline_sql("upgrade", "head")


def test_migration_matches_orm_columns(migration_sql: str):
    """Every ORM column (type + nullability) must appear in the migration."""
    import sqlalchemy.dialects.postgresql as postgresql
    from sqlalchemy.schema import CreateColumn

    for table in Base.metadata.tables.values():
        assert f"CREATE TABLE {table.name}" in migration_sql, table.name
        for column in table.columns:
            column_ddl = _normalize(
                str(CreateColumn(column).compile(dialect=postgresql.dialect()))
            )
            assert column_ddl in migration_sql, (
                f"Migration DDL does not match ORM column "
                f"{table.name}.{column.name!r}: {column_ddl!r}"
            )
        pk = ", ".join(c.name for c in table.primary_key.columns)
        assert f"PRIMARY KEY ({pk})" in migration_sql, table.name


def test_migration_creates_all_orm_indexes(migration_sql: str):
    import sqlalchemy.dialects.postgresql as postgresql
    for table in Base.metadata.tables.values():
        for index in table.indexes:
            assert (
                _normalize(str(CreateIndex(index).compile(dialect=postgresql.dialect())))
                in migration_sql
            ), (
                f"Missing index {index.name!r} in migration DDL"
            )


def test_downgrade_reverses_schema():
    down_sql = _offline_sql("downgrade", "head:base")
    for table_name in Base.metadata.tables:
        assert f"DROP TABLE {table_name}" in re.sub(r"\s+", " ", down_sql)
