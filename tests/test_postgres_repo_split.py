"""Tests for PostgresDocumentRepository read/write splitting."""

import uuid
from datetime import datetime, timezone

import pytest

from infrastructure.repositories.postgres_repo import PostgresDocumentRepository
from domain.entities.document import Document, DocStatus


class FakeSession:
    """Records which factory created it; mimics AsyncSession surface."""

    def __init__(self, log: list[str], name: str):
        self._log = log
        self._name = name

    async def __aenter__(self):
        self._log.append(self._name)
        return self

    async def __aexit__(self, *args):
        return False

    async def execute(self, stmt):
        return SimpleNamespace(scalar_one_or_none=lambda: None, scalars=lambda: SimpleNamespace(all=lambda: []))

    async def commit(self):
        pass

    def add(self, obj):
        pass


class FakeSessionFactory:
    def __init__(self, log: list[str], name: str):
        self._log = log
        self._name = name

    def __call__(self):
        return FakeSession(self._log, self._name)


from types import SimpleNamespace  # noqa: E402


def make_repo():
    log: list[str] = []
    primary = FakeSessionFactory(log, "primary")
    replica = FakeSessionFactory(log, "replica")
    return PostgresDocumentRepository(primary, read_session_factory=replica), log


async def test_get_document_uses_read_replica():
    repo, log = make_repo()
    await repo.get_document(uuid.uuid4())
    assert log == ["replica"]


async def test_get_all_active_uses_read_replica():
    repo, log = make_repo()
    await repo.get_all_active()
    assert log == ["replica"]


async def test_save_uses_primary():
    repo, log = make_repo()
    doc = Document(
        id=uuid.uuid4(),
        file_name="f.md",
        file_path="docs/f.md",
        file_hash="h",
        owner_id="u",
        access_group=None,
        uploaded_at=datetime.now(timezone.utc),
        status=DocStatus.PENDING,
        deleted=False,
    )
    await repo.save(doc)
    assert log == ["primary"]


async def test_without_replica_reads_use_primary():
    log: list[str] = []
    primary = FakeSessionFactory(log, "primary")
    repo = PostgresDocumentRepository(primary)
    await repo.get_document(uuid.uuid4())
    assert log == ["primary"]