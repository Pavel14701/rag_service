from typing import Any

"""Tests for PostgresDocumentRepository read/write splitting."""

import uuid
from datetime import datetime, timezone

import pytest

from infrastructure.repositories import PostgresDocumentRepository
from domain.model import Document, DocStatus


class FakeSession:
    """Records which factory created it; mimics AsyncSession surface."""

    def __init__(self, log: list[str], name: str) -> None:
        self._log = log
        self._name = name

    async def __aenter__(self):
        self._log.append(self._name)
        return self

    async def __aexit__(self, *args):
        return False

    async def execute(self, stmt: Any):
        return SimpleNamespace(scalar_one_or_none=lambda: None, scalars=lambda: SimpleNamespace(all=lambda: []))

    async def commit(self) -> None:
        pass

    def add(self, obj) -> None:
        pass


class FakeSessionFactory:
    def __init__(self, log: list[str], name: str) -> None:
        self._log = log
        self._name = name

    def __call__(self):
        return FakeSession(self._log, self._name)


from types import SimpleNamespace  # noqa: E402

pytestmark = pytest.mark.infra


def make_repo():
    log: list[str] = []
    primary = FakeSessionFactory(log, "primary")
    replica = FakeSessionFactory(log, "replica")
    return PostgresDocumentRepository(
        primary,  # type: ignore[arg-type]
        read_session_factory=replica,  # type: ignore[arg-type]
    ), log


async def test_get_document_uses_read_replica() -> None:
    repo, log = make_repo()
    await repo.get_document(uuid.uuid4())
    assert log == ["replica"]


async def test_get_all_active_uses_read_replica() -> None:
    repo, log = make_repo()
    await repo.get_all_active()
    assert log == ["replica"]


async def test_save_uses_primary() -> None:
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


async def test_without_replica_reads_use_primary() -> None:
    log: list[str] = []
    primary = FakeSessionFactory(log, "primary")
    repo = PostgresDocumentRepository(
        primary,  # type: ignore[arg-type]
    )
    await repo.get_document(uuid.uuid4())
    assert log == ["primary"]
