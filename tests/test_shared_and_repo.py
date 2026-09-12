"""Tests for hashing utility and Postgres<->Domain mapping."""

import hashlib

from domain.model import DocStatus
from infrastructure.file_storage import compute_file_hash
from infrastructure.repositories import DocumentORM, PostgresDocumentRepository

from conftest import make_document

import pytest

pytestmark = pytest.mark.infra



def test_compute_file_hash(tmp_path) -> None:
    path = tmp_path / "file.bin"
    path.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert compute_file_hash(path) == expected


def test_compute_file_hash_chunked_read(tmp_path) -> None:
    path = tmp_path / "big.bin"
    data = b"x" * (4096 * 3 + 7)  # spans several 4096-byte chunks
    path.write_bytes(data)
    assert compute_file_hash(path) == hashlib.sha256(data).hexdigest()


def test_orm_domain_roundtrip() -> None:
    doc = make_document(status=DocStatus.INDEXED)
    orm = PostgresDocumentRepository._from_domain(doc)
    assert isinstance(orm, DocumentORM)
    assert orm.status == "indexed"
    assert orm.owner_id == doc.owner_id

    restored = PostgresDocumentRepository._to_domain(orm)
    assert restored == doc


def test_deleted_filter_compiles() -> None:
    """Regression: `not DocumentORM.deleted` crashed with TypeError in SQLAlchemy."""
    from sqlalchemy import select

    stmt = select(DocumentORM).where(DocumentORM.deleted.is_(False))
    compiled = str(stmt)
    assert "deleted" in compiled
