from pathlib import Path

"""Tests for DocumentManager."""

import uuid

import pytest

from application.services import DocumentManager
from application.services import IndexerService

from infrastructure.parsing import DocumentParserSelector


class FakeSelector:
    """Delegates to the real extension-based selector."""

    def __init__(self) -> None:
        self._selector = DocumentParserSelector()

    def get_parser(self, file_path: Path):
        return self._selector.get_parser(file_path)
from domain.model import DocumentNotFoundError, PermissionDeniedError

from conftest import (
    FakeDocumentRepository,
    FakeEmbedding,
    FakeFileStorage,
    FakeVectorStore,
    make_document,
)

pytestmark = pytest.mark.indexing


@pytest.fixture
def indexer(repo: FakeDocumentRepository, storage: FakeFileStorage, vector_store: FakeVectorStore, embedding: FakeEmbedding):
    return IndexerService(storage, vector_store, repo, embedding, FakeSelector())


@pytest.fixture
def manager(repo: FakeDocumentRepository, storage: FakeFileStorage, vector_store: FakeVectorStore, indexer: IndexerService, embedding: FakeEmbedding):
    return DocumentManager(storage, vector_store, repo, indexer, embedding)


async def test_delete_document_success(manager, repo, vector_store) -> None:
    doc = make_document(owner_id="u1")
    await repo.save(doc)

    await manager.delete_document(doc.id, user_id="u1")

    assert vector_store.deleted_filters == [
        {"key": "doc_id", "match": {"value": str(doc.id)}}
    ]
    assert doc.id in repo.deleted_ids


async def test_delete_document_not_found(manager) -> None:
    with pytest.raises(DocumentNotFoundError):
        await manager.delete_document(uuid.uuid4(), user_id="u1")


async def test_delete_document_already_deleted(manager, repo) -> None:
    doc = make_document(deleted=True)
    repo.documents[doc.id] = doc
    with pytest.raises(DocumentNotFoundError):
        await manager.delete_document(doc.id, user_id="u1")


async def test_delete_document_not_owner(manager, repo) -> None:
    doc = make_document(owner_id="u1")
    await repo.save(doc)
    with pytest.raises(PermissionDeniedError):
        await manager.delete_document(doc.id, user_id="intruder")


async def test_delete_document_with_file_removal(manager, repo, storage) -> None:
    doc = make_document(owner_id="u1")
    await repo.save(doc)

    await manager.delete_document(doc.id, user_id="u1", remove_file=True)
    assert storage.deleted_keys == [doc.file_path]


async def test_delete_document_keeps_file_by_default(manager, repo, storage) -> None:
    doc = make_document(owner_id="u1")
    await repo.save(doc)

    await manager.delete_document(doc.id, user_id="u1")
    assert storage.deleted_keys == []


async def test_reindex_all_admin_blue_green(
    manager, repo, vector_store, embedding,
):
    await repo.save(make_document(owner_id="admin_1"))
    await repo.save(make_document(owner_id="admin_1"))
    await repo.save(make_document(owner_id="someone", deleted=True))

    await manager.reindex_all("admin_1")

    # live collection is never dropped: the shadow flips atomically
    assert vector_store.dropped == 0
    assert vector_store.promoted == 1
    assert vector_store.created_dimensions == [embedding.dim]
    assert vector_store.shadow_name is None  # promoted, not dangling


async def test_reindex_all_legacy_path_when_blue_green_disabled(
    storage, repo, vector_store, embedding, indexer,
):
    from application.services import DocumentManager

    manager = DocumentManager(
        storage,
        vector_store,
        repo,
        indexer,
        embedding,
        blue_green=False,
    )
    await repo.save(make_document(owner_id="admin_1"))

    await manager.reindex_all("admin_1")

    assert vector_store.dropped == 1
    assert vector_store.created_dimensions == [embedding.dim]
    assert vector_store.promoted == 0


async def test_reindex_all_non_admin_denied(manager) -> None:
    with pytest.raises(PermissionDeniedError):
        await manager.reindex_all("regular_user")


async def test_reindex_all_admin_via_group(manager, repo) -> None:
    repo.user_groups["member"] = ["admin", "other"]
    await repo.save(make_document())
    await manager.reindex_all("member")


async def test_reindex_all_explicit_dimension(manager, repo, vector_store) -> None:
    await repo.save(make_document(owner_id="admin_1"))
    await manager.reindex_all("admin_1", vector_dimension=128)
    assert vector_store.created_dimensions == [128]
    assert vector_store.promoted == 1


async def test_reindex_all_continues_after_failure(
    manager, repo, vector_store, embedding, monkeypatch
):
    doc1 = make_document(owner_id="admin_1")
    doc2 = make_document(owner_id="admin_1")
    await repo.save(doc1)
    await repo.save(doc2)

    original = manager._indexer.index_document
    calls = []

    async def flaky_index(doc_id) -> None:
        calls.append(doc_id)
        if doc_id == doc1.id:
            raise RuntimeError("boom")
        await original(doc_id)

    monkeypatch.setattr(manager._indexer, "index_document", flaky_index)

    # Must not raise even though the first document failed
    await manager.reindex_all("admin_1")
    assert calls == [doc1.id, doc2.id]
