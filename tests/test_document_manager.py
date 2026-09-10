"""Tests for DocumentManager."""

import uuid

import pytest

from application.services.document_manager import DocumentManager
from application.services.indexer import IndexerService
from domain.exceptions import DocumentNotFoundError, PermissionDeniedError

from conftest import make_document


@pytest.fixture
def indexer(repo, storage, vector_store, embedding):
    return IndexerService(storage, vector_store, repo, embedding)


@pytest.fixture
def manager(repo, storage, vector_store, indexer, embedding):
    return DocumentManager(storage, vector_store, repo, indexer, embedding)


async def test_delete_document_success(manager, repo, vector_store):
    doc = make_document(owner_id="u1")
    await repo.save(doc)

    await manager.delete_document(doc.id, user_id="u1")

    assert vector_store.deleted_filters == [
        {"key": "doc_id", "match": {"value": str(doc.id)}}
    ]
    assert doc.id in repo.deleted_ids


async def test_delete_document_not_found(manager):
    with pytest.raises(DocumentNotFoundError):
        await manager.delete_document(uuid.uuid4(), user_id="u1")


async def test_delete_document_already_deleted(manager, repo):
    doc = make_document(deleted=True)
    repo.documents[doc.id] = doc
    with pytest.raises(DocumentNotFoundError):
        await manager.delete_document(doc.id, user_id="u1")


async def test_delete_document_not_owner(manager, repo):
    doc = make_document(owner_id="u1")
    await repo.save(doc)
    with pytest.raises(PermissionDeniedError):
        await manager.delete_document(doc.id, user_id="intruder")


async def test_delete_document_with_file_removal(manager, repo, storage):
    doc = make_document(owner_id="u1")
    await repo.save(doc)

    await manager.delete_document(doc.id, user_id="u1", remove_file=True)
    assert storage.deleted_keys == [doc.file_path]


async def test_delete_document_keeps_file_by_default(manager, repo, storage):
    doc = make_document(owner_id="u1")
    await repo.save(doc)

    await manager.delete_document(doc.id, user_id="u1")
    assert storage.deleted_keys == []


async def test_reindex_all_admin(manager, repo, vector_store, embedding):
    await repo.save(make_document(owner_id="admin_1"))
    await repo.save(make_document(owner_id="admin_1"))
    await repo.save(make_document(owner_id="someone", deleted=True))

    await manager.reindex_all("admin_1")

    assert vector_store.dropped == 1
    assert vector_store.created_dimensions == [embedding.dim]


async def test_reindex_all_non_admin_denied(manager):
    with pytest.raises(PermissionDeniedError):
        await manager.reindex_all("regular_user")


async def test_reindex_all_admin_via_group(manager, repo):
    repo.user_groups["member"] = ["admin", "other"]
    await repo.save(make_document())
    await manager.reindex_all("member")


async def test_reindex_all_explicit_dimension(manager, repo, vector_store):
    await repo.save(make_document(owner_id="admin_1"))
    await manager.reindex_all("admin_1", vector_dimension=128)
    assert vector_store.created_dimensions == [128]


async def test_reindex_all_continues_after_failure(
    manager, repo, vector_store, embedding, monkeypatch
):
    doc1 = make_document(owner_id="admin_1")
    doc2 = make_document(owner_id="admin_1")
    await repo.save(doc1)
    await repo.save(doc2)

    original = manager._indexer.index_document
    calls = []

    async def flaky_index(doc_id):
        calls.append(doc_id)
        if doc_id == doc1.id:
            raise RuntimeError("boom")
        await original(doc_id)

    monkeypatch.setattr(manager._indexer, "index_document", flaky_index)

    # Must not raise even though the first document failed
    await manager.reindex_all("admin_1")
    assert calls == [doc1.id, doc2.id]