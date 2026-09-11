"""Tests for the embedding-model version marker and migration reindexing."""

import asyncio

import pytest
import infrastructure.parsing.factory as parsing_factory
from infrastructure.parsing.factory import ParserFactory

from application.services.document_manager import DocumentManager
from application.services.indexer import IndexerService

from conftest import (
    FakeDocumentRepository,
    FakeEmbedding,
    FakeFileStorage,
    FakeVectorStore,
    make_document,
)


class FakeParser:
    def parse(self, file_path):
        return [{"text": "hello world", "metadata": {}}]


def _setup(monkeypatch, version=None):
    repo = FakeDocumentRepository()
    storage = FakeFileStorage()
    vector_store = FakeVectorStore()
    embedding = FakeEmbedding()
    if version is not None:
        indexer = IndexerService(storage, vector_store, repo, embedding, embedding_version=version)
    else:
        indexer = IndexerService(storage, vector_store, repo, embedding)
    if monkeypatch is not None:
        monkeypatch.setattr(
            parsing_factory.ParserFactory, "get_parser", staticmethod(lambda p: FakeParser())
        )
    return repo, storage, vector_store, embedding, indexer


def _manager(vector_store, repo, indexer, embedding, model_name):
    return DocumentManager(None, vector_store, repo, indexer, embedding, model_name)


async def test_indexer_stamps_embedding_model_version(monkeypatch):
    repo, storage, vector_store, embedding, indexer = _setup(
        monkeypatch, version="e5-large-v2"
    )
    doc = make_document()
    await repo.save(doc)

    await indexer.index_document(doc.id)

    payload = vector_store.upserts[0]["payloads"][0]
    assert payload["embedding_model"] == "e5-large-v2"


async def test_indexer_without_version_has_no_marker(monkeypatch):
    repo, storage, vector_store, embedding, indexer = _setup(monkeypatch, version=None)
    doc = make_document()
    await repo.save(doc)

    await indexer.index_document(doc.id)

    assert "embedding_model" not in vector_store.upserts[0]["payloads"][0]


async def test_check_embedding_version_reads_marker_from_points():
    vector_store = FakeVectorStore()
    vector_store.upserts.append(
        {"ids": ["p1"], "vectors": [[0.0]], "payloads": [{"text": "t", "embedding_model": "e5-v1"}]}
    )
    manager = _manager(
        vector_store,
        FakeDocumentRepository(),
        IndexerService(FakeFileStorage(), vector_store, FakeDocumentRepository(), FakeEmbedding()),
        FakeEmbedding(),
        "e5-v2",
    )
    stored, expected = await manager.check_embedding_version()
    assert (stored, expected) == ("e5-v1", "e5-v2")
    assert (await manager.needs_reindex()) is True


async def test_needs_reindex_false_when_same_model():
    vector_store = FakeVectorStore()
    vector_store.upserts.append(
        {"ids": ["p1"], "vectors": [[0.0]], "payloads": [{"text": "t", "embedding_model": "e5"}]}
    )
    manager = _manager(
        vector_store,
        FakeDocumentRepository(),
        IndexerService(FakeFileStorage(), vector_store, FakeDocumentRepository(), FakeEmbedding()),
        FakeEmbedding(),
        "e5",
    )
    assert (await manager.needs_reindex()) is False


async def test_needs_reindex_false_when_collection_empty():
    manager = _manager(
        FakeVectorStore(),
        FakeDocumentRepository(),
        IndexerService(FakeFileStorage(), FakeVectorStore(), FakeDocumentRepository(), FakeEmbedding()),
        FakeEmbedding(),
        "e5",
    )
    stored, expected = await manager.check_embedding_version()
    assert stored is None and expected == "e5"
    assert (await manager.needs_reindex()) is False


async def test_reindex_all_detects_model_migration(monkeypatch, capsys):
    """Admin reindex over a collection built by an older model works."""
    repo, storage, vector_store, embedding, indexer = _setup(
        monkeypatch, version="new-model"
    )
    doc = make_document()
    await repo.save(doc)
    repo.user_groups["admin_1"] = []

    manager = _manager(vector_store, repo, indexer, embedding, "new-model")
    # Simulate points built by the previous model
    vector_store.upserts.append(
        {"ids": ["old"], "vectors": [[0.0]], "payloads": [{"embedding_model": "old-model"}]}
    )

    await manager.reindex_all("admin_1")

    # Collection was dropped and recreated, docs re-indexed with the new marker
    assert vector_store.dropped == 1
    assert len(vector_store.created_dimensions) == 1
    assert vector_store.upserts[-1]["payloads"][0]["embedding_model"] == "new-model"


async def test_reindex_all_requires_admin():
    from domain.exceptions import PermissionDeniedError

    manager = _manager(
        FakeVectorStore(),
        FakeDocumentRepository(),
        IndexerService(FakeFileStorage(), FakeVectorStore(), FakeDocumentRepository(), FakeEmbedding()),
        FakeEmbedding(),
        "e5",
    )
    with pytest.raises(PermissionDeniedError):
        await manager.reindex_all("regular-user")