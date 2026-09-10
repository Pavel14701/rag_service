"""Tests for BaseConsumer message handling and consumer `handle` methods."""

import json
import uuid
from types import SimpleNamespace

import pytest

from entrypoints.base_consumer import BaseConsumer
from entrypoints.ingest_consumer import IngestConsumer
from entrypoints.query_consumer import QueryConsumer
from entrypoints.delete_consumer import DeleteConsumer
from entrypoints.reindex_consumer import ReindexConsumer
from application.services.indexer import IndexerService
from application.services.retriever import RetrieverService
from application.services.document_manager import DocumentManager
from application.interfaces import TokenValidator, DocumentRepository
from domain.entities.document import DocStatus
from domain.exceptions import PermissionDeniedError

from conftest import (
    FakeContainer,
    FakeTokenValidator,
    FakeEmbedding,
    FakeFileStorage,
    FakeLLM,
    FakeVectorStore,
    make_document,
)


class RecordingConsumer(BaseConsumer):
    def __init__(self, container):
        super().__init__("recording_queue", container)
        self.handled = []
        self.fail_with = None

    async def handle(self, data):
        if self.fail_with:
            raise self.fail_with
        self.handled.append(data)


class FakeMessage:
    """Minimal IncomingMessage double."""

    def __init__(self, body: bytes):
        self.body = body

    def process(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


async def test_on_message_parses_json_and_calls_handle():
    consumer = RecordingConsumer(FakeContainer({}))
    await consumer._on_message(FakeMessage(json.dumps({"a": 1}).encode()))
    assert consumer.handled == [{"a": 1}]


async def test_on_message_invalid_json_does_not_raise():
    consumer = RecordingConsumer(FakeContainer({}))
    await consumer._on_message(FakeMessage(b"not-json"))
    assert consumer.handled == []


async def test_on_message_handler_error_swallowed():
    consumer = RecordingConsumer(FakeContainer({}))
    consumer.fail_with = RuntimeError("boom")
    await consumer._on_message(FakeMessage(b"{}"))
    # no exception propagated


@pytest.fixture
def deps(repo, storage, vector_store, embedding, llm):
    indexer = IndexerService(storage, vector_store, repo, embedding)
    retriever = RetrieverService(vector_store, repo, embedding, llm)
    manager = DocumentManager(storage, vector_store, repo, indexer, embedding)
    container = FakeContainer(
        {
            TokenValidator: FakeTokenValidator(),
            DocumentRepository: repo,
            IndexerService: indexer,
            RetrieverService: retriever,
            DocumentManager: manager,
        }
    )
    return SimpleNamespace(repo=repo, storage=storage, container=container)


async def test_ingest_consumer_indexes_owned_document(deps):
    doc = make_document(owner_id="user-1")
    await deps.repo.save(doc)
    # Real MarkdownParser is used downstream, so provide parseable content
    deps.storage.files[doc.file_path] = b"# Header\n\nSome text to index.\n"
    consumer = IngestConsumer(deps.container)

    await consumer.handle({"token": "tok:user-1", "doc_id": str(doc.id)})
    assert deps.repo.status_updates[-1] == (doc.id, DocStatus.INDEXED)


async def test_ingest_consumer_missing_token_raises(deps):
    consumer = IngestConsumer(deps.container)
    with pytest.raises(ValueError, match="Missing token"):
        await consumer.handle({"doc_id": str(uuid.uuid4())})


async def test_ingest_consumer_not_owner_raises(deps):
    doc = make_document(owner_id="user-1")
    await deps.repo.save(doc)
    consumer = IngestConsumer(deps.container)

    with pytest.raises(PermissionDeniedError):
        await consumer.handle({"token": "tok:other-user", "doc_id": str(doc.id)})


async def test_query_consumer_calls_retriever(deps, repo, vector_store, llm, capsys):
    doc = make_document(owner_id="user-1")
    await repo.save(doc)
    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {"doc_id": str(doc.id)}, "text": "t"}
    ]
    consumer = QueryConsumer(deps.container)

    await consumer.handle({"token": "tok:user-1", "query": "hi"})
    assert len(llm.calls) == 1
    assert "Result for" in capsys.readouterr().out


async def test_query_consumer_missing_query_raises(deps):
    consumer = QueryConsumer(deps.container)
    with pytest.raises(KeyError):
        await consumer.handle({"token": "tok:user-1"})


async def test_delete_consumer_deletes_owned_document(deps):
    doc = make_document(owner_id="user-1")
    await deps.repo.save(doc)
    consumer = DeleteConsumer(deps.container)

    await consumer.handle({"token": "tok:user-1", "doc_id": str(doc.id), "remove_file": True})
    assert doc.id in deps.repo.deleted_ids
    assert deps.storage.deleted_keys == [doc.file_path]


async def test_reindex_consumer_calls_manager(deps, repo, vector_store):
    await repo.save(make_document(owner_id="admin_1"))
    consumer = ReindexConsumer(deps.container)

    await consumer.handle({"token": "tok:admin_1"})
    assert vector_store.dropped == 1