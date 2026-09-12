from pathlib import Path

"""Tests for BaseConsumer message handling and consumer `handle` methods."""

import json
import uuid
from types import SimpleNamespace

import pytest

from config import Settings
from entrypoints.consumers import BaseConsumer
from entrypoints.consumers import IngestConsumer
from entrypoints.consumers import QueryConsumer
from entrypoints.consumers import DeleteConsumer
from entrypoints.consumers import ReindexConsumer
from application.services import IndexerService


class FakeSelector:
    """Delegates to the real extension-based selector (tests index .md)."""

    def __init__(self) -> None:
        self._selector = DocumentParserSelector()

    def get_parser(self, file_path: Path):
        return self._selector.get_parser(file_path)
from application.services import RetrieverService
from application.services import DocumentManager
from application.interfaces import TokenValidator, DocumentRepository
from infrastructure.parsing import DocumentParserSelector
from domain.model import DocStatus, PermanentError, TransientError
from domain.model import PermissionDeniedError

from conftest import (
    FakeDocumentRepository,
    FakeContainer,
    FakeMessage,
    FakeTokenValidator,
    FakeEmbedding,
    FakeFileStorage,
    FakeLLM,
    FakeVectorStore,
    make_document,
)

pytestmark = pytest.mark.consumers


class RecordingConsumer(BaseConsumer):
    def __init__(self, container) -> None:
        super().__init__("recording_queue", container)
        self.handled = []
        self.contexts = []
        self.fail_with = None

    async def handle(self, data) -> None:
        from structlog.contextvars import get_contextvars

        if self.fail_with:
            raise self.fail_with
        self.handled.append(data)
        self.contexts.append(dict(get_contextvars()))


async def test_on_message_parses_json_and_calls_handle() -> None:
    consumer = RecordingConsumer(FakeContainer({}))
    await consumer._on_message(FakeMessage(json.dumps({"a": 1}).encode()))
    assert consumer.handled == [{"a": 1}]


async def test_on_message_invalid_json_does_not_raise() -> None:
    consumer = RecordingConsumer(FakeContainer({}))
    await consumer._on_message(FakeMessage(b"not-json"))
    assert consumer.handled == []


async def test_on_message_handler_error_schedules_retry() -> None:
    consumer = RecordingConsumer(FakeContainer({}))
    consumer.fail_with = RuntimeError("boom")
    msg = FakeMessage(b"{}")
    await consumer._on_message(msg)
    # Message republished to the first retry queue with incremented counter
    assert len(msg.published) == 1
    routing_key, _, headers = msg.published[0]
    assert routing_key == "recording_queue.retry.5000"
    assert headers["x-retry-count"] == 1


async def test_on_message_retries_exhausted_moves_to_dlq() -> None:
    consumer = RecordingConsumer(FakeContainer({}))
    consumer.fail_with = RuntimeError("boom")
    msg = FakeMessage(b"{}", headers={"x-retry-count": 3})
    await consumer._on_message(msg)
    assert len(msg.published) == 1
    routing_key, _, headers = msg.published[0]
    assert routing_key == "recording_queue.dlq"
    assert headers["x-retry-count"] == 3


async def test_on_message_success_does_not_republish() -> None:
    consumer = RecordingConsumer(FakeContainer({}))
    msg = FakeMessage(json.dumps({"a": 1}).encode())
    await consumer._on_message(msg)
    assert msg.published == []
    assert consumer.handled == [{"a": 1}]


async def test_request_id_taken_from_header_and_bound() -> None:
    consumer = RecordingConsumer(FakeContainer({}))
    msg = FakeMessage(b"{}", headers={"x-request-id": "req-42"})
    await consumer._on_message(msg)
    assert consumer.contexts[0]["request_id"] == "req-42"
    assert consumer.contexts[0]["queue"] == "recording_queue"


async def test_request_id_generated_when_missing() -> None:
    consumer = RecordingConsumer(FakeContainer({}))
    await consumer._on_message(FakeMessage(b"{}"))
    request_id = consumer.contexts[0]["request_id"]
    assert isinstance(request_id, str) and len(request_id) == 36  # uuid4
    # context cleared after processing
    from structlog.contextvars import get_contextvars
    assert get_contextvars() == {}


async def test_retry_republish_propagates_request_id() -> None:
    consumer = RecordingConsumer(FakeContainer({}))
    consumer.fail_with = RuntimeError("boom")
    msg = FakeMessage(b"{}", headers={"x-request-id": "req-7"})
    await consumer._on_message(msg)
    _, _, headers = msg.published[0]
    assert headers["x-request-id"] == "req-7"


@pytest.fixture
def deps(repo: FakeDocumentRepository, storage: FakeFileStorage, vector_store: FakeVectorStore, embedding: FakeEmbedding, llm: FakeLLM):
    indexer = IndexerService(storage, vector_store, repo, embedding, FakeSelector())
    retriever = RetrieverService(vector_store, repo, embedding, llm)
    manager = DocumentManager(storage, vector_store, repo, indexer, embedding)
    container = FakeContainer(
        {
            TokenValidator: FakeTokenValidator(),
            DocumentRepository: repo,
            IndexerService: indexer,
            Settings: Settings(),
            RetrieverService: retriever,
            DocumentManager: manager,
        }
    )
    return SimpleNamespace(repo=repo, storage=storage, container=container)


async def test_ingest_consumer_indexes_owned_document(deps) -> None:
    doc = make_document(owner_id="user-1")
    await deps.repo.save(doc)
    # Real MarkdownParser is used downstream, so provide parseable content
    deps.storage.files[doc.file_path] = b"# Header\n\nSome text to index.\n"
    consumer = IngestConsumer(deps.container)

    await consumer.handle({"token": "tok:user-1", "doc_id": str(doc.id)})
    assert deps.repo.status_updates[-1] == (doc.id, DocStatus.INDEXED)


async def test_ingest_consumer_missing_token_raises(deps) -> None:
    consumer = IngestConsumer(deps.container)
    with pytest.raises(ValueError, match="Missing token"):
        await consumer.handle({"doc_id": str(uuid.uuid4())})


async def test_ingest_consumer_not_owner_raises(deps) -> None:
    doc = make_document(owner_id="user-1")
    await deps.repo.save(doc)
    consumer = IngestConsumer(deps.container)

    with pytest.raises(PermissionDeniedError):
        await consumer.handle({"token": "tok:other-user", "doc_id": str(doc.id)})


async def test_query_consumer_calls_retriever(deps, repo, vector_store, llm) -> None:
    doc = make_document(owner_id="user-1")
    await repo.save(doc)
    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {"doc_id": str(doc.id)}, "text": "t"}
    ]
    consumer = QueryConsumer(deps.container)

    await consumer.handle({"token": "tok:user-1", "query": "hi"})
    assert len(llm.calls) == 1


async def test_query_consumer_routes_llm_provider_per_request(deps, repo, vector_store, llm) -> None:
    doc = make_document(owner_id="user-1")
    await repo.save(doc)
    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {"doc_id": str(doc.id)}, "text": "t"}
    ]
    deps.container._instances[Settings] = Settings(
        llm_enabled_providers="deepseek,openai",
        llm_allowed_models="gpt-4o",
    )
    consumer = QueryConsumer(deps.container)

    await consumer.handle(
        {
            "token": "tok:user-1",
            "query": "hi",
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
        }
    )

    assert llm.calls[-1]["provider"] == "openai"
    assert llm.calls[-1]["model"] == "gpt-4o"


async def test_query_consumer_publishes_reply_to_reply_to_queue(
    deps, repo, vector_store
):
    doc = make_document(owner_id="user-1")
    await repo.save(doc)
    vector_store.search_results = [
        {"id": "c1", "score": 0.9, "payload": {"doc_id": str(doc.id)}, "text": "t"}
    ]
    consumer = QueryConsumer(deps.container)
    msg = FakeMessage(
        json.dumps({"token": "tok:user-1", "query": "hi", "query_id": "q-1"}).encode(),
        reply_to="rpc-reply-1",
        correlation_id="corr-42",
    )

    await consumer._on_message(msg)

    replies = [p for p in msg.published if p[0] == "rpc-reply-1"]
    assert len(replies) == 1
    _, body, _ = replies[0]
    payload = json.loads(body)
    assert payload["query_id"] == "q-1"
    assert payload["answer"] == "fake answer"
    assert payload["sources"][0]["doc_id"] == str(doc.id)


async def test_query_consumer_reply_falls_back_to_shared_queue(deps, repo) -> None:
    consumer = QueryConsumer(deps.container)
    msg = FakeMessage(json.dumps({"token": "tok:user-1", "query": "hi"}).encode())

    await consumer._on_message(msg)

    routing_keys = [p[0] for p in msg.published]
    assert "query_reply_queue" in routing_keys


async def test_query_consumer_passes_groups_claim_to_retriever(
    deps, repo, vector_store
):
    doc = make_document(owner_id="someone-else", access_group="team-a")
    await repo.save(doc)
    validator = FakeTokenValidator({"tok:g": {"sub": "user-1", "groups": ["team-a"]}})
    container = FakeContainer(
        {**deps.container._instances, TokenValidator: validator}
    )
    consumer = QueryConsumer(container)

    await consumer.handle({"token": "tok:g", "query": "hi"})

    filter_cond = vector_store.searches[-1]["filter_condition"]
    assert filter_cond["should"][1] == {
        "key": "access_group",
        "match": {"value": ["team-a"]},
    }


async def test_query_consumer_missing_query_raises(deps) -> None:
    consumer = QueryConsumer(deps.container)
    with pytest.raises(KeyError):
        await consumer.handle({"token": "tok:user-1"})


async def test_delete_consumer_deletes_owned_document(deps) -> None:
    doc = make_document(owner_id="user-1")
    await deps.repo.save(doc)
    consumer = DeleteConsumer(deps.container)

    await consumer.handle({"token": "tok:user-1", "doc_id": str(doc.id), "remove_file": True})
    assert doc.id in deps.repo.deleted_ids
    assert deps.storage.deleted_keys == [doc.file_path]


async def test_reindex_consumer_calls_manager(deps, repo, vector_store) -> None:
    await repo.save(make_document(owner_id="admin_1"))
    consumer = ReindexConsumer(deps.container)

    await consumer.handle({"token": "tok:admin_1"})
    # blue-green: the shadow collection is promoted, the live one stays
    assert vector_store.dropped == 0
    assert vector_store.promoted == 1

async def test_permanent_error_goes_straight_to_dlq() -> None:
    """Poison messages must not be retried: straight to the DLQ."""
    consumer = RecordingConsumer(FakeContainer({}))
    consumer.fail_with = PermanentError('bad input')
    msg = FakeMessage(b'{}')

    await consumer._on_message(msg)

    routing_keys = [r for r, _, _ in msg.published]
    assert routing_keys == ['recording_queue.dlq']


async def test_transient_error_is_retried_not_dlqed() -> None:
    consumer = RecordingConsumer(FakeContainer({}))
    consumer.fail_with = TransientError('flaky')
    msg = FakeMessage(b'{}')

    await consumer._on_message(msg)

    routing_keys = [r for r, _, _ in msg.published]
    assert routing_keys == ['recording_queue.retry.5000']


class FakeIdemCache:
    """RedisCache double with dict storage (get/set protocol)."""

    def __init__(self) -> None:
        self.store = {}

    def get(self, key: str):
        return self.store.get(key)

    def set(self, key, value) -> None:
        self.store[key] = value


async def test_redelivered_duplicate_is_skipped() -> None:
    from infrastructure.caching import SharedCaches, TTLCache

    idem = FakeIdemCache()
    consumer = RecordingConsumer(
        FakeContainer({SharedCaches: SimpleNamespace(redis=idem)})
    )
    msg = FakeMessage(b'{}', redelivered=True, message_id='m-1')
    idem.store[TTLCache.make_key('idem', 'recording_queue', 'm-1')] = '1'

    await consumer._on_message(msg)

    # Already processed before the crash: handler must not run again
    assert consumer.handled == []


async def test_redelivered_unseen_message_is_processed_and_marked() -> None:
    from infrastructure.caching import SharedCaches, TTLCache

    idem = FakeIdemCache()
    consumer = RecordingConsumer(
        FakeContainer({SharedCaches: SimpleNamespace(redis=idem)})
    )
    msg = FakeMessage(b'{"a": 1}', redelivered=True, message_id='m-2')

    await consumer._on_message(msg)

    assert consumer.handled == [{'a': 1}]
    assert idem.store[TTLCache.make_key('idem', 'recording_queue', 'm-2')] == '1'


async def test_first_delivery_is_processed_without_redelivery_flag() -> None:
    from infrastructure.caching import SharedCaches, TTLCache

    idem = FakeIdemCache()
    idem.store[TTLCache.make_key('idem', 'recording_queue', 'm-3')] = '1'
    consumer = RecordingConsumer(
        FakeContainer({SharedCaches: SimpleNamespace(redis=idem)})
    )
    # message_id matches a processed key, but redelivered=False:
    # this is a NEW message from the producer, not a broker redelivery
    msg = FakeMessage(b'{"b": 2}', redelivered=False, message_id='m-3')

    await consumer._on_message(msg)

    assert consumer.handled == [{'b': 2}]


async def test_query_consumer_rejects_model_outside_allowlist(deps) -> None:
    settings = Settings(llm_allowed_models="gpt-4o-mini")
    deps.container._instances[Settings] = settings
    consumer = QueryConsumer(deps.container)

    with pytest.raises(ValueError, match="not allowed"):
        await consumer.handle(
            {"token": "tok:user-1", "query": "hi", "llm_model": "claude-3-opus"}
        )


async def test_query_consumer_rejects_disabled_provider(deps) -> None:
    settings = Settings(
        llm_provider="deepseek", llm_enabled_providers="deepseek"
    )
    deps.container._instances[Settings] = settings
    consumer = QueryConsumer(deps.container)

    with pytest.raises(ValueError, match="not enabled"):
        await consumer.handle(
            {"token": "tok:user-1", "query": "hi", "llm_provider": "openai"}
        )


async def test_query_consumer_without_allowlist_accepts_any_model(deps) -> None:
    deps.container._instances[Settings] = Settings()
    consumer = QueryConsumer(deps.container)

    # empty allowlist = unrestricted; must not raise
    await consumer.handle(
        {"token": "tok:user-1", "query": "hi", "llm_model": "any-model-x"}
    )


def test_main_queue_arguments_include_max_length_when_set() -> None:
    args = BaseConsumer._main_queue_arguments(100)
    assert args["x-max-length"] == 100
    assert args["x-overflow"] == "reject-publish"
    # DLX wiring stays intact alongside the cap
    assert args["x-dead-letter-exchange"] == ""


def test_main_queue_arguments_unlimited_by_default() -> None:
    args = BaseConsumer._main_queue_arguments(0)
    assert "x-max-length" not in args
    assert "x-overflow" not in args
