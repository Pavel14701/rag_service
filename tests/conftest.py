"""Shared test doubles, factories and fixtures.

Layout (top to bottom):

1. Unstructured availability probe — ``requires_unstructured`` skip marker
   (libmagic crashes the interpreter on some Windows setups, so the probe
   runs in a subprocess).
2. Domain factory — ``make_document``.
3. Fake doubles grouped by the port they implement:

   - RabbitMQ transport   — ``FakeMessage``
   - DocumentRepository   — ``FakeDocumentRepository``
   - FileStorage          — ``FakeFileStorage``
   - VectorStore          — ``FakeVectorStore``
   - EmbeddingModel       — ``FakeEmbedding``
   - LLMGenerator         — ``FakeLLM``
   - TokenValidator       — ``FakeTokenValidator``
   - DI container         — ``FakeContainer``

4. Shared fixtures used across the suite (``repo``, ``storage``,
   ``vector_store``, ``embedding``, ``llm``).

Everything is in-memory; no database, broker or Redis is required.
"""

import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from domain.model import Document, DocStatus

# ---------------------------------------------------------------------------
# 1. Unstructured availability probe
# ---------------------------------------------------------------------------


@cache
def unstructured_available() -> bool:
    """Check whether `unstructured` (and python-magic) can be imported.

    The check runs in a subprocess because a broken libmagic installation
    can crash the interpreter with an access violation while loading the
    DLL, which would take down the whole pytest process.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import unstructured, magic"],
            capture_output=True,
            timeout=120,
        )
        return result.returncode == 0
    except Exception:
        return False


requires_unstructured = pytest.mark.skipif(
    not unstructured_available(),
    reason="unstructured/python-magic (libmagic) is not importable on this platform",
)


# ---------------------------------------------------------------------------
# 2. Domain factory
# ---------------------------------------------------------------------------


def make_document(
    doc_id: uuid.UUID | None = None,
    owner_id: str = "user-1",
    access_group: str | None = None,
    deleted: bool = False,
    status: DocStatus = DocStatus.PENDING,
) -> Document:
    """Build a Document domain entity for tests."""
    return Document(
        id=doc_id or uuid.uuid4(),
        file_name="test.md",
        file_path=f"docs/{owner_id}/test.md",
        file_hash="abc123",
        owner_id=owner_id,
        access_group=access_group,
        uploaded_at=datetime.now(timezone.utc),
        status=status,
        deleted=deleted,
    )


# ---------------------------------------------------------------------------
# 3a. Fake doubles — RabbitMQ transport (FakeMessage)
# ---------------------------------------------------------------------------


class FakeMessage:
    """Minimal aio-pika IncomingMessage double.

    Records republished messages in ``published`` so retry/DLQ tests can
    assert routing keys and headers.
    """

    def __init__(
        self,
        body: bytes,
        headers: dict | None = None,
        reply_to: str | None = None,
        correlation_id: str | None = None,
        redelivered: bool = False,
        message_id: str | None = None,
    ) -> None:
        self.body = body
        self.headers = headers or {}
        self.reply_to = reply_to
        self.correlation_id = correlation_id
        self.redelivered = redelivered
        self.message_id = message_id
        self.published: list[tuple[str, bytes, dict]] = []
        self.channel = SimpleNamespace(
            default_exchange=SimpleNamespace(publish=self._publish)
        )

    async def _publish(self, message, routing_key) -> None:
        self.published.append((routing_key, message.body, dict(message.headers)))

    def process(self, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


# ---------------------------------------------------------------------------
# 3b. Fake doubles — DocumentRepository (FakeDocumentRepository)
# ---------------------------------------------------------------------------


class FakeDocumentRepository:
    """In-memory DocumentRepository double."""

    def __init__(self) -> None:
        self.documents: dict[uuid.UUID, Document] = {}
        self.user_groups: dict[str, list[str]] = {}
        self.conversations: list[dict[str, Any]] = []
        self.status_updates: list[tuple[uuid.UUID, DocStatus]] = []
        self.deleted_ids: list[uuid.UUID] = []

    async def get_document(self, doc_id: uuid.UUID) -> Document | None:
        doc = self.documents.get(doc_id)
        if doc is None or doc.deleted:
            return None
        return doc

    async def get_all_active(self) -> list[Document]:
        return [d for d in self.documents.values() if not d.deleted]

    async def iter_active_documents(self, batch_size: int = 500) -> AsyncIterator[Document]:
        docs = [d for d in self.documents.values() if not d.deleted]
        for d in docs:
            yield d

    async def save(self, document: Document) -> None:
        self.documents[document.id] = document

    async def update_status(self, doc_id: uuid.UUID, status: DocStatus) -> None:
        self.status_updates.append((doc_id, status))
        if doc_id in self.documents:
            self.documents[doc_id].status = status

    async def mark_deleted(self, doc_id: uuid.UUID) -> None:
        self.deleted_ids.append(doc_id)
        if doc_id in self.documents:
            self.documents[doc_id].deleted = True

    async def get_user_groups(self, user_id: str) -> list[str]:
        return self.user_groups.get(user_id, [])

    async def set_user_groups(self, user_id: str, groups: list[str]) -> None:
        self.user_groups[user_id] = list(groups)

    async def save_conversation(
        self,
        user_id: str,
        query: str,
        response: str,
        sources: list[dict[str, Any]],
    ) -> None:
        self.conversations.append(
            {"user_id": user_id, "query": query, "response": response, "sources": sources}
        )


# ---------------------------------------------------------------------------
# 3c. Fake doubles — FileStorage (FakeFileStorage)
# ---------------------------------------------------------------------------


class FakeFileStorage:
    """In-memory FileStorage double."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.deleted_keys: list[str] = []

    async def upload_file(self, local_path, destination_key: str) -> str:
        self.files[destination_key] = local_path.read_bytes()
        return destination_key

    async def download_file(self, source_key: str, destination_path) -> None:
        destination_path.write_bytes(self.files.get(source_key, b""))

    async def delete_file(self, source_key: str) -> None:
        self.deleted_keys.append(source_key)
        self.files.pop(source_key, None)


# ---------------------------------------------------------------------------
# 3d. Fake doubles — VectorStore (FakeVectorStore)
# ---------------------------------------------------------------------------


class FakeVectorStore:
    """In-memory VectorStore double recording all calls."""

    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []
        self.searches: list[dict[str, Any]] = []
        self.deleted_filters: list[dict[str, Any]] = []
        self.created_dimensions: list[int] = []
        self.dropped = 0
        self.search_results: list[dict[str, Any]] = []
        self.deleted_doc_ids: list[uuid.UUID] = []
        self.collection_dimension: int | None = None
        self.optimize_calls = 0
        self.shadow_name: str | None = None
        self.promoted = 0
        self.discarded = 0

    async def create_collection(self, dimension: int) -> None:
        self.created_dimensions.append(dimension)

    async def drop_collection(self) -> None:
        self.dropped += 1

    async def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        self.upserts.append({"ids": ids, "vectors": vectors, "payloads": payloads})

    async def search(
        self,
        vector: list[float],
        top_k: int | None,
        filter_condition: dict[str, Any] | None = None,
        keyword_query: str | None = None,
    ) -> list[dict[str, Any]]:
        self.searches.append(
            {
                "vector": vector,
                "top_k": top_k,
                "filter_condition": filter_condition,
                "keyword_query": keyword_query,
            }
        )
        return list(self.search_results)

    async def delete_by_filter(self, filter_condition) -> None:
        self.deleted_filters.append(filter_condition)

    async def delete_by_doc_id(self, doc_id) -> None:
        self.deleted_doc_ids.append(doc_id)

    async def optimize_collection(self) -> None:
        self.optimize_calls += 1

    async def create_shadow_collection(self, dimension: int) -> str:
        self.created_dimensions.append(dimension)
        self.shadow_name = f'green_{len(self.created_dimensions)}'
        return self.shadow_name

    async def promote_shadow(self) -> None:
        self.promoted += 1
        self.shadow_name = None

    async def discard_shadow(self) -> None:
        self.discarded += 1
        self.shadow_name = None

    async def scroll_first_payload(self):
        """Model-version marker: payload of the first indexed point."""
        if self.upserts and self.upserts[0]["payloads"]:
            return dict(self.upserts[0]["payloads"][0])
        return None

    async def get_collection_dimension(self):
        return self.collection_dimension


# ---------------------------------------------------------------------------
# 3e. Fake doubles - EmbeddingModel (FakeEmbedding)
# ---------------------------------------------------------------------------


class FakeEmbedding:
    """EmbeddingModel double returning deterministic vectors."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []
        self.query_calls: list[list[str]] = []
        self.passage_calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t) % 10)] * self.dim for t in texts]

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        self.query_calls.append(list(texts))
        return await self.embed(texts)

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.passage_calls.append(list(texts))
        return await self.embed(texts)

    def dimension(self) -> int:
        return self.dim


# ---------------------------------------------------------------------------
# 3f. Fake doubles - LLMGenerator (FakeLLM)
# ---------------------------------------------------------------------------


class FakeLLM:
    """LLMGenerator double recording every generate/generate_with call."""

    def __init__(self, response: str = 'fake answer') -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        self.calls.append(
            {
                'provider': None,
                'model': None,
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'temperature': temperature,
            }
        )
        return self.response

    async def generate_with(
        self,
        provider: str | None,
        model: str | None,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        self.calls.append(
            {
                'provider': provider,
                'model': model,
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'temperature': temperature,
            }
        )
        return self.response


# ---------------------------------------------------------------------------
# 3g. Fake doubles - TokenValidator (FakeTokenValidator)
# ---------------------------------------------------------------------------


class FakeTokenValidator:
    """TokenValidator double: decodes user from '<token>:<user>' strings.

    Custom payloads (e.g. with a ``groups`` claim) can be registered
    per token string via the constructor.
    """

    def __init__(self, payloads: dict[str, dict[str, Any]] | None = None) -> None:
        self.payloads = payloads or {}

    def validate(self, token: str) -> dict[str, Any]:
        if token in self.payloads:
            return self.payloads[token]
        if ':' not in token:
            raise ValueError('Invalid token')
        return {'sub': token.split(':', 1)[1]}


# ---------------------------------------------------------------------------
# 3h. Fake doubles - DI container (FakeContainer)
# ---------------------------------------------------------------------------


class FakeContainer:
    """Minimal AsyncContainer double resolving from a type -> instance map."""

    def __init__(self, instances: dict[type, Any]) -> None:
        self._instances = instances

    async def get(self, provider_type: type) -> Any:
        return self._instances[provider_type]



# ---------------------------------------------------------------------------
# 4. Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo() -> FakeDocumentRepository:
    """Empty in-memory document repository."""
    return FakeDocumentRepository()


@pytest.fixture
def storage() -> FakeFileStorage:
    """Empty in-memory file storage."""
    return FakeFileStorage()


@pytest.fixture
def vector_store() -> FakeVectorStore:
    """In-memory vector store recording every write."""
    return FakeVectorStore()


@pytest.fixture
def embedding() -> FakeEmbedding:
    """Deterministic 4-dimensional embedding double."""
    return FakeEmbedding()


@pytest.fixture
def llm() -> FakeLLM:
    """LLM double returning a fixed 'fake answer'."""
    return FakeLLM()
