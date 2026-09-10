"""Shared test doubles and helpers."""

import subprocess
import sys
import uuid
from datetime import datetime, timezone
from functools import cache
from typing import Any

import pytest

from domain.entities.document import Document, DocStatus


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


class FakeVectorStore:
    """In-memory VectorStore double recording calls."""

    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []
        self.searches: list[dict[str, Any]] = []
        self.deleted_filters: list[dict[str, Any]] = []
        self.created_dimensions: list[int] = []
        self.dropped = 0
        self.search_results: list[dict[str, Any]] = []

    async def create_collection(self, dimension: int) -> None:
        self.created_dimensions.append(dimension)

    async def drop_collection(self) -> None:
        self.dropped += 1

    async def upsert(self, ids, vectors, payloads) -> None:
        self.upserts.append({"ids": ids, "vectors": vectors, "payloads": payloads})

    async def search(self, vector, top_k, filter_condition=None):
        self.searches.append(
            {"vector": vector, "top_k": top_k, "filter_condition": filter_condition}
        )
        return self.search_results[:top_k]

    async def delete_by_filter(self, filter_condition) -> None:
        self.deleted_filters.append(filter_condition)


class FakeEmbedding:
    """EmbeddingModel double returning deterministic vectors."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t) % 10)] * self.dim for t in texts]

    def dimension(self) -> int:
        return self.dim


class FakeLLM:
    """LLMGenerator double."""

    def __init__(self, response: str = "fake answer") -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
            }
        )
        return self.response


class FakeTokenValidator:
    """TokenValidator double: decodes user from '<token>:<user>' strings."""

    def validate(self, token: str) -> dict[str, str]:
        if ":" not in token:
            raise ValueError("Invalid token")
        return {"sub": token.split(":", 1)[1]}


class FakeContainer:
    """Minimal AsyncContainer double resolving from a type -> instance map."""

    def __init__(self, instances: dict[type, Any]) -> None:
        self._instances = instances

    async def get(self, provider_type: type) -> Any:
        return self._instances[provider_type]


@pytest.fixture
def repo() -> FakeDocumentRepository:
    return FakeDocumentRepository()


@pytest.fixture
def storage() -> FakeFileStorage:
    return FakeFileStorage()


@pytest.fixture
def vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def embedding() -> FakeEmbedding:
    return FakeEmbedding()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()