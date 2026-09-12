"""Tests for IndexerService."""

import uuid
from pathlib import Path

import pytest

from application.services import IndexerService
from domain.model import (
    DocStatus,
    DocumentNotFoundError,
    IndexLockedError,
    IndexingError,
    PermanentIndexingError,
)

from conftest import (
    FakeDocumentRepository,
    FakeEmbedding,
    FakeFileStorage,
    FakeVectorStore,
    make_document,
)

pytestmark = pytest.mark.indexing


class FakeParser:
    def __init__(self, elements) -> None:
        self._elements = elements

    def parse(self, file_path: Path):
        return self._elements


class FakeSelector:
    """ParserSelector stub whose parser is swapped per test."""

    def __init__(self, parser=None) -> None:
        self.parser = parser

    def get_parser(self, file_path: Path):
        assert self.parser is not None, 'no parser configured for test'
        return self.parser


class FakeLock:
    """DistributedLock double with controllable availability."""

    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.acquired: list[str] = []
        self.released: list[str] = []

    async def acquire(self, name: str, ttl: float = 300.0) -> bool:
        if not self.available:
            return False
        self.acquired.append(name)
        return True

    async def release(self, name: str) -> None:
        self.released.append(name)


@pytest.fixture
def parser_selector():
    return FakeSelector()


@pytest.fixture
def lock():
    return FakeLock()


@pytest.fixture
def service(repo: FakeDocumentRepository, storage: FakeFileStorage, vector_store: FakeVectorStore, embedding: FakeEmbedding, parser_selector: FakeSelector, lock: FakeLock):
    return IndexerService(
        storage, vector_store, repo, embedding, parser_selector, lock=lock
    )


async def test_index_document_success(
    service, repo, storage, vector_store, embedding, parser_selector, lock
):
    doc = make_document()
    await repo.save(doc)
    storage.files[doc.file_path] = b"content"

    elements = [{"text": "hello world", "metadata": {"page": 2, "header": "Intro"}}]
    parser_selector.parser = FakeParser(elements)

    await service.index_document(doc.id)

    # Ghost-chunk prevention: old chunks of this doc are purged before upsert
    assert vector_store.deleted_doc_ids == [doc.id]
    assert len(vector_store.upserts) == 1
    upsert = vector_store.upserts[0]
    assert len(upsert["ids"]) == 1
    # ids must be parseable UUIDs (Qdrant requirement)
    uuid.UUID(upsert["ids"][0])
    assert upsert["payloads"][0]["text"] == "hello world"
    assert upsert["payloads"][0]["doc_id"] == str(doc.id)
    assert upsert["payloads"][0]["owner_id"] == doc.owner_id
    assert upsert["payloads"][0]["page"] == 2

    # Embedding generated from the chunk text
    assert embedding.calls == [["hello world"]]

    # Status updated to INDEXED
    assert (doc.id, DocStatus.INDEXED) in repo.status_updates

    # Lock was taken and released
    assert lock.acquired == [f"ingest:{doc.id}"]
    assert lock.released == [f"ingest:{doc.id}"]


async def test_index_document_not_found(service, repo) -> None:
    with pytest.raises(DocumentNotFoundError):
        await service.index_document(uuid.uuid4())
    assert repo.status_updates == []


async def test_index_document_deleted(service, repo) -> None:
    doc = make_document(deleted=True)
    repo.documents[doc.id] = doc
    with pytest.raises(DocumentNotFoundError):
        await service.index_document(doc.id)


async def test_index_document_no_chunks_raises(
    service, repo, storage, parser_selector
):
    doc = make_document()
    await repo.save(doc)
    storage.files[doc.file_path] = b"content"
    parser_selector.parser = FakeParser([])

    # Empty extraction is permanent: retrying cannot produce chunks
    with pytest.raises(PermanentIndexingError):
        await service.index_document(doc.id)
    assert (doc.id, DocStatus.FAILED_INVALID) in repo.status_updates


async def test_index_document_parser_failure_marks_failed_invalid(
    service, repo, storage, parser_selector
):
    doc = make_document()
    await repo.save(doc)
    storage.files[doc.file_path] = b"content"

    class BrokenParser:
        def parse(self, file_path) -> None:
            raise RuntimeError("boom")

    parser_selector.parser = BrokenParser()

    with pytest.raises(PermanentIndexingError, match="Parsing failed"):
        await service.index_document(doc.id)
    assert (doc.id, DocStatus.FAILED_INVALID) in repo.status_updates


async def test_lock_busy_raises_without_side_effects(
    repo, storage, vector_store, embedding, parser_selector
):
    doc = make_document()
    await repo.save(doc)
    storage.files[doc.file_path] = b"content"
    busy_lock = FakeLock(available=False)
    service = IndexerService(
        storage,
        vector_store,
        repo,
        embedding,
        parser_selector,
        lock=busy_lock,
    )

    with pytest.raises(IndexLockedError):
        await service.index_document(doc.id)

    # Nothing was indexed and the document is NOT marked failed:
    # another worker is doing the same job right now.
    assert vector_store.upserts == []
    assert vector_store.deleted_doc_ids == []
    assert repo.status_updates == []


async def test_lock_released_on_failure(
    repo, storage, vector_store, embedding, parser_selector, lock
):
    doc = make_document()
    await repo.save(doc)
    storage.files[doc.file_path] = b"content"
    parser_selector.parser = FakeParser([])  # -> PermanentIndexingError

    with pytest.raises(PermanentIndexingError):
        await IndexerService(
            storage,
            vector_store,
            repo,
            embedding,
            parser_selector,
            lock=lock,
        ).index_document(doc.id)

    assert lock.released == [f"ingest:{doc.id}"]


def test_split_text_respects_max_tokens() -> None:
    text = " ".join(["word"] * 100)  # 100 * 5 chars incl. spaces
    chunks = IndexerService._split_text(text, max_tokens=50)
    assert len(chunks) > 1
    for chunk in chunks:
        # spaces are counted, so joined chunks truly respect the limit
        assert len(chunk) <= 50
    # No words lost
    assert sum(len(c.split()) for c in chunks) == 100


def test_split_text_single_word_chunk() -> None:
    """A word longer than max_tokens becomes its own chunk (cannot be split)."""
    long_word = "x" * 80
    text = f"short {long_word} tail"
    chunks = IndexerService._split_text(text, max_tokens=50)
    assert chunks == ["short", long_word, "tail"]


def test_prepare_chunks_deterministic_ids(repo) -> None:
    service = IndexerService(
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        FakeSelector(),
    )
    doc = make_document()
    elements = [{"text": "some text", "metadata": {}}]

    chunks1 = service._prepare_chunks(elements, doc.id, doc)
    chunks2 = service._prepare_chunks(elements, doc.id, doc)

    assert chunks1 == chunks2  # deterministic -> re-index updates, not duplicates


async def test_tiny_text_chunks_are_merged(
    repo, storage, vector_store, embedding, parser_selector
):
    """Short text elements merge into the next chunk (no stub chunks)."""
    doc = make_document()
    await repo.save(doc)
    storage.files[doc.file_path] = b"content"
    parser_selector.parser = FakeParser(
        [
            {"text": "tiny intro", "metadata": {"type": "text"}},
            {
                "text": "long enough body paragraph with real content here",
                "metadata": {"type": "text"},
            },
        ]
    )
    service = IndexerService(
        storage,
        vector_store,
        repo,
        embedding,
        parser_selector,
        chunk_min_chars=50,
    )

    await service.index_document(doc.id)

    texts = [u["payloads"][0]["text"] for u in vector_store.upserts]
    assert len(texts) == 1
    assert "tiny intro" in texts[0]
    assert "long enough body" in texts[0]


async def test_table_chunks_repeat_header_and_keep_type(
    repo, storage, vector_store, embedding, parser_selector
):
    """Tables are chunked row-wise; every chunk carries the header."""
    doc = make_document()
    await repo.save(doc)
    storage.files[doc.file_path] = b"content"
    table_html = (
        "<table><tr><th>Quarter</th><th>Revenue</th></tr>"
        "<tr><td>Q1</td><td>100</td></tr>"
        "<tr><td>Q2</td><td>200</td></tr>"
        "<tr><td>Q3</td><td>300</td></tr></table>"
    )
    parser_selector.parser = FakeParser(
        [
            {
                "text": "Quarter | Revenue Q1 100 Q2 200 Q3 300",
                "metadata": {
                    "type": "table",
                    "page": 4,
                    "table_html": table_html,
                },
            }
        ]
    )
    # tiny budget: each data row must go to its own chunk
    service = IndexerService(
        storage,
        vector_store,
        repo,
        embedding,
        parser_selector,
    )
    chunks = service._prepare_chunks(
        parser_selector.parser.parse(Path("x")), doc.id, doc, max_tokens=40
    )

    assert len(chunks) == 3  # header repeated for each data row
    for chunk in chunks:
        assert chunk["metadata"]["type"] == "table"
        assert chunk["metadata"]["page"] == 4
        assert "Quarter | Revenue" in chunk["text"]
        assert "<th>Quarter</th>" in chunk["metadata"]["table_html"]
    assert "Q1" in chunks[0]["text"]
    assert "Q2" in chunks[1]["text"]
    assert "Q3" in chunks[2]["text"]


async def test_parse_timeout_is_permanent(
    repo, storage, vector_store, embedding, parser_selector
):
    import time as time_mod

    doc = make_document()
    await repo.save(doc)
    storage.files[doc.file_path] = b"content"

    class SlowParser:
        def parse(self, file_path: Path):
            time_mod.sleep(0.5)
            return []

    parser_selector.parser = SlowParser()
    service = IndexerService(
        storage,
        vector_store,
        repo,
        embedding,
        parser_selector,
        parse_timeout=0.05,
    )

    with pytest.raises(PermanentIndexingError, match="timed out"):
        await service.index_document(doc.id)
    assert (doc.id, DocStatus.FAILED_INVALID) in repo.status_updates


async def test_mime_mismatch_is_permanent(repo, storage, embedding) -> None:
    import sys
    from types import SimpleNamespace as NS

    from infrastructure.parsing import DocumentParserSelector

    doc = make_document()
    doc.file_name = "doc.pdf"
    await repo.save(doc)
    storage.files[doc.file_path] = b"MZ binary garbage"

    fake_magic = NS(from_file=lambda path, mime=True: "application/x-msdownload")
    monkey_target = "magic"
    import sys as sys_mod

    sys_mod.modules[monkey_target] = fake_magic
    try:
        service = IndexerService(
            storage,
            FakeVectorStore(),
            repo,
            embedding,
            DocumentParserSelector(),
        )
        with pytest.raises(PermanentIndexingError, match="Parser selection failed"):
            await service.index_document(doc.id)
    finally:
        sys_mod.modules.pop(monkey_target, None)
    assert (doc.id, DocStatus.FAILED_INVALID) in repo.status_updates
