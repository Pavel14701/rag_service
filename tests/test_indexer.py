"""Tests for IndexerService."""

import uuid
from pathlib import Path

import pytest

import infrastructure.parsing.factory as parsing_factory
from application.services.indexer import IndexerService
from domain.entities.document import DocStatus
from domain.exceptions import DocumentNotFoundError, IndexingError

from conftest import make_document


class FakeParser:
    def __init__(self, elements):
        self._elements = elements

    def parse(self, file_path: Path):
        return self._elements


@pytest.fixture
def service(repo, storage, vector_store, embedding):
    return IndexerService(storage, vector_store, repo, embedding)


async def test_index_document_success(service, repo, storage, vector_store, embedding, monkeypatch):
    doc = make_document()
    await repo.save(doc)
    storage.files[doc.file_path] = b"content"

    elements = [{"text": "hello world", "metadata": {"page": 2, "header": "Intro"}}]
    monkeypatch.setattr(
        parsing_factory.ParserFactory, "get_parser", staticmethod(lambda p: FakeParser(elements))
    )

    await service.index_document(doc.id)

    # Vectors upserted with valid UUID ids and text stored in the payload
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


async def test_index_document_not_found(service, repo):
    with pytest.raises(DocumentNotFoundError):
        await service.index_document(uuid.uuid4())
    assert repo.status_updates == []


async def test_index_document_deleted(service, repo):
    doc = make_document(deleted=True)
    repo.documents[doc.id] = doc
    with pytest.raises(DocumentNotFoundError):
        await service.index_document(doc.id)


async def test_index_document_no_chunks_raises(service, repo, storage, monkeypatch):
    doc = make_document()
    await repo.save(doc)
    storage.files[doc.file_path] = b"content"
    monkeypatch.setattr(
        parsing_factory.ParserFactory, "get_parser", staticmethod(lambda p: FakeParser([]))
    )

    with pytest.raises(IndexingError):
        await service.index_document(doc.id)
    assert (doc.id, DocStatus.FAILED) in repo.status_updates


async def test_index_document_parser_failure_marks_failed(service, repo, storage, monkeypatch):
    doc = make_document()
    await repo.save(doc)
    storage.files[doc.file_path] = b"content"

    class BrokenParser:
        def parse(self, file_path):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        parsing_factory.ParserFactory, "get_parser", staticmethod(lambda p: BrokenParser())
    )

    with pytest.raises(IndexingError, match="Indexing failed"):
        await service.index_document(doc.id)
    assert (doc.id, DocStatus.FAILED) in repo.status_updates


def test_split_text_respects_max_tokens():
    text = " ".join(["word"] * 100)  # 100 * 5 chars incl. spaces
    chunks = IndexerService._split_text(text, max_tokens=50)
    assert len(chunks) > 1
    for chunk in chunks:
        # spaces are counted, so joined chunks truly respect the limit
        assert len(chunk) <= 50
    # No words lost
    assert sum(len(c.split()) for c in chunks) == 100


def test_split_text_single_word_chunk():
    """A word longer than max_tokens becomes its own chunk (cannot be split)."""
    long_word = "x" * 80
    text = f"short {long_word} tail"
    chunks = IndexerService._split_text(text, max_tokens=50)
    assert chunks == ["short", long_word, "tail"]


def test_prepare_chunks_deterministic_ids(repo):
    service = IndexerService(None, None, None, None)
    doc = make_document()
    elements = [{"text": "some text", "metadata": {}}]

    chunks1 = service._prepare_chunks(elements, doc.id, doc)
    chunks2 = service._prepare_chunks(elements, doc.id, doc)

    assert chunks1 == chunks2  # deterministic -> re-index updates, not duplicates