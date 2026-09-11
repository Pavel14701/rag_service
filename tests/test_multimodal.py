"""Tests for multimodal document support: tables and element typing."""

from typing import Any

from application.services.indexer import IndexerService
from conftest import requires_unstructured


class _FakeDoc:
    owner_id = "owner-1"
    access_group = "group-1"


def _make_indexer() -> IndexerService:
    return IndexerService(
        file_storage=None,
        vector_store=None,
        repo=None,
        embedding=None,
    )


def test_prepare_chunks_preserves_element_type_and_table_html():
    indexer = _make_indexer()
    elements: list[dict[str, Any]] = [
        {"text": "Intro paragraph about revenue.", "metadata": {"page": 1, "type": "text"}},
        {
            "text": "Quarter | Revenue\nQ1 | 100",
            "metadata": {
                "page": 2,
                "type": "table",
                "table_html": "<table><tr><th>Quarter</th><th>Revenue</th></tr><tr><td>Q1</td><td>100</td></tr></table>",
            },
        },
        {"text": "Figure caption text.", "metadata": {"page": 3, "type": "image"}},
    ]
    chunks = indexer._prepare_chunks(elements, doc_id=None, doc=_FakeDoc())
    assert len(chunks) == 3

    text_chunk, table_chunk, image_chunk = chunks
    assert text_chunk["metadata"]["type"] == "text"
    assert "table_html" not in text_chunk["metadata"]

    assert table_chunk["metadata"]["type"] == "table"
    assert table_chunk["metadata"]["table_html"].startswith("<table>")

    assert image_chunk["metadata"]["type"] == "image"


def test_prepare_chunks_defaults_type_to_text():
    indexer = _make_indexer()
    elements = [{"text": "plain chunk", "metadata": {"page": 1}}]
    chunks = indexer._prepare_chunks(elements, doc_id=None, doc=_FakeDoc())
    assert chunks[0]["metadata"]["type"] == "text"


@requires_unstructured
def test_pdf_parser_tags_tables_and_images(tmp_path, monkeypatch):
    """PDFParser maps unstructured categories into metadata type/table_html."""
    from infrastructure.parsing.pdf_parser import PDFParser

    class _Meta:
        def __init__(self, data: dict) -> None:
            self._data = data

        def to_dict(self) -> dict:
            return self._data

    class _El:
        def __init__(self, text: str, data: dict) -> None:
            self._text = text
            self.metadata = _Meta(data)

        def __str__(self) -> str:
            return self._text

    def fake_partition_pdf(**kwargs):  # noqa: ANN003
        return [
            _El("Just narrative text.", {"page_number": 1, "category": "NarrativeText"}),
            _El(
                "Col A | Col B",
                {
                    "page_number": 2,
                    "category": "Table",
                    "text_as_html": "<table><tr><td>Col A</td></tr></table>",
                },
            ),
            _El("", {"page_number": 3}),  # empty -> skipped
        ]

    import infrastructure.parsing.pdf_parser as pdf_module

    monkeypatch.setattr(pdf_module, "partition_pdf", fake_partition_pdf)

    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")
    elements = PDFParser(strategy="fast").parse(pdf_file)

    assert [e["metadata"]["type"] for e in elements] == ["narrativetext", "table"]
    assert elements[1]["metadata"]["table_html"] == "<table><tr><td>Col A</td></tr></table>"
    assert "table_html" not in elements[0]["metadata"]