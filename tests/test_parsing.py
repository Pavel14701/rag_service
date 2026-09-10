"""Tests for MarkdownParser and ParserFactory."""

from pathlib import Path

import pytest

from infrastructure.parsing.factory import ParserFactory
from infrastructure.parsing.markdown_parser import MarkdownParser

from conftest import requires_unstructured


def write_md(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "doc.md"
    path.write_text(content, encoding="utf-8")
    return path


def test_markdown_parser_splits_by_headers(tmp_path):
    path = write_md(
        tmp_path,
        "# Title\n\nIntro paragraph.\n\n## Section A\n\nBody of A.\n\n## Section B\n\nBody of B.\n",
    )
    chunks = MarkdownParser().parse(path)

    assert len(chunks) == 3
    assert chunks[0]["metadata"]["header"] == "Title"
    assert "Intro paragraph." in chunks[0]["text"]
    assert chunks[1]["metadata"]["header"] == "Section A"
    assert "Body of A." in chunks[1]["text"]
    assert chunks[2]["metadata"]["header"] == "Section B"


def test_markdown_parser_empty_file_returns_no_chunks(tmp_path):
    path = write_md(tmp_path, "")
    assert MarkdownParser().parse(path) == []


def test_markdown_parser_code_blocks_kept(tmp_path):
    path = write_md(tmp_path, "# H\n\n```python\nprint('hi')\n```\n")
    chunks = MarkdownParser().parse(path)
    assert any("print" in c["text"] for c in chunks)


def test_factory_selects_markdown_parser():
    assert isinstance(ParserFactory.get_parser(Path("file.md")), MarkdownParser)
    assert isinstance(ParserFactory.get_parser(Path("file.MD")), MarkdownParser)


@requires_unstructured
def test_factory_selects_pdf_parser():
    from infrastructure.parsing.pdf_parser import PDFParser

    assert isinstance(ParserFactory.get_parser(Path("file.pdf")), PDFParser)


@requires_unstructured
def test_factory_selects_docx_parser():
    from infrastructure.parsing.docx_parser import DocxParser

    assert isinstance(ParserFactory.get_parser(Path("file.docx")), DocxParser)


@requires_unstructured
def test_factory_falls_back_to_unstructured_parser():
    from infrastructure.parsing.unstructured_parser import UnstructuredParser

    assert isinstance(ParserFactory.get_parser(Path("file.txt")), UnstructuredParser)
    assert isinstance(ParserFactory.get_parser(Path("file.html")), UnstructuredParser)


@requires_unstructured
def test_lazy_exports_resolve():
    import infrastructure.parsing as parsing_pkg

    assert parsing_pkg.PDFParser.__name__ == "PDFParser"
    assert parsing_pkg.DocxParser.__name__ == "DocxParser"
    assert parsing_pkg.UnstructuredParser.__name__ == "UnstructuredParser"


def test_lazy_exports_missing_attribute():
    import infrastructure.parsing as parsing_pkg

    with pytest.raises(AttributeError):
        parsing_pkg.NoSuchParser  # noqa: B018