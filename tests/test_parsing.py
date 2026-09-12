"""Tests for MarkdownParser and DocumentParserSelector."""

from pathlib import Path

import pytest

from infrastructure.parsing import (
    DocumentParserSelector,
    MarkdownParser,
)

from conftest import requires_unstructured

pytestmark = pytest.mark.parsing


def write_md(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "doc.md"
    path.write_text(content, encoding="utf-8")
    return path


def test_markdown_parser_splits_by_headers(tmp_path) -> None:
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


def test_markdown_parser_empty_file_returns_no_chunks(tmp_path) -> None:
    path = write_md(tmp_path, "")
    assert MarkdownParser().parse(path) == []


def test_markdown_parser_code_blocks_kept(tmp_path) -> None:
    path = write_md(tmp_path, "# H\n\n```python\nprint('hi')\n```\n")
    chunks = MarkdownParser().parse(path)
    assert any("print" in c["text"] for c in chunks)


def test_selector_selects_markdown_parser() -> None:
    selector = DocumentParserSelector()
    assert isinstance(selector.get_parser(Path("file.md")), MarkdownParser)
    assert isinstance(selector.get_parser(Path("file.MD")), MarkdownParser)


@requires_unstructured
def test_selector_selects_pdf_parser() -> None:
    from infrastructure.parsing import PDFParser

    selector = DocumentParserSelector()
    assert isinstance(selector.get_parser(Path("file.pdf")), PDFParser)


@requires_unstructured
def test_selector_selects_docx_parser() -> None:
    from infrastructure.parsing import DocxParser

    selector = DocumentParserSelector()
    assert isinstance(selector.get_parser(Path("file.docx")), DocxParser)


@requires_unstructured
def test_selector_falls_back_to_unstructured_parser() -> None:
    from infrastructure.parsing import UnstructuredParser

    selector = DocumentParserSelector()
    assert isinstance(selector.get_parser(Path("file.txt")), UnstructuredParser)
    assert isinstance(selector.get_parser(Path("file.html")), UnstructuredParser)


def test_selector_classes_are_module_attributes() -> None:
    """The merged parsing module exposes every parser class directly."""
    import infrastructure.parsing as parsing_module

    assert parsing_module.PDFParser.__name__ == "PDFParser"
    assert parsing_module.DocxParser.__name__ == "DocxParser"
    assert parsing_module.UnstructuredParser.__name__ == "UnstructuredParser"
    assert parsing_module.MarkdownParser.__name__ == "MarkdownParser"


def test_selector_missing_attribute_raises() -> None:
    import infrastructure.parsing as parsing_module

    try:
        parsing_module.NoSuchParser  # noqa: B018
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError")


def _install_fake_magic(monkeypatch, detected: str) -> None:
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules,
        "magic",
        SimpleNamespace(from_file=lambda path, mime=True: detected),
    )


def test_selector_rejects_exe_renamed_to_pdf(tmp_path, monkeypatch) -> None:
    from infrastructure.parsing import FileMimeTypeError

    _install_fake_magic(monkeypatch, "application/x-msdownload")
    fake_pdf = tmp_path / "doc.pdf"
    fake_pdf.write_bytes(b"MZ garbage")

    with pytest.raises(FileMimeTypeError, match="does not match"):
        DocumentParserSelector().get_parser(fake_pdf)


def test_selector_accepts_real_pdf(tmp_path, monkeypatch) -> None:
    from infrastructure.parsing import PDFParser

    _install_fake_magic(monkeypatch, "application/pdf")
    fake_pdf = tmp_path / "doc.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")

    assert isinstance(
        DocumentParserSelector().get_parser(fake_pdf), PDFParser
    )


def test_selector_text_family_flexibility(tmp_path, monkeypatch) -> None:
    """Any text/* content is acceptable for text-like formats."""
    _install_fake_magic(monkeypatch, "text/plain; charset=utf-8".split(";")[0])
    fake_md = tmp_path / "doc.md"
    fake_md.write_bytes(b"# hi")

    assert isinstance(
        DocumentParserSelector().get_parser(fake_md), MarkdownParser
    )


def test_selector_skips_mime_check_for_missing_files() -> None:
    # extension-based selection keeps working for in-memory paths
    assert isinstance(
        DocumentParserSelector().get_parser(Path("file.md")), MarkdownParser
    )
