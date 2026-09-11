"""Tests for PDF OCR configuration (strategy / languages passthrough)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from infrastructure.parsing.factory import ParserFactory

from conftest import requires_unstructured


@pytest.fixture(autouse=True)
def reset_factory():
    yield
    ParserFactory.configure("auto", "eng")


def test_factory_configure_sets_defaults():
    ParserFactory.configure("ocr_only", "eng,rus")
    assert ParserFactory.pdf_ocr_strategy == "ocr_only"
    assert ParserFactory.pdf_ocr_languages == "eng,rus"


@requires_unstructured
def test_pdf_parser_receives_configured_strategy():
    ParserFactory.configure("ocr_only", "eng,rus")
    parser = ParserFactory.get_parser(Path("file.pdf"))
    assert parser._strategy == "ocr_only"
    assert parser._languages == "eng,rus"


@requires_unstructured
def test_pdf_parser_passes_strategy_and_languages_to_partition():
    from infrastructure.parsing.pdf_parser import PDFParser

    element = MagicMock()
    element.__str__ = lambda self: "page text"
    element.metadata.to_dict.return_value = {"page_number": 3}

    with patch(
        "infrastructure.parsing.pdf_parser.partition_pdf"
    ) as partition:
        partition.return_value = [element]
        result = PDFParser(strategy="ocr_only", languages="eng,rus").parse(Path("x.pdf"))

    kwargs = partition.call_args.kwargs
    assert kwargs["strategy"] == "ocr_only"
    assert kwargs["languages"] == ["eng", "rus"]
    assert kwargs["infer_table_structure"] is True
    assert result[0]["metadata"]["page"] == 3


@requires_unstructured
def test_pdf_parser_default_strategy_is_auto():
    from infrastructure.parsing.pdf_parser import PDFParser

    with patch("infrastructure.parsing.pdf_parser.partition_pdf") as partition:
        partition.return_value = []
        PDFParser().parse(Path("x.pdf"))
    assert partition.call_args.kwargs["strategy"] == "auto"
    assert partition.call_args.kwargs["languages"] == ["eng"]