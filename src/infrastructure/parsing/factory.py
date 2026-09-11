"""Factory to select appropriate parser based on file extension."""

from importlib import import_module
from pathlib import Path
from typing import cast

from .base import DocumentParser
from .markdown_parser import MarkdownParser

# Parsers backed by the heavy `unstructured` library are imported lazily:
# importing `unstructured` pulls in python-magic/libmagic, which may be
# unavailable (e.g. on Windows without libmagic installed).

_LAZY_PARSERS = {
    '.pdf': ('infrastructure.parsing.pdf_parser', 'PDFParser'),
    '.docx': ('infrastructure.parsing.docx_parser', 'DocxParser'),
}


class ParserFactory:
    """Factory for creating document parsers.

    OCR behavior for scanned PDFs is configured process-wide via
    ``configure`` (called from the DI container at startup with the
    ``PDF_OCR_STRATEGY`` / ``PDF_OCR_LANGUAGES`` settings).
    """

    pdf_ocr_strategy: str = 'auto'
    pdf_ocr_languages: str = 'eng'

    @classmethod
    def configure(cls, strategy: str, languages: str) -> None:
        """Set the process-wide PDF OCR strategy and languages."""
        cls.pdf_ocr_strategy = strategy
        cls.pdf_ocr_languages = languages

    @staticmethod
    def get_parser(file_path: Path) -> DocumentParser:
        """Return a parser instance based on file extension.

        Args:
            file_path: Path to the file.

        Returns:
            A DocumentParser implementation.

        """
        ext = file_path.suffix.lower()
        if ext == '.md':
            return MarkdownParser()
        if ext == '.pdf':
            pdf_parser_cls = getattr(
                import_module('infrastructure.parsing.pdf_parser'), 'PDFParser'
            )
            return cast(
                DocumentParser,
                pdf_parser_cls(
                    strategy=ParserFactory.pdf_ocr_strategy,
                    languages=ParserFactory.pdf_ocr_languages,
                ),
            )
        module_name, class_name = _LAZY_PARSERS.get(
            ext,
            (
                'infrastructure.parsing.unstructured_parser',
                'UnstructuredParser',
            ),
        )
        parser_cls = getattr(import_module(module_name), class_name)
        return cast(DocumentParser, parser_cls())
