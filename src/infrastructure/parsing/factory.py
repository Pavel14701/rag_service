"""Factory to select appropriate parser based on file extension."""

from importlib import import_module
from pathlib import Path

from .base import DocumentParser
from .markdown_parser import MarkdownParser

# Parsers backed by the heavy `unstructured` library are imported lazily:
# importing `unstructured` pulls in python-magic/libmagic, which may be
# unavailable (e.g. on Windows without libmagic installed).

_LAZY_PARSERS = {
    ".pdf": ("infrastructure.parsing.pdf_parser", "PDFParser"),
    ".docx": ("infrastructure.parsing.docx_parser", "DocxParser"),
}


class ParserFactory:
    """Factory for creating document parsers."""

    @staticmethod
    def get_parser(file_path: Path) -> DocumentParser:
        """
        Return a parser instance based on file extension.

        Args:
            file_path: Path to the file.

        Returns:
            A DocumentParser implementation.
        """
        ext = file_path.suffix.lower()
        if ext == ".md":
            return MarkdownParser()
        module_name, class_name = _LAZY_PARSERS.get(
            ext, ("infrastructure.parsing.unstructured_parser", "UnstructuredParser")
        )
        parser_cls = getattr(import_module(module_name), class_name)
        return parser_cls()
