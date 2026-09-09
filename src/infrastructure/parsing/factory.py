"""Factory to select appropriate parser based on file extension."""

from pathlib import Path

from .base import DocumentParser
from .markdown_parser import MarkdownParser
from .pdf_parser import PDFParser
from .docx_parser import DocxParser
from .unstructured_parser import UnstructuredParser


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
        if ext == ".pdf":
            return PDFParser()
        if ext == ".docx":
            return DocxParser()
        # Default fallback using unstructured
        return UnstructuredParser()
