"""Document parsers package.

Heavy parsers (PDF/DOCX/unstructured) are exposed lazily so that simply
importing this package does not pull in `unstructured` (and transitively
python-magic/libmagic, which can be missing on some platforms).
"""

from typing import Any

from .base import DocumentParser
from .markdown_parser import MarkdownParser
from .factory import ParserFactory

_LAZY_EXPORTS = {
    "PDFParser": "infrastructure.parsing.pdf_parser",
    "DocxParser": "infrastructure.parsing.docx_parser",
    "UnstructuredParser": "infrastructure.parsing.unstructured_parser",
}

__all__ = [
    "DocumentParser",
    "MarkdownParser",
    "PDFParser",
    "DocxParser",
    "UnstructuredParser",
    "ParserFactory",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        return getattr(import_module(_LAZY_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
