"""Fallback parser for any file format using unstructured auto-partition."""

from pathlib import Path
from typing import Any

from unstructured.partition.auto import partition

from .base import DocumentParser


class UnstructuredParser(DocumentParser):
    """Universal parser that uses unstructured's auto-detection.

    Works for many formats: .txt, .html, .epub, etc.
    """

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        """Parse any supported file format and return text elements.

        Args:
            file_path: Path to the file.

        Returns:
            list of dicts with 'text' and 'metadata'.

        """
        elements = partition(
            filename=str(file_path),
            infer_table_structure=True,
            strategy='auto',
        )
        result: list[dict[str, Any]] = []
        for el in elements:
            text = str(el).strip()
            if not text:
                continue
            metadata = el.metadata.to_dict() if hasattr(el, 'metadata') else {}
            result.append(
                {
                    'text': text,
                    'metadata': {
                        'page': metadata.get('page_number', 0),
                        'category': metadata.get('category', ''),
                        'filetype': metadata.get('filetype', ''),
                    },
                }
            )
        return result
