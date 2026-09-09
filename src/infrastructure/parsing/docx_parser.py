"""Parser for DOCX files using unstructured library."""

from pathlib import Path
from typing import Any

from unstructured.partition.docx import partition_docx

from .base import DocumentParser


class DocxParser(DocumentParser):
    """Parser for DOCX documents that extracts text and metadata."""

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        """
        Parse a DOCX file and return a list of text elements with metadata.

        Args:
            file_path: Path to the DOCX file.

        Returns:
            list of dicts with keys 'text' and 'metadata'
            (includes page numbers if available).
        """
        elements = partition_docx(
            filename=str(file_path),
            infer_table_structure=True,
        )
        result: list[dict[str, Any]] = []
        for el in elements:
            text = str(el).strip()
            if not text:
                continue
            metadata = el.metadata.to_dict() if hasattr(el, "metadata") else {}
            result.append({
                "text": text,
                "metadata": {
                    "page": metadata.get("page_number", 0),
                    "category": metadata.get("category", ""),
                }
            })
        return result
