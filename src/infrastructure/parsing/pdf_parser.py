"""Parser for PDF files using unstructured library."""

from pathlib import Path
from typing import Any

from unstructured.partition.pdf import partition_pdf

from .base import DocumentParser


class PDFParser(DocumentParser):
    """Parser for PDF documents that extracts text and metadata."""

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        """
        Parse a PDF file and return a list of text elements with metadata.

        Args:
            file_path: Path to the PDF file.

        Returns:
            list of dicts with keys 'text'
            and 'metadata' (includes page numbers).
        """
        elements = partition_pdf(
            filename=str(file_path),
            extract_images_in_pdf=False,
            infer_table_structure=True,
            strategy="auto",
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
                    "header": metadata.get("header", ""),
                    "footer": metadata.get("footer", ""),
                }
            })
        return result
