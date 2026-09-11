"""Parser for PDF files using unstructured library."""

from pathlib import Path
from typing import Any

from unstructured.partition.pdf import partition_pdf

from .base import DocumentParser


class PDFParser(DocumentParser):
    """Parser for PDF documents that extracts text and metadata.

    ``strategy`` controls the OCR handling for scanned documents
    (unstructured partition strategy):
    - ``auto``        — OCR when needed (default);
    - ``ocr_only``    — force OCR for all pages;
    - ``hi_res``      — high-resolution layout+OCR pipeline;
    - ``fast``        — plain text extraction, no OCR.

    ``languages`` is a comma-separated list of tesseract languages
    (e.g. ``eng`` or ``eng,rus``) used when OCR runs.
    """

    def __init__(self, strategy: str = "auto", languages: str = "eng") -> None:
        self._strategy = strategy
        self._languages = languages

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        """
        Parse a PDF file and return a list of text elements with metadata.

        Args:
            file_path: Path to the PDF file.

        Returns:
            list of dicts with keys 'text'
            and 'metadata' (includes page numbers).
        """
        languages = [lang for lang in self._languages.split(",") if lang]
        elements = partition_pdf(
            filename=str(file_path),
            extract_images_in_pdf=False,
            infer_table_structure=True,
            strategy=self._strategy,
            languages=languages or None,
        )
        result: list[dict[str, Any]] = []
        for el in elements:
            text = str(el).strip()
            if not text:
                continue
            metadata = el.metadata.to_dict() if hasattr(el, "metadata") else {}
            # Multimodality: tag the element category ("table", "image",
            # "title", ...) and keep the HTML representation of tables so
            # they can be embedded and stored with structure preserved.
            el_type = str(metadata.get("category") or "text").lower()
            el_metadata: dict[str, Any] = {
                "page": metadata.get("page_number", 0),
                "header": metadata.get("header", ""),
                "footer": metadata.get("footer", ""),
                "type": el_type,
            }
            table_html = metadata.get("text_as_html")
            if el_type == "table" and table_html:
                el_metadata["table_html"] = table_html
            result.append({
                "text": text,
                "metadata": el_metadata,
            })
        return result
