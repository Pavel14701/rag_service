"""Parser for Markdown files, preserving header hierarchy."""

from pathlib import Path
from typing import Any
import markdown
from bs4 import BeautifulSoup

from .base import DocumentParser


class MarkdownParser(DocumentParser):
    """Parser for .md files that splits by headers."""

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        """
        Parse markdown, splitting on headers (h1-h3).

        Returns:
            list of chunks, each with 'text' and metadata including 'header'.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        html = markdown.markdown(content)
        soup = BeautifulSoup(html, "html.parser")

        chunks: list[dict[str, Any]] = []
        current_section: list[str] = []
        header = "Root"

        for el in soup.find_all(["h1", "h2", "h3", "p", "ul", "ol", "pre"]):
            if el.name in ("h1", "h2", "h3"):
                if current_section:
                    chunks.append({
                        "text": "\n".join(current_section),
                        "metadata": {"header": header},
                    })
                    current_section = []
                header = el.get_text(strip=True)
                current_section.append(header)
            else:
                current_section.append(el.get_text(strip=True))

        if current_section:
            chunks.append({
                "text": "\n".join(current_section),
                "metadata": {"header": header},
            })

        return chunks
