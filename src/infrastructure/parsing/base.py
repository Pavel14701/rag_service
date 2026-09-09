"""Parser interface and factory for document parsing."""

from pathlib import Path
from typing import Protocol, runtime_checkable, Any


@runtime_checkable
class DocumentParser(Protocol):
    """Interface for parsing different document formats."""

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        """
        Parse a document and return a list of elements.

        Each element is a dict with 'text' and 'metadata' keys.
        """
        ...
