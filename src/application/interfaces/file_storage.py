"""Interface for file storage operations."""

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class FileStorage(Protocol):
    """Abstract interface for file storage (e.g., MinIO, local FS)."""

    async def upload_file(self, local_path: Path, destination_key: str) -> str:
        """
        Upload a file to storage.

        Args:
            local_path: Path to the local file.
            destination_key: Key (path) under which to store the file.

        Returns:
            The storage key of the uploaded file.
        """
        ...

    async def download_file(
        self,
        source_key: str,
        destination_path: Path
    ) -> None:
        """
        Download a file from storage to a local path.

        Args:
            source_key: Storage key of the file.
            destination_path: Local path to save the file.
        """
        ...

    async def delete_file(self, source_key: str) -> None:
        """Delete a file from storage."""
        ...
