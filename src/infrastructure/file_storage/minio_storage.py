"""MinIO implementation of FileStorage."""

from pathlib import Path

from minio import Minio

from application.interfaces import FileStorage


class MinioStorage(FileStorage):
    """File storage adapter for MinIO."""

    def __init__(self, client: Minio, bucket_name: str) -> None:
        self._client = client
        self._bucket = bucket_name
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    async def upload_file(self, local_path: Path, destination_key: str) -> str:
        """Upload a local file and return its storage key."""
        self._client.fput_object(
            self._bucket, destination_key, str(local_path)
        )
        return destination_key

    async def download_file(
        self, source_key: str, destination_path: Path
    ) -> None:
        """Download a stored object to a local path."""
        self._client.fget_object(
            self._bucket, source_key, str(destination_path)
        )

    async def delete_file(self, source_key: str) -> None:
        """Delete a stored object."""
        self._client.remove_object(self._bucket, source_key)
