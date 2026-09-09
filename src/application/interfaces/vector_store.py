"""Interface for vector database operations."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VectorStore(Protocol):
    """Abstract interface for a vector store (e.g., Qdrant)."""

    async def create_collection(self, dimension: int) -> None:
        """Create a new collection with given vector dimension."""
        ...

    async def drop_collection(self) -> None:
        """Delete the entire collection."""
        ...

    async def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """Insert or update vectors with associated payloads."""
        ...

    async def search(
        self,
        vector: list[float],
        top_k: int,
        filter_condition: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform similarity search.

        Returns:
            list of hits, each containing 'id', 'score', 'payload', and optionally 'text'.
        """  # noqa: E501
        ...

    async def delete_by_filter(self, filter_condition: dict[str, Any]) -> None:
        """Delete vectors matching a filter."""
        ...
