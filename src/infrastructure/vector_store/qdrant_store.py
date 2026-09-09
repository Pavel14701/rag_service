"""Qdrant implementation of VectorStore."""

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from application.interfaces.vector_store import VectorStore


class QdrantStore(VectorStore):
    """Vector store implementation using Qdrant."""

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection = collection_name

    async def create_collection(self, dimension: int) -> None:
        """Create a new collection with given vector dimension."""
        self._client.recreate_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
        )

    async def drop_collection(self) -> None:
        """Delete the collection."""
        self._client.delete_collection(collection_name=self._collection)

    async def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """Insert or update vectors."""
        points = [
            models.PointStruct(
                id=id,
                vector=vector,
                payload=payload,
            )
            for id, vector, payload in zip(ids, vectors, payloads)
        ]
        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )

    async def search(
        self,
        vector: list[float],
        top_k: int,
        filter_condition: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Perform similarity search with optional filter."""
        qdrant_filter = None
        if filter_condition:
            # Convert filter condition to Qdrant filter (simplified)
            # For production, implement a proper conversion.
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=filter_condition["key"],
                        match=models.MatchValue(value=filter_condition["match"]["value"]),
                    )
                ]
            )

        hits = self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
                "text": hit.payload.get("text", ""),
            }
            for hit in hits
        ]

    async def delete_by_filter(self, filter_condition: dict[str, Any]) -> None:
        """Delete vectors matching a filter."""
        # Convert to Qdrant filter
        q_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=filter_condition["key"],
                    match=models.MatchValue(value=filter_condition["match"]["value"]),
                )
            ]
        )
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=q_filter),
        )