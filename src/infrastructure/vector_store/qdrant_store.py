"""Qdrant implementation of VectorStore."""

from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from application.interfaces.vector_store import VectorStore


class QdrantStore(VectorStore):
    """Vector store implementation using Qdrant."""

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection = collection_name

    def _build_field_condition(self, cond: dict[str, Any]) -> models.FieldCondition:
        """Convert ``{"key": ..., "match": {"value": ...}}`` to a Qdrant condition."""
        value = cond["match"]["value"]
        if isinstance(value, (list, tuple, set)):
            match: models.MatchValue | models.MatchAny = models.MatchAny(any=list(value))
        else:
            match = models.MatchValue(value=value)
        return models.FieldCondition(key=cond["key"], match=match)

    def _build_filter(self, condition: dict[str, Any]) -> models.Filter:
        """Convert a generic filter dict into a Qdrant Filter.

        Supported forms:
          - ``{"key": ..., "match": {"value": ...}}`` (single condition -> must)
          - ``{"must": [...], "should": [...], "must_not": [...]}`` (recursive)
        """
        if "key" in condition:
            return models.Filter(must=[self._build_field_condition(condition)])

        kwargs: dict[str, Any] = {}
        for clause in ("must", "should", "must_not"):
            sub_conditions = condition.get(clause)
            if not sub_conditions:
                continue
            converted = []
            for sub in sub_conditions:
                if "key" in sub:
                    converted.append(self._build_field_condition(sub))
                else:
                    converted.append(self._build_filter(sub))
            kwargs[clause] = converted
        return models.Filter(**kwargs)

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
                id=point_id,
                vector=vector,
                payload=payload,
            )
            for point_id, vector, payload in zip(ids, vectors, payloads)
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
            qdrant_filter = self._build_filter(filter_condition)

        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [
            {
                "id": point.id,
                "score": point.score,
                "payload": point.payload or {},
                "text": (point.payload or {}).get("text", ""),
            }
            for point in response.points
        ]

    async def delete_by_filter(self, filter_condition: dict[str, Any]) -> None:
        """Delete vectors matching a filter."""
        q_filter = self._build_filter(filter_condition)
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=q_filter),
        )