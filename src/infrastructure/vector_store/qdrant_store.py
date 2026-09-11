"""Qdrant implementation of VectorStore."""

from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from application.interfaces.vector_store import VectorStore
from shared.hybrid import bm25_rank, rrf_fuse
from shared.metrics import VECTOR_SEARCH_SECONDS
from shared.tracing import get_tracer


class QdrantStore(VectorStore):
    """Vector store implementation using Qdrant.

    Optional HNSW tuning: ``hnsw_m``/``hnsw_ef_construct`` are applied
    at collection creation, ``hnsw_ef`` at query time.

    Hybrid search: when ``fulltext_enabled`` is True a full-text index
    is created on the ``text`` payload field; ``search`` with a
    ``keyword_query`` then fetches lexical candidates, ranks them with
    BM25 client-side and fuses both ranked lists with RRF.
    """

    _TEXT_FIELD = 'text'

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        hnsw_m: int | None = None,
        hnsw_ef_construct: int | None = None,
        hnsw_ef: int | None = None,
        fulltext_enabled: bool = False,
        hybrid_candidates: int = 50,
        hybrid_rrf_k: int = 60,
    ) -> None:
        self._client = client
        self._collection = collection_name
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construct = hnsw_ef_construct
        self._hnsw_ef = hnsw_ef
        self._fulltext_enabled = fulltext_enabled
        self._hybrid_candidates = hybrid_candidates
        self._hybrid_rrf_k = hybrid_rrf_k

    def _build_field_condition(
        self, cond: dict[str, Any]
    ) -> models.FieldCondition:
        """Convert a ``match`` dict into a Qdrant field condition."""
        value = cond['match']['value']
        if isinstance(value, (list, tuple, set)):
            match: models.MatchValue | models.MatchAny = models.MatchAny(
                any=list(value)
            )
        else:
            match = models.MatchValue(value=value)
        return models.FieldCondition(key=cond['key'], match=match)

    def _build_filter(self, condition: dict[str, Any]) -> models.Filter:
        """Convert a generic filter dict into a Qdrant Filter.

        Supported forms:
          - ``{"key": ..., "match": {"value": ...}}`` -> single ``must``
          - ``{"must": [...], "should": [...], "must_not": [...]}`` (recursive)
        """
        if 'key' in condition:
            return models.Filter(must=[self._build_field_condition(condition)])

        kwargs: dict[str, Any] = {}
        for clause in ('must', 'should', 'must_not'):
            sub_conditions = condition.get(clause)
            if not sub_conditions:
                continue
            converted: list[models.FieldCondition | models.Filter] = []
            for sub in sub_conditions:
                if 'key' in sub:
                    converted.append(self._build_field_condition(sub))
                else:
                    converted.append(self._build_filter(sub))
            kwargs[clause] = converted
        return models.Filter(**kwargs)

    async def create_collection(self, dimension: int) -> None:
        """Create a new collection with given vector dimension."""
        kwargs: dict[str, Any] = {
            'collection_name': self._collection,
            'vectors_config': models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
        }
        if self._hnsw_m is not None or self._hnsw_ef_construct is not None:
            kwargs['hnsw_config'] = models.HnswConfigDiff(
                m=self._hnsw_m,
                ef_construct=self._hnsw_ef_construct,
            )
        self._client.recreate_collection(**kwargs)
        if self._fulltext_enabled:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=self._TEXT_FIELD,
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WORD,
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
        keyword_query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Similarity search; fused with BM25 when ``keyword_query`` given."""
        qdrant_filter = None
        if filter_condition:
            qdrant_filter = self._build_filter(filter_condition)

        with get_tracer().start_as_current_span('vector.search') as span:
            span.set_attribute('qdrant.collection', self._collection)
            span.set_attribute('search.top_k', top_k)
            span.set_attribute('search.hybrid', keyword_query is not None)
            with VECTOR_SEARCH_SECONDS.time():
                query_kwargs: dict[str, Any] = {}
                if self._hnsw_ef is not None:
                    query_kwargs['search_params'] = models.SearchParams(
                        hnsw_ef=self._hnsw_ef
                    )
                response = self._client.query_points(
                    collection_name=self._collection,
                    query=vector,
                    limit=top_k,
                    query_filter=qdrant_filter,
                    with_payload=True,
                    **query_kwargs,
                )
                vector_hits = [
                    self._point_to_hit(point) for point in response.points
                ]
                if keyword_query:
                    hits = self._hybrid_fuse(
                        vector_hits, keyword_query, top_k, qdrant_filter
                    )
                else:
                    hits = vector_hits
            span.set_attribute('search.results', len(hits))
            return hits

    @staticmethod
    def _point_to_hit(point: Any) -> dict[str, Any]:
        payload = point.payload or {}
        return {
            'id': point.id,
            'score': point.score,
            'payload': payload,
            'text': payload.get('text', ''),
        }

    def _hybrid_fuse(
        self,
        vector_hits: list[dict[str, Any]],
        keyword_query: str,
        top_k: int,
        qdrant_filter: Optional[models.Filter],
    ) -> list[dict[str, Any]]:
        """Fuse vector ranking with a client-side BM25 lexical ranking."""
        keyword_points = self._scroll_keyword_candidates(
            keyword_query, qdrant_filter
        )
        if not keyword_points:
            return vector_hits

        corpus = [
            p.payload.get('text', '') if p.payload else ''
            for p in keyword_points
        ]
        keyword_ranking = bm25_rank(keyword_query, corpus)
        keyword_ids = [keyword_points[i].id for i in keyword_ranking]

        id_to_hit = {hit['id']: hit for hit in vector_hits}
        for point in keyword_points:
            id_to_hit.setdefault(point.id, self._point_to_hit(point))

        fused_ids = rrf_fuse(
            [
                [hit['id'] for hit in vector_hits],
                keyword_ids,
            ],
            k=self._hybrid_rrf_k,
            top_k=top_k,
        )
        fused = []
        for position, point_id in enumerate(fused_ids, start=1):
            hit = id_to_hit[point_id]
            fused.append({**hit, 'score': 1 / (self._hybrid_rrf_k + position)})
        return fused

    def _scroll_keyword_candidates(
        self,
        keyword_query: str,
        qdrant_filter: Optional[models.Filter],
    ) -> list[Any]:
        """Fetch lexical candidates via the full-text index on ``text``."""
        must: list[Any] = (
            list(qdrant_filter.must)
            if qdrant_filter and qdrant_filter.must
            else []
        )
        must.append(
            models.FieldCondition(
                key=self._TEXT_FIELD,
                match=models.MatchText(text=keyword_query),
            )
        )
        try:
            points, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=models.Filter(must=must),
                limit=self._hybrid_candidates,
                with_payload=True,
            )
        except Exception:  # noqa: BLE001 - hybrid degrades to pure vector
            return []
        return points

    async def scroll_first_payload(self) -> Optional[dict[str, Any]]:
        """Payload of an arbitrary point; used for the model-version marker."""
        try:
            points, _ = self._client.scroll(
                collection_name=self._collection,
                limit=1,
                with_payload=True,
            )
        except Exception:  # noqa: BLE001 - missing collection -> no version
            return None
        if not points:
            return None
        return points[0].payload or {}

    async def delete_by_filter(self, filter_condition: dict[str, Any]) -> None:
        """Delete vectors matching a filter."""
        q_filter = self._build_filter(filter_condition)
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=q_filter),
        )
