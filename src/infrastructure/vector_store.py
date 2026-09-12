"""Vector store: Qdrant adapter and hybrid search (BM25 + RRF).

The BM25/RRF helpers support the optional hybrid search mode: semantic
(vector) hits are fused with lexical (keyword) hits fetched via the
Qdrant full-text index, without external dependencies.
"""

import math
import re
import time
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from application.interfaces import VectorStore
from infrastructure.observability import VECTOR_SEARCH_SECONDS, get_tracer

_TOKEN_RE = re.compile(r'[a-z0-9а-яё]+', re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase tokenization over latin/cyrillic words and digits."""
    return _TOKEN_RE.findall(text.lower())


def bm25_scores(
    query: str,
    corpus: list[str],
    k: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """BM25 score per corpus document (0.0 when terms do not match).

    Args:
        query: Keyword query text.
        corpus: Document texts; index in the corpus is the document ID.
        k: BM25 term-frequency saturation parameter.
        b: BM25 length normalization parameter.

    Returns:
        One score per corpus document, in corpus order.

    """
    doc_tokens = [tokenize(doc) for doc in corpus]
    doc_count = len(doc_tokens)
    if doc_count == 0:
        return []
    avg_len = sum(len(toks) for toks in doc_tokens) / doc_count or 1.0

    df: dict[str, int] = {}
    for toks in doc_tokens:
        for term in set(toks):
            df[term] = df.get(term, 0) + 1

    query_terms = tokenize(query)
    if not query_terms:
        return [0.0] * doc_count

    scores: list[float] = [0.0] * doc_count
    for index, toks in enumerate(doc_tokens):
        tf: dict[str, int] = {}
        for term in toks:
            tf[term] = tf.get(term, 0) + 1
        length_norm = len(toks) / avg_len
        for term in query_terms:
            occurrences = tf.get(term, 0)
            if not occurrences:
                continue
            idf = _idf(df.get(term, 0), doc_count)
            scores[index] += (
                idf
                * occurrences
                * (k + 1)
                / (occurrences + k * (1 - b + b * length_norm))
            )
    return scores


def bm25_rank(
    query: str,
    corpus: list[str],
    k: float = 1.5,
    b: float = 0.75,
) -> list[int]:
    """Rank corpus documents against the query with BM25.

    Returns:
        Corpus indices sorted by descending BM25 score (documents with
        zero score are excluded).

    """
    scores = bm25_scores(query, corpus, k, b)
    return sorted(
        (i for i, score in enumerate(scores) if score > 0),
        key=lambda i: scores[i],
        reverse=True,
    )


def strip_stopwords(text: str, stopwords: frozenset[str]) -> str:
    """Drop stop-word tokens (lowercased match) from ``text``.

    Popular terms (company names, boilerplate) flood the lexical
    candidates with noise that would crowd out relevant hits.
    """
    if not stopwords:
        return text
    return ' '.join(
        token for token in tokenize(text) if token not in stopwords
    )


def _idf(df: int, doc_count: int) -> float:
    """BM25 inverse document frequency."""
    return math.log(1 + (doc_count - df + 0.5) / (df + 0.5))


def rrf_fuse(
    ranked_lists: list[list[Any]],
    k: int = 60,
    top_k: int | None = None,
) -> list[Any]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion.

    score(item) = sum over lists of 1 / (k + rank), rank starting at 1.

    Args:
        ranked_lists: Ordered items per list (best first).
        k: RRF damping constant (higher reduces the impact of top ranks).
        top_k: Limit the fused result length; None = all fused items.

    Returns:
        Items ordered by descending fused score (ties keep the order of
        first appearance in the first list).

    """
    scores: dict[Any, float] = {}
    first_seen: dict[Any, int] = {}
    for items in ranked_lists:
        for rank, item in enumerate(items, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
            first_seen.setdefault(item, len(first_seen))
    ordered = sorted(
        scores, key=lambda item: (-scores[item], first_seen[item])
    )
    return ordered if top_k is None else ordered[:top_k]


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
        bm25_stopwords: frozenset[str] | None = None,
        bm25_score_threshold: float = 0.0,
    ) -> None:
        self._client = client
        self._collection = collection_name
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construct = hnsw_ef_construct
        self._hnsw_ef = hnsw_ef
        self._fulltext_enabled = fulltext_enabled
        self._hybrid_candidates = hybrid_candidates
        self._hybrid_rrf_k = hybrid_rrf_k
        self._bm25_stopwords = bm25_stopwords or frozenset()
        self._bm25_score_threshold = bm25_score_threshold
        # Blue-Green: name of the offline green collection being built.
        self._shadow: str | None = None

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

    def _write_targets(self) -> list[str]:
        """Collections receiving writes: active + shadow (dual-write)."""
        if self._shadow is not None:
            return [self._collection, self._shadow]
        return [self._collection]

    async def create_collection(self, dimension: int) -> None:
        """Create the active collection with the given dimension."""
        self._create_collection_named(self._collection, dimension)

    def _create_collection_named(self, name: str, dimension: int) -> None:
        """Create a named collection with vectors/HNSW/fulltext config."""
        kwargs: dict[str, Any] = {
            'collection_name': name,
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
                collection_name=name,
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
        for target in self._write_targets():
            self._client.upsert(collection_name=target, points=points)

    async def retrieve_by_ids(
        self,
        ids: list[str],
        filter_condition: dict[str, Any] | None = None,
    ) -> list[str]:
        """Return the subset of ``ids`` visible under the ACL filter.

        Qdrant retrieve skips points excluded by the read filter, so
        the returned list contains exactly the ids the requester is
        allowed to see (cache-then-validate ACL gateway).
        """
        if not ids:
            return []
        read_filter = (
            self._build_filter(filter_condition) if filter_condition else None
        )
        points = self._client.retrieve(
            collection_name=self._collection,
            ids=ids,
            with_payload=False,
            with_vectors=False,
            read_filter=read_filter,
        )
        return [str(point.id) for point in points]

    async def search(
        self,
        vector: list[float],
        top_k: int,
        filter_condition: dict[str, Any] | None = None,
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
        qdrant_filter: models.Filter | None,
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
        # Stop-word cleanup + score threshold keep BM25 noise (hits
        # matching only a popular term) out of the RRF fusion.
        query_text = strip_stopwords(keyword_query, self._bm25_stopwords)
        scores = bm25_scores(query_text, corpus)
        keyword_ranking = sorted(
            (
                i
                for i, score in enumerate(scores)
                if score > self._bm25_score_threshold
            ),
            key=lambda i: scores[i],
            reverse=True,
        )
        keyword_ids = [keyword_points[i].id for i in keyword_ranking]
        if not keyword_ids:
            return vector_hits

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
        qdrant_filter: models.Filter | None,
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

    async def scroll_first_payload(self) -> dict[str, Any] | None:
        """Payload of an arbitrary point; used for the model-version marker."""
        try:
            points, _ = self._client.scroll(
                collection_name=self._collection,
                limit=1,
                with_payload=True,
            )
        except Exception:  # noqa: BLE001 - missing collection -> no version
            return None
        return points[0].payload or {} if points else None

    async def delete_by_filter(self, filter_condition: dict[str, Any]) -> None:
        """Delete vectors matching a filter."""
        q_filter = self._build_filter(filter_condition)
        for target in self._write_targets():
            self._client.delete(
                collection_name=target,
                points_selector=models.FilterSelector(filter=q_filter),
            )

    async def create_shadow_collection(self, dimension: int) -> str:
        """Create the offline green collection for a blue-green reindex.

        Writes become dual-routed (active + green) so concurrent ingest
        is not lost; reads keep serving the active collection.
        """
        shadow = f'{self._collection}__green_{int(time.time())}'
        self._create_collection_named(shadow, dimension)
        self._shadow = shadow
        return shadow

    async def promote_shadow(self) -> None:
        """Atomically flip the collection alias to the green collection.

        The configured collection name acts as the public alias. The
        superseded collection is dropped after the flip.
        """
        if self._shadow is None:
            raise ValueError('No shadow collection to promote')
        alias = self._collection
        shadow = self._shadow
        current = {
            entry.alias_name: entry.collection_name
            for entry in self._client.get_aliases().aliases
        }
        if current.get(alias) == shadow:
            self._shadow = None  # already promoted
            return
        if alias in current:
            # Atomic flip: drop + create within a single alias operation.
            operation: list[
                models.CreateAliasOperation | models.DeleteAliasOperation
            ] = [
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=alias),
                ),
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=shadow,
                        alias_name=alias,
                    ),
                ),
            ]
        elif alias in {
            c.name for c in self._client.get_collections().collections
        }:
            raise RuntimeError(
                f'{alias!r} is a physical collection left from an older ',
                'deployment; blue-green promotion needs that name for ',
                'the alias. Move it aside manually once (reindex it ',
                'under a versioned name or delete it).',
            )
        else:
            operation = [
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=shadow,
                        alias_name=alias,
                    ),
                ),
            ]
        self._client.update_collection_aliases(
            change_aliases_operations=operation
        )
        superseded = current.get(alias)
        if superseded and superseded != shadow:
            self._client.delete_collection(collection_name=superseded)
        self._collection = shadow
        self._shadow = None

    async def discard_shadow(self) -> None:
        """Drop the green collection without promoting it."""
        if self._shadow is not None:
            self._client.delete_collection(collection_name=self._shadow)
        self._shadow = None

    async def optimize_collection(self) -> None:
        """Trigger segment merge / HNSW rebuild after bulk indexing."""
        self._client.update_collection(
            collection_name=self._collection,
            optimizer_config=models.OptimizersConfigDiff(),
        )

    async def delete_by_doc_id(self, doc_id: uuid.UUID) -> None:
        """Delete all chunks of a document (prevents ghost chunks)."""
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key='doc_id',
                        match=models.MatchValue(value=str(doc_id)),
                    )
                ]
            )
        )
        for target in self._write_targets():
            self._client.delete(
                collection_name=target, points_selector=selector
            )

    async def get_collection_dimension(self) -> int | None:
        """Vector dimension of the collection; None when it is absent."""
        try:
            info = self._client.get_collection(
                collection_name=self._collection
            )
        except Exception:  # noqa: BLE001 - missing collection -> fresh deploy
            return None
        vectors = getattr(
            getattr(getattr(info, 'config', None), 'params', None),
            'vectors',
            None,
        )
        size = getattr(vectors, 'size', None)
        return int(size) if size is not None else None
