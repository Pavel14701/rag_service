"""GraphRAG: entity pseudo-graph stored in Qdrant.

Entities are points in a dedicated collection (name embedded as vector,
payload: name/kind/doc_ids/relations), living next to the document
collection: ``<collection>__entities``. No separate graph database is
required; see docs/graphrag.md for optional backends.
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models

from application.interfaces import EmbeddingModel, ExtractedRelation


class QdrantGraphStore:
    """Entity pseudo-graph backed by a dedicated Qdrant collection."""

    def __init__(
        self,
        client: QdrantClient,
        entities_collection: str,
        embedding: EmbeddingModel,
    ) -> None:
        self._client = client
        self._collection = entities_collection
        self._embedding = embedding
        self._collection_ready = False
        # cache of entity name (lowercased) -> embedding vector
        self._vector_cache: dict[str, list[float]] = {}

    async def add_relations(
        self,
        doc_id: uuid.UUID,
        relations: list[ExtractedRelation],
    ) -> None:
        """Store triples; entity points are created/merged idempotently."""
        await self._ensure_collection()
        for rel in relations:
            # the relation is stored on the subject point
            await self._add_entity_point(
                rel.subject, str(doc_id), rel.predicate, rel.obj
            )
            # the object gets its own point so multi-hop traversal works
            await self._add_entity_point(rel.obj, str(doc_id), None, None)

    async def related_doc_ids(
        self,
        entity_names: list[str],
        max_hops: int = 1,
    ) -> list[uuid.UUID]:
        """Doc ids reachable from the entities within ``max_hops`` hops."""
        doc_ids: set[str] = set()
        frontier = {
            name.strip().lower() for name in entity_names if name.strip()
        }
        visited: set[str] = set()
        for _ in range(max(1, max_hops)):
            if not frontier:
                break
            points = self._scroll_by_names(sorted(frontier))
            visited |= frontier
            next_frontier: set[str] = set()
            for point in points:
                payload = point.payload or {}
                doc_ids.update(str(d) for d in payload.get('doc_ids', []))
                for relation in payload.get('relations', []):
                    neighbor = str(relation.get('object', '')).strip().lower()
                    if neighbor and neighbor not in visited:
                        next_frontier.add(neighbor)
            frontier = next_frontier
        result: list[uuid.UUID] = []
        for raw in doc_ids:
            try:
                result.append(uuid.UUID(raw))
            except ValueError:
                continue
        return result

    # -- internals ----------------------------------------------------------

    def _scroll_by_names(self, names: list[str]) -> list[models.Record]:
        result, _offset = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key='name',
                        match=models.MatchAny(any=names),
                    )
                ]
            ),
            with_payload=True,
            limit=100,
        )
        return list(result)

    async def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._embedding.dimension(),
                    distance=models.Distance.COSINE,
                ),
            )
        self._collection_ready = True

    async def _add_entity_point(
        self,
        name: str,
        doc_id: str,
        predicate: str | None,
        obj: str | None,
    ) -> None:
        key = name.strip().lower()
        if not key:
            return
        vector = self._vector_cache.get(key)
        if vector is None:
            vector = (await self._embedding.embed([name]))[0]
            self._vector_cache[key] = vector
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f'graph-entity:{key}'))

        doc_ids: set[str] = {doc_id}
        relations: list[dict[str, str]] = []
        existing = self._client.retrieve(
            collection_name=self._collection,
            ids=[point_id],
            with_payload=True,
        )
        if existing:
            old = existing[0].payload or {}
            doc_ids.update(str(d) for d in old.get('doc_ids', []))
            relations = [
                {
                    'predicate': str(r.get('predicate')),
                    'object': str(r.get('object')),
                }
                for r in old.get('relations', [])
                if isinstance(r, dict)
            ]
        if predicate and obj:
            entry = {'predicate': predicate, 'object': obj}
            if entry not in relations:
                relations.append(entry)

        self._client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        'name': name,
                        'kind': 'entity',
                        'doc_ids': sorted(doc_ids),
                        'relations': relations,
                    },
                )
            ],
        )
