"""GraphRAG: optional Neo4j backend for the GraphStore port.

Requires the optional ``graph-neo4j`` extra
(``pip install rag[graph-neo4j]``). It is NOT installed by default: the
bundled Qdrant pseudo-graph covers the documented use cases, and the Neo4j
Community Edition is licensed under GPLv3 - see docs/graphrag.md for the
decision guidance (edge volume, traversal depth, Cypher needs).

The sync ``neo4j`` driver is wrapped with ``asyncio.to_thread`` so the
async retrieval loop is never blocked.
"""

import asyncio
import uuid

from application.interfaces import ExtractedRelation

_MERGE_QUERY = (
    'MERGE (s:Entity {name: $subject}) '
    'MERGE (o:Entity {name: $object}) '
    'MERGE (s)-[r:RELATION {predicate: $predicate}]->(o) '
    'SET s.doc_ids = CASE WHEN s.doc_ids IS NULL THEN [$doc_id] '
    'ELSE CASE WHEN $doc_id IN s.doc_ids THEN s.doc_ids '
    'ELSE s.doc_ids + $doc_id END END '
    'SET o.doc_ids = CASE WHEN o.doc_ids IS NULL THEN [$doc_id] '
    'ELSE CASE WHEN $doc_id IN o.doc_ids THEN o.doc_ids '
    'ELSE o.doc_ids + $doc_id END END'
)


class Neo4jGraphStore:
    """GraphStore on Neo4j (opt-in backend, lazy driver import)."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover - extra not installed
            raise RuntimeError(
                'The Neo4j graph backend requires the optional extra '
                "'graph-neo4j' (pip install rag[graph-neo4j])"
            ) from exc
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    async def add_relations(
        self,
        doc_id: uuid.UUID,
        relations: list[ExtractedRelation],
    ) -> None:
        """MERGE entity nodes and typed relations with doc id lists."""
        rows = [(r.subject, r.predicate, r.obj) for r in relations]
        await asyncio.to_thread(self._add_relations_sync, str(doc_id), rows)

    def _add_relations_sync(
        self, doc_id: str, rows: list[tuple[str, str, str]]
    ) -> None:
        with self._driver.session() as session:
            for subject, predicate, obj in rows:
                session.run(
                    _MERGE_QUERY,
                    doc_id=doc_id,
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                )

    async def related_doc_ids(
        self,
        entity_names: list[str],
        max_hops: int = 1,
    ) -> list[uuid.UUID]:
        """Cypher variable-length traversal, ``1..max_hops`` hops."""
        names = [name.strip().lower() for name in entity_names if name.strip()]
        if not names:
            return []
        raw = await asyncio.to_thread(self._related_sync, names, max_hops)
        result: list[uuid.UUID] = []
        for doc_id in raw:
            try:
                result.append(uuid.UUID(doc_id))
            except ValueError:
                continue
        return result

    def _related_sync(self, names: list[str], max_hops: int) -> list[str]:
        hops = max(1, int(max_hops))
        query = (
            'MATCH (e:Entity) WHERE toLower(e.name) IN $names '
            f'MATCH (e)-[:RELATION*1..{hops}]-(n) '
            'UNWIND n.doc_ids AS doc_id RETURN DISTINCT doc_id'
        )
        with self._driver.session() as session:
            return [
                str(record['doc_id'])
                for record in session.run(query, names=names)
            ]

    def close(self) -> None:
        """Release the driver connection pool."""
        self._driver.close()
