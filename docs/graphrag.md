# GraphRAG

GraphRAG complements vector search with an **entity graph**: documents are
indexed as usual, but an optional extraction step pulls
`(subject; predicate; object)` triples out of the text. At query time the
question's entities are matched against the graph, and chunks from
*graph-neighboring* documents (documents that mention related entities) are
merged into the retrieval context — capturing answers that no single chunk
contains.

Both directions are **opt-in** and independent:

- `GRAPH_EXTRACT_ENABLED=true` — extraction at indexing time;
- `GRAPH_EXPAND_ENABLED=true` — expansion at query time.

## Storage model: a pseudo-graph in Qdrant

The default backend stores entities in a dedicated Qdrant collection next to
the document collection: `<collection>__entities`.

- **One point per entity.** The point id is deterministic
  (`uuid5`, namespace URL, `graph-entity:<lowercased name>`), so repeated
  mentions upsert/merge instead of duplicating.
- **Vector** = embedding of the entity name (the same embedding model as the
  documents).
- **Payload**: `name`, `kind`, `doc_ids` (documents mentioning the entity),
  `relations` (`[{predicate, object}, ...]` outgoing edges).
- Relations are stored on the *subject* point; the *object* also gets its own
  point so multi-hop traversal works.
- No extra infrastructure is required, and the graph lives and dies with the
  same Qdrant deployment (same snapshots, same backups).

Traversal (`related_doc_ids`) is a BFS over the payload relations with a
per-hop payload filter (`MatchAny` on lowercased entity names), returning the
union of `doc_ids` within `GRAPH_MAX_HOPS`.

## Indexing pipeline integration

After the normal chunk upsert succeeds (see
[Retrieval](retrieval.md#indexing-pipeline)), `IndexerService` extracts
triples from the document text and stores them via the `GraphStore` port.
The graph is **best-effort**: any extraction or storage failure is logged
(`graph_extraction_failed`) and indexing still completes with `INDEXED`.

## Query pipeline integration

After the (possibly agentic) search rounds, `RetrieverService` extracts
entities from the search query, asks the graph for related documents, and
runs one additional search restricted to those `doc_id`s — with the **same
ACL filter** as the main search. Neighbor chunks are deduplicated against the
already-retrieved ids. Expansion is also best-effort: failures simply yield
no extra hits.

## Ports and adapters

| Port (`application/interfaces.py`) | Adapters |
|---|---|
| `EntityExtractor.extract(text) -> list[ExtractedRelation]` | `LLMEntityExtractor`, `NoOpEntityExtractor` (`infrastructure/graph_extract.py`) |
| `GraphStore.add_relations / related_doc_ids` | `QdrantGraphStore` (`infrastructure/graph_store.py`), `Neo4jGraphStore` (`infrastructure/graph_store_neo4j.py`, optional) |

`ExtractedRelation` is a frozen dataclass: `subject`, `predicate`, `obj`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GRAPH_EXTRACT_ENABLED` | `false` | Extract triples while indexing |
| `GRAPH_EXPAND_ENABLED` | `false` | Expand retrieval via graph neighbors |
| `GRAPH_MAX_HOPS` | `1` | Graph traversal depth at query time |
| `GRAPH_BACKEND` | `qdrant` | `qdrant` \| `neo4j` |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | empty | Required when `GRAPH_BACKEND=neo4j` |

Note: with `GRAPH_EXPAND_ENABLED=true` but `GRAPH_EXTRACT_ENABLED=false` the
graph stays empty, so expansion is a no-op. Extraction at least once is what
populates the graph.

## Cost model

- **Indexing**: one LLM extraction call per indexed document
  (`GRAPH_EXTRACT_ENABLED`).
- **Query**: one LLM extraction call per query plus one extra vector search
  (`GRAPH_EXPAND_ENABLED`).

## Optional backends & dependencies

The core service has **no new dependencies**: the Qdrant pseudo-graph uses the
existing client. Two extras are declared in `pyproject.toml` for scale-ups —
install them only when you actually outgrow the pseudo-graph:

| Extra | Packages | When you need it |
|---|---|---|
| `graph-neo4j` | `neo4j>=5.0` | Millions of edges, traversal depth ≥ 3, or you need real Cypher queries |
| `graph-communities` | `igraph>=0.11`, `leidenalg>=0.10` | Community detection (Leiden) over large entity graphs for global "theme" answers |

```bash
pip install rag[graph-neo4j]
# or
pip install rag[graph-communities]
```

`Neo4jGraphStore` imports the driver **lazily** and raises a `RuntimeError`
with install instructions when the extra is missing. The sync driver is
wrapped with `asyncio.to_thread` so the async event loop is never blocked.

Up to roughly 100k entities / 1M relations the pure-Python BFS in Qdrant is
fast enough; prefer Neo4j only beyond that, or when you need multi-hop Cypher
analytics the payload-based traversal cannot express.

> **License note:** Neo4j Community Edition is licensed under **GPLv3**.
> Using it as a separate server process from your (Apache-2.0) application is
> the usual setup; embedding it into a distributed Apache-2.0 product is not.
> Check your legal requirements before adding the extra to a commercial image.

## Testing

`tests/test_graph.py` (marker `retrieval`) covers triple parsing and graceful
degradation, entity point creation/merging in the Qdrant store, multi-hop
traversal, query-time expansion (including ACL filtering and dedup) and the
best-effort indexing hook.
