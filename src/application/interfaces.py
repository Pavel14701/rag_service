"""Ports (interfaces) implemented by infrastructure adapters.

Every application service depends only on these protocols; concrete
implementations live in :mod:`infrastructure` and are wired in the
DI container (composition root).
"""

import uuid
from dataclasses import dataclass
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from domain.model import Document, DocStatus


@runtime_checkable
class DocumentRepository(Protocol):
    """Abstract interface for document metadata storage."""

    async def get_document(self, doc_id: uuid.UUID) -> Document | None:
        """Retrieve a document by its ID."""
        ...

    async def get_all_active(self) -> list[Document]:
        """Get all non-deleted documents."""
        ...

    async def save(self, document: Document) -> None:
        """Persist a document record."""
        ...

    async def update_status(
        self, doc_id: uuid.UUID, status: DocStatus
    ) -> None:
        """Update the indexing status of a document."""
        ...

    async def mark_deleted(self, doc_id: uuid.UUID) -> None:
        """Soft-delete a document."""
        ...

    async def get_user_groups(self, user_id: str) -> list[str]:
        """Retrieve access groups for a given user from the DB."""
        ...

    async def set_user_groups(self, user_id: str, groups: list[str]) -> None:
        """Replace the set of access groups for a given user."""
        ...

    async def save_conversation(
        self,
        user_id: str,
        query: str,
        response: str,
        sources: list[dict[str, Any]],
    ) -> None:
        """Store a conversation record."""
        ...

    def iter_active_documents(
        self, batch_size: int = 500
    ) -> AsyncIterator[Document]:
        """Stream active documents in keyset batches.

        Keyset pagination (``WHERE id > last ORDER BY id LIMIT n``)
        avoids the growing-offset scans and unbounded lists of a full
        ``get_all_active`` on large corpora.
        """
        ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """Abstract interface for text embedding generation."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors (list of floats) of the same length.

        """
        ...

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for search queries.

        Models such as E5 require a ``query: `` prefix for queries.
        Default implementation delegates to ``embed`` for models that
        do not use prefixes.
        """
        return await self.embed(texts)

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for document passages.

        Models such as E5 require a ``passage: `` prefix for passages.
        Default implementation delegates to ``embed`` for models that
        do not use prefixes.
        """
        return await self.embed(texts)

    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        ...


@runtime_checkable
class FileStorage(Protocol):
    """Abstract interface for file storage (e.g., MinIO, local FS)."""

    async def upload_file(self, local_path: Path, destination_key: str) -> str:
        """Upload a file to storage.

        Args:
            local_path: Path to the local file.
            destination_key: Key (path) under which to store the file.

        Returns:
            The storage key of the uploaded file.

        """
        ...

    async def download_file(
        self, source_key: str, destination_path: Path
    ) -> None:
        """Download a file from storage to a local path.

        Args:
            source_key: Storage key of the file.
            destination_path: Local path to save the file.

        """
        ...

    async def delete_file(self, source_key: str) -> None:
        """Delete a file from storage."""
        ...


@runtime_checkable
class LLMGenerator(Protocol):
    """Abstract interface for a language model."""

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            system_prompt: System instruction for the model.
            user_prompt: User's query with context.
            temperature: Sampling temperature (0-1).

        Returns:
            The generated text.

        """
        ...


@runtime_checkable
class TokenValidator(Protocol):
    """Abstract interface for validating authentication tokens."""

    def validate(self, token: str) -> dict[str, str]:
        """Validate a JWT token and extract payload.

        Args:
            token: JWT string.

        Returns:
            dictionary with token claims (e.g., 'sub' for user_id).

        Raises:
            ValueError: If token is invalid or expired.

        """
        ...


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
        keyword_query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform similarity search.

        When ``keyword_query`` is provided, results are fused with a
        lexical (BM25) ranking over full-text candidates (hybrid search).

        Returns:
            list of hits, each containing 'id', 'score', 'payload',
            and optionally 'text'.

        """
        ...

    async def scroll_first_payload(self) -> dict[str, Any] | None:
        """Return the payload of an arbitrary stored point, or None.

        Used to detect which embedding model version the stored vectors
        were built with (model-version marker for migration reindexing).
        """
        ...

    async def delete_by_filter(self, filter_condition: dict[str, Any]) -> None:
        """Delete vectors matching a filter."""
        ...

    async def create_shadow_collection(self, dimension: int) -> str:
        """Create an offline green collection for blue-green reindex.

        After this call writes are dual-routed into the shadow while
        reads keep serving the active collection. Returns its name.
        """
        ...

    async def promote_shadow(self) -> None:
        """Atomically switch the live collection to the shadow one."""
        ...

    async def discard_shadow(self) -> None:
        """Drop the shadow collection without promoting it."""
        ...

    async def optimize_collection(self) -> None:
        """Run the store optimizer after bulk (re)indexing.


        Mass upserts leave unmerged segments; a full reindex should
        trigger HNSW rebuild so search latency/quality is restored.
        """
        ...

    async def delete_by_doc_id(self, doc_id: uuid.UUID) -> None:
        """Delete all vector chunks belonging to a document.

        Must be called before re-upserting a document: chunk ids are
        deterministic (uuid5), so a shortened document would otherwise
        leave stale "ghost" chunks behind.
        """
        ...

    async def get_collection_dimension(self) -> int | None:
        """Return the vector dimension of the collection, or None.

        ``None`` when the collection does not exist (fresh deploy).
        Used for the startup fail-fast check against the configured
        embedding model dimension.
        """
        ...


@runtime_checkable
class DocumentParser(Protocol):
    """Interface for parsing different document formats."""

    def parse(self, file_path: Path) -> list[dict[str, Any]]:
        """Parse a document and return a list of elements.

        Each element is a dict with 'text' and 'metadata' keys.
        """
        ...


@runtime_checkable
class ParserSelector(Protocol):
    """Interface for choosing a parser instance for a given file."""

    def get_parser(self, file_path: Path) -> DocumentParser:
        """Return a parser implementation based on the file."""
        ...


@runtime_checkable
class QueryRewriter(Protocol):
    """Rewrites a user question into a search-friendly form.

    Implementations must degrade gracefully: on any failure they
    return the original query unchanged (search must never break
    because of an auxiliary LLM call).
    """

    async def rewrite(self, query: str, feedback: str | None = None) -> str:
        """Return the rewritten query (or the original one).

        ``feedback`` is the grader reason from the previous agentic
        round (agentic retrieval); rewriters may use it to rephrase.
        """
        ...


@runtime_checkable
class SemanticCache(Protocol):
    """Near-duplicate answer cache keyed by query embedding.

    Lookup returns a previously generated answer for a question
    whose embedding is semantically close to the current one.
    """

    async def lookup(self, vector: list[float]) -> str | None:
        """Best cached answer within the similarity threshold."""
        ...

    async def store(self, vector: list[float], answer: str) -> None:
        """Remember the answer for this query embedding."""
        ...


@runtime_checkable
class DistributedLock(Protocol):
    """Mutual exclusion across workers (e.g. Redis SET NX EX)."""

    async def acquire(self, name: str, ttl: float = 300.0) -> bool:
        """Try to lock ``name`` for ``ttl`` seconds; True when acquired."""
        ...

    async def release(self, name: str) -> None:
        """Release the lock (only if still owned by this instance)."""
        ...


# ---------------------------------------------------------------------------
# Agentic retrieval (Corrective RAG)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GradeVerdict:
    """Verdict of a DocumentGrader on the retrieved hits."""

    relevant: bool
    reason: str = ''
    score: float = 0.0


@runtime_checkable
class DocumentGrader(Protocol):
    """Agentic retrieval: grades retrieved hits against the query.

    Used by the corrective loop: a negative verdict with a reason triggers
    a corrective query rewrite and another search round.
    """

    async def grade(
        self, query: str, hits: list[dict[str, Any]]
    ) -> GradeVerdict:
        """Return whether the hits are good enough to answer."""
        ...


@runtime_checkable
class QueryPlanner(Protocol):
    """Agentic retrieval: splits a question into sub-questions.

    Sub-queries are searched in parallel and fused. Implementations must
    degrade gracefully (return at least the original question).
    """

    async def plan(self, query: str) -> list[str]:
        """Return sub-questions (at least one)."""
        ...


# ---------------------------------------------------------------------------
# GraphRAG
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedRelation:
    """One (subject; predicate; object) triple extracted from text."""

    subject: str
    predicate: str
    obj: str


@runtime_checkable
class EntityExtractor(Protocol):
    """GraphRAG: extracts relation triples from free text."""

    async def extract(self, text: str) -> list[ExtractedRelation]:
        """Extract triples; empty list when nothing is found."""
        ...


@runtime_checkable
class GraphStore(Protocol):
    """GraphRAG: entity/relation store supporting graph expansion."""

    async def add_relations(
        self,
        doc_id: uuid.UUID,
        relations: list[ExtractedRelation],
    ) -> None:
        """Store triples extracted from a document."""
        ...

    async def related_doc_ids(
        self,
        entity_names: list[str],
        max_hops: int = 1,
    ) -> list[uuid.UUID]:
        """Doc ids of documents mentioning neighbors of the entities."""
        ...
