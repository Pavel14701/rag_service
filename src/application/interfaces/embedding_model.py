"""Interface for embedding generation."""

from typing import Protocol, runtime_checkable


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
