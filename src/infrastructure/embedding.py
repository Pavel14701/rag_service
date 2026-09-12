"""Sentence-transformers embedding model."""

from typing import Any
import asyncio
from functools import partial

from sentence_transformers import SentenceTransformer

from application.interfaces import EmbeddingModel
from infrastructure.caching import Cache, TTLCache


class SentenceTransformerEmbedding(EmbeddingModel):
    """Wrapper for sentence-transformers models.

    Applies E5-style prefixes (``query: `` / ``passage: ``) when the
    model requires them. Prefixes are configurable so the model can be
    swapped via settings without code changes.

    Performance options:
    - ``device``: force "cpu"/"cuda" instead of auto-detection;
    - ``batch_size``: batching for the underlying encoder;
    - ``query_cache``: TTL cache for query embeddings (queries are
      pure functions of the text, unlike search results which depend
      on per-user access filters).
    """

    def __init__(
        self,
        model_name: str,
        query_prefix: str = '',
        passage_prefix: str = '',
        device: str | None = None,
        batch_size: int | None = None,
        query_cache: Cache | None = None,
    ) -> None:
        if device:
            self._model = SentenceTransformer(model_name, device=device)
        else:
            self._model = SentenceTransformer(model_name)
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._batch_size = batch_size
        self._query_cache = query_cache

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed raw texts without any prefix handling."""
        loop = asyncio.get_running_loop()
        kwargs: dict[str, Any] = {
            'convert_to_numpy': True,
            'show_progress_bar': False,
        }
        if self._batch_size is not None:
            kwargs['batch_size'] = self._batch_size
        encode = partial(self._model.encode, texts, **kwargs)
        embeddings = await loop.run_in_executor(None, encode)
        return [emb.tolist() for emb in embeddings]

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Embed query texts with the query prefix and caching."""
        if self._query_cache is None:
            return await self.embed(texts)

        keys = [TTLCache.make_key(self._query_prefix, t) for t in texts]
        if missing := [
            (index, text, key)
            for index, (text, key) in enumerate(zip(texts, keys))
            if self._query_cache.get(key) is None
        ]:
            computed = await self.embed([text for _, text, _ in missing])
            for (_, _, key), vector in zip(missing, computed):
                self._query_cache.set(key, vector)
        results: list[list[float] | None] = [
            self._query_cache.get(key) for key in keys
        ]
        return [vector for vector in results if vector is not None]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed passage texts with the passage prefix."""
        return await self.embed([f'{self._passage_prefix}{t}' for t in texts])

    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        dimension = self._model.get_sentence_embedding_dimension()
        return int(dimension) if dimension is not None else 0
