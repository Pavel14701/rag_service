"""Sentence-transformers embedding model."""

from typing import List
import asyncio
from functools import partial

from sentence_transformers import SentenceTransformer

from application.interfaces import EmbeddingModel


class SentenceTransformerEmbedding(EmbeddingModel):
    """Wrapper for sentence-transformers models."""

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_running_loop()
        encode = partial(
            self._model.encode,
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embeddings = await loop.run_in_executor(None, encode)
        return [emb.tolist() for emb in embeddings]

    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()