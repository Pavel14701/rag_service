"""Tests for SentenceTransformerEmbedding (model is mocked)."""

from unittest.mock import MagicMock, patch

import numpy as np

from infrastructure.embedding.sentence_transformer import SentenceTransformerEmbedding


async def test_embed_returns_lists_and_calls_encode_with_kwargs():
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

    with patch(
        "infrastructure.embedding.sentence_transformer.SentenceTransformer",
        return_value=fake_model,
    ):
        model = SentenceTransformerEmbedding("fake-model")

    result = await model.embed(["a", "b"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    kwargs = fake_model.encode.call_args.kwargs
    assert kwargs["convert_to_numpy"] is True
    assert kwargs["show_progress_bar"] is False
    assert fake_model.encode.call_args.args == (["a", "b"],)


def test_dimension_from_model():
    fake_model = MagicMock()
    fake_model.get_sentence_embedding_dimension.return_value = 384

    with patch(
        "infrastructure.embedding.sentence_transformer.SentenceTransformer",
        return_value=fake_model,
    ):
        model = SentenceTransformerEmbedding("fake-model")

    assert model.dimension() == 384