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


def _make_model(**kwargs) -> SentenceTransformerEmbedding:
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.0, 1.0]])
    fake_model.get_sentence_embedding_dimension.return_value = 2
    with patch(
        "infrastructure.embedding.sentence_transformer.SentenceTransformer",
        return_value=fake_model,
    ):
        return SentenceTransformerEmbedding("fake-model", **kwargs)


async def test_embed_query_applies_e5_prefix():
    model = _make_model(query_prefix="query: ")
    result = await model.embed_query(["what is rag?"])
    assert result == [[0.0, 1.0]]


async def test_embed_passages_applies_e5_prefix():
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.0, 1.0]])
    fake_model.get_sentence_embedding_dimension.return_value = 2
    with patch(
        "infrastructure.embedding.sentence_transformer.SentenceTransformer",
        return_value=fake_model,
    ):
        model = SentenceTransformerEmbedding(
            "fake-model", passage_prefix="passage: "
        )

    await model.embed_passages(["some text"])
    assert fake_model.encode.call_args.args == (["passage: some text"],)


async def test_no_prefix_by_default():
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.0, 1.0]])
    fake_model.get_sentence_embedding_dimension.return_value = 2
    with patch(
        "infrastructure.embedding.sentence_transformer.SentenceTransformer",
        return_value=fake_model,
    ):
        model = SentenceTransformerEmbedding("fake-model")

    await model.embed_query(["q"])
    assert fake_model.encode.call_args.args == (["q"],)


def _make_cached_model(cache) -> tuple[SentenceTransformerEmbedding, MagicMock]:
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.0, 1.0]])
    fake_model.get_sentence_embedding_dimension.return_value = 2
    with patch(
        "infrastructure.embedding.sentence_transformer.SentenceTransformer",
        return_value=fake_model,
    ):
        return SentenceTransformerEmbedding("fake-model", query_cache=cache), fake_model


async def test_query_cache_prevents_reencode():
    from shared.caching import TTLCache

    model, fake_model = _make_cached_model(TTLCache(maxsize=8, ttl=60))

    first = await model.embed_query(["same question"])
    second = await model.embed_query(["same question"])

    assert first == second == [[0.0, 1.0]]
    assert fake_model.encode.call_count == 1  # second call served from cache


async def test_query_cache_partial_batch():
    from shared.caching import TTLCache

    model, fake_model = _make_cached_model(TTLCache(maxsize=8, ttl=60))
    await model.embed_query(["cached"])

    result = await model.embed_query(["cached", "fresh"])

    assert len(result) == 2
    # only the missing text is encoded
    assert fake_model.encode.call_args.args == (["fresh"],)


async def test_batch_size_passed_to_encode():
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.0, 1.0]])
    fake_model.get_sentence_embedding_dimension.return_value = 2
    with patch(
        "infrastructure.embedding.sentence_transformer.SentenceTransformer",
        return_value=fake_model,
    ):
        model = SentenceTransformerEmbedding("fake-model", batch_size=8)

    await model.embed(["a", "b"])
    assert fake_model.encode.call_args.kwargs["batch_size"] == 8


async def test_device_passed_to_model_constructor():
    with patch(
        "infrastructure.embedding.sentence_transformer.SentenceTransformer"
    ) as ctor:
        ctor.return_value.get_sentence_embedding_dimension.return_value = 2
        SentenceTransformerEmbedding("fake-model", device="cuda")

    assert ctor.call_args.kwargs.get("device") == "cuda"