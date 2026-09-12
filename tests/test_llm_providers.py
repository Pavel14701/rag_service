"""Tests for the multi-provider LLM clients and provider selection."""

import json

import httpx
import pytest

from config import Settings
from infrastructure.llm import OpenAIChatClient
from infrastructure.llm import AnthropicClient
from infrastructure.llm import DeepSeekClient

pytestmark = pytest.mark.llm


def _ok_openai_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "the answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def _ok_anthropic_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "content": [
                {"type": "text", "text": "the "},
                {"type": "text", "text": "answer"},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )


async def test_openai_client_sends_model_and_bearer_auth():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.content)
        return _ok_openai_response()

    client = OpenAIChatClient(
        "sk-test",
        "http://test/v1",
        model="gpt-4o-mini",
        max_tokens=300,
        http_transport=httpx.MockTransport(handler),
    )

    result = await client.generate("sys", "user", temperature=0.2)

    assert result == "the answer"
    assert captured["url"] == "http://test/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer sk-test"
    assert captured["payload"]["model"] == "gpt-4o-mini"
    assert captured["payload"]["max_tokens"] == 300
    assert captured["payload"]["temperature"] == 0.2
    assert captured["payload"]["messages"][0] == {"role": "system", "content": "sys"}


async def test_openai_client_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    client = OpenAIChatClient(
        "sk-test",
        "http://test",
        model="m",
        http_transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RuntimeError, match="LLM API error: 429"):
        await client.generate("s", "u")


async def test_anthropic_client_sends_messages_api_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.content)
        return _ok_anthropic_response()

    client = AnthropicClient(
        "ak-test",
        "http://test",
        model="claude-3-5-haiku-latest",
        http_transport=httpx.MockTransport(handler),
    )

    result = await client.generate("sys", "user", temperature=0.4)

    # Text blocks are concatenated
    assert result == "the answer"
    assert captured["url"] == "http://test/v1/messages"
    assert captured["headers"]["x-api-key"] == "ak-test"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    # Anthropic uses "system" as a top-level field, not a message role
    assert captured["payload"]["system"] == "sys"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "user"}]
    assert captured["payload"]["model"] == "claude-3-5-haiku-latest"
    assert captured["payload"]["temperature"] == 0.4


async def test_anthropic_client_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key")

    client = AnthropicClient(
        "ak-test",
        "http://test",
        http_transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RuntimeError, match="Anthropic API error: 401"):
        await client.generate("s", "u")


async def test_deepseek_client_still_sends_deepseek_chat_model():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _ok_openai_response()

    client = DeepSeekClient(
        "key",
        "http://test",
        http_transport=httpx.MockTransport(handler),
    )
    assert isinstance(client, OpenAIChatClient)  # unified implementation
    await client.generate("s", "u")
    assert captured["payload"]["model"] == "deepseek-chat"


# ---------- provider selection (config + container) ----------


def test_settings_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="llm_provider"):
        Settings(llm_provider="gemini", jwt_secret="s")


def test_settings_normalizes_openai_compatible_alias() -> None:
    settings = Settings(llm_provider="OpenAI-Compatible", jwt_secret="s")
    assert settings.llm_provider == "openai"


def _container_env(monkeypatch, **extra) -> None:
    for key, value in {"JWT_SECRET": "test-secret", **extra}.items():
        monkeypatch.setenv(key, value)


async def test_container_selects_openai_provider(monkeypatch) -> None:
    from container import create_container
    from application.interfaces import LLMGenerator

    _container_env(
        monkeypatch,
        LLM_PROVIDER="openai",
        OPENAI_API_KEY="sk-test",
        LLM_MODEL="gpt-4o-mini",
    )
    client = (await create_container().get(LLMGenerator)).default_client
    assert type(client) is OpenAIChatClient
    assert client._model == "gpt-4o-mini"


async def test_container_selects_anthropic_provider(monkeypatch) -> None:
    from container import create_container
    from application.interfaces import LLMGenerator

    _container_env(monkeypatch, LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY="ak-test")
    client = (await create_container().get(LLMGenerator)).default_client
    assert type(client) is AnthropicClient
    assert client._model == "claude-3-5-haiku-latest"


async def test_container_defaults_to_deepseek(monkeypatch) -> None:
    from container import create_container
    from application.interfaces import LLMGenerator

    _container_env(monkeypatch, DEEPSEEK_API_KEY="dk-test")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    client = (await create_container().get(LLMGenerator)).default_client
    assert type(client) is DeepSeekClient


async def test_container_missing_provider_key_raises(monkeypatch) -> None:
    from container import create_container
    from application.interfaces import LLMGenerator

    _container_env(monkeypatch, LLM_PROVIDER="openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        await create_container().get(LLMGenerator)
