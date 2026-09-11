"""DeepSeek API client for LLM generation.

Thin specialization of the generic OpenAI-compatible
:class:`OpenAIChatClient` (DeepSeek exposes an OpenAI-style chat
completions API) pinned to the ``deepseek-chat`` model.
"""

import httpx

from infrastructure.llm.openai_client import OpenAIChatClient
from shared.caching import TTLCache
from shared.circuit_breaker import CircuitBreaker


class DeepSeekClient(OpenAIChatClient):
    """Client for the DeepSeek chat completion API.

    Kept for backward compatibility: existing call sites construct it
    with ``(api_key, base_url)`` and the provider default model.
    """

    error_prefix = "DeepSeek API error"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = 30.0,
        http_transport: httpx.AsyncBaseTransport | None = None,
        cache: TTLCache | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        model: str = "deepseek-chat",
        max_tokens: int = 500,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            http_transport=http_transport,
            cache=cache,
            circuit_breaker=circuit_breaker,
        )

