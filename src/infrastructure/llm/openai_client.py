"""OpenAI chat-completions API client for LLM generation.

Works with any OpenAI-compatible chat completions endpoint: OpenAI,
vLLM, Ollama (``/v1``), Together, Groq, DeepSeek, etc. — the provider
is selected via ``base_url`` + ``model``.
"""

import httpx

from application.interfaces import LLMGenerator
from shared.caching import TTLCache
from shared.circuit_breaker import CircuitBreaker, CircuitOpenError
from shared.metrics import LLM_CACHE_TOTAL, LLM_GENERATION_SECONDS, LLM_TOKENS_TOTAL
from shared.tracing import get_tracer


class OpenAIChatClient(LLMGenerator):
    """Client for an OpenAI-compatible chat completion API.

    Supports an optional in-process TTL response cache keyed by
    ``(system_prompt, user_prompt, temperature)`` to cut cost and
    latency for repeated questions.

    An optional ``circuit_breaker`` guards the remote API: after a
    configured number of consecutive failures, calls fail fast with
    :class:`CircuitOpenError` (no network round-trip) until the reset
    timeout elapses.
    """

    error_prefix = "LLM API error"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int = 500,
        timeout: float = 30.0,
        http_transport: httpx.AsyncBaseTransport | None = None,
        cache: TTLCache | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._transport = http_transport
        self._cache = cache
        self._breaker = circuit_breaker

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }

        with get_tracer().start_as_current_span("llm.generate") as span:
            span.set_attribute("llm.provider", "openai-compatible")
            span.set_attribute("llm.model", self._model)
            span.set_attribute("llm.temperature", temperature)

            cache_key = None
            if self._cache is not None:
                cache_key = TTLCache.make_key(system_prompt, user_prompt, temperature)
                cached = self._cache.get(cache_key)
                if cached is not None:
                    span.set_attribute("llm.cache_hit", True)
                    LLM_CACHE_TOTAL.labels(result="hit").inc()
                    if self._breaker is not None:
                        self._breaker.record_success()
                    return cached
                LLM_CACHE_TOTAL.labels(result="miss").inc()

            if self._breaker is not None and not self._breaker.allow():
                span.set_attribute("llm.circuit", "open")
                raise CircuitOpenError(
                    "LLM API circuit is open; generation rejected"
                )

            with LLM_GENERATION_SECONDS.time():
                try:
                    async with httpx.AsyncClient(
                        timeout=self._timeout, transport=self._transport
                    ) as client:
                        response = await client.post(url, headers=headers, json=payload)
                        if response.status_code != 200:
                            raise RuntimeError(
                                f"{self.error_prefix}: {response.status_code} - {response.text}"
                            )
                        data = response.json()
                except Exception:
                    if self._breaker is not None:
                        self._breaker.record_failure()
                    raise
            if self._breaker is not None:
                self._breaker.record_success()

            usage = data.get("usage") or {}
            if usage.get("prompt_tokens"):
                LLM_TOKENS_TOTAL.labels(kind="prompt").inc(usage["prompt_tokens"])
            if usage.get("completion_tokens"):
                LLM_TOKENS_TOTAL.labels(kind="completion").inc(
                    usage["completion_tokens"]
                )
            span.set_attribute("llm.completion_tokens", usage.get("completion_tokens", 0))
            content = data["choices"][0]["message"]["content"]
            if self._cache is not None and cache_key is not None:
                self._cache.set(cache_key, content)
            return content