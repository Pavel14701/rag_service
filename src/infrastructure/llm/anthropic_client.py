"""Anthropic Messages API client for LLM generation."""

import httpx

from application.interfaces import LLMGenerator
from shared.caching import Cache, TTLCache
from shared.circuit_breaker import CircuitBreaker, CircuitOpenError
from shared.metrics import (
    LLM_CACHE_TOTAL,
    LLM_GENERATION_SECONDS,
    LLM_TOKENS_TOTAL,
)
from shared.tracing import get_tracer


class AnthropicClient(LLMGenerator):
    """Client for the Anthropic Messages API (``POST /v1/messages``).

    Implements the same :class:`LLMGenerator` contract as the
    OpenAI-compatible clients, including the optional TTL response
    cache and the circuit breaker guarding the remote API.

    Authentication uses the ``x-api-key`` header (not ``Authorization``)
    and requires the ``anthropic-version`` header.
    """

    error_prefix = 'Anthropic API error'

    def __init__(
        self,
        api_key: str,
        base_url: str = 'https://api.anthropic.com',
        model: str = 'claude-3-5-haiku-latest',
        max_tokens: int = 500,
        timeout: float = 30.0,
        http_transport: httpx.AsyncBaseTransport | None = None,
        cache: Cache | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        api_version: str = '2023-06-01',
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip('/')
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._transport = http_transport
        self._cache = cache
        self._breaker = circuit_breaker
        self._api_version = api_version

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """Generate a completion, with caching and circuit breaking."""
        url = f'{self._base_url}/v1/messages'
        headers = {
            'x-api-key': self._api_key,
            'anthropic-version': self._api_version,
            'Content-Type': 'application/json',
        }
        payload = {
            'model': self._model,
            'max_tokens': self._max_tokens,
            'temperature': temperature,
            'system': system_prompt,
            'messages': [{'role': 'user', 'content': user_prompt}],
        }

        with get_tracer().start_as_current_span('llm.generate') as span:
            span.set_attribute('llm.provider', 'anthropic')
            span.set_attribute('llm.model', self._model)
            span.set_attribute('llm.temperature', temperature)

            cache_key = None
            if self._cache is not None:
                cache_key = TTLCache.make_key(
                    system_prompt, user_prompt, temperature
                )
                cached = self._cache.get(cache_key)
                if cached is not None:
                    span.set_attribute('llm.cache_hit', True)
                    LLM_CACHE_TOTAL.labels(result='hit').inc()
                    if self._breaker is not None:
                        self._breaker.record_success()
                    return str(cached)
                LLM_CACHE_TOTAL.labels(result='miss').inc()

            if self._breaker is not None and not self._breaker.allow():
                span.set_attribute('llm.circuit', 'open')
                raise CircuitOpenError(
                    'LLM API circuit is open; generation rejected'
                )

            with LLM_GENERATION_SECONDS.time():
                try:
                    async with httpx.AsyncClient(
                        timeout=self._timeout, transport=self._transport
                    ) as client:
                        response = await client.post(
                            url, headers=headers, json=payload
                        )
                        if response.status_code != 200:
                            raise RuntimeError(
                                f'{self.error_prefix}: '
                                f'{response.status_code} - {response.text}'
                            )
                        data = response.json()
                except Exception:
                    if self._breaker is not None:
                        self._breaker.record_failure()
                    raise
            if self._breaker is not None:
                self._breaker.record_success()

            usage = data.get('usage') or {}
            # Anthropic calls them input/output tokens;
            # map to the shared metrics.
            if usage.get('input_tokens'):
                LLM_TOKENS_TOTAL.labels(kind='prompt').inc(
                    usage['input_tokens']
                )
            if usage.get('output_tokens'):
                LLM_TOKENS_TOTAL.labels(kind='completion').inc(
                    usage['output_tokens']
                )
            span.set_attribute(
                'llm.completion_tokens', usage.get('output_tokens', 0)
            )

            # The response content is a list of typed blocks;
            # concatenate text ones.
            content = ''.join(
                block.get('text', '')
                for block in data.get('content', [])
                if block.get('type') == 'text'
            )
            if self._cache is not None and cache_key is not None:
                self._cache.set(cache_key, content)
            return content
