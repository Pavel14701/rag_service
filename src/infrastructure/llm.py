"""LLM infrastructure: provider clients and per-request routing.

All HTTP chat providers share one request pipeline (cache lookup ->
circuit breaker -> pooled HTTP POST with continuation for truncated
answers -> usage accounting) implemented once in
:class:`_PooledChatClient`. Concrete providers only adapt the wire
format:

- ``OpenAIChatClient`` — any OpenAI-compatible chat completions
  endpoint (OpenAI, vLLM, Ollama, Together, Groq, DeepSeek, ...) via
  ``base_url`` + ``model``;
- ``AnthropicClient`` — Anthropic Messages API;
- ``DeepSeekClient`` — thin specialization of the OpenAI-compatible
  client pinned to ``deepseek-chat``;
- ``LLMRouter`` — routes per-request provider/model choices among the
  *enabled* providers only (allowlist — tenants cannot reach
  unconfigured providers).

``LLMQueryRewriter`` / ``NoOpQueryRewriter`` implement the
``QueryRewriter`` port for optional question rewriting before search.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from application.interfaces import LLMGenerator
from infrastructure.caching import Cache, TTLCache
from infrastructure.observability import (
    LLM_CACHE_TOTAL,
    LLM_GENERATION_SECONDS,
    LLM_TOKENS_TOTAL,
    LLM_TRUNCATIONS_TOTAL,
    get_tracer,
)
from infrastructure.resilience import CircuitBreaker, CircuitOpenError

_CONTINUATION_PROMPT = (
    'Continue exactly where you left off. Do not repeat any '
    'previously generated text.'
)
_MAX_ROUNDS = 3  # initial request + up to 2 continuations


class _PooledChatClient(LLMGenerator):
    """Shared request pipeline for HTTP chat providers.

    Owns everything generic: the pooled HTTP client, TTL answer cache,
    circuit breaker, the continuation loop for answers cut by the
    ``max_tokens`` limit (truncated answers are never cached) and usage
    accounting. Wire-format details are delegated to the subclasses via
    the ``_*_`` adapter methods.
    """

    provider = 'openai-compatible'
    error_prefix = 'LLM API error'
    url_path = '/chat/completions'

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int = 500,
        timeout: float = 30.0,
        http_transport: httpx.AsyncBaseTransport | None = None,
        cache: Cache | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        continue_on_truncation: bool = False,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip('/')
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._transport = http_transport
        self._cache = cache
        self._breaker = circuit_breaker
        self._continue_on_truncation = continue_on_truncation
        # Shared pooled client (None = per-call client, tests/CLI).
        self._http_client = http_client

    # ----- wire-format adapters (override in subclasses) -----

    def _headers(self) -> dict[str, str]:
        """Auth/content headers for the request."""
        raise NotImplementedError

    def _initial_messages(
        self, system_prompt: str, user_prompt: str
    ) -> list[dict[str, str]]:
        """Build the provider message list for a fresh request."""
        raise NotImplementedError

    def _request_payload(
        self,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> dict[str, Any]:
        """Build the request body for the given messages."""
        raise NotImplementedError

    def _stop_reason(self, data: dict[str, Any]) -> str | None:
        """Provider-specific truncation marker from the response."""
        raise NotImplementedError

    def _extract_text(self, data: dict[str, Any]) -> str:
        """Extract the generated text from a response."""
        raise NotImplementedError

    def _record_usage(
        self,
        data: dict[str, Any],
        span: Any,
    ) -> None:
        """Feed provider token usage into metrics and the span."""
        raise NotImplementedError

    # ----- shared pipeline -----

    async def _post(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> httpx.Response:
        """Send via the shared pooled client (fallback: per-call one)."""
        if self._http_client is not None:
            return await self._http_client.post(
                url, headers=headers, json=payload
            )
        async with httpx.AsyncClient(
            timeout=self._timeout, transport=self._transport
        ) as client:
            return await client.post(url, headers=headers, json=payload)

    @staticmethod
    def _continuation_messages(
        messages: list[dict[str, str]], content: str
    ) -> list[dict[str, str]]:
        """Messages for a follow-up request after truncation."""
        return messages + [
            {'role': 'assistant', 'content': content},
            {'role': 'user', 'content': _CONTINUATION_PROMPT},
        ]

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """Generate a completion, with caching and circuit breaking."""
        headers = self._headers()
        messages = self._initial_messages(system_prompt, user_prompt)

        with get_tracer().start_as_current_span('llm.generate') as span:
            span.set_attribute('llm.provider', self.provider)
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
                    content = ''
                    stop_reason = None
                    for _ in range(_MAX_ROUNDS):
                        response = await self._post(
                            self._url,
                            headers,
                            self._request_payload(messages, temperature),
                        )
                        if response.status_code != 200:
                            raise RuntimeError(
                                f'{self.error_prefix}: '
                                f'{response.status_code} - {response.text}'
                            )
                        data = response.json()
                        stop_reason = self._stop_reason(data)
                        part = self._extract_text(data)
                        content = f'{content}{part}' if content else part
                        if not self._is_truncated(stop_reason):
                            break
                        LLM_TRUNCATIONS_TOTAL.labels(
                            provider=self.provider
                        ).inc()
                        span.set_attribute('llm.truncated', True)
                        if not self._continue_on_truncation:
                            break
                        messages = self._continuation_messages(
                            messages, content
                        )
                except Exception:
                    if self._breaker is not None:
                        self._breaker.record_failure()
                    raise
            if self._breaker is not None:
                self._breaker.record_success()

            self._record_usage(data, span)
            if self._is_truncated(stop_reason):
                # Truncated answers must never poison the answer cache.
                span.set_attribute('llm.truncated_final', True)
            elif self._cache is not None and cache_key is not None:
                self._cache.set(cache_key, content)
            return content

    @property
    def _url(self) -> str:
        """Full endpoint URL."""
        return f'{self._base_url}{self.url_path}'

    def _is_truncated(self, stop_reason: str | None) -> bool:
        """Whether the provider cut the answer by the token limit."""
        raise NotImplementedError


class OpenAIChatClient(_PooledChatClient):
    """Client for an OpenAI-compatible chat completion API.

    Works with any endpoint speaking the chat completions protocol;
    the provider is selected via ``base_url`` + ``model``.
    """

    provider = 'openai-compatible'
    error_prefix = 'LLM API error'
    url_path = '/chat/completions'

    def _headers(self) -> dict[str, str]:
        return {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json',
        }

    def _initial_messages(
        self, system_prompt: str, user_prompt: str
    ) -> list[dict[str, str]]:
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]

    def _request_payload(
        self,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> dict[str, Any]:
        return {
            'model': self._model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': self._max_tokens,
        }

    def _stop_reason(self, data: dict[str, Any]) -> str | None:
        reason: str | None = data['choices'][0].get('finish_reason')
        return reason

    def _is_truncated(self, stop_reason: str | None) -> bool:
        return stop_reason == 'length'

    def _extract_text(self, data: dict[str, Any]) -> str:
        return str(data['choices'][0]['message']['content'] or '')

    def _record_usage(self, data: dict[str, Any], span: Any) -> None:
        usage = data.get('usage') or {}
        if usage.get('prompt_tokens'):
            LLM_TOKENS_TOTAL.labels(kind='prompt').inc(usage['prompt_tokens'])
        if usage.get('completion_tokens'):
            LLM_TOKENS_TOTAL.labels(kind='completion').inc(
                usage['completion_tokens']
            )
        span.set_attribute(
            'llm.completion_tokens', usage.get('completion_tokens', 0)
        )


class AnthropicClient(_PooledChatClient):
    """Client for the Anthropic Messages API (``POST /v1/messages``).

    Authentication uses the ``x-api-key`` header (not ``Authorization``)
    and requires the ``anthropic-version`` header. The system prompt is
    a top-level request field, not a message.
    """

    provider = 'anthropic'
    error_prefix = 'Anthropic API error'
    url_path = '/v1/messages'

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
        continue_on_truncation: bool = False,
        http_client: httpx.AsyncClient | None = None,
        api_version: str = '2023-06-01',
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
            continue_on_truncation=continue_on_truncation,
            http_client=http_client,
        )
        self._api_version = api_version
        self._system_prompt = ''

    def _headers(self) -> dict[str, str]:
        return {
            'x-api-key': self._api_key,
            'anthropic-version': self._api_version,
            'Content-Type': 'application/json',
        }

    def _initial_messages(
        self, system_prompt: str, user_prompt: str
    ) -> list[dict[str, str]]:
        # the system prompt is a top-level request field, not a message
        self._system_prompt = system_prompt
        return [{'role': 'user', 'content': user_prompt}]

    def _request_payload(
        self,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> dict[str, Any]:
        return {
            'model': self._model,
            'max_tokens': self._max_tokens,
            'temperature': temperature,
            'system': self._system_prompt,
            'messages': messages,
        }

    def _stop_reason(self, data: dict[str, Any]) -> str | None:
        return data.get('stop_reason')

    def _is_truncated(self, stop_reason: str | None) -> bool:
        return stop_reason == 'max_tokens'

    def _extract_text(self, data: dict[str, Any]) -> str:
        # content is a list of typed blocks; concatenate the text ones
        return ''.join(
            block.get('text', '')
            for block in data.get('content', [])
            if block.get('type') == 'text'
        )

    def _record_usage(self, data: dict[str, Any], span: Any) -> None:
        usage = data.get('usage') or {}
        # Anthropic calls them input/output tokens; map to shared metrics.
        if usage.get('input_tokens'):
            LLM_TOKENS_TOTAL.labels(kind='prompt').inc(usage['input_tokens'])
        if usage.get('output_tokens'):
            LLM_TOKENS_TOTAL.labels(kind='completion').inc(
                usage['output_tokens']
            )
        span.set_attribute(
            'llm.completion_tokens', usage.get('output_tokens', 0)
        )


class DeepSeekClient(OpenAIChatClient):
    """Client for the DeepSeek chat completion API.

    Kept for backward compatibility: existing call sites construct it
    with ``(api_key, base_url)`` and the provider default model.
    """

    error_prefix = 'DeepSeek API error'

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = 30.0,
        http_transport: httpx.AsyncBaseTransport | None = None,
        cache: Cache | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        continue_on_truncation: bool = False,
        http_client: httpx.AsyncClient | None = None,
        model: str = 'deepseek-chat',
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
            continue_on_truncation=continue_on_truncation,
            http_client=http_client,
        )


# Builds a concrete client bound to a specific model name.
ProviderFactory = Callable[[str], LLMGenerator]


@dataclass(frozen=True)
class LLMProviderSpec:
    """Construction recipe for one provider.

    ``factory`` creates a client for a given model; ``default_model`` is
    used when the request does not explicitly ask for another model.
    """

    factory: ProviderFactory
    default_model: str


class LLMRouter:
    """LLMGenerator that also routes per-request provider/model choices.

    Implements the ``LLMGenerator`` port: plain ``generate`` goes to the
    default provider (configured by ``LLM_PROVIDER``). ``generate_with``
    selects among the *enabled* providers only (allowlist) and may
    override the model, restricted by ``allowed_models`` when set.
    """

    def __init__(
        self,
        providers: dict[str, LLMProviderSpec],
        default_provider: str,
        allowed_models: list[str] | None = None,
    ) -> None:
        if default_provider not in providers:
            raise ValueError(
                f'Default LLM provider {default_provider!r} is not among '
                f'enabled providers: {sorted(providers)}'
            )
        self._providers = dict(providers)
        self._default = default_provider
        self._allowed_models = set(allowed_models) if allowed_models else None
        self._instances: dict[tuple[str, str], LLMGenerator] = {}
        self._lock = threading.Lock()

    @property
    def enabled_providers(self) -> list[str]:
        """Sorted names of providers available for per-request routing."""
        return sorted(self._providers)

    @property
    def default_client(self) -> LLMGenerator:
        """Client for the default provider and its default model."""
        return self._client(
            self._default, self._providers[self._default].default_model
        )

    def _client(self, provider: str, model: str) -> LLMGenerator:
        key = (provider, model)
        with self._lock:
            client = self._instances.get(key)
            if client is None:
                client = self._providers[provider].factory(model)
                self._instances[key] = client
            return client

    async def generate(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.1
    ) -> str:
        """Default generation path (default provider, default model)."""
        client = self._client(
            self._default, self._providers[self._default].default_model
        )
        return await client.generate(system_prompt, user_prompt, temperature)

    async def generate_with(
        self,
        provider: str | None,
        model: str | None,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """Generate via an explicitly requested provider/model.

        Raises ``ValueError`` for providers outside the enabled set or
        models outside ``allowed_models``.
        """
        provider = provider or self._default
        if provider not in self._providers:
            raise ValueError(
                f'LLM provider {provider!r} is not enabled '
                f'(enabled: {", ".join(self.enabled_providers)})'
            )
        if model is None:
            spec = self._providers[provider]
            model = spec.default_model
        elif (
            self._allowed_models is not None
            and model not in self._allowed_models
        ):
            raise ValueError(
                f'LLM model {model!r} is not allowed '
                f'(allowed: {sorted(self._allowed_models)})'
            )
        client = self._client(provider, model)
        return await client.generate(system_prompt, user_prompt, temperature)


class LLMQueryRewriter:
    """Rewrites user questions into search-friendly form via the LLM.

    Degrades gracefully: any LLM failure returns the original query, so
    search availability never depends on this auxiliary call.
    """

    _SYSTEM_PROMPT = (
        'You rewrite user questions for a document search engine. '
        'Expand abbreviations, fix typos, resolve vague references, '
        'make the question self-contained. Reply with the rewritten '
        'question ONLY - no explanations and no quotes.'
    )

    def __init__(self, llm: LLMGenerator, temperature: float = 0.0) -> None:
        self._llm = llm
        self._temperature = temperature

    async def rewrite(self, query: str, feedback: str | None = None) -> str:
        """Return the rewritten query, or the original on any failure.

        ``feedback`` (agentic retrieval) is appended to the prompt so the
        corrective round rephrases away from the failed attempt.
        """
        user_prompt = query
        if feedback:
            user_prompt = (
                f'{query}\n\n(A previous search with a similar query found '
                f'irrelevant results: {feedback}. Rephrase differently.)'
            )
        try:
            rewritten = (
                await self._llm.generate(
                    self._SYSTEM_PROMPT, user_prompt, self._temperature
                )
            ).strip()
        except Exception:  # noqa: BLE001 - search must not depend on it
            return query
        return rewritten or query


class NoOpQueryRewriter:
    """Pass-through rewriter used when rewriting is disabled."""

    async def rewrite(self, query: str, feedback: str | None = None) -> str:
        """Return the query unchanged."""
        return query
