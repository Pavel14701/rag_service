"""Per-request LLM routing across multiple configured providers."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from application.interfaces import LLMGenerator

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
    selects among the *enabled* providers only (allowlist — tenants cannot
    reach unconfigured providers) and may override the model, restricted
    by ``allowed_models`` when set (cost control).
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
        spec = self._providers[provider]
        if model is None:
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
