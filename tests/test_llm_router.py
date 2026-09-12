import sys

import pytest

sys.path.insert(0, "src")

from infrastructure.llm import LLMProviderSpec, LLMRouter  # noqa: E402

pytestmark = pytest.mark.llm


class StubClient:
    """LLMGenerator double bound to one model."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[dict] = []

    async def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        self.calls.append(
            {"system_prompt": system_prompt, "user_prompt": user_prompt, "temperature": temperature}
        )
        return f"answer[{self.model}]"


def recorder(bucket: list):
    def factory(model: str) -> StubClient:
        client = StubClient(model)
        bucket.append(client)
        return client

    return factory


def make_router(allowed_models: list[str] | None = None):
    ds, oa = [], []
    specs = {
        "deepseek": LLMProviderSpec(factory=recorder(ds), default_model="deepseek-chat"),
        "openai": LLMProviderSpec(factory=recorder(oa), default_model="gpt-4o-mini"),
    }
    router = LLMRouter(specs, "deepseek", allowed_models=allowed_models)
    return router, ds, oa


async def test_generate_uses_default_provider() -> None:
    router, ds, _ = make_router()

    answer = await router.generate("sys", "user", 0.2)

    assert answer == "answer[deepseek-chat]"
    assert len(ds) == 1 and ds[0].model == "deepseek-chat"
    assert ds[0].calls[0]["system_prompt"] == "sys"
    assert ds[0].calls[0]["temperature"] == 0.2


async def test_generate_with_selects_requested_provider() -> None:
    router, ds, oa = make_router()

    answer = await router.generate_with("openai", None, "sys", "user")

    assert answer == "answer[gpt-4o-mini]"
    assert ds == [] and len(oa) == 1 and oa[0].model == "gpt-4o-mini"


async def test_generate_with_none_provider_falls_back_to_default() -> None:
    router, ds, oa = make_router()

    answer = await router.generate_with(None, None, "sys", "user")

    assert answer == "answer[deepseek-chat]"
    assert oa == []


async def test_model_override_creates_and_caches_client() -> None:
    router, _, oa = make_router()

    await router.generate_with("openai", "gpt-4o", "sys", "q1")
    await router.generate_with("openai", "gpt-4o", "sys", "q2")
    await router.generate_with("openai", "gpt-4o-mini", "sys", "q3")

    # one client per unique (provider, model); the gpt-4o client is cached
    assert [c.model for c in oa] == ["gpt-4o", "gpt-4o-mini"]
    assert [c.calls[0]["user_prompt"] for c in oa] == ["q1", "q3"]
    assert oa[0].calls[1]["user_prompt"] == "q2"


async def test_unknown_provider_rejected() -> None:
    router, _, _ = make_router()

    with pytest.raises(ValueError, match="not enabled"):
        await router.generate_with("claude", None, "sys", "user")


async def test_model_outside_allowlist_rejected() -> None:
    router, _, _ = make_router(allowed_models=["gpt-4o-mini", "gpt-4o"])

    with pytest.raises(ValueError, match="not allowed"):
        await router.generate_with("openai", "o1-preview", "sys", "user")


async def test_no_allowlist_permits_any_model() -> None:
    router, _, _ = make_router(allowed_models=None)

    answer = await router.generate_with("openai", "arbitrary-model", "sys", "user")

    assert answer == "answer[arbitrary-model]"


async def test_default_provider_must_be_among_enabled() -> None:
    specs = {
        "deepseek": LLMProviderSpec(factory=recorder([]), default_model="deepseek-chat"),
    }
    with pytest.raises(ValueError, match="not among"):
        LLMRouter(specs, "openai")


async def test_enabled_providers_property() -> None:
    router, _, _ = make_router()
    assert router.enabled_providers == ["deepseek", "openai"]
