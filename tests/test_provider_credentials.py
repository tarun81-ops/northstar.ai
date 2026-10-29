from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from northstar.ai import providers


def test_provider_registry_only_accepts_openrouter(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "direct-openai-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "direct-gemini-key")
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")

    assert providers.list_providers() == ["openrouter"]
    assert providers.get_api_key("openrouter") == "openrouter-test-key"
    for provider in ("openai", "google_genai"):
        with pytest.raises(providers.ProviderUnavailableError):
            providers.get_api_key(provider)


def test_openrouter_client_uses_only_openrouter_key_and_fixed_endpoint(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=FakeChatOpenAI),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "direct-openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-only-key")

    providers.build_client("openrouter", "openrouter:vendor/model:free", max_tokens=50)

    assert captured["api_key"] == "router-only-key"
    assert captured["base_url"] == providers.OPENROUTER_BASE_URL
    assert captured["model"] == "vendor/model:free"
    assert captured["model_kwargs"]["max_tokens"] == 50
    assert captured["max_retries"] == 0


@pytest.mark.parametrize(
    "override",
    [
        {"base_url": "https://attacker.example/v1"},
        {"openai_api_base": "https://attacker.example/v1"},
        {"api_key": "attacker-key"},
        {"openai_api_key": "attacker-key"},
    ],
)
def test_openrouter_client_rejects_endpoint_or_key_overrides(override, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-only-key")
    with pytest.raises(ValueError, match="connection settings are fixed"):
        providers.build_client("openrouter", "openrouter:vendor/model:free", **override)


@pytest.mark.parametrize("provider", ["openai", "google_genai"])
def test_direct_provider_client_routes_are_unreachable(provider, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "direct-openai-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "direct-gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

    with pytest.raises(providers.ProviderUnavailableError):
        providers.build_client(provider, f"{provider}:model")
