from __future__ import annotations

from dataclasses import dataclass

import pytest

from northstar.ai import model_factory, model_router, providers
from northstar.ai.catalog import ModelEntry, catalog

FREE_MODEL = "openrouter:fixture/agent-model:free"


@dataclass
class CapturedCall:
    provider: str
    model: str
    temperature: float | None
    max_tokens: int | None
    streaming: bool
    extra: dict


class FakeBuildClient:
    def __init__(self):
        self.calls: list[CapturedCall] = []

    def __call__(self, provider, model, *, streaming=False, **kwargs):
        self.calls.append(
            CapturedCall(
                provider=provider,
                model=model,
                temperature=kwargs.get("temperature"),
                max_tokens=kwargs.get("max_tokens"),
                streaming=streaming,
                extra={
                    key: value
                    for key, value in kwargs.items()
                    if key not in {"temperature", "max_tokens"}
                },
            )
        )
        return self


@pytest.fixture
def free_model_factory(monkeypatch):
    fake = FakeBuildClient()
    monkeypatch.setattr(providers, "build_client", fake)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "direct-openai-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "direct-gemini-key")
    entry = ModelEntry(
        provider="openrouter",
        model_id=FREE_MODEL,
        name="Fixture agent model",
        verified=False,
        source="catalog",
        pricing={"prompt": "0", "completion": "0"},
        supports_tool_calling=True,
        supports_structured_output=True,
        supported_parameters=frozenset(
            {"tools", "response_format", "structured_outputs", "max_tokens", "temperature"}
        ),
    )
    monkeypatch.setattr(catalog, "get_provider_models", lambda provider: [entry] if provider == "openrouter" else [])
    monkeypatch.setattr(model_factory, "selected_model_for_session", lambda: None)
    return fake


@pytest.mark.parametrize(
    "task_type",
    [
        "planning",
        "research_analysis",
        "notebook_generation",
        "final_explanation",
        "complex_reasoning",
        "intent_classification",
        "summarization",
        "field_extraction",
        "formatting",
        "short_chat",
        "hard_reasoning",
    ],
)
def test_all_agent_tasks_use_the_free_openrouter_factory(free_model_factory, task_type):
    model_router.get_llm(task_type, required_capabilities={"structured_output"})
    call = free_model_factory.calls[-1]
    assert call.provider == "openrouter"
    assert call.model == FREE_MODEL


def test_supported_agent_task_defaults_are_sent(free_model_factory):
    model_router.get_llm("planning", required_capabilities={"structured_output"})
    call = free_model_factory.calls[-1]
    assert call.temperature == 0.3
    assert call.max_tokens == 1500


def test_openrouter_free_router_is_default_for_basic_legacy_chat(free_model_factory):
    model_router.get_llm("short_chat")
    assert free_model_factory.calls[-1].model == "openrouter:openrouter/free"


@pytest.mark.parametrize(
    "model",
    [
        "openai:fixture/model:free",
        "google_genai:fixture/model:free",
        "openrouter:fixture/paid-model",
        "openrouter:fixture/unknown-model:free",
    ],
)
def test_factory_rejects_direct_paid_or_unknown_models(free_model_factory, model):
    with pytest.raises(ValueError):
        model_router.get_llm(
            "planning",
            model=model,
            required_capabilities={"structured_output"},
        )
    assert not free_model_factory.calls


def test_factory_cannot_be_unlocked_to_paid_or_direct_routes(free_model_factory):
    with pytest.raises(model_factory.OpenRouterModelSelectionError):
        model_factory.get_llm(model=FREE_MODEL, free_only=False)
    with pytest.raises(model_factory.OpenRouterModelSelectionError):
        model_router.get_llm("planning", free_only=False)
    assert not free_model_factory.calls


def test_saved_direct_model_is_rejected_not_used_as_a_route(free_model_factory, monkeypatch):
    monkeypatch.setattr(model_factory, "selected_model_for_session", lambda: "openai:model")
    with pytest.raises(model_factory.OpenRouterModelSelectionError):
        model_router.get_llm("planning", required_capabilities={"structured_output"})
    assert not free_model_factory.calls


def test_missing_openrouter_key_does_not_fall_back_to_other_provider_keys(
    free_model_factory, monkeypatch
):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(providers.ProviderUnavailableError) as error:
        model_router.get_llm("planning", required_capabilities={"structured_output"})
    assert error.value.provider == "openrouter"
    assert not free_model_factory.calls
