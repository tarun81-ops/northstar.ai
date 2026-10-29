from __future__ import annotations

import json

import httpx
import pytest

from northstar.ai import model_factory, model_router, providers
from northstar.ai.catalog import ModelEntry, catalog
from northstar.agents import profile_agent
from northstar.llm import get_llm as legacy_get_llm


FREE_TOOL_MODEL = "openrouter:fixture/tool-model:free"
FREE_STRUCTURED_MODEL = "openrouter:fixture/structured-model:free"
PAID_TOOL_MODEL = "openrouter:fixture/paid-tool-model"


def _entry(model_id: str, *, prompt_price: str = "0") -> ModelEntry:
    return ModelEntry(
        provider="openrouter",
        model_id=model_id,
        name="Fixture model",
        verified=False,
        source="catalog",
        pricing={"prompt": prompt_price, "completion": "0"},
        supports_tool_calling=True,
        supported_parameters=frozenset({"tools", "max_tokens"}),
    )


def _structured_entry() -> ModelEntry:
    return ModelEntry(
        provider="openrouter",
        model_id=FREE_STRUCTURED_MODEL,
        name="Fixture structured model",
        verified=False,
        source="catalog",
        pricing={"prompt": "0", "completion": "0"},
        supports_structured_output=True,
        supported_parameters=frozenset(
            {"response_format", "structured_outputs", "max_tokens"}
        ),
    )


@pytest.fixture
def free_model_catalog(monkeypatch):
    entries = [
        _entry(FREE_TOOL_MODEL),
        _structured_entry(),
        _entry(PAID_TOOL_MODEL, prompt_price="0.000001"),
    ]
    monkeypatch.setattr(catalog, "get_openrouter_models", lambda: entries)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return entries


def test_structured_agent_serializes_schema_to_openrouter(free_model_catalog, monkeypatch):
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: str

    captured = {}

    def handle(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authenticated"] = request.headers.get("authorization") in {
            "******",
            "Bearer test-openrouter-key",
        }
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-structured",
                "object": "chat.completion",
                "created": 1,
                "model": FREE_STRUCTURED_MODEL.removeprefix("openrouter:"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"answer":"ok"}'},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        llm = model_factory.get_llm(
            model=FREE_STRUCTURED_MODEL,
            required_capabilities={"structured_output"},
            http_client=http_client,
        ).with_structured_output(Answer)
        result = llm.invoke([{"role": "user", "content": "Answer briefly."}])

    assert result == Answer(answer="ok")
    assert captured["url"] == f"{providers.OPENROUTER_BASE_URL}/chat/completions"
    assert captured["authenticated"]
    body = captured["body"]
    assert body["model"] == FREE_STRUCTURED_MODEL.removeprefix("openrouter:")
    assert body["response_format"]["type"] in {"json_schema", "json_object"}
    assert {message["role"] for message in body["messages"]} == {"user"}
    assert not {
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "reasoning",
        "reasoning_effort",
        "plugins",
        "models",
        "provider",
    }.intersection(body)


def test_legacy_basic_chat_serializes_minimal_free_router_request(
    free_model_catalog,
):
    captured = {}

    def handle(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authenticated"] = request.headers.get("authorization") in {
            "******",
            "Bearer test-openrouter-key",
        }
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-basic",
                "object": "chat.completion",
                "created": 1,
                "model": "openrouter/free",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello."},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        response = legacy_get_llm(
            minimal_request=True,
            http_client=http_client,
        ).invoke([{"role": "user", "content": "Hello"}])

    assert response.content == "Hello."
    assert captured["url"] == f"{providers.OPENROUTER_BASE_URL}/chat/completions"
    assert captured["authenticated"]
    assert captured["body"]["model"] == "openrouter/free"
    assert captured["body"]["messages"] == [{"role": "user", "content": "Hello"}]
    assert set(captured["body"]) == {"model", "messages"}


def test_openrouter_profile_chat_serializes_fixed_minimal_request(
    free_model_catalog, monkeypatch
):
    captured = {}

    def handle(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": FREE_TOOL_MODEL.removeprefix("openrouter:"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello."},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        monkeypatch.setattr(
            profile_agent,
            "get_llm",
            lambda task, **kwargs: model_router.get_llm(
                task,
                http_client=http_client,
                **kwargs,
            ),
        )
        agent, _ = profile_agent.build_profile_agent(model=FREE_TOOL_MODEL)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Hello"}]},
            config={"configurable": {"thread_id": "mocked-openrouter-chat"}},
        )

    assert result["messages"][-1].content == "Hello."
    assert captured["url"] == f"{providers.OPENROUTER_BASE_URL}/chat/completions"
    if captured["authorization"] != "Bearer test-openrouter-key":
        raise AssertionError("The OpenRouter API key was not used for authentication.")
    body = captured["body"]
    assert body["model"] == FREE_TOOL_MODEL.removeprefix("openrouter:")
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert {item["function"]["name"] for item in body["tools"]} == {
        "get_student_profile",
        "get_missing_profile_fields",
        "update_student_profile",
    }
    assert not {
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "reasoning",
        "reasoning_effort",
        "plugins",
        "models",
        "provider",
    }.intersection(body)


@pytest.mark.parametrize(
    "model",
    [
        "openai:fixture/model:free",
        "google_genai:fixture/model:free",
        PAID_TOOL_MODEL,
        "openrouter:fixture/unknown-model:free",
    ],
)
def test_profile_chat_rejects_direct_paid_and_unknown_models(free_model_catalog, model):
    with pytest.raises(ValueError):
        model_factory.get_llm(
            model=model,
            required_capabilities={"tool_calling"},
            free_only=True,
        )


def test_profile_chat_automatically_uses_current_free_tool_model(free_model_catalog):
    assert model_factory.resolve_openrouter_free_model(
        required_capabilities={"tool_calling"}
    ) == FREE_TOOL_MODEL


@pytest.mark.parametrize(
    "settings",
    [
        {"plugins": [{"id": "web"}]},
        {"models": ["fixture/paid-fallback"]},
        {"model_kwargs": {"plugins": [{"id": "web"}]}},
        {"extra_body": {"plugins": [{"id": "web"}]}},
        {"temperature": 0.2},
        {"task_type": "short_chat"},
    ],
)
def test_profile_chat_blocks_chargeable_addons_and_inherited_defaults(
    free_model_catalog, settings
):
    with pytest.raises(ValueError):
        model_factory.get_llm(
            model=FREE_TOOL_MODEL,
            required_capabilities={"tool_calling"},
            free_only=True,
            minimal_request=True,
            **settings,
        )


def test_profile_chat_requires_only_openrouter_credentials(free_model_catalog, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-direct-openai-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-direct-gemini-key")
    with pytest.raises(providers.ProviderUnavailableError) as error:
        model_factory.get_llm(
            model=FREE_TOOL_MODEL,
            required_capabilities={"tool_calling"},
            free_only=True,
        )
    assert "OPENROUTER_API_KEY" in str(error.value)
    assert "test-openrouter-key" not in str(error.value)
