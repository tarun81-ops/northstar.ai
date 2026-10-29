from __future__ import annotations

import pytest

from northstar.ai import llm_config


def test_task_settings_defaults_and_fallback():
    assert llm_config.task_settings("planning").temperature == 0.3
    assert llm_config.task_settings("planning").max_output_tokens == 1500
    assert llm_config.task_settings("short_chat").max_output_tokens == 400
    assert llm_config.task_settings("unknown") == llm_config.task_settings("short_chat")


def test_provider_prefix_defaults_to_openrouter_and_preserves_explicit_provider():
    assert llm_config.provider_prefix("vendor/model:free") == "openrouter:vendor/model:free"
    assert llm_config.provider_prefix("openrouter:vendor/model:free") == (
        "openrouter:vendor/model:free"
    )
    assert llm_config.provider_prefix("openai/vendor-model") == "openrouter:openai/vendor-model"
    assert llm_config.provider_prefix("openai:model") == "openai:model"


def test_openrouter_api_key_is_read_from_environment(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert llm_config.openrouter_api_key() is None

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    assert llm_config.openrouter_api_key() == "test-openrouter-key"


def test_sanitizer_normalizes_token_limit_alias_and_rejects_bad_params():
    model = "openrouter:vendor/model:free"
    kwargs = llm_config.sanitize_llm_kwargs(
        model,
        {"model": model, "max_output_tokens": 400},
        supported_params={"max_tokens", "temperature"},
    )
    assert kwargs == {"model": model, "max_tokens": 400}
    with pytest.raises(ValueError, match="Unsupported parameter"):
        llm_config.sanitize_llm_kwargs(
            model,
            {"model": model, "reasoning_effort": "high"},
            supported_params={"max_tokens", "temperature"},
        )


def test_explicit_sampling_options_are_rejected_when_metadata_omits_them():
    model = "openrouter:vendor/model:free"
    with pytest.raises(ValueError, match="Unsupported parameter"):
        llm_config.sanitize_llm_kwargs(
            model,
            {"model": model, "temperature": 0.2},
            supported_params={"max_tokens"},
        )
