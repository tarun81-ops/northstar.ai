from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage

from northstar.ai import chat_recovery
from northstar.ai.catalog import ModelEntry, catalog
from northstar.ai.providers import ProviderUnavailableError

FREE_MODEL = "openrouter:fixture/model-a:free"
OTHER_FREE_MODEL = "openrouter:fixture/model-b:free"
PAID_MODEL = "openrouter:fixture/model-paid"


class ProviderError(Exception):
    def __init__(self, status_code: int, message: str, retry_after: str = "10"):
        self.status_code = status_code
        self.response = type("Response", (), {"headers": {"Retry-After": retry_after}})()
        super().__init__(message)


def _entry(model: str, pricing: dict | None = None) -> ModelEntry:
    return ModelEntry(
        provider="openrouter",
        model_id=model,
        name=model,
        verified=False,
        source="catalog",
        pricing=pricing or {"prompt": "0", "completion": "0"},
    )


def _patch_catalog(monkeypatch, *models: str) -> None:
    entries = {model: _entry(model) for model in models}
    entries[PAID_MODEL] = _entry(
        PAID_MODEL,
        {"prompt": "0.01", "completion": "0.02"},
    )
    monkeypatch.setattr(catalog, "find", lambda model: entries.get(model))


def test_fallback_is_disabled_by_default(monkeypatch):
    _patch_catalog(monkeypatch, FREE_MODEL, OTHER_FREE_MODEL)
    calls = []

    def invoke(model: str, resume: bool):
        calls.append((model, resume))
        raise ProviderError(429, "rate limited")

    with pytest.raises(chat_recovery.ChatRecoveryError) as error:
        chat_recovery.invoke_with_fallback(
            invoke,
            FREE_MODEL,
            [OTHER_FREE_MODEL],
        )

    assert error.value.category == "rate_limit"
    assert calls == [(FREE_MODEL, False)]


def test_rate_limit_can_resume_on_an_explicit_current_free_model(monkeypatch, caplog):
    _patch_catalog(monkeypatch, FREE_MODEL, OTHER_FREE_MODEL)
    calls = []
    cooldowns = {}

    def invoke(model: str, resume: bool):
        calls.append((model, resume))
        if model == FREE_MODEL:
            raise ProviderError(429, "rate limited secret=must-not-be-logged")
        return {
            "messages": [
                AIMessage(
                    "ok",
                    usage_metadata={
                        "input_tokens": 4,
                        "output_tokens": 5,
                        "total_tokens": 9,
                    },
                )
            ]
        }

    with caplog.at_level(logging.INFO):
        result = chat_recovery.invoke_with_fallback(
            invoke,
            FREE_MODEL,
            [OTHER_FREE_MODEL],
            enabled=True,
            cooldowns=cooldowns,
            now=100,
        )

    assert calls == [(FREE_MODEL, False), (OTHER_FREE_MODEL, True)]
    assert result.actual_model == OTHER_FREE_MODEL
    assert result.attempts == 2
    assert result.fallback_reason == "rate_limit"
    assert cooldowns[FREE_MODEL] == 110
    assert "must-not-be-logged" not in caplog.text


def test_paid_or_unknown_fallback_is_skipped(monkeypatch):
    _patch_catalog(monkeypatch, FREE_MODEL)
    calls = []

    def invoke(model: str, resume: bool):
        calls.append(model)
        raise ProviderError(429, "rate limited")

    with pytest.raises(chat_recovery.ChatRecoveryError):
        chat_recovery.invoke_with_fallback(
            invoke,
            FREE_MODEL,
            [PAID_MODEL, "openrouter:fixture/unknown:free"],
            enabled=True,
        )

    assert calls == [FREE_MODEL]


@pytest.mark.parametrize(
    "error, expected",
    [
        (ProviderError(429, "daily quota exhausted"), "daily_quota"),
        (ProviderError(401, "invalid key"), "credentials"),
        (ProviderError(503, "provider down"), "outage"),
        (ValueError("unsupported temperature"), "configuration"),
    ],
)
def test_failure_categories(error, expected):
    assert chat_recovery.classify_error(error) == expected


def test_account_quota_exhaustion_does_not_try_another_model(monkeypatch):
    _patch_catalog(monkeypatch, FREE_MODEL, OTHER_FREE_MODEL)
    calls = []

    def invoke(model: str, resume: bool):
        calls.append(model)
        raise ProviderError(429, "daily quota exhausted")

    with pytest.raises(chat_recovery.ChatRecoveryError) as error:
        chat_recovery.invoke_with_fallback(
            invoke,
            FREE_MODEL,
            [OTHER_FREE_MODEL],
            enabled=True,
        )
    assert error.value.category == "daily_quota"
    assert calls == [FREE_MODEL]


def test_direct_provider_selected_model_is_unreachable(monkeypatch):
    calls = []
    monkeypatch.setattr(catalog, "find", lambda model: None)

    with pytest.raises(chat_recovery.ChatRecoveryError) as error:
        chat_recovery.invoke_with_fallback(
            lambda model, resume: calls.append(model),
            "openai:fixture/model",
            ["google_genai:fixture/model"],
            enabled=True,
        )
    assert error.value.category == "configuration"
    assert calls == []


def test_recovery_cannot_be_configured_to_allow_paid_routes(monkeypatch):
    _patch_catalog(monkeypatch, FREE_MODEL)
    with pytest.raises(ValueError, match="only permits validated"):
        chat_recovery.invoke_with_fallback(
            lambda model, resume: None,
            FREE_MODEL,
            zero_priced_only=False,
        )


def test_missing_openrouter_key_is_sanitized(monkeypatch):
    _patch_catalog(monkeypatch, FREE_MODEL)
    secret = "sensitive-provider-detail"

    def invoke(model: str, resume: bool):
        raise ProviderUnavailableError("openrouter", secret)

    with pytest.raises(chat_recovery.ChatRecoveryError) as error:
        chat_recovery.invoke_with_fallback(invoke, FREE_MODEL)

    assert error.value.category == "credentials"
    assert secret not in str(error.value)
    assert error.value.__cause__ is None


def test_model_cooldown_skips_fallback_until_expiry(monkeypatch):
    _patch_catalog(monkeypatch, FREE_MODEL, OTHER_FREE_MODEL)
    calls = []

    def invoke(model: str, resume: bool):
        calls.append(model)
        raise ProviderError(429, "rate limited")

    with pytest.raises(chat_recovery.ChatRecoveryError):
        chat_recovery.invoke_with_fallback(
            invoke,
            FREE_MODEL,
            [OTHER_FREE_MODEL],
            enabled=True,
            cooldowns={OTHER_FREE_MODEL: 200},
            now=100,
        )
    assert calls == [FREE_MODEL]


def test_total_attempt_time_budget_is_not_reset_between_models(monkeypatch):
    _patch_catalog(monkeypatch, FREE_MODEL, OTHER_FREE_MODEL)
    calls = []
    clock_values = iter([0.0, 0.0, 91.0])
    monkeypatch.setattr(chat_recovery.time, "monotonic", lambda: next(clock_values))

    def invoke(model: str, resume: bool):
        calls.append(model)
        raise ProviderError(429, "rate limited")

    with pytest.raises(chat_recovery.ChatRecoveryError):
        chat_recovery.invoke_with_fallback(
            invoke,
            FREE_MODEL,
            [OTHER_FREE_MODEL],
            enabled=True,
        )

    assert calls == [FREE_MODEL]
