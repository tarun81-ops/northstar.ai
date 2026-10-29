"""Bounded model recovery for a single checkpointed chat inference."""

from __future__ import annotations

import logging
import time
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from typing import Any, Callable

from northstar.ai.catalog import catalog
from northstar.ai.providers import provider_for_model

MAX_FALLBACK_ATTEMPTS = 2
MAX_ATTEMPTS = MAX_FALLBACK_ATTEMPTS + 1
MAX_TOTAL_ATTEMPT_SECONDS = 90
_TRANSIENT_CATEGORIES = {"rate_limit", "daily_quota", "outage"}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatRecoveryResult:
    value: Any
    actual_model: str
    attempts: int
    fallback_reason: str | None


class ChatRecoveryError(Exception):
    """Sanitized failure raised after one bounded inference recovery sequence."""

    def __init__(
        self,
        category: str,
        attempts: int,
        retry_after_seconds: int | None = None,
    ):
        self.category = category
        self.attempts = attempts
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Chat inference failed ({category}) after {attempts} attempt(s).")


def classify_error(error: Exception) -> str:
    """Classify provider failures without returning their potentially sensitive text."""
    name = type(error).__name__.casefold()
    response = getattr(error, "response", None)
    status = getattr(error, "status_code", None) or getattr(response, "status_code", None)
    detail = str(error).casefold()

    if any(token in name for token in ("safety", "refusal", "contentfilter")):
        return "safety"
    if status in {401, 403} or any(
        token in name
        for token in ("authentication", "permission", "providerunavailable")
    ):
        return "credentials"
    if status == 429 or "ratelimit" in name or "resourceexhausted" in name:
        if any(token in detail for token in ("quota", "credit", "daily limit", "exhausted")):
            return "daily_quota"
        return "rate_limit"
    if status == 400 or any(token in name for token in ("invalidargument", "badrequest")):
        return "configuration"
    if status is not None and status >= 500:
        return "outage"
    if any(token in name for token in ("timeout", "connection", "serviceunavailable", "internalserver")):
        return "outage"
    if isinstance(error, (TypeError, ValueError)):
        return "configuration"
    return "unknown"


def _retry_after_hint(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        return None
    try:
        delay = int(value)
    except (TypeError, ValueError):
        try:
            delay = int((parsedate_to_datetime(value).timestamp()) - time.time())
        except (TypeError, ValueError, OverflowError):
            return None
    return min(max(delay, 5), 3600)


def _retry_after_seconds(error: Exception) -> int:
    return _retry_after_hint(error) or (
        3600 if classify_error(error) == "daily_quota" else 60
    )


def _is_current_free_model(model: str) -> bool:
    if provider_for_model(model) != "openrouter":
        return False
    bare_model = model.split(":", 1)[1] if ":" in model else ""
    if not bare_model.endswith(":free"):
        return False
    entry = catalog.find(model)
    return entry is not None and entry.zero_priced


def _cooldown_key(model: str) -> str:
    return model


def invoke_with_fallback(
    invoke: Callable[[str, bool], Any],
    selected_model: str,
    approved_models: list[str] | tuple[str, ...] = (),
    *,
    enabled: bool = False,
    zero_priced_only: bool = True,
    cooldowns: dict[str, float] | None = None,
    now: float | None = None,
) -> ChatRecoveryResult:
    """Run one inference and, when enabled, try at most two free model alternatives.

    ``invoke(model, resume)`` must resume from the graph checkpoint after the
    first failure; it must not replay side-effecting nodes.
    """
    if not zero_priced_only:
        raise ValueError("Chat recovery only permits validated zero-priced OpenRouter models.")
    if not _is_current_free_model(selected_model):
        raise ChatRecoveryError("configuration", 0)

    cooldowns = cooldowns if cooldowns is not None else {}
    current_time = time.time() if now is None else now
    sequence_started = time.monotonic()
    attempts = 0
    fallback_reason = None
    last_error: Exception | None = None
    candidates = [selected_model]
    if enabled:
        seen = {selected_model}
        for model in approved_models:
            if model not in seen:
                candidates.append(model)
                seen.add(model)
            if len(candidates) >= MAX_ATTEMPTS:
                break

    for candidate in candidates:
        if time.monotonic() - sequence_started >= MAX_TOTAL_ATTEMPT_SECONDS:
            break
        if attempts:
            category = classify_error(last_error)
            if (
                not enabled
                or category not in _TRANSIENT_CATEGORIES
                or category == "daily_quota"
            ):
                break
            if not _is_current_free_model(candidate):
                continue
            entry = catalog.find(candidate)
            if entry is None or not entry.zero_priced:
                continue
            if cooldowns.get(_cooldown_key(candidate), 0) > current_time:
                continue
            fallback_reason = classify_error(last_error)

        attempts += 1
        attempt_started = time.perf_counter()
        try:
            value = invoke(candidate, attempts > 1)
            usage = None
            messages = value.get("messages", []) if isinstance(value, dict) else []
            if messages:
                usage = getattr(messages[-1], "usage_metadata", None)
            tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
            logger.info(
                "llm_inference provider=%s model=%s task=profile_chat attempts=%d "
                "fallback_reason=%s latency_ms=%.0f tokens=%s",
                provider_for_model(candidate),
                candidate,
                attempts,
                fallback_reason or "-",
                (time.perf_counter() - attempt_started) * 1000,
                tokens if tokens is not None else "-",
            )
            return ChatRecoveryResult(value, candidate, attempts, fallback_reason)
        except Exception as error:
            last_error = error
            category = classify_error(error)
            logger.info(
                "llm_inference provider=%s model=%s task=profile_chat attempts=%d "
                "error_category=%s latency_ms=%.0f",
                provider_for_model(candidate),
                candidate,
                attempts,
                category,
                (time.perf_counter() - attempt_started) * 1000,
            )
            if category in _TRANSIENT_CATEGORIES:
                cooldowns[_cooldown_key(candidate)] = (
                    current_time + _retry_after_seconds(error)
                )
            if category not in _TRANSIENT_CATEGORIES:
                break

    category = classify_error(last_error) if last_error else "unavailable"
    retry_after = _retry_after_hint(last_error) if last_error else None
    raise ChatRecoveryError(category, attempts, retry_after) from None
