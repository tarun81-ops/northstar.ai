"""OpenRouter-only credentials and client construction."""

from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

try:
    import streamlit as st

    _HAS_STREAMLIT = True
except ImportError:  # Streamlit is optional for non-UI consumers.
    _HAS_STREAMLIT = False


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ProviderUnavailableError(Exception):
    """Raised when OpenRouter is unavailable or a non-OpenRouter route is requested."""

    def __init__(self, provider: str, reason: str):
        self.provider = provider
        self.reason = reason
        super().__init__(f"Provider '{provider}' is unavailable: {reason}")


@dataclass(frozen=True)
class ProviderConfig:
    provider: str = "openrouter"
    env_var: str = "OPENROUTER_API_KEY"
    st_secret_key: str = "openrouter"
    base_url: str = OPENROUTER_BASE_URL
    client_kind: str = "openai"
    max_tokens_param: str = "max_tokens"


_OPENROUTER = ProviderConfig()


def _openrouter_chat_client_class():
    from langchain_openai import ChatOpenAI

    class OpenRouterChatOpenAI(ChatOpenAI):
        @property
        def _default_params(self) -> dict[str, Any]:
            params = super()._default_params
            if not self.streaming:
                params.pop("stream", None)
            return params

        def _get_request_payload(self, input_, *, stop=None, **kwargs: Any) -> dict:
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            if "response_format" in payload:
                # LangChain's structured-output path removes this before sending.
                payload["stream"] = False
            return payload

    return OpenRouterChatOpenAI


def get_provider_config(provider: str) -> ProviderConfig:
    if provider != "openrouter":
        raise ProviderUnavailableError(
            provider,
            "NorthStar only supports the OpenRouter endpoint.",
        )
    return _OPENROUTER


def list_providers() -> list[str]:
    return ["openrouter"]


def provider_for_model(model: str) -> str:
    return model.split(":", 1)[0] if ":" in model else "openrouter"


def model_without_prefix(model: str) -> str:
    return model.split(":", 1)[1] if ":" in model else model


def get_api_key(provider: str = "openrouter") -> str | None:
    cfg = get_provider_config(provider)
    value = os.getenv(cfg.env_var, "").strip()
    if value:
        return value
    if _HAS_STREAMLIT:
        try:
            secret = st.secrets.get(cfg.st_secret_key, "") or st.secrets.get(cfg.env_var, "")
            return str(secret).strip() or None
        except Exception:
            return None
    return None


def has_api_key(provider: str = "openrouter") -> bool:
    return get_api_key(provider) is not None


def build_client(
    provider: str,
    model: str,
    *,
    streaming: bool = False,
    **kwargs,
):
    """Construct a client pinned to the OpenRouter API and key."""
    cfg = get_provider_config(provider)
    locked_overrides = {
        "api_key",
        "openai_api_key",
        "base_url",
        "openai_api_base",
        "default_headers",
        "organization",
        "openai_organization",
    }
    attempted_overrides = locked_overrides.intersection(kwargs)
    if attempted_overrides:
        names = ", ".join(sorted(attempted_overrides))
        raise ValueError(
            f"OpenRouter connection settings are fixed and cannot be overridden: {names}."
        )
    api_key = get_api_key(provider)
    if not api_key:
        raise ProviderUnavailableError(
            provider,
            f"no API key configured (set {cfg.env_var} or add it to Streamlit secrets)",
        )

    client_kwargs = {
        "model": model_without_prefix(model),
        "api_key": api_key,
        "base_url": OPENROUTER_BASE_URL,
        "default_headers": {
            "HTTP-Referer": "https://northstar.ai",
            "X-Title": "NorthStar",
        },
        "max_retries": 0,
        "timeout": 30,
        "streaming": streaming,
    }

    model_kwargs = dict(kwargs.pop("model_kwargs", {}) or {})
    if "max_tokens" in kwargs:
        model_kwargs["max_tokens"] = kwargs.pop("max_tokens")
    if model_kwargs:
        client_kwargs["model_kwargs"] = model_kwargs

    for key in (
        "model",
        "api_key",
        "base_url",
        "default_headers",
        "streaming",
        "timeout",
        "max_retries",
    ):
        kwargs.pop(key, None)
    client_kwargs.update(kwargs)
    logger.debug("building chat client provider=openrouter model=%s streaming=%s", model, streaming)
    return _openrouter_chat_client_class()(**client_kwargs)


def supported_constructor_parameters(provider: str) -> set[str]:
    get_provider_config(provider)
    from langchain_openai import ChatOpenAI

    locked = {
        "model",
        "api_key",
        "base_url",
        "default_headers",
        "streaming",
        "max_retries",
        "timeout",
    }
    return set(inspect.signature(ChatOpenAI).parameters) - locked | {"max_tokens"}
