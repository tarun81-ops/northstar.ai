"""Shared free-only OpenRouter factory for NorthStar.

All agents route through here.  Responsibilities:

* Validate a current free OpenRouter catalog entry or the free router.
* Build the LangChain OpenAI-compatible client pinned to OpenRouter.
* Apply task defaults only when the model catalog advertises the parameters.
* Validate parameters against model metadata so unsupported settings
  surface as errors instead of being silently dropped.
* Log only provider/model, task, attempts, latency, token usage, and
  error/fallback category -- never secrets or prompt content.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from northstar.ai import llm_config
from northstar.ai import providers
from northstar.ai.model_router import selected_model_for_session

logger = logging.getLogger(__name__)
_MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}\Z")

# Models that the app knows do not support streaming.  Empty until a real
# integration is confirmed; stream capability is validated at build time.
_NO_STREAMING_MODELS: frozenset[str] = frozenset()


class LLMError(Exception):
    """Base error for the shared factory."""


class ModelUnavailableError(LLMError):
    """Raised when the selected model/provider cannot be used right now."""

    def __init__(self, provider: str, model: str, reason: str, category: str):
        self.provider = provider
        self.model = model
        self.category = category  # "credentials" | "rate_limit" | "outage" | "config"
        super().__init__(f"{provider}:{model} unavailable ({category}): {reason}")


class OpenRouterModelSelectionError(ValueError):
    """Raised when a current free OpenRouter model cannot be selected safely."""


def resolve_openrouter_free_model(
    model: str | None = None,
    *,
    required_capabilities: set[str] | None = None,
) -> str:
    """Resolve or validate a current zero-priced OpenRouter catalog entry."""
    from northstar.ai.catalog import catalog

    capabilities = required_capabilities or set()
    free_router = "openrouter:openrouter/free"
    if model is not None:
        model = llm_config.provider_prefix(model)
        if providers.provider_for_model(model) != "openrouter":
            raise OpenRouterModelSelectionError("This chat only accepts OpenRouter model IDs.")
        if model == free_router:
            if capabilities:
                raise OpenRouterModelSelectionError(
                    "The OpenRouter free router has no verified metadata for the required capabilities."
                )
            return model
        entry = catalog.find(model)
        if entry is None:
            raise OpenRouterModelSelectionError(
                "The selected OpenRouter model has no current catalog metadata."
            )
        if not model.split(":", 1)[1].endswith(":free") or not entry.zero_priced:
            raise OpenRouterModelSelectionError(
                "This chat only accepts current zero-priced OpenRouter :free models."
            )
        unsupported = {capability for capability in capabilities if not entry.supports(capability)}
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise OpenRouterModelSelectionError(
                f"The selected free OpenRouter model does not support: {names}."
            )
        return model

    if not capabilities:
        return free_router

    for entry in catalog.get_provider_models("openrouter"):
        model_id = entry.model_id
        if (
            model_id.split(":", 1)[-1].endswith(":free")
            and entry.zero_priced
            and all(entry.supports(capability) for capability in capabilities)
        ):
            return model_id
    needed = ", ".join(sorted(capabilities))
    qualifier = f" with {needed}" if needed else ""
    raise OpenRouterModelSelectionError(
        f"No current zero-priced OpenRouter :free model is listed{qualifier}."
    )


def _log_llm_call(
    *,
    provider: str,
    model: str,
    task_type: str | None,
    attempts: int,
    latency_ms: float | None = None,
    tokens: int | None = None,
    category: str | None = None,
    error: str | None = None,
) -> None:
    """Emit a safe log line for an LLM call -- no secrets, no prompt content."""
    parts = [
        f"provider={provider}",
        f"model={model}",
        f"task={task_type or '-'}",
        f"attempts={attempts}",
    ]
    if latency_ms is not None:
        parts.append(f"latency_ms={latency_ms:.0f}")
    if tokens is not None:
        parts.append(f"tokens={tokens}")
    if category:
        parts.append(f"category={category}")
    if error:
        parts.append(f"error={error}")
    logger.info("llm_call " + " ".join(parts))


def get_llm(
    model: str | None = None,
    *,
    task_type: str | None = None,
    streaming: bool = False,
    required_capabilities: set[str] | None = None,
    free_only: bool = True,
    minimal_request: bool = False,
    **kwargs: Any,
):
    """Build a chat client for a selected or catalog-compatible free model.

    Resolution order:
    1. Validate ``model`` when given -- user selection wins.
    2. Otherwise use the session selection or find a free model satisfying
       ``required_capabilities``. Basic chat uses the OpenRouter free router.

    Per-task defaults are added only when catalog metadata advertises them.
    Unsupported parameters raise :class:`ValueError`.

    Raises
    ------
    ProviderUnavailableError: the OpenRouter API key is missing.
    ValueError: an invalid or unsupported parameter was requested.
    """
    start = time.perf_counter()

    if not free_only:
        raise OpenRouterModelSelectionError("NorthStar only permits free OpenRouter models.")
    blocked_request_options = {
        "plugins",
        "models",
        "route",
        "provider",
        "web_search",
        "web_search_options",
        "extra_body",
        "model_kwargs",
        "default_query",
        "api_key",
        "openai_api_key",
        "base_url",
        "openai_api_base",
        "default_headers",
        "organization",
        "openai_organization",
    }
    if blocked_request_options.intersection(kwargs):
        names = ", ".join(sorted(blocked_request_options.intersection(kwargs)))
        raise OpenRouterModelSelectionError(
            f"Chargeable or alternate OpenRouter request options are disabled: {names}."
        )

    if minimal_request:
        if task_type is not None:
            raise ValueError("Minimal OpenRouter requests do not use task generation defaults.")
        if streaming:
            raise ValueError("Minimal OpenRouter requests use non-streaming requests.")
        unsupported = set(kwargs) - {"http_client"}
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(
                f"Minimal OpenRouter requests do not accept optional request settings: {names}."
            )

    # Resolve the provider:model identifier.
    if model is None:
        # A session-scoped choice overrides task routing, never a global value.
        model = selected_model_for_session()
        if model is not None:
            normalized = llm_config.provider_prefix(model)
            if providers.provider_for_model(normalized) != "openrouter":
                raise OpenRouterModelSelectionError(
                    "The saved model selection is not an OpenRouter model. Choose a free OpenRouter model."
                )
            else:
                model = normalized
    if not providers.has_api_key("openrouter"):
        raise providers.ProviderUnavailableError(
            "openrouter",
            "no API key configured (set OPENROUTER_API_KEY or add it to Streamlit secrets)",
        )
    model = resolve_openrouter_free_model(
        model,
        required_capabilities=required_capabilities,
    )

    provider = providers.provider_for_model(model)
    bare_model = providers.model_without_prefix(model)
    is_free_router = model == "openrouter:openrouter/free"
    if not _MODEL_ID_PATTERN.fullmatch(bare_model):
        raise ValueError("The selected model ID has an invalid format.")
    providers.get_provider_config(provider)

    # Credential check: missing key disables only this provider.
    if not providers.has_api_key(provider):
        latency_ms = (time.perf_counter() - start) * 1000
        _log_llm_call(
            provider=provider,
            model=model,
            task_type=task_type,
            attempts=1,
            latency_ms=latency_ms,
            category="credentials",
            error="missing_api_key",
        )
        raise providers.ProviderUnavailableError(
            provider,
            f"no API key configured for provider '{provider}'",
        )

    entry = None
    if required_capabilities or provider == "openrouter":
        from northstar.ai.catalog import catalog

        entry = catalog.find(model)
    if required_capabilities:
        unknown = set(required_capabilities)
        if entry is not None:
            unknown = {
                capability
                for capability in required_capabilities
                if not entry.supports(capability)
            }
        if entry is None or unknown:
            needed = ", ".join(sorted(unknown or required_capabilities))
            raise ValueError(
                f"Model '{model}' has no verified compatibility metadata for "
                f"required capability/capabilities: {needed}. Choose a listed "
                "compatible model or configure a verified model allowlist."
            )

    model_parameters = entry.supported_parameters if entry is not None else None
    if provider == "openrouter" and model_parameters is None and not is_free_router:
        raise ValueError(
            f"OpenRouter has no current capability metadata for model '{model}'. "
            "Refresh the catalog or choose a listed compatible model."
        )

    # Merge task defaults with caller overrides, applying model-specific
    # parameters only when the catalog advertises them.
    merged: dict[str, Any] = {"model": model}
    token_param = llm_config.max_tokens_param_for_provider(provider)
    if task_type is not None and model_parameters is not None:
        settings = llm_config.task_settings(task_type)
        supports_temperature = llm_config.model_supports_temperature(model)
        supports_temperature = supports_temperature and "temperature" in model_parameters
        if supports_temperature:
            merged["temperature"] = settings.temperature
        if "max_tokens" in model_parameters:
            merged[token_param] = settings.max_output_tokens

    token_aliases = ("max_tokens", "max_output_tokens", "max_completion_tokens")
    explicit_token_limit = next(
        (kwargs[key] for key in (token_param, *token_aliases) if key in kwargs),
        None,
    )
    for key in token_aliases:
        kwargs.pop(key, None)
    if explicit_token_limit is not None:
        merged.pop(token_param, None)
        kwargs[token_param] = explicit_token_limit
    merged.update(kwargs)

    if model_parameters is not None:
        model_options = {
            "temperature",
            "top_p",
            "reasoning_effort",
            "reasoning",
        }
        unsupported = {
            key
            for key in kwargs
            if (
                key in model_options and key not in model_parameters
            )
            or (
                key in {"max_tokens", "max_output_tokens", "max_completion_tokens"}
                and "max_tokens" not in model_parameters
            )
        }
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"Model '{model}' does not advertise support for: {names}.")

    # Validate parameters -- unsupported settings surface as errors.
    try:
        clean = llm_config.sanitize_llm_kwargs(
            model,
            merged,
            supported_params=providers.supported_constructor_parameters(provider),
        )
    except ValueError:
        latency_ms = (time.perf_counter() - start) * 1000
        _log_llm_call(
            provider=provider,
            model=model,
            task_type=task_type,
            attempts=1,
            latency_ms=latency_ms,
            category="config",
            error="unsupported_parameter",
        )
        raise

    # Streaming validation: don't stream models known not to support it.
    if streaming and model in _NO_STREAMING_MODELS:
        latency_ms = (time.perf_counter() - start) * 1000
        _log_llm_call(
            provider=provider,
            model=model,
            task_type=task_type,
            attempts=1,
            latency_ms=latency_ms,
            category="config",
            error="streaming_not_supported",
        )
        raise ValueError(f"streaming is not supported for model '{model}'")

    # Build the client.  ``model`` is passed as a named parameter, so strip it
    # from the sanitized kwargs to avoid passing it twice.
    clean.pop("model", None)
    try:
        client = providers.build_client(provider, model, streaming=streaming, **clean)
    except providers.ProviderUnavailableError:
        latency_ms = (time.perf_counter() - start) * 1000
        _log_llm_call(
            provider=provider,
            model=model,
            task_type=task_type,
            attempts=1,
            latency_ms=latency_ms,
            category="credentials",
            error="missing_api_key",
        )
        raise
    except Exception as exc:  # pragma: no cover - client construction failure
        latency_ms = (time.perf_counter() - start) * 1000
        _log_llm_call(
            provider=provider,
            model=model,
            task_type=task_type,
            attempts=1,
            latency_ms=latency_ms,
            category="outage",
            error=type(exc).__name__,
        )
        raise ModelUnavailableError(
            provider,
            model,
            f"client construction failed ({type(exc).__name__})",
            "outage",
        ) from exc

    latency_ms = (time.perf_counter() - start) * 1000
    _log_llm_call(
        provider=provider,
        model=model,
        task_type=task_type,
        attempts=1,
        latency_ms=latency_ms,
    )
    return client
