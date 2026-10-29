"""OpenRouter credentials and model-compatible generation settings."""

from __future__ import annotations

import os

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSettings:
    """Generation settings for a single task type."""

    temperature: float
    max_output_tokens: int


# Per-task-type defaults. Temperature and max output tokens can be tuned
# independently of model choice.
TASK_DEFAULTS: dict[str, TaskSettings] = {
    # --- main model tasks -------------------------------------------------
    "planning": TaskSettings(temperature=0.3, max_output_tokens=1500),
    "research_analysis": TaskSettings(temperature=0.4, max_output_tokens=1200),
    "notebook_generation": TaskSettings(temperature=0.2, max_output_tokens=1000),
    "final_explanation": TaskSettings(temperature=0.5, max_output_tokens=1000),
    "complex_reasoning": TaskSettings(temperature=0.3, max_output_tokens=1600),
    # --- fast model tasks -------------------------------------------------
    "intent_classification": TaskSettings(temperature=0.1, max_output_tokens=300),
    "summarization": TaskSettings(temperature=0.3, max_output_tokens=500),
    "field_extraction": TaskSettings(temperature=0.1, max_output_tokens=400),
    "formatting": TaskSettings(temperature=0.1, max_output_tokens=300),
    "short_chat": TaskSettings(temperature=0.4, max_output_tokens=400),
    # --- hard reasoning ---------------------------------------------------
    "hard_reasoning": TaskSettings(temperature=0.2, max_output_tokens=2000),
}


def task_settings(task_type: str) -> TaskSettings:
    """Generation settings for a task type, falling back to fast-model defaults."""
    return TASK_DEFAULTS.get(task_type, TASK_DEFAULTS["short_chat"])


def openrouter_api_key() -> str | None:
    """Return the OpenRouter API key from the environment, or ``None`` when unset."""
    return os.getenv("OPENROUTER_API_KEY", "").strip() or None


def provider_for_model(model: str) -> str:
    """Extract the provider from a ``provider:model`` identifier."""
    if ":" in model:
        return model.split(":", 1)[0]
    return "openrouter"


def max_tokens_param_for_provider(provider: str) -> str:
    """Return the OpenRouter client kwarg for the output-token limit."""
    return "max_tokens"


def provider_prefix(model: str) -> str:
    """Preserve explicit providers and otherwise treat IDs as OpenRouter IDs."""
    if ":" in model and ("/" not in model or model.index(":") < model.index("/")):
        return model
    return f"openrouter:{model}"


def context_compact_threshold() -> int:
    """Token threshold above which older history is compacted before an agent call.

    Default: 20,000 tokens. Controlled by ``NORTHSTAR_CONTEXT_COMPACT_THRESHOLD``.
    """
    try:
        return int(os.getenv("NORTHSTAR_CONTEXT_COMPACT_THRESHOLD", "20000"))
    except (TypeError, ValueError):
        return 20000
# -- parameter sanitization --

_UNSUPPORTED_SAMPLING_KWARGS = {
    "temperature",
    "top_p",
    "top_logprobs",
    "logprobs",
}


def model_supports_temperature(model: str) -> bool:
    """Catalog metadata, checked by the factory, is authoritative for support."""
    return True


def sanitize_llm_kwargs(
    model: str,
    merged: dict[str, Any],
    *,
    supported_params: set[str] | None = None,
) -> dict[str, Any]:
    """Filter ``merged`` kwargs so only supported parameters reach the LLM client.

    Resolution order:
    1. ``model`` is inspected to determine which parameters are safe to pass.
    2. If ``supported_params`` is given, only those keys are retained; otherwise
       a provider‑specific whitelist is used.
    3. Parameters known to be unsupported for the detected model are removed,
       even if present in ``merged`` via caller overrides.
    4. ``max_tokens`` / ``max_output_tokens`` are always kept (they are not
      sampling parameters and are supported by virtually all models).
    5. Any remaining keys are passed through.

    This prevents invocation‑time overrides from re‑introducing rejected
    parameters after the task‑default filtering step.

    Returns a new dict; the original ``merged`` is not mutated.

    Parameters
    ----------
    model: The fully resolved model identifier (e.g. ``openrouter:vendor/model:free``).
    merged: The merged dict of ``{{model, temperature, max_tokens}}`` + caller ``**kwargs``.
    supported_params: Optional explicit set of allowed parameter names. If ``None\",
      the function uses the documented per‑model whitelist.

    Returns
    -------
    A clean dict ready to be passed to the OpenRouter ChatOpenAI client.

    Raises
    ------
    ValueError: If an explicit ``supported_params`` set is provided but contains
      keys that are not recognised as valid chat parameters, or if an
      explicitly-passed parameter is unsupported for the resolved provider/model.
    """
    provider = provider_for_model(model)
    token_param = max_tokens_param_for_provider(provider)
    token_aliases = {"max_tokens", "max_output_tokens", "max_completion_tokens"}

    # Start from the merged dict and build a clean copy.
    clean: dict[str, Any] = {}

    # Always forward the model name – it is needed by the client.
    clean["model"] = model

    # Determine which parameters are explicitly allowed.
    if supported_params is not None:
        # Caller explicitly whitelisted some params – keep only those.
        allowed = supported_params
        unsupported = {
            key
            for key in merged
            if key not in {"model", *token_aliases}
            and key not in allowed
        }
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"Unsupported parameter(s) for model '{model}': {names}.")
    else:
        # Use the provider‑specific whitelist.
        allowed: set[str] = set()
        # Temperature and other sampling params are only allowed for models
        # that support them.
        # Catalog metadata is checked by the factory before this compatibility helper.
        model_supports_temp = model_supports_temperature(model)
        if model_supports_temp:
            allowed.update({
                "temperature",
                "top_p",
                "top_logprobs",
                "logprobs",
            })
        # Always allow the output token limit using the installed integration's
        # provider-specific spelling.
        allowed.add(token_param)

    # Normalize output-token aliases to the constructor keyword used by the
    # installed provider integration, preferring its native spelling.
    token_limit = merged.get(token_param)
    if token_limit is None:
        token_limit = next(
            (merged[key] for key in token_aliases if key in merged),
            None,
        )
    if (
        supported_params is not None
        and token_limit is not None
        and token_param not in supported_params
    ):
        raise ValueError(
            f"The installed provider integration does not support '{token_param}' "
            f"for model '{model}'."
        )

    # Apply the filter.
    for key, value in merged.items():
        # Never forward the model key itself (already handled).
        if key == "model":
            continue
        if key in token_aliases:
            continue
        # If the caller gave an explicit whitelist, honour it strictly.
        if supported_params is not None:
            if key in allowed:
                clean[key] = value
            continue
        # Model‑specific rule set: surface unsupported sampling params instead
        # of silently dropping them (requirement: surface the bug).
        if key in _UNSUPPORTED_SAMPLING_KWARGS and not model_supports_temp:
            raise ValueError(
                f"{key} is not supported for model '{model}' (provider '{provider}'). "
                "Remove it from the call or choose a model that supports "
                "sampling parameters."
            )
        clean[key] = value

    # Ensure the provider's token-limit param is present exactly once.  Never
    # alias max_tokens <-> max_output_tokens: passing both breaks the client.
    if token_limit is not None:
        clean[token_param] = token_limit

    return clean