"""Legacy chat-model API, delegated to the shared free OpenRouter factory.

This module preserves the legacy ``get_llm()`` and ``ai_problem()`` entry points
used by pages, while delegating all client construction to the shared,
provider-aware factory in :mod:`northstar.ai.model_factory`.
"""
from northstar.ai.model_factory import get_llm as _factory_get_llm
from northstar.ai import providers


def get_llm(**kwargs):
    """Return a free OpenRouter chat model with optional caller overrides.

    Delegates to the shared factory so provider awareness, credential checks,
    and parameter validation are consistent across all entry points.
    """
    if kwargs.pop("free_only", True) is not True:
        from northstar.ai.model_factory import OpenRouterModelSelectionError

        raise OpenRouterModelSelectionError("NorthStar only permits free OpenRouter models.")
    model = kwargs.pop("model", None)
    return _factory_get_llm(model=model, free_only=True, **kwargs)


def ai_problem() -> str | None:
    """A student-friendly reason the AI can't run (missing key), or None if it should work.

    Pages show it and disable AI actions; everything that doesn't need the AI
    keeps working.
    """
    if not providers.has_api_key("openrouter"):
        return (
            "NorthStar's AI is switched off: no OPENROUTER_API_KEY was found. "
            "Add it to the .env file or Streamlit secrets. "
            "You can still use your plan, check-ins and the forms."
        )
    return None
