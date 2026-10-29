"""Legacy model-router API, delegated to the free-only OpenRouter factory."""
from __future__ import annotations

def selected_model_for_session() -> str | None:
    """Return a model choice only inside an active, user-scoped Streamlit run."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:  # Streamlit is optional for non-UI consumers.
        return None
    if get_script_run_ctx() is None:
        return None
    import streamlit as st

    return st.session_state.get("northstar_selected_model")


def get_llm(task_type: str = "short_chat", **kwargs):
    """Return a free-only OpenRouter chat model for the legacy task call shape."""
    user_model = kwargs.pop("model", None)
    if kwargs.pop("free_only", True) is not True:
        from northstar.ai.model_factory import OpenRouterModelSelectionError

        raise OpenRouterModelSelectionError("NorthStar only permits free OpenRouter models.")

    # Lazy import to avoid a circular import.
    from northstar.ai.model_factory import get_llm as _factory_get_llm

    return _factory_get_llm(
        model=user_model,
        task_type=task_type,
        free_only=True,
        **kwargs,
    )
