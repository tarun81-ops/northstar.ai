"""Session-scoped selection of current, explicitly free OpenRouter models."""

from __future__ import annotations

import streamlit as st

from northstar.ai.catalog import ModelEntry, catalog
from northstar.ai.model_factory import OpenRouterModelSelectionError, resolve_openrouter_free_model

SELECTED_MODEL_KEY = "northstar_selected_model"
LAST_RESPONDING_MODEL_KEY = "northstar_last_responding_model"
ERROR_STATUS_KEY = "northstar_model_error"
FALLBACK_NOTICE_KEY = "northstar_fallback_notice"

_FREE_ROUTER_ENTRY = ModelEntry(
    provider="openrouter",
    model_id="openrouter:openrouter/free",
    name="OpenRouter free models router",
    verified=True,
    source="OpenRouter free-only router",
    pricing={"prompt": "0", "completion": "0"},
)


def free_model_entries(
    entries: list[ModelEntry],
    required_capabilities: set[str] | None = None,
) -> list[ModelEntry]:
    """Keep only catalog-listed :free models with complete zero-price metadata."""
    capabilities = required_capabilities or set()
    return [
        entry
        for entry in entries
        if entry.provider == "openrouter"
        and entry.model_id.startswith("openrouter:")
        and entry.model_id.split(":", 1)[1].endswith(":free")
        and entry.zero_priced
        and all(entry.supports(capability) for capability in capabilities)
    ]


def _model_label(entry: ModelEntry) -> str:
    capabilities = []
    if entry.supports_tool_calling:
        capabilities.append("tools")
    if entry.supports_structured_output:
        capabilities.append("structured output")
    suffix = f" · {', '.join(capabilities)}" if capabilities else ""
    return f"{entry.name} ({entry.model_id.split(':', 1)[1]}){suffix}"


def render_model_selector(
    *,
    show_fallback_controls: bool = False,
    required_capabilities: set[str] | None = None,
) -> str:
    """Render a selector limited to current catalog-verified free OpenRouter models."""
    del show_fallback_controls
    for key in (
        "northstar_auto_fallback",
        "northstar_fallback_models",
        "northstar_zero_priced_fallback",
    ):
        st.session_state.pop(key, None)
    st.sidebar.divider()
    st.sidebar.subheader("Free OpenRouter model")

    if st.sidebar.button("Refresh model catalog", key="northstar_refresh_catalog"):
        catalog.refresh()
        st.rerun()

    entries = free_model_entries(
        catalog.get_provider_models("openrouter"),
        required_capabilities,
    )
    if not required_capabilities:
        entries.insert(0, _FREE_ROUTER_ENTRY)
    if not catalog.catalog_available:
        if required_capabilities:
            st.sidebar.warning("The OpenRouter model catalog is unavailable. Refresh it before choosing a model.")
        else:
            st.sidebar.caption("The OpenRouter free router remains available while the explicit model catalog is offline.")
    if not entries:
        st.session_state.pop(SELECTED_MODEL_KEY, None)
        st.sidebar.info(
            "No current free OpenRouter models match the required capabilities. "
            "The chat is unavailable until the catalog has a compatible entry."
        )
        return ""

    current = st.session_state.get(SELECTED_MODEL_KEY)
    try:
        selected = (
            resolve_openrouter_free_model(
                current,
                required_capabilities=required_capabilities,
            )
            if current
            else None
        )
    except OpenRouterModelSelectionError:
        selected = None
    options = [entry.model_id for entry in entries]
    if selected not in options:
        selected = options[0]

    widget_key = "northstar_free_model_choice"
    if st.session_state.get(widget_key) not in options:
        st.session_state[widget_key] = selected
    model_by_id = {entry.model_id: entry for entry in entries}
    chosen = st.sidebar.selectbox(
        "Model",
        options,
        format_func=lambda value: _model_label(model_by_id[value]),
        key=widget_key,
    )
    st.session_state[SELECTED_MODEL_KEY] = chosen
    st.sidebar.caption("Only current catalog-listed :free entries with zero-priced metadata are selectable.")
    st.sidebar.caption(f"Selected: {chosen}")
    if last_model := st.session_state.get(LAST_RESPONDING_MODEL_KEY):
        st.sidebar.caption(f"Last answered with: {last_model}")
    if error := st.session_state.get(ERROR_STATUS_KEY):
        st.sidebar.error(error)
    return chosen


def set_error(message: str) -> None:
    """Store a short, sanitized model status message for the next rerun."""
    st.session_state[ERROR_STATUS_KEY] = message


def clear_error() -> None:
    st.session_state.pop(ERROR_STATUS_KEY, None)
