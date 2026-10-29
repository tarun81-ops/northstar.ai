from __future__ import annotations

from northstar.ai.catalog import ModelCatalog, ModelEntry


def test_openrouter_catalog_parses_capabilities_and_pricing(monkeypatch):
    response = type(
        "Response",
        (),
        {
            "raise_for_status": lambda self: None,
            "json": lambda self: {
                "data": [
                    {
                        "id": "vendor/agent-model",
                        "name": "Agent Model",
                        "context_length": 1000,
                        "pricing": {"prompt": "0", "completion": "0"},
                        "supported_parameters": [
                            "tools",
                            "tool_choice",
                            "response_format",
                            "max_tokens",
                        ],
                    }
                ]
            },
        },
    )()
    monkeypatch.setattr("northstar.ai.catalog.requests.get", lambda *args, **kwargs: response)

    model_catalog = ModelCatalog()
    entries = model_catalog.get_openrouter_models()

    assert model_catalog.catalog_available
    assert entries[0].model_id == "openrouter:vendor/agent-model"
    assert entries[0].supports_tool_calling
    assert entries[0].supports_structured_output
    assert entries[0].zero_priced


def test_zero_priced_requires_all_price_fields_and_overrides_to_be_zero():
    zero = ModelEntry(
        provider="openrouter",
        model_id="openrouter:vendor/free:free",
        name="Free model",
        verified=False,
        source="catalog",
        pricing={
            "prompt": "0",
            "completion": "0",
            "web_search": "0",
            "overrides": [{"prompt": "0", "completion": "0"}],
        },
    )
    paid_addon = ModelEntry(
        provider="openrouter",
        model_id="openrouter:vendor/free:free",
        name="Ambiguous add-on price",
        verified=False,
        source="catalog",
        pricing={"prompt": "0", "completion": "0", "web_search": "0.01"},
    )
    unknown_override = ModelEntry(
        provider="openrouter",
        model_id="openrouter:vendor/free:free",
        name="Unknown override price",
        verified=False,
        source="catalog",
        pricing={"prompt": "0", "completion": "0", "overrides": [{"prompt": "0"}]},
    )

    assert zero.zero_priced
    assert not paid_addon.zero_priced
    assert not unknown_override.zero_priced


def test_free_model_selector_requires_catalog_free_variant_and_capability():
    from northstar.ui.model_selector import free_model_entries

    free_tools = ModelEntry(
        provider="openrouter",
        model_id="openrouter:vendor/tools:free",
        name="Free tools",
        verified=False,
        source="catalog",
        pricing={"prompt": "0", "completion": "0"},
        supports_tool_calling=True,
    )
    paid = ModelEntry(
        provider="openrouter",
        model_id="openrouter:vendor/paid",
        name="Paid",
        verified=False,
        source="catalog",
        pricing={"prompt": "0.01", "completion": "0.01"},
        supports_tool_calling=True,
    )
    free_without_tools = ModelEntry(
        provider="openrouter",
        model_id="openrouter:vendor/no-tools:free",
        name="Free without tools",
        verified=False,
        source="catalog",
        pricing={"prompt": "0", "completion": "0"},
    )

    assert free_model_entries([free_tools, paid, free_without_tools]) == [
        free_tools,
        free_without_tools,
    ]
    assert free_model_entries(
        [free_tools, paid, free_without_tools],
        required_capabilities={"tool_calling"},
    ) == [free_tools]


def test_openrouter_catalog_failure_degrades_without_throwing(monkeypatch):
    monkeypatch.setattr(
        "northstar.ai.catalog.requests.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError()),
    )
    model_catalog = ModelCatalog()
    assert model_catalog.get_openrouter_models() == []
    assert not model_catalog.catalog_available


def test_direct_provider_catalogs_are_not_available():
    model_catalog = ModelCatalog()
    assert model_catalog.get_provider_models("openai") == []
    assert model_catalog.get_provider_models("google_genai") == []
