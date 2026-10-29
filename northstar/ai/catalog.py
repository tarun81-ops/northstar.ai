"""Current OpenRouter model catalog for NorthStar's free-only selector.

Responsibilities:
* Cache OpenRouter model metadata (fetched from
  ``https://openrouter.ai/api/v1/models``) with a time-to-live so the selector
  doesn't hit the network on every render.
* Keep listed models and their declared pricing/capabilities as catalog data;
  selection code separately requires verified zero pricing.
* Handle catalog failures gracefully -- return an empty list rather than
  inventing model IDs or assuming a model is free.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import requests

logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Default TTL for the OpenRouter catalog cache (5 minutes).
CATALOG_TTL_SECONDS = 300

@dataclass
class ModelEntry:
    """A single model available for selection."""

    provider: str
    model_id: str
    name: str
    verified: bool
    source: str
    context_tokens: int | None = None
    pricing: dict | None = None
    supports_tool_calling: bool = False
    supports_structured_output: bool = False
    supported_parameters: frozenset[str] | None = None

    def supports(self, capability: str) -> bool:
        if capability == "tool_calling":
            return self.supports_tool_calling
        if capability == "structured_output":
            return self.supports_structured_output
        return False

    @property
    def zero_priced(self) -> bool:
        if not isinstance(self.pricing, dict) or not {"prompt", "completion"} <= self.pricing.keys():
            return False

        overrides = self.pricing.get("overrides", [])
        if not isinstance(overrides, list) or any(
            not isinstance(item, dict)
            or not {"prompt", "completion"} <= item.keys()
            for item in overrides
        ):
            return False

        def all_prices_are_zero(value) -> bool:
            if isinstance(value, dict):
                return all(all_prices_are_zero(item) for item in value.values())
            if isinstance(value, list):
                return all(all_prices_are_zero(item) for item in value)
            try:
                amount = Decimal(str(value))
                return amount.is_finite() and amount == 0
            except (InvalidOperation, TypeError, ValueError):
                return False

        try:
            return all_prices_are_zero(self.pricing)
        except (TypeError, ValueError):
            return False


class ModelCatalog:
    """In-memory cache of model metadata per provider."""

    def __init__(self, ttl_seconds: int = CATALOG_TTL_SECONDS):
        self._ttl = ttl_seconds
        self._cache = {}
        self.catalog_available = False

    def _expired(self, provider: str) -> bool:
        entry = self._cache.get(provider)
        if entry is None:
            return True
        return time.time() - entry[0] > self._ttl

    def get_openrouter_models(self) -> list:
        """Return OpenRouter models from the cached catalog, or [] on failure."""
        if not self._expired("openrouter"):
            return self._cache["openrouter"][1]

        models = []
        try:
            resp = requests.get(OPENROUTER_MODELS_URL, params={"limit": 1000}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("data", []):
                model_id = item.get("id", "")
                if not isinstance(model_id, str) or not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}", model_id
                ):
                    continue
                supported = set(item.get("supported_parameters") or [])
                models.append(
                    ModelEntry(
                        provider="openrouter",
                        model_id="openrouter:" + model_id,
                        name=item.get("name", model_id),
                        verified=False,
                        source="catalog",
                        context_tokens=item.get("context_length"),
                        pricing=item.get("pricing"),
                        supports_tool_calling="tools" in supported,
                        supports_structured_output=bool(
                            {"response_format", "structured_outputs"} & supported
                        ),
                        supported_parameters=frozenset(supported),
                    )
                )
            self.catalog_available = True
        except Exception as exc:
            logger.warning("openrouter catalog unavailable: %s", type(exc).__name__)
            self.catalog_available = False
            models = []

        self._cache["openrouter"] = (time.time(), models)
        return models

    def get_provider_models(self, provider: str) -> list:
        """Return only the OpenRouter catalog; direct providers have no route."""
        if provider == "openrouter":
            return self.get_openrouter_models()
        return []

    def all_models(self) -> list:
        """Return all known models across all providers."""
        all_models = []
        all_models.extend(self.get_provider_models("openrouter"))
        return all_models

    def refresh(self) -> None:
        """Expire all cached entries so the next read re-fetches."""
        self._cache.clear()
        self.catalog_available = False

    def find(self, model_id: str):
        """Find a model entry by its full ``provider:model`` id."""
        provider, separator, _ = model_id.partition(":")
        if not separator:
            return None
        for entry in self.get_provider_models(provider):
            if entry.model_id == model_id:
                return entry
        return None


catalog = ModelCatalog()
