"""The one place the app creates a chat model. Switch provider via NORTHSTAR_MODEL."""
import os

from langchain.chat_models import init_chat_model

from northstar import config

# provider prefix in NORTHSTAR_MODEL -> the environment variable its API key lives in
KEY_FOR_PROVIDER = {"openai": "OPENAI_API_KEY", "google_genai": "GOOGLE_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def get_llm(**kwargs):
    return init_chat_model(config.MODEL, **kwargs)


def ai_problem() -> str | None:
    """A student-friendly reason the AI can't run (missing key), or None if it should work.
    Pages show it and disable AI buttons; everything that doesn't need the AI keeps working."""
    provider = config.MODEL.split(":", 1)[0]
    key = KEY_FOR_PROVIDER.get(provider)
    if key and not os.getenv(key, "").strip():
        return (f"NorthStar's AI is switched off: no {key} was found. Add it to the .env file and restart the app. "
                "You can still use your plan, check-ins and the forms.")
    return None
