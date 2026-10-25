"""All settings come from environment variables (loaded from .env locally)."""
import os

from dotenv import load_dotenv

load_dotenv()

# Format "provider:model", e.g. "openai:gpt-6-luna" or "google_genai:gemini-2.5-flash".
MODEL = os.getenv("NORTHSTAR_MODEL", "openai:gpt-6-luna")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/northstar.db")
