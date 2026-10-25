"""All settings come from environment variables (loaded from .env locally)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Format "provider:model", e.g. "openai:gpt-6-luna" or "google_genai:gemini-2.5-flash".
MODEL = os.getenv("NORTHSTAR_MODEL", "openai:gpt-6-luna")
# Absolute default path, so the DB is the same file whichever folder you run from.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(PROJECT_ROOT / 'data' / 'northstar.db').as_posix()}")
