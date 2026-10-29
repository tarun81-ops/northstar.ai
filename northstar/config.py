"""All settings come from environment variables (loaded from .env locally)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Absolute default path, so the DB is the same file whichever folder you run from.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(PROJECT_ROOT / 'data' / 'northstar.db').as_posix()}")

# Kaggle credentials and dataset settings
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")
KAGGLE_MAX_DATASET_SIZE_MB = float(os.getenv("KAGGLE_MAX_DATASET_SIZE_MB", "50.0"))
KAGGLE_STUDENT_STORAGE_CAP_MB = float(os.getenv("KAGGLE_STUDENT_STORAGE_CAP_MB", "100.0"))
DATASET_WORKSPACE_ROOT = Path(os.getenv("DATASET_WORKSPACE_ROOT", str(PROJECT_ROOT / "data" / "workspaces")))
