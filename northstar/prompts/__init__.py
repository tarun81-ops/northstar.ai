"""Prompt files live next to this module. Every agent's system prompt = shared safety rules + its role + today."""
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent


def system_prompt(role: str) -> str:
    """e.g. system_prompt("profile") -> shared_safety.md + profile.md + today's date."""
    parts = [(HERE / f"{name}.md").read_text(encoding="utf-8") for name in ("shared_safety", role)]
    return "\n\n".join(parts + [f"Today is {date.today():%A %d %B %Y}."])
