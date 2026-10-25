"""Profile Agent: a tool-calling loop (model -> tool -> model ...) that interviews the student
and writes the profile to the database through profile_tools.

    agent, state = build_profile_agent(student_id)
    result = agent.invoke({"messages": history})   # history = list of chat messages
    result["messages"]                            # full history incl. tool calls and the new reply
"""
from datetime import date
from pathlib import Path

from langchain.agents import create_agent

from northstar.db.session import get_session
from northstar.llm import get_llm
from northstar.tools.profile_tools import make_profile_tools

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"

def greeting(alias: str | None = None) -> str:
    """Fixed first message (no LLM call needed)."""
    if alias:
        return f"Welcome back, {alias}! Has anything changed - new interests, a different schedule, upcoming exams? Tell me and I'll update your profile."
    return ("Hi! I'm NorthStar, your study mentor. I'll ask a few quick questions so I can plan around your life. "
            "What nickname should I call you, and what grade or year are you in?")


def system_prompt() -> str:
    parts = [(PROMPTS / name).read_text(encoding="utf-8") for name in ("shared_safety.md", "profile.md")]
    return "\n\n".join(parts + [f"Today is {date.today():%A %d %B %Y}."])


def build_profile_agent(student_id: int | None = None, session_factory=get_session):
    """Returns (agent, state). state["student_id"] is set once a new student has been saved."""
    tools, state = make_profile_tools(student_id, session_factory)
    return create_agent(get_llm(), tools, system_prompt=system_prompt()), state
