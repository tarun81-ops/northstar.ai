"""LangGraph orchestrator: one entry point for NorthStar's AI flows.

    START --route(intent)--+--> weekly_review --> END
                           +--> plan_week ------> END
                           +--> suggest_niches -> END

A StateGraph passes a `State` dict between nodes. The router is plain Python (the page says what it wants),
so routing never costs an LLM call. Each node loads what it needs from the database, calls one agent,
and puts its output in state["result"]. Pages just call run(...).
The profile chat keeps its own create_agent loop and the Roadmap Agent is a direct call; they join this
graph when free-form Mentor Chat needs an LLM router.
"""
from datetime import date
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from northstar.agents.niche_agent import suggest_niches
from northstar.agents.progress_agent import write_review
from northstar.agents.schedule_agent import plan_week
from northstar.db.session import get_session
from northstar.schemas import MilestoneIn
from northstar.services.goals import get_active_goal
from northstar.services.niches import refresh_signals, save_suggestions, suggestions
from northstar.services.patterns import detect
from northstar.services.profile import get_profile
from northstar.services.progress import week_summary
from northstar.services.review import adjustments_for_week, save_review
from northstar.services.tasks import busy_minutes, get_week_tasks

Intent = Literal["weekly_review", "plan_week", "suggest_niches"]


class State(TypedDict, total=False):
    intent: Intent
    student_id: int
    week_start: date
    today: date
    result: Any


def weekly_review_node(state: State) -> dict:
    """Numbers -> patterns (Python) -> review + proposed changes (LLM) -> saved for the student to accept or dismiss."""
    sid, start = state["student_id"], state["week_start"]
    with get_session() as s:
        summary = week_summary(s, sid, start)
        goal = get_active_goal(s, sid)
    if not summary["tasks_planned"]:
        raise ValueError(f"No tasks were planned in the week of {start}, so there is nothing to review")
    patterns = detect(summary)
    review = write_review(summary, patterns, goal.title if goal else None)
    with get_session() as s:
        return {"result": save_review(s, sid, start, review, patterns)}


def plan_week_node(state: State) -> dict:
    """Profile + milestones + finished tasks + accepted changes -> a week plan preview (not saved)."""
    sid, start = state["student_id"], state["week_start"]
    with get_session() as s:
        profile = get_profile(s, sid)
        goal = get_active_goal(s, sid)
        milestones = [MilestoneIn.model_validate(m, from_attributes=True) for m in goal.milestones]
        busy = busy_minutes(get_week_tasks(s, sid, start))
        adjust = adjustments_for_week(s, sid, start)
    return {"result": plan_week(profile, goal.title, milestones, start, state["today"], busy, adjust)}


def suggest_niches_node(state: State) -> dict:
    """Fresh signals (Python) -> niches citing them, each with a curiosity experiment (LLM) -> saved as suggestions."""
    sid = state["student_id"]
    with get_session() as s:
        profile = get_profile(s, sid)
        signals = refresh_signals(s, sid, profile)
        avoid = [n.niche_name for status in ("chosen", "dismissed") for n in suggestions(s, sid, status)]
    if not signals:
        raise ValueError("Add something you're curious about first - NorthStar needs at least one signal")
    niches = suggest_niches(profile, signals, avoid)
    with get_session() as s:
        save_suggestions(s, sid, niches)
    return {"result": len(niches)}


def route(state: State) -> Intent:
    return state["intent"]


builder = StateGraph(State)
builder.add_node("weekly_review", weekly_review_node)
builder.add_node("plan_week", plan_week_node)
builder.add_node("suggest_niches", suggest_niches_node)
builder.add_conditional_edges(START, route, ["weekly_review", "plan_week", "suggest_niches"])
for node in ("weekly_review", "plan_week", "suggest_niches"):
    builder.add_edge(node, END)
graph = builder.compile()


def run(intent: Intent, student_id: int, week_start: date | None = None, today: date | None = None) -> Any:
    return graph.invoke({"intent": intent, "student_id": student_id, "week_start": week_start,
                         "today": today or date.today()})["result"]
