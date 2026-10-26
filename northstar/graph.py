"""LangGraph orchestrator: one entry point for NorthStar's AI flows.

    START --route(intent)--+--> weekly_review --> END
                           +--> plan_week ------> END

A StateGraph passes a `State` dict between nodes. The router is plain Python (the page says what it wants),
so routing never costs an LLM call. Each node loads what it needs from the database, calls one agent,
and puts its output in state["result"]. Pages just call run(...).
The profile chat keeps its own create_agent loop and the Roadmap Agent is a direct call; they join this
graph when free-form Mentor Chat needs an LLM router.
"""
from datetime import date
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from northstar.agents.progress_agent import write_review
from northstar.agents.schedule_agent import plan_week
from northstar.db.session import get_session
from northstar.schemas import MilestoneIn
from northstar.services.goals import get_active_goal
from northstar.services.patterns import detect
from northstar.services.profile import get_profile
from northstar.services.progress import week_summary
from northstar.services.review import adjustments_for_week, save_review
from northstar.services.tasks import busy_minutes, get_week_tasks

Intent = Literal["weekly_review", "plan_week"]


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


def route(state: State) -> Intent:
    return state["intent"]


builder = StateGraph(State)
builder.add_node("weekly_review", weekly_review_node)
builder.add_node("plan_week", plan_week_node)
builder.add_conditional_edges(START, route, ["weekly_review", "plan_week"])
builder.add_edge("weekly_review", END)
builder.add_edge("plan_week", END)
graph = builder.compile()


def run(intent: Intent, student_id: int, week_start: date, today: date | None = None) -> Any:
    return graph.invoke({"intent": intent, "student_id": student_id, "week_start": week_start,
                         "today": today or date.today()})["result"]
