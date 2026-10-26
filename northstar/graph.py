"""LangGraph orchestrator: one entry point for NorthStar's AI flows.

    START --route(intent)--+--> weekly_review --> END
                           +--> plan_week ------> END
                           +--> suggest_niches -> END
                           +--> coach_step -----> END
                           +--> match_opportunities --> END

A StateGraph passes a `State` dict between nodes. The router is plain Python (the page says what it wants),
so routing never costs an LLM call. Each node loads what it needs from the database, calls one agent,
and puts its output in state["result"]. Pages just call run(...).
The profile chat keeps its own create_agent loop and the Roadmap Agent is a direct call; they join this
graph when free-form Mentor Chat needs an LLM router.
"""
from datetime import date
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from northstar.agents.dataset_agent import match_datasets
from northstar.agents.niche_agent import suggest_niches
from northstar.agents.opportunity_agent import match_opportunities
from northstar.agents.progress_agent import write_review
from northstar.agents.research_agent import coach_step
from northstar.agents.schedule_agent import plan_week
from northstar.db.session import get_session
from northstar.schemas import MilestoneIn
from northstar.services.goals import get_active_goal
from northstar.services.kaggle_dataset_service import filter_and_rank_candidates, save_dataset_recommendations
from northstar.services.niches import refresh_signals, save_suggestions, suggestions
from northstar.services.opportunities import CANDIDATES, dismissed_keys, eligible, load_opportunities, rank, save_matches
from northstar.services.patterns import detect
from northstar.services.profile import get_profile
from northstar.services.progress import week_summary
from northstar.services.research import get_project, save_coach_reply
from northstar.services.review import adjustments_for_week, save_review
from northstar.services.tasks import busy_minutes, get_week_tasks

Intent = Literal["weekly_review", "plan_week", "suggest_niches", "coach_step", "match_opportunities", "recommend_datasets"]
NODES = ("weekly_review", "plan_week", "suggest_niches", "coach_step", "match_opportunities", "recommend_datasets")


class State(TypedDict, total=False):
    intent: Intent
    student_id: int
    week_start: date
    today: date
    project_id: int  # coach_step
    step_id: int     # coach_step
    only_free: bool  # match_opportunities
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


def coach_step_node(state: State) -> dict:
    """The student's project + notes for one step -> the Research Coach's guidance (saved on the step)."""
    sid = state["student_id"]
    with get_session() as s:
        profile = get_profile(s, sid)
        project = get_project(s, sid, state["project_id"])
        step = next((st for st in project.steps if st.id == state["step_id"]), None)
    if step is None:
        raise ValueError("That step isn't part of this project")
    reply = coach_step(profile, project, step.step)
    with get_session() as s:
        save_coach_reply(s, sid, step.id, reply)
    return {"result": reply}


def match_opportunities_node(state: State) -> dict:
    """Eligible (age, deadline, cost) and ranked by interests (Python) -> best fits with "why" (LLM) -> saved."""
    sid = state["student_id"]
    with get_session() as s:
        profile = get_profile(s, sid)
        signals = [sig.signal for sig in refresh_signals(s, sid, profile)]
        niches = [n.niche_name for n in suggestions(s, sid, "chosen")]
        skip = dismissed_keys(s, sid)
    open_to_them = [o for o in eligible(load_opportunities(), profile, state["today"], state.get("only_free", False))
                    if o.key not in skip]
    candidates = rank(open_to_them, profile.interests + profile.subjects + profile.strengths + signals + niches)[:CANDIDATES]
    if not candidates:
        return {"result": 0}  # nothing eligible matches their interests: no AI call
    matches = match_opportunities(profile, candidates, signals, niches)
    with get_session() as s:
        save_matches(s, sid, matches)
    return {"result": len(matches)}


def recommend_datasets_node(state: State) -> dict:
    """Pre-filtered & ranked catalog (Python) -> top 2-5 best fits with student-friendly reasons (LLM) -> saved."""
    sid, pid = state["student_id"], state["project_id"]
    with get_session() as s:
        project = get_project(s, sid, pid)
        ranked = filter_and_rank_candidates(s, sid, pid)
        candidates = [d for d, _ in ranked]
        if not candidates:
            return {"result": 0}
        question_step = next((st for st in project.steps if st.step == "question"), None)
        question_notes = question_step.notes if question_step else ""
        niche_name = ""
        if project.niche_id:
            from northstar.db.models import NicheSuggestion

            niche = s.get(NicheSuggestion, project.niche_id)
            if niche:
                niche_name = f"{niche.niche_name}: {niche.description}"

    matches = match_datasets(project.title, question_notes, niche_name, candidates)
    with get_session() as s:
        saved = save_dataset_recommendations(s, sid, pid, matches)
    return {"result": len(saved)}


def route(state: State) -> Intent:
    return state["intent"]


builder = StateGraph(State)
builder.add_node("weekly_review", weekly_review_node)
builder.add_node("plan_week", plan_week_node)
builder.add_node("suggest_niches", suggest_niches_node)
builder.add_node("coach_step", coach_step_node)
builder.add_node("match_opportunities", match_opportunities_node)
builder.add_node("recommend_datasets", recommend_datasets_node)
builder.add_conditional_edges(START, route, list(NODES))
for node in NODES:
    builder.add_edge(node, END)
graph = builder.compile()


def run(intent: Intent, student_id: int, week_start: date | None = None, today: date | None = None, **inputs) -> Any:
    """inputs: extra state for a node, e.g. project_id and step_id for coach_step."""
    return graph.invoke({"intent": intent, "student_id": student_id, "week_start": week_start,
                         "today": today or date.today(), **inputs})["result"]
