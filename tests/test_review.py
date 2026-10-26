from datetime import date, timedelta

import pytest
from sqlalchemy.orm import Session, sessionmaker

from northstar import graph
from northstar.agents import progress_agent, schedule_agent
from northstar.db.models import ProgressLog, Task
from northstar.db.seed import demo_profile, seed_week
from northstar.db.session import init_db, make_engine
from northstar.schemas import GoalIn, MilestoneIn, ProposedChange, TaskDraft, TaskDraftList, WeeklyReviewOut
from northstar.services.goals import create_goal, save_milestones
from northstar.services.patterns import detect
from northstar.services.profile import save_profile
from northstar.services.progress import summarize, week_summary
from northstar.services.review import (
    Adjustments, adjustments_for_week, check_review, decide_change, get_review, save_review,
)

WEEK = date(2026, 11, 2)  # a Monday
NEXT = WEEK + timedelta(days=7)


@pytest.fixture
def engine():
    e = make_engine("sqlite://")
    init_db(e)
    return e


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


@pytest.fixture
def sid(session):
    return save_profile(session, demo_profile())


def review(*changes: ProposedChange) -> WeeklyReviewOut:
    return WeeklyReviewOut(win="You checked in every day.", what_the_data_shows="Tasks took longer than planned.",
                           changes=list(changes), question="What made the algebra tricky?")


# ---- patterns --------------------------------------------------------------------

@pytest.mark.parametrize("kind, expected", [
    ("good", {"too_easy", "strong_interest"}),
    ("bad", {"bad_week", "skipping_subject"}),
    ("stressed", {"overloaded", "too_hard", "strong_interest"}),
])
def test_each_seeded_week_triggers_its_patterns(session, sid, kind, expected):
    seed_week(session, sid, WEEK, kind)
    assert {p.name for p in detect(week_summary(session, sid, WEEK))} == expected


def make_week(*outcomes) -> tuple[list[Task], list[ProgressLog]]:
    """outcomes: (completed, minutes_spent, difficulty, enjoyment, stress) per 30-minute task."""
    tasks, logs = [], []
    for i, (completed, spent, difficulty, enjoyment, stress) in enumerate(outcomes):
        status = {"yes": "done", "partial": "partial", "no": "skipped"}[completed]
        tasks.append(Task(id=i, category="academics", duration_minutes=30, status=status))
        logs.append(ProgressLog(task_id=i, date=WEEK + timedelta(days=i), completed=completed, time_spent_minutes=spent,
                                difficulty_rating=difficulty, enjoyment=enjoyment, stress=stress, blockers=""))
    return tasks, logs


def test_steady_week_has_no_patterns():
    s = summarize(*make_week(*[("yes", 30, "just_right", 3, 2)] * 4))
    assert s["completion_rate"] == 1.0 and detect(s) == []


def test_too_few_checkins_means_no_patterns():
    assert detect(summarize(*make_week(("no", 0, None, None, 5), ("no", 0, None, None, 5)))) == []


def test_losing_interest():
    s = summarize(*make_week(("yes", 30, "just_right", 4, 2), ("yes", 30, "just_right", 3, 2), ("yes", 30, "just_right", 2, 2)))
    assert [p.name for p in detect(s)] == ["losing_interest"]


# ---- review service -------------------------------------------------------------

def test_check_review():
    ok = review(ProposedChange(kind="lighten_load", amount=20, reason="Stress was high."))
    assert check_review(ok) == []
    assert "between 10 and 40" in check_review(review(ProposedChange(kind="lighten_load", amount=80, reason="x")))[0]
    assert "needs a category" in check_review(review(ProposedChange(kind="easier_steps", reason="x")))[0]
    four = review(*[ProposedChange(kind="shorter_tasks", amount=30, reason="x")] * 4)
    assert "at most 3" in check_review(four)[0]


def test_rerunning_review_replaces_only_undecided_changes(session, sid):
    save_review(session, sid, WEEK, review(ProposedChange(kind="lighten_load", amount=20, reason="a"),
                                          ProposedChange(kind="shorter_tasks", amount=30, reason="b")), [])
    _, changes = get_review(session, sid, WEEK)
    decide_change(session, sid, changes[0].id, accept=True)

    save_review(session, sid, WEEK, review(ProposedChange(kind="easier_steps", category="academics", reason="c")), [])
    _, changes = get_review(session, sid, WEEK)
    assert [(c.kind, c.status) for c in changes] == [("lighten_load", "applied"), ("easier_steps", "proposed")]


def test_decide_change_guards(session, sid):
    save_review(session, sid, WEEK, review(ProposedChange(kind="lighten_load", amount=20, reason="a")), [])
    change = get_review(session, sid, WEEK)[1][0]
    other = save_profile(session, demo_profile().model_copy(update={"alias": "Meera"}))
    with pytest.raises(ValueError, match="not this student's"):
        decide_change(session, other, change.id, accept=True)
    decide_change(session, sid, change.id, accept=False)
    with pytest.raises(ValueError, match="already dismissed"):
        decide_change(session, sid, change.id, accept=True)


def test_adjustments_use_only_accepted_changes_from_the_latest_earlier_review(session, sid):
    save_review(session, sid, WEEK - timedelta(days=7), review(ProposedChange(kind="lighten_load", amount=40, reason="old")), [])
    decide_change(session, sid, get_review(session, sid, WEEK - timedelta(days=7))[1][0].id, accept=True)
    save_review(session, sid, WEEK, review(ProposedChange(kind="lighten_load", amount=20, reason="a"),
                                          ProposedChange(kind="shorter_tasks", amount=30, reason="b"),
                                          ProposedChange(kind="easier_steps", category="academics", reason="c")), [])
    lighten, shorter, easier = get_review(session, sid, WEEK)[1]
    decide_change(session, sid, lighten.id, accept=True)
    decide_change(session, sid, easier.id, accept=True)
    decide_change(session, sid, shorter.id, accept=False)

    adjust = adjustments_for_week(session, sid, NEXT)
    assert adjust.load_factor == 0.8 and adjust.max_minutes is None and adjust.easier == {"academics"}
    assert adjustments_for_week(session, sid, WEEK).load_factor == 0.6  # the week before used the older review
    assert adjustments_for_week(session, sid, WEEK - timedelta(days=7)) == Adjustments()


def test_planner_enforces_accepted_changes():
    adjust = Adjustments(max_minutes=30, easier={"academics"})
    long = TaskDraft(title="Long session", category="technical", duration_minutes=45, difficulty=2)
    hard = TaskDraft(title="Hard algebra", category="academics", duration_minutes=20, difficulty=4)
    problems = schedule_agent.check_drafts([long, hard], {WEEK: 200}, set(), adjust)
    assert any("at most 30 minutes" in p for p in problems)
    assert any("difficulty must be 2 or less" in p for p in problems)


# ---- the whole loop, through the LangGraph graph (fake LLM, no API key) -----------

class FakeLLM:
    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.outputs.pop(0)


def test_review_accept_then_replan_through_the_graph(engine, session, sid, monkeypatch):
    monkeypatch.setattr(graph, "get_session", sessionmaker(engine))
    goal_id = create_goal(session, sid, GoalIn(title="Maths + robot", category="academics",
                                               target_date=date.today() + timedelta(days=90)))
    save_milestones(session, goal_id, [MilestoneIn(title="Algebra baseline", due_date=date.today() + timedelta(days=30),
                                                   completion_criteria="12 problems marked", category="academics")])
    seed_week(session, sid, WEEK, "stressed")

    # 1. review: patterns found in Python, the (fake) LLM proposes lightening the load
    reviewer = FakeLLM(review(ProposedChange(kind="lighten_load", amount=30, reason="Stress averaged 4.4/5.")))
    monkeypatch.setattr(progress_agent, "get_llm", lambda: reviewer)
    graph.run("weekly_review", sid, WEEK)
    saved_review, changes = get_review(session, sid, WEEK)
    assert {p["name"] for p in saved_review.patterns} >= {"overloaded", "too_hard"}
    assert '"overloaded"' in reviewer.calls[0][1].content  # the LLM was told what Python detected
    assert changes[0].status == "proposed"

    # 2. before accepting, next week is planned at full capacity
    planner = FakeLLM(*[TaskDraftList(tasks=[TaskDraft(title="Revise algebra", category="academics",
                                                       duration_minutes=30, difficulty=2)])] * 2)
    monkeypatch.setattr(schedule_agent, "get_llm", lambda: planner)
    full = graph.run("plan_week", sid, NEXT, today=NEXT)

    # 3. the student accepts -> the next plan has 30% less time, and the agent is told why
    decide_change(session, sid, changes[0].id, accept=True)
    lighter = graph.run("plan_week", sid, NEXT, today=NEXT)
    assert sum(lighter.capacity.values()) < sum(full.capacity.values()) * 0.75
    assert "Plan 30% less work" in planner.calls[1][1].content
