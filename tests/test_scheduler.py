from collections import Counter
from datetime import date, timedelta

import pytest
from sqlalchemy.orm import Session

from northstar.agents import schedule_agent
from northstar.db.seed import demo_profile
from northstar.db.session import init_db, make_engine
from northstar.schemas import Exam, MilestoneIn, TaskDraft, TaskDraftList
from northstar.services.profile import save_profile
from northstar.services.scheduler import capacity, exams_soon, place, week_days, week_start
from northstar.services.tasks import get_week_tasks, save_week

START = date(2026, 11, 2)  # a Monday
MON, TUE, WED, THU, FRI, SAT, SUN = week_days(START)


def task(minutes: int, title: str = "Do a task", category: str = "academics", **kw) -> TaskDraft:
    return TaskDraft(title=title, category=category, duration_minutes=minutes, difficulty=2, **kw)


# ---- capacity ------------------------------------------------------------------

def test_capacity_subtracts_activities_and_rest_buffer():
    # demo: 120 free on weekdays, football Tue/Thu 90, Sat 240, Sun 180; 20% buffer, rounded down to 5
    cap = capacity(demo_profile(), START, today=START)
    assert cap == {MON: 95, TUE: 20, WED: 95, THU: 20, FRI: 95, SAT: 190, SUN: 140}


def test_capacity_past_days_and_busy_minutes():
    cap = capacity(demo_profile(), START, today=WED, busy={WED: 50, SAT: 500})
    assert cap[MON] == cap[TUE] == 0  # already gone
    assert cap[WED] == 45 and cap[SAT] == 0  # busy time removed, never negative


def test_lighter_load_counts_finished_work_against_the_smaller_limit():
    # Monday: 120 free x 0.8 x 0.65 = 62 -> 60, minus 40 already done = 20 (not (95 - 40) x 0.65 = 35)
    cap = capacity(demo_profile(), START, today=START, busy={MON: 40}, load_factor=0.65)
    assert cap[MON] == 20 and cap[SUN] == 90


def test_week_start_and_exam_mode():
    assert week_start(date(2026, 11, 5)) == START
    profile = demo_profile().model_copy(update={"exams": [
        Exam(subject="Maths", date=START + timedelta(days=10)),
        Exam(subject="Physics", date=START + timedelta(days=20)),
        Exam(subject="Old", date=START - timedelta(days=1)),
    ]})
    assert [e.subject for e in exams_soon(profile, START)] == ["Maths"]


# ---- place ---------------------------------------------------------------------

def test_place_never_overloads_and_never_drops():
    cap = capacity(demo_profile(), START, today=START)
    drafts = [task(m, f"Task {i}", category=c) for i, (m, c) in enumerate(
        [(60, "academics"), (45, "technical"), (30, "academics"), (90, "technical"), (25, "wellbeing"),
         (60, "academics"), (40, "technical"), (15, "academics"), (90, "academics"), (30, "creativity")])]
    placed, overflow = place(drafts, cap)

    used = Counter()
    for d, t in placed:
        used[d] += t.duration_minutes
    assert all(used[d] <= cap[d] for d in cap)
    assert len(placed) + len(overflow) == len(drafts)
    assert [d for d, _ in placed] == sorted(d for d, _ in placed)  # returned in day order


def test_required_tasks_win_over_optional():
    placed, overflow = place([task(30, "Optional first", is_optional=True), task(30, "Required")], {MON: 30})
    assert [t.title for _, t in placed] == ["Required"]
    assert [t.title for t in overflow] == ["Optional first"]


def test_at_most_two_of_a_category_per_day():
    placed, overflow = place([task(10), task(10), task(10)], {MON: 100})
    assert len(placed) == 2 and len(overflow) == 1


def test_milestone_order_is_kept():
    # Without the order rule, step 2 would go on Monday (emptier), before step 1 on Tuesday.
    drafts = [task(50, "Other work", milestone_id=None), task(70, "Step 1", milestone_id=1),
              task(20, "Step 2", milestone_id=1)]
    placed, _ = place(drafts, {MON: 100, TUE: 100})
    days = {t.title: d for d, t in placed}
    assert days == {"Other work": MON, "Step 1": TUE, "Step 2": TUE}


def test_spreads_work_instead_of_piling_on_monday():
    placed, _ = place([task(30, category=c) for c in ("academics", "technical", "creativity")],
                      {MON: 100, TUE: 100, WED: 100})
    assert {d for d, _ in placed} == {MON, TUE, WED}


# ---- save_week -----------------------------------------------------------------

@pytest.fixture
def session():
    engine = make_engine("sqlite://")
    init_db(engine)
    with Session(engine) as s:
        yield s


def test_save_week_replaces_only_unstarted_tasks(session):
    sid = save_profile(session, demo_profile())
    save_week(session, sid, START, [(MON, task(30, "Finished one")), (TUE, task(20, "Not started"))])
    get_week_tasks(session, sid, START)[0].status = "done"
    session.commit()

    save_week(session, sid, START, [(WED, task(40, "New plan"))])
    assert [(t.title, t.status) for t in get_week_tasks(session, sid, START)] == [("Finished one", "done"), ("New plan", "todo")]

    with pytest.raises(ValueError, match="outside the week"):
        save_week(session, sid, START, [(START + timedelta(days=7), task(30))])


# ---- schedule agent (fake LLM, no API key) ---------------------------------------

class FakeLLM:
    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.outputs.pop(0)


MILESTONES = [
    MilestoneIn(id=1, title="Old finished one", due_date=START, completion_criteria="Done already", category="academics", status="done"),
    MilestoneIn(id=2, title="Algebra baseline", due_date=FRI, completion_criteria="12 problems marked", category="academics"),
    MilestoneIn(id=3, title="Robot chassis", due_date=SUN, completion_criteria="Wheels turn", category="technical"),
    MilestoneIn(id=4, title="Later one", due_date=SUN, completion_criteria="Later", category="technical"),
]


def test_plan_week_retries_on_bad_milestone_and_places_tasks(monkeypatch):
    bad = TaskDraftList(tasks=[task(30, milestone_id=99)])
    good = TaskDraftList(tasks=[task(40, "Algebra baseline quiz", milestone_id=2), task(45, "Build chassis", "technical", milestone_id=3)])
    fake = FakeLLM(bad, good)
    monkeypatch.setattr(schedule_agent, "get_llm", lambda *args, **kwargs: fake)

    plan = schedule_agent.plan_week(demo_profile(), "Maths + robot", MILESTONES, START, today=START)
    assert [t.title for _, t in plan.placed] == ["Algebra baseline quiz", "Build chassis"]
    assert "milestone_id must be one of [2, 3]" in fake.calls[1][-1].content  # only the 2 current, unfinished ones
    assert "Aarav" not in fake.calls[0][1].content


def test_check_drafts_enforces_required_budget():
    cap = {MON: 100}  # budget = 80
    assert "budget is 80" in schedule_agent.check_drafts([task(90)], cap, set())[0]
    assert schedule_agent.check_drafts([task(60), task(30, is_optional=True)], cap, set()) == []


def test_no_free_time_means_no_ai_call(monkeypatch):
    monkeypatch.setattr(schedule_agent, "get_llm", lambda *args, **kwargs: FakeLLM())  # would fail if called
    plan = schedule_agent.plan_week(demo_profile(), "Goal", MILESTONES, START, today=START + timedelta(days=7))
    assert plan.placed == [] and sum(plan.capacity.values()) == 0
