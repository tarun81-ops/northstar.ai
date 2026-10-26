from datetime import date, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from northstar import config
from northstar.db import seed
from northstar.db.models import Milestone, ProgressLog, Student, Task
from northstar.db.session import make_engine
from northstar.llm import ai_problem
from northstar.schemas import TaskDraft
from northstar.services.scheduler import place, week_start

MON = date(2026, 11, 2)
TUE = MON + timedelta(days=1)


def test_ai_problem_names_the_missing_key(monkeypatch):
    monkeypatch.setattr(config, "MODEL", "openai:gpt-6-luna")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert "OPENAI_API_KEY" in ai_problem()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert ai_problem() is None

    monkeypatch.setattr(config, "MODEL", "google_genai:gemini-2.5-flash")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert "GOOGLE_API_KEY" in ai_problem()


def test_later_milestone_tasks_never_come_before_earlier_ones():
    # The agent lists the second milestone's task first; the scheduler still keeps milestone order.
    t = lambda title, minutes, m: TaskDraft(title=title, category="academics", duration_minutes=minutes, difficulty=2, milestone_id=m)
    drafts = [t("Next milestone task", 30, 2), t("Baseline part 1", 70, 1), t("Baseline part 2", 20, 1)]
    placed, _ = place(drafts, {MON: 100, TUE: 100}, milestone_rank={1: 0, 2: 1})
    day = {task.title: d for d, task in placed}
    assert day["Next milestone task"] >= max(day["Baseline part 1"], day["Baseline part 2"])


def test_reset_demo(monkeypatch):
    engine = make_engine("sqlite://")
    monkeypatch.setattr(seed, "engine", engine)  # never touch the real data/northstar.db
    monkeypatch.setattr(seed, "get_session", sessionmaker(engine))
    count = lambda model, *where: sessionmaker(engine)().scalar(select(func.count()).select_from(model).where(*where))

    seed.reset_demo(full=True)
    last_monday = week_start(date.today()) - timedelta(days=7)
    assert count(Student) == 1 and count(Milestone) == len(seed.DEMO_MILESTONES)
    assert count(Task, Task.week_start == last_monday) == count(ProgressLog) == 7  # a checked-in past week, ready to review
    assert count(Task, Task.week_start == week_start(date.today())) >= 1  # this week's plan

    seed.reset_demo(full=False)
    assert count(Student) == count(Task) == 0
