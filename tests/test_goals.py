from datetime import date, timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from northstar.agents import roadmap_agent
from northstar.db.seed import demo_profile
from northstar.db.session import init_db, make_engine
from northstar.schemas import GoalIn, MilestoneDraft, MilestoneIn, MilestoneList
from northstar.services.goals import (
    ActiveGoalExists, archive_goal, check_milestones, create_goal, get_active_goal, save_milestones,
)
from northstar.services.profile import save_profile

TODAY = date.today()
GOAL = GoalIn(title="Score 85% in maths", category="academics", target_date=TODAY + timedelta(days=60))


def ms(days: int, title: str = "Step", **kw) -> dict:
    return dict(title=title, due_date=TODAY + timedelta(days=days), completion_criteria="Scores 8/10 on a quiz",
                category="academics", **kw)


@pytest.fixture
def session():
    engine = make_engine("sqlite://")
    init_db(engine)
    with Session(engine) as s:
        yield s


@pytest.fixture
def goal_id(session):
    return create_goal(session, save_profile(session, demo_profile()), GOAL)


def test_check_milestones():
    good = [MilestoneDraft(**ms(d)) for d in (7, 20, 35, 50)]
    assert check_milestones(good, TODAY, GOAL.target_date) == []

    assert "need 4-8" in check_milestones(good[:3], TODAY, GOAL.target_date)[0]
    assert "ascending" in check_milestones(good[::-1], TODAY, GOAL.target_date)[0]
    assert "after today" in check_milestones([MilestoneDraft(**ms(0))] + good[1:], TODAY, GOAL.target_date)[0]
    assert "target date" in check_milestones(good[:3] + [MilestoneDraft(**ms(90))], TODAY, GOAL.target_date)[0]


def test_goal_target_must_be_in_future():
    with pytest.raises(ValidationError):
        GoalIn(title="Old goal", category="academics", target_date=TODAY)


def test_only_one_active_goal(session, goal_id):
    student_id = get_active_goal(session, 1).student_id
    with pytest.raises(ActiveGoalExists):
        create_goal(session, student_id, GOAL)
    archive_goal(session, goal_id)
    assert create_goal(session, student_id, GOAL) != goal_id


def test_save_milestones_updates_in_place(session, goal_id):
    save_milestones(session, goal_id, [MilestoneIn(**ms(d, f"Step {d}")) for d in (30, 7, 20, 45)])
    first = get_active_goal(session, 1).milestones
    assert [m.title for m in first] == ["Step 7", "Step 20", "Step 30", "Step 45"]  # ordered by due date
    ids = {m.title: m.id for m in first}

    # rename Step 20, drop Step 45, add a new one
    edited = [MilestoneIn(**ms(7, "Step 7"), id=ids["Step 7"]),
              MilestoneIn(**ms(20, "Renamed", status="done"), id=ids["Step 20"]),
              MilestoneIn(**ms(30, "Step 30"), id=ids["Step 30"]), MilestoneIn(**ms(50, "New one"))]
    save_milestones(session, goal_id, edited)
    after = {m.title: (m.id, m.status) for m in get_active_goal(session, 1).milestones}
    assert after["Renamed"] == (ids["Step 20"], "done")  # same row, updated
    assert "Step 45" not in after and "New one" in after
    assert len(after) == 4


def test_save_milestones_rejects_bad_input(session, goal_id):
    with pytest.raises(ValueError, match="target date"):
        save_milestones(session, goal_id, [MilestoneIn(**ms(90))])
    with pytest.raises(ValueError, match="don't belong"):
        save_milestones(session, goal_id, [MilestoneIn(**ms(10), id=999)])


class FakeLLM:
    """Stands in for the real model: returns canned outputs and records what it was sent."""

    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.outputs.pop(0)


def test_generate_retries_once_with_the_problems(monkeypatch):
    bad = MilestoneList(milestones=[MilestoneDraft(**ms(d)) for d in (40, 10, 20, 30)])  # out of order
    good = MilestoneList(milestones=[MilestoneDraft(**ms(d)) for d in (10, 20, 30, 40)])
    fake = FakeLLM(bad, good)
    monkeypatch.setattr(roadmap_agent, "get_llm", lambda: fake)

    assert roadmap_agent.generate_milestones(demo_profile(), GOAL) == good.milestones
    assert "ascending" in fake.calls[1][-1].content  # the fix request told the model what was wrong
    assert "Aarav" not in fake.calls[0][1].content  # nickname is not sent to the LLM


def test_generate_gives_up_after_one_retry(monkeypatch):
    bad = MilestoneList(milestones=[MilestoneDraft(**ms(10))])
    monkeypatch.setattr(roadmap_agent, "get_llm", lambda: FakeLLM(bad, bad))
    with pytest.raises(roadmap_agent.RoadmapError, match="need 4-8"):
        roadmap_agent.generate_milestones(demo_profile(), GOAL)
