from datetime import date

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from northstar.db.models import ProgressLog, Task
from northstar.db.seed import demo_profile, seed_week
from northstar.db.session import init_db, make_engine
from northstar.schemas import CheckIn
from northstar.services.profile import save_profile
from northstar.services.progress import encouragement, log_checkin, logs_for_day, tasks_for_day, undo_checkin

DAY = date(2026, 11, 2)
GUILT_WORDS = ("lazy", "failed", "should have", "disappointed", "behind", "missed")


@pytest.fixture
def session():
    engine = make_engine("sqlite://")
    init_db(engine)
    with Session(engine) as s:
        yield s


@pytest.fixture
def student_and_task(session):
    sid = save_profile(session, demo_profile())
    task = Task(student_id=sid, title="Solve 10 problems", category="academics", week_start=DAY,
                scheduled_date=DAY, duration_minutes=40)
    session.add(task)
    session.commit()
    return sid, task


def test_checkin_sets_status_and_updates_instead_of_duplicating(session, student_and_task):
    sid, task = student_and_task
    log_checkin(session, sid, DAY, CheckIn(task_id=task.id, completed="partial", time_spent_minutes=20))
    assert task.status == "partial"

    log_checkin(session, sid, DAY, CheckIn(task_id=task.id, completed="yes", time_spent_minutes=45, enjoyment=4,
                                           evidence_url="https://github.com/aarav/robot"))
    logs = list(session.scalars(select(ProgressLog)))
    assert len(logs) == 1
    assert (logs[0].completed, logs[0].time_spent_minutes, logs[0].evidence_url) == ("yes", 45, "https://github.com/aarav/robot")
    assert task.status == "done"
    assert list(logs_for_day(session, sid, DAY)) == [task.id]
    assert tasks_for_day(session, sid, DAY) == [task]


def test_no_maps_to_skipped_and_undo_restores_todo(session, student_and_task):
    sid, task = student_and_task
    log_checkin(session, sid, DAY, CheckIn(task_id=task.id, completed="no", blockers="Too tired"))
    assert task.status == "skipped"
    undo_checkin(session, sid, task.id, DAY)
    assert task.status == "todo" and logs_for_day(session, sid, DAY) == {}


def test_cannot_check_in_someone_elses_task(session, student_and_task):
    _, task = student_and_task
    other = save_profile(session, demo_profile().model_copy(update={"alias": "Meera"}))
    with pytest.raises(ValueError, match="not this student's task"):
        log_checkin(session, other, DAY, CheckIn(task_id=task.id, completed="yes"))


@pytest.mark.parametrize("bad", [{"stress": 6}, {"enjoyment": 0}, {"evidence_url": "not a link"},
                                 {"time_spent_minutes": -5}, {"completed": "maybe"}])
def test_checkin_validation(bad):
    with pytest.raises(ValidationError):
        CheckIn(**{"task_id": 1, "completed": "yes"} | bad)


@pytest.mark.parametrize("checkins", [
    [CheckIn(task_id=1, completed="yes")],
    [CheckIn(task_id=1, completed="yes"), CheckIn(task_id=2, completed="no", blockers="Too tired")],
    [CheckIn(task_id=1, completed="no"), CheckIn(task_id=2, completed="no", stress=5)],
    [CheckIn(task_id=1, completed="partial", difficulty_rating="too_hard")],
])
def test_encouragement_is_never_guilt_tripping(checkins):
    message = encouragement(checkins).lower()
    assert message and not any(word in message for word in GUILT_WORDS)


def test_encouragement_reacts_to_the_day():
    assert "everything" in encouragement([CheckIn(task_id=1, completed="yes")])
    assert "1 of 2" in encouragement([CheckIn(task_id=1, completed="partial"), CheckIn(task_id=2, completed="no")])
    assert "Rest" in encouragement([CheckIn(task_id=1, completed="yes", stress=4)])
    assert encouragement([]) == ""


@pytest.mark.parametrize("kind, statuses", [
    ("good", {"done"}),
    ("bad", {"done", "skipped"}),
    ("stressed", {"done", "partial"}),
])
def test_seed_week_shapes(session, kind, statuses):
    sid = save_profile(session, demo_profile())
    seed_week(session, sid, DAY, kind)
    seed_week(session, sid, DAY, kind)  # re-seeding replaces, never duplicates
    tasks = list(session.scalars(select(Task).where(Task.student_id == sid)))
    logs = list(session.scalars(select(ProgressLog).where(ProgressLog.student_id == sid)))
    assert len(tasks) == len(logs) == 7
    assert {t.status for t in tasks} == statuses
    if kind == "stressed":
        assert all(log.stress >= 4 for log in logs)
