"""Daily check-ins: write progress logs and keep each task's status in step with them."""
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from northstar.db.models import ProgressLog, Task
from northstar.schemas import CheckIn

STATUS_FOR = {"yes": "done", "partial": "partial", "no": "skipped"}


def tasks_for_day(session: Session, student_id: int, day: date) -> list[Task]:
    return list(session.scalars(
        select(Task).where(Task.student_id == student_id, Task.scheduled_date == day).order_by(Task.is_optional, Task.id)
    ))


def logs_for_day(session: Session, student_id: int, day: date) -> dict[int, ProgressLog]:
    """Existing check-ins for a day, keyed by task id (to pre-fill the form when editing)."""
    logs = session.scalars(select(ProgressLog).where(ProgressLog.student_id == student_id, ProgressLog.date == day))
    return {log.task_id: log for log in logs}


def _own_task(session: Session, student_id: int, task_id: int) -> Task:
    task = session.get(Task, task_id)
    if task is None or task.student_id != student_id:
        raise ValueError(f"Task {task_id} is not this student's task")
    return task


def log_checkin(session: Session, student_id: int, day: date, checkin: CheckIn) -> None:
    """Save a check-in. Checking in again for the same task and day updates it instead of adding a duplicate."""
    task = _own_task(session, student_id, checkin.task_id)
    log = session.scalar(select(ProgressLog).where(
        ProgressLog.student_id == student_id, ProgressLog.task_id == task.id, ProgressLog.date == day
    )) or ProgressLog(student_id=student_id, task_id=task.id, date=day)
    for field, value in checkin.model_dump(exclude={"task_id"}).items():
        setattr(log, field, str(value) if field == "evidence_url" and value else value)
    task.status = STATUS_FOR[checkin.completed]
    session.add(log)
    session.commit()


def undo_checkin(session: Session, student_id: int, task_id: int, day: date) -> None:
    """Un-tick: remove that day's check-in and put the task back to 'todo'."""
    task = _own_task(session, student_id, task_id)
    for log in session.scalars(select(ProgressLog).where(
        ProgressLog.student_id == student_id, ProgressLog.task_id == task_id, ProgressLog.date == day
    )):
        session.delete(log)
    task.status = "todo"
    session.commit()


def encouragement(checkins: list[CheckIn]) -> str:
    """A short, never-guilt-tripping reply after a check-in. Rule-based on purpose: free, instant, predictable."""
    if not checkins:
        return ""
    total = len(checkins)
    done = sum(c.completed == "yes" for c in checkins)
    progressed = sum(c.completed in ("yes", "partial") for c in checkins)

    if done == total:
        lines = ["You did everything you planned today - great work! 🎉"]
    elif progressed:
        lines = [f"You made progress on {progressed} of {total} tasks. Every step counts."]
    else:
        lines = ["Thanks for checking in. Some days don't go to plan - that's normal, and it's useful to know."]
    if any(c.blockers for c in checkins):
        lines.append("Thanks for saying what got in the way; NorthStar will plan around it.")
    if any((c.stress or 0) >= 4 for c in checkins):
        lines.append("It sounds like a heavy day. Rest is part of the plan too 🌙")
    elif any(c.difficulty_rating == "too_hard" for c in checkins):
        lines.append("Hard tasks will be split into smaller steps.")
    return " ".join(lines)
