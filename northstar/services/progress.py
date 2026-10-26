"""Daily check-ins: write progress logs and keep each task's status in step with them.
Also summarises a week of check-ins (plain numbers the Progress Agent and pattern detection read)."""
from collections import Counter
from datetime import date
from statistics import mean

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


def _avg(values: list[int]) -> float | None:
    return round(mean(values), 1) if values else None


def summarize(tasks: list[Task], logs: list[ProgressLog]) -> dict:
    """One week in numbers. Pure function: easy to test and safe to show the LLM (no names, no free-form IDs)."""
    log_for = {log.task_id: log for log in logs}
    status = Counter(t.status for t in tasks)
    checked = [t for t in tasks if t.id in log_for]
    ratings = [log.difficulty_rating for log in logs if log.difficulty_rating]

    by_category = {}
    for t in tasks:
        c = by_category.setdefault(t.category, {"planned": 0, "done": 0, "partial": 0, "skipped": 0, "enjoyment": []})
        c["planned"] += 1
        if t.status in c:
            c[t.status] += 1
        if t.id in log_for and log_for[t.id].enjoyment:
            c["enjoyment"].append(log_for[t.id].enjoyment)

    planned = len(tasks)
    return {
        "tasks_planned": planned,
        "done": status["done"], "partial": status["partial"], "skipped": status["skipped"], "not_checked_in": status["todo"],
        "completion_rate": round((status["done"] + 0.5 * status["partial"]) / planned, 2) if planned else 0.0,
        "planned_minutes": sum(t.duration_minutes for t in tasks),
        "planned_minutes_checked_in": sum(t.duration_minutes for t in checked),
        "actual_minutes": sum(log.time_spent_minutes for log in logs),
        "difficulty": dict(Counter(ratings)),
        "struggled": sum(log.difficulty_rating == "too_hard" or log.completed == "partial" for log in logs),
        "avg_stress": _avg([log.stress for log in logs if log.stress]),
        "avg_enjoyment": _avg([log.enjoyment for log in logs if log.enjoyment]),
        "enjoyment_by_day": [log.enjoyment for log in sorted(logs, key=lambda l: l.date) if log.enjoyment],
        "by_category": {cat: {k: v for k, v in c.items() if k != "enjoyment"} | {"avg_enjoyment": _avg(c["enjoyment"]),
                                                                             "rated": len(c["enjoyment"])}
                        for cat, c in by_category.items()},
        "blockers": [log.blockers for log in logs if log.blockers],
    }


def week_summary(session: Session, student_id: int, start: date) -> dict:
    tasks = list(session.scalars(select(Task).where(Task.student_id == student_id, Task.week_start == start)))
    logs = list(session.scalars(select(ProgressLog).where(ProgressLog.task_id.in_([t.id for t in tasks])))) if tasks else []
    return summarize(tasks, logs)


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
