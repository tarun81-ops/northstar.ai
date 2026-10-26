"""Reading and saving a week's tasks."""
from collections import Counter
from datetime import date

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from northstar.db.models import Task
from northstar.schemas import TaskDraft
from northstar.services.scheduler import week_days


def get_week_tasks(session: Session, student_id: int, start: date) -> list[Task]:
    return list(session.scalars(
        select(Task).where(Task.student_id == student_id, Task.week_start == start).order_by(Task.scheduled_date, Task.id)
    ))


def busy_minutes(tasks: list[Task]) -> dict[date, int]:
    """Minutes per day already used by finished or partly finished tasks (so re-planning doesn't double-book)."""
    busy = Counter()
    for t in tasks:
        if t.status in ("done", "partial"):
            busy[t.scheduled_date] += t.duration_minutes
    return dict(busy)


def save_week(session: Session, student_id: int, start: date, placed: list[tuple[date, TaskDraft]]) -> None:
    """Replace the week's unstarted ("todo") tasks with `placed`. Done, partial and skipped tasks are history: kept."""
    days = set(week_days(start))
    outside = [t.title for d, t in placed if d not in days]
    if outside:
        raise ValueError(f"Tasks scheduled outside the week of {start}: {', '.join(outside)}")

    session.execute(delete(Task).where(Task.student_id == student_id, Task.week_start == start, Task.status == "todo"))
    session.add_all(
        Task(student_id=student_id, milestone_id=t.milestone_id, title=t.title, category=t.category, week_start=start,
             scheduled_date=d, duration_minutes=t.duration_minutes, difficulty=t.difficulty, is_optional=t.is_optional)
        for d, t in placed
    )
    session.commit()
