"""Goals and milestones. MVP rule: one active goal per student."""
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from northstar.db.models import Goal, Milestone
from northstar.schemas import GoalIn, MilestoneDraft, MilestoneIn

MIN_MILESTONES, MAX_MILESTONES = 4, 8


class ActiveGoalExists(Exception):
    pass


def check_milestones(milestones: list[MilestoneDraft], today: date, target_date: date) -> list[str]:
    """Problems with AI-generated milestones, in plain words (empty list = OK). Python checks dates, not the LLM."""
    problems = []
    if not MIN_MILESTONES <= len(milestones) <= MAX_MILESTONES:
        problems.append(f"need {MIN_MILESTONES}-{MAX_MILESTONES} milestones, got {len(milestones)}")
    dates = [m.due_date for m in milestones]
    if dates != sorted(dates):
        problems.append("due dates must be in ascending order")
    if dates and dates[0] <= today:
        problems.append(f"the first due date must be after today ({today})")
    if dates and dates[-1] > target_date:
        problems.append(f"the last due date must be on or before the goal's target date ({target_date})")
    return problems


def get_active_goal(session: Session, student_id: int) -> Goal | None:
    return session.scalar(
        select(Goal).where(Goal.student_id == student_id, Goal.status == "active").options(selectinload(Goal.milestones))
    )


def create_goal(session: Session, student_id: int, goal: GoalIn) -> int:
    if get_active_goal(session, student_id):
        raise ActiveGoalExists("Archive the current goal before starting a new one.")
    row = Goal(student_id=student_id, **goal.model_dump())
    session.add(row)
    session.commit()
    return row.id


def archive_goal(session: Session, goal_id: int) -> None:
    session.get(Goal, goal_id).status = "archived"
    session.commit()


def save_milestones(session: Session, goal_id: int, milestones: list[MilestoneIn]) -> None:
    """Make the goal's milestones exactly `milestones`: update by id, add new ones, delete the ones left out.
    Updating in place keeps ids stable, so tasks linked to a milestone stay linked."""
    goal = session.get(Goal, goal_id)
    late = [m.title for m in milestones if m.due_date > goal.target_date]
    if late:
        raise ValueError(f"Due after the goal's target date ({goal.target_date}): {', '.join(late)}")
    existing = {m.id: m for m in goal.milestones}
    unknown = {m.id for m in milestones if m.id} - existing.keys()
    if unknown:
        raise ValueError(f"Milestone ids {unknown} don't belong to this goal")

    rows = []
    for i, m in enumerate(sorted(milestones, key=lambda m: m.due_date)):
        row = existing.get(m.id) or Milestone()
        for field, value in m.model_dump(exclude={"id"}).items():
            setattr(row, field, value)
        row.order_index = i
        rows.append(row)
    goal.milestones = rows  # rows no longer in the list are deleted (cascade="delete-orphan")
    session.commit()
