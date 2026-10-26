"""Weekly reviews and the plan changes they propose.

The student stays in control: the AI can only PROPOSE changes (status "proposed"). decide_change() is the
only way to apply one, and only the student's Accept button calls it - it is deliberately not an LLM tool.
Accepted changes become rules for the next weekly plan (see Adjustments), enforced in Python.
"""
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from northstar.db.models import PlanChange, WeeklyReview
from northstar.schemas import WeeklyReviewOut
from northstar.services.patterns import Pattern

MAX_CHANGES = 3
AMOUNT_RANGE = {"lighten_load": (10, 40), "shorter_tasks": (15, 60)}
NEEDS_CATEGORY = {"easier_steps", "harder_challenge", "deeper_project"}


def check_review(review: WeeklyReviewOut) -> list[str]:
    """Problems with an AI review, in plain words (empty = OK)."""
    problems = []
    if len(review.changes) > MAX_CHANGES:
        problems.append(f"propose at most {MAX_CHANGES} changes, not {len(review.changes)}")
    for c in review.changes:
        if c.kind in AMOUNT_RANGE:
            low, high = AMOUNT_RANGE[c.kind]
            if c.amount is None or not low <= c.amount <= high:
                problems.append(f"{c.kind} needs an amount between {low} and {high}")
        if c.kind in NEEDS_CATEGORY and c.category is None:
            problems.append(f"{c.kind} needs a category")
    return problems


def describe(kind: str, payload: dict) -> str:
    """A plan change in student-friendly words."""
    cat, amount = payload.get("category"), payload.get("amount")
    return {
        "lighten_load": f"Plan {amount}% less work",
        "shorter_tasks": f"Keep tasks to {amount} minutes or less",
        "easier_steps": f"Break {cat} into smaller, easier steps",
        "harder_challenge": f"Make {cat} tasks a bit more challenging",
        "deeper_project": f"Add an optional deeper project in {cat}",
    }[kind]


def get_review(session: Session, student_id: int, start: date) -> tuple[WeeklyReview | None, list[PlanChange]]:
    review = session.scalar(select(WeeklyReview).where(WeeklyReview.student_id == student_id, WeeklyReview.week_start == start))
    if review is None:
        return None, []
    return review, list(session.scalars(select(PlanChange).where(PlanChange.review_id == review.id).order_by(PlanChange.id)))


def latest_review(session: Session, student_id: int) -> WeeklyReview | None:
    return session.scalar(select(WeeklyReview).where(WeeklyReview.student_id == student_id)
                          .order_by(WeeklyReview.week_start.desc()).limit(1))


def save_review(session: Session, student_id: int, start: date, review: WeeklyReviewOut, patterns: list[Pattern]) -> int:
    """Store the review and its proposed changes. Re-running a week's review replaces its *proposed* changes only:
    changes the student already accepted or dismissed are kept."""
    row, _ = get_review(session, student_id, start)
    row = row or WeeklyReview(student_id=student_id, week_start=start)
    row.summary = review.model_dump_json(exclude={"changes"})
    row.patterns = [p.__dict__ for p in patterns]
    session.add(row)
    session.flush()
    session.execute(delete(PlanChange).where(PlanChange.review_id == row.id, PlanChange.status == "proposed"))
    session.add_all(
        PlanChange(student_id=student_id, review_id=row.id, kind=c.kind, reason=c.reason,
                   payload={"category": c.category, "amount": c.amount})
        for c in review.changes
    )
    session.commit()
    return row.id


def decide_change(session: Session, student_id: int, change_id: int, accept: bool) -> None:
    """The student's Accept / Dismiss. Only proposed changes belonging to this student can be decided."""
    change = session.get(PlanChange, change_id)
    if change is None or change.student_id != student_id:
        raise ValueError(f"Change {change_id} is not this student's")
    if change.status != "proposed":
        raise ValueError(f"Change {change_id} was already {change.status}")
    change.status = "applied" if accept else "dismissed"
    session.commit()


@dataclass
class Adjustments:
    """Accepted changes, turned into rules the weekly planner must follow."""

    load_factor: float = 1.0            # multiply each day's capacity
    max_minutes: int | None = None      # longest allowed task
    easier: set[str] = field(default_factory=set)   # categories capped at difficulty 2
    notes: list[str] = field(default_factory=list)  # all accepted changes in words, for the Schedule Agent


EASIER_MAX_DIFFICULTY = 2


def adjustments_for_week(session: Session, student_id: int, start: date) -> Adjustments:
    """Rules for planning the week starting `start`: the changes accepted in the latest review of an earlier week."""
    latest = session.scalar(select(WeeklyReview).where(WeeklyReview.student_id == student_id, WeeklyReview.week_start < start)
                            .order_by(WeeklyReview.week_start.desc()).limit(1))
    adjust = Adjustments()
    if latest is None:
        return adjust
    for c in session.scalars(select(PlanChange).where(PlanChange.review_id == latest.id, PlanChange.status == "applied")):
        amount, cat = c.payload.get("amount"), c.payload.get("category")
        if c.kind == "lighten_load":
            adjust.load_factor = min(adjust.load_factor, 1 - amount / 100)
        elif c.kind == "shorter_tasks":
            adjust.max_minutes = min(adjust.max_minutes or amount, amount)
        elif c.kind == "easier_steps":
            adjust.easier.add(cat)
        adjust.notes.append(describe(c.kind, c.payload))
    return adjust
