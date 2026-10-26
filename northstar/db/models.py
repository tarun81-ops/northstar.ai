"""Database tables for the MVP. The database, not the LLM, is the source of truth.

Allowed values (statuses, ratings) are enforced by the database itself via CHECK constraints,
so bad data from any code path - including an LLM tool call - is rejected.
"""
import datetime as dt
from datetime import date, datetime, time

from sqlalchemy import JSON, CheckConstraint, Enum, ForeignKey, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def choice(*values: str, name: str) -> Enum:
    """A string column limited to `values` (stored as VARCHAR + CHECK, works on SQLite and Postgres)."""
    return Enum(*values, name=name, native_enum=False, create_constraint=True)


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(primary_key=True)
    alias: Mapped[str] = mapped_column(String(60))  # a nickname, never a legal name
    age_band: Mapped[str] = mapped_column(choice("11-13", "14-16", "17-19", "college", name="age_band"))
    grade: Mapped[str] = mapped_column(String(40))
    language: Mapped[str] = mapped_column(String(40), default="English")
    subjects: Mapped[list] = mapped_column(JSON, default=list)
    interests: Mapped[list] = mapped_column(JSON, default=list)
    strengths: Mapped[list] = mapped_column(JSON, default=list)
    challenges: Mapped[list] = mapped_column(JSON, default=list)
    learning_preferences: Mapped[list] = mapped_column(JSON, default=list)
    # {"mon": {"school": "08:00-15:00", "free_minutes": 120}, ...} - what the scheduler reads
    weekly_availability: Mapped[dict] = mapped_column(JSON, default=dict)
    activities: Mapped[list] = mapped_column(JSON, default=list)  # [{"name", "day", "minutes"}]
    exams: Mapped[list] = mapped_column(JSON, default=list)  # [{"subject", "date"}]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    goals: Mapped[list["Goal"]] = relationship(back_populates="student", cascade="all, delete-orphan")


class Goal(Base):
    __tablename__ = "goals"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    category: Mapped[str] = mapped_column(String(40))
    target_date: Mapped[date]
    status: Mapped[str] = mapped_column(
        choice("active", "completed", "paused", "archived", name="goal_status"), default="active"
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    student: Mapped[Student] = relationship(back_populates="goals")
    milestones: Mapped[list["Milestone"]] = relationship(
        back_populates="goal", cascade="all, delete-orphan", order_by="Milestone.order_index"
    )


class Milestone(Base):
    __tablename__ = "milestones"

    id: Mapped[int] = mapped_column(primary_key=True)
    goal_id: Mapped[int] = mapped_column(ForeignKey("goals.id", ondelete="CASCADE"), index=True)
    order_index: Mapped[int]
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    due_date: Mapped[date]
    completion_criteria: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(choice("todo", "in_progress", "done", name="milestone_status"), default="todo")

    goal: Mapped[Goal] = relationship(back_populates="milestones")


class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        CheckConstraint("duration_minutes BETWEEN 5 AND 240", name="task_duration_range"),
        CheckConstraint("difficulty BETWEEN 1 AND 5", name="task_difficulty_range"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id", ondelete="CASCADE"), index=True)
    milestone_id: Mapped[int | None] = mapped_column(ForeignKey("milestones.id", ondelete="SET NULL"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(40))
    week_start: Mapped[date] = mapped_column(index=True)  # Monday; lets us replace one week cleanly
    scheduled_date: Mapped[date] = mapped_column(index=True)
    start_time: Mapped[time | None]
    duration_minutes: Mapped[int]
    difficulty: Mapped[int] = mapped_column(default=2)
    is_optional: Mapped[bool] = mapped_column(default=False)
    status: Mapped[str] = mapped_column(choice("todo", "done", "partial", "skipped", name="task_status"), default="todo")


class ProgressLog(Base):
    __tablename__ = "progress_logs"
    __table_args__ = (
        CheckConstraint("time_spent_minutes >= 0", name="log_time_nonnegative"),
        CheckConstraint("enjoyment BETWEEN 1 AND 5", name="log_enjoyment_range"),
        CheckConstraint("stress BETWEEN 1 AND 5", name="log_stress_range"),
        CheckConstraint("confidence BETWEEN 1 AND 5", name="log_confidence_range"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id", ondelete="CASCADE"), index=True)
    task_id: Mapped[int | None] = mapped_column(ForeignKey("tasks.id", ondelete="SET NULL"), index=True)
    # dt.date, not date: the column is itself called `date`, which would shadow the type here.
    date: Mapped[dt.date] = mapped_column(index=True)
    completed: Mapped[str] = mapped_column(choice("yes", "partial", "no", name="log_completed"))
    time_spent_minutes: Mapped[int] = mapped_column(default=0)
    difficulty_rating: Mapped[str | None] = mapped_column(
        choice("too_easy", "just_right", "too_hard", name="log_difficulty")
    )
    enjoyment: Mapped[int | None]
    stress: Mapped[int | None]
    confidence: Mapped[int | None]
    student_notes: Mapped[str] = mapped_column(Text, default="")
    evidence_url: Mapped[str | None] = mapped_column(String(500))
    blockers: Mapped[str] = mapped_column(Text, default="")


class WeeklyReview(Base):
    __tablename__ = "weekly_reviews"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id", ondelete="CASCADE"), index=True)
    week_start: Mapped[date]
    summary: Mapped[str] = mapped_column(Text)
    patterns: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class PlanChange(Base):
    """A change the AI *proposes*. Only the student's Accept button applies it."""

    __tablename__ = "plan_changes"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id", ondelete="CASCADE"), index=True)
    review_id: Mapped[int | None] = mapped_column(ForeignKey("weekly_reviews.id", ondelete="SET NULL"))
    kind: Mapped[str] = mapped_column(String(40))
    reason: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(
        choice("proposed", "applied", "dismissed", name="plan_change_status"), default="proposed"
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


# ---- Phase 9: niche discovery ---------------------------------------------------------

class InterestSignal(Base):
    """Evidence of what the student is curious about. Niche suggestions must cite these by id."""

    __tablename__ = "interest_signals"
    __table_args__ = (CheckConstraint("strength BETWEEN 1 AND 5", name="signal_strength_range"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id", ondelete="CASCADE"), index=True)
    signal: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(choice("profile", "checkins", "student", "experiment", name="signal_source"))
    strength: Mapped[int] = mapped_column(default=3)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class NicheSuggestion(Base):
    __tablename__ = "niche_suggestions"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id", ondelete="CASCADE"), index=True)
    niche_name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text)
    why: Mapped[str] = mapped_column(Text)
    matched_signals: Mapped[list] = mapped_column(JSON, default=list)  # [{"id", "signal"}] snapshot of the cited evidence
    skills_required: Mapped[list] = mapped_column(JSON, default=list)
    experiment: Mapped[dict] = mapped_column(JSON, default=dict)  # the curiosity experiment proposed with it
    status: Mapped[str] = mapped_column(
        choice("suggested", "chosen", "dismissed", name="niche_status"), default="suggested"
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class CuriosityExperiment(Base):
    """A short, free, safe try-out of a niche, ending in a reflection the student scores."""

    __tablename__ = "curiosity_experiments"
    __table_args__ = tuple(
        CheckConstraint(f"{s} BETWEEN 1 AND 5", name=f"experiment_{s}_range")
        for s in ("curiosity_score", "energy_score", "confidence_score", "meaning_score")
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id", ondelete="CASCADE"), index=True)
    niche_id: Mapped[int] = mapped_column(ForeignKey("niche_suggestions.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    steps: Mapped[list] = mapped_column(JSON, default=list)  # [{"text": "...", "done": false}]
    duration_days: Mapped[int]
    started_on: Mapped[date]
    status: Mapped[str] = mapped_column(choice("active", "done", "stopped", name="experiment_status"), default="active")
    reflection: Mapped[str] = mapped_column(Text, default="")
    curiosity_score: Mapped[int | None]
    energy_score: Mapped[int | None]
    confidence_score: Mapped[int | None]
    meaning_score: Mapped[int | None]
