"""Pydantic schemas: the single definition of 'valid data', shared by forms, LLM tools and tests."""
from datetime import date
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, BaseModel, Field, StringConstraints, field_validator

Day = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
DAYS: tuple[Day, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60)]
# "08:00-15:00", or empty when there is no school that day
SchoolHours = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^$|^([01]\d|2[0-3]):[0-5]\d-([01]\d|2[0-3]):[0-5]\d$")]


class DayAvailability(BaseModel):
    school: SchoolHours = ""
    free_minutes: int = Field(0, ge=0, le=720)


class Activity(BaseModel):
    name: ShortText
    day: Day
    minutes: int = Field(ge=0, le=600)


class Exam(BaseModel):
    subject: ShortText
    date: date


class ProfileIn(BaseModel):
    alias: ShortText = Field(description="Nickname - never a full legal name")
    age_band: Literal["11-13", "14-16", "17-19", "college"]
    grade: ShortText
    language: ShortText = "English"
    subjects: list[ShortText] = Field(default_factory=list, max_length=20)
    interests: list[ShortText] = Field(default_factory=list, max_length=20)
    strengths: list[ShortText] = Field(default_factory=list, max_length=20)
    challenges: list[ShortText] = Field(default_factory=list, max_length=20)
    learning_preferences: list[ShortText] = Field(default_factory=list, max_length=10)
    weekly_availability: dict[Day, DayAvailability] = Field(
        default_factory=lambda: {d: DayAvailability() for d in DAYS}
    )
    activities: list[Activity] = Field(default_factory=list, max_length=20)
    exams: list[Exam] = Field(default_factory=list, max_length=20)


Category = Literal["academics", "technical", "communication", "creativity", "wellbeing", "interests", "research"]
CATEGORIES: tuple[str, ...] = Category.__args__
LongText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=1000)]


class GoalIn(BaseModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=200)]
    description: LongText = ""
    category: Category
    target_date: date

    @field_validator("target_date")
    @classmethod
    def in_the_future(cls, v: date) -> date:
        if v <= date.today():
            raise ValueError("target date must be in the future")
        return v


class MilestoneDraft(BaseModel):
    """What the Roadmap Agent produces. Counts and dates are checked in Python (services/goals.py)."""

    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=200)]
    description: LongText = ""
    due_date: date
    completion_criteria: Annotated[str, StringConstraints(strip_whitespace=True, min_length=5, max_length=1000)] = Field(
        description="Something observable, e.g. 'scores 8/10 on a past-paper section' or 'robot follows a line for 2 m'"
    )
    category: Category


class MilestoneList(BaseModel):
    milestones: list[MilestoneDraft]


class TaskDraft(BaseModel):
    """A task the Schedule Agent proposes. It does NOT pick the day - services/scheduler.py does."""

    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=200)] = Field(
        description="Concrete verb first, e.g. 'Solve 8 factorising problems'"
    )
    category: Category
    duration_minutes: int = Field(ge=10, le=90)
    difficulty: int = Field(ge=1, le=5)
    is_optional: bool = False
    milestone_id: int | None = Field(None, description="Id of the milestone this task moves forward, if any")
    why: LongText = Field("", description="One line: why this task, this week")


class TaskDraftList(BaseModel):
    tasks: list[TaskDraft]


Rating = Annotated[int, Field(ge=1, le=5)]


class CheckIn(BaseModel):
    """One task's check-in. Only `completed` is required - everything else is optional to keep it under a minute."""

    task_id: int
    completed: Literal["yes", "partial", "no"]
    time_spent_minutes: int = Field(0, ge=0, le=600)
    difficulty_rating: Literal["too_easy", "just_right", "too_hard"] | None = None
    enjoyment: Rating | None = None
    stress: Rating | None = None
    confidence: Rating | None = None
    student_notes: LongText = ""
    blockers: LongText = ""
    evidence_url: AnyHttpUrl | None = None  # a link the student chooses to share (photo, repo, doc)


ChangeKind = Literal["lighten_load", "shorter_tasks", "easier_steps", "harder_challenge", "deeper_project"]


class ProposedChange(BaseModel):
    """A change to how future weeks are planned. The menu of kinds is fixed: the AI can't invent new powers."""

    kind: ChangeKind
    category: Category | None = Field(None, description="Required for easier_steps, harder_challenge and deeper_project")
    amount: int | None = Field(None, description="lighten_load: percent 10-40. shorter_tasks: max minutes per task 15-60")
    reason: LongText = Field(description="One sentence, based on the data, kind")


class WeeklyReviewOut(BaseModel):
    win: LongText = Field(description="One genuine win from the week")
    what_the_data_shows: LongText = Field(description="2-3 sentences, no blame")
    changes: list[ProposedChange] = Field(description="At most 3. Empty if the week went well and nothing needs changing")
    question: LongText = Field(description="One reflective question for the student")


class MilestoneIn(MilestoneDraft):
    """A milestone the student edits/confirms. id=None means new."""

    id: int | None = None
    status: Literal["todo", "in_progress", "done"] = "todo"
