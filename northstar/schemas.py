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


class ExperimentDraft(BaseModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=200)]
    duration_days: int = Field(description="3 to 14 days")
    steps: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=300)]] = Field(
        description="3 to 6 small, concrete steps"
    )


class NicheDraft(BaseModel):
    """A niche the Niche Mentor suggests. matched_signal_ids must be real signal ids - Python checks."""

    niche_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=100)]
    description: LongText = Field(description="2 sentences: what this niche is about")
    why: LongText = Field(description="1-2 sentences linking it to the student's signals")
    matched_signal_ids: list[int] = Field(description="Ids of the signals this niche builds on (at least 1)")
    skills_required: list[ShortText] = Field(default_factory=list, max_length=5)
    experiment: ExperimentDraft


class NicheList(BaseModel):
    niches: list[NicheDraft]


class Reflection(BaseModel):
    curiosity: Rating = Field(description="How curious did it make you?")
    energy: Rating = Field(description="Did it give you energy or drain you?")
    confidence: Rating = Field(description="How confident do you feel in it now?")
    meaning: Rating = Field(description="Does it feel meaningful to you?")
    notes: LongText = ""


class SourceIn(BaseModel):
    """A source the student adds themselves."""

    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=200)]
    url: AnyHttpUrl
    publisher: Annotated[str, StringConstraints(strip_whitespace=True, max_length=120)] = ""
    note: LongText = ""


class CoachReply(BaseModel):
    """The Research Coach's guidance for one step. It asks; it never does the work or invents sources."""

    feedback: LongText = Field(description="1-3 sentences on the student's notes so far: what's good, what's missing")
    questions: list[LongText] = Field(description="1-3 questions that help the student think the next part through")
    next_action: LongText = Field(description="One small concrete thing to do next")
    ethics_flags: list[LongText] = Field(default_factory=list, description="Any safety or ethics concerns, else empty")


class MilestoneIn(MilestoneDraft):
    """A milestone the student edits/confirms. id=None means new."""

    id: int | None = None
    status: Literal["todo", "in_progress", "done"] = "todo"
