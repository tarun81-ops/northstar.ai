"""Pydantic schemas: the single definition of 'valid data', shared by forms, LLM tools and tests."""
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

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
