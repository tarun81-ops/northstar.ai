"""Tools the Profile Agent can call. They are the ONLY way the LLM touches profile data.

The student id lives in `state`, captured by the closure - the LLM never sees or chooses it.
Answers arrive one chat turn at a time, so until the required fields (nickname, age band, grade)
are known they are kept in `state["draft"]`; after that every update is saved straight to the DB.
"""
from datetime import date

from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError

from northstar.db.session import get_session
from northstar.schemas import DAYS, Activity, Day, DayAvailability, Exam, ProfileIn, ShortText
from northstar.services.profile import get_profile, save_profile

REQUIRED = ("alias", "age_band", "grade")
SHOULD_ASK = ("subjects", "interests", "strengths", "challenges", "learning_preferences",
              "weekly_availability", "activities", "exams")


class ProfileUpdate(BaseModel):
    """Only the fields the student just told you. Lists REPLACE the old list, so include existing items you want to keep."""

    alias: ShortText | None = Field(None, description="Nickname only - never a full name")
    age_band: str | None = Field(None, description='One of "11-13", "14-16", "17-19", "college"')
    grade: ShortText | None = Field(None, description='e.g. "Grade 10", "First-year college"')
    language: ShortText | None = None
    subjects: list[ShortText] | None = None
    interests: list[ShortText] | None = None
    strengths: list[ShortText] | None = None
    challenges: list[ShortText] | None = None
    learning_preferences: list[ShortText] | None = Field(None, description="e.g. Practice problems, Projects, Videos, Reading")
    weekly_availability: dict[Day, DayAvailability] | None = Field(
        None, description='Only the days mentioned, e.g. {"mon": {"school": "08:00-15:00", "free_minutes": 120}}'
    )
    activities: list[Activity] | None = None
    exams: list[Exam] | None = None


def make_profile_tools(student_id: int | None = None, session_factory=get_session):
    """Returns (tools, state). state["student_id"] is filled in once the student is created."""
    state = {"student_id": student_id, "draft": {}}

    def current() -> dict:
        if state["student_id"] is None:
            return dict(state["draft"])
        with session_factory() as s:
            return get_profile(s, state["student_id"]).model_dump(mode="json")

    def missing(data: dict) -> dict:
        week = data.get("weekly_availability") or {}
        return {
            "required": [f for f in REQUIRED if not data.get(f)],
            "not_yet_discussed": [
                f for f in SHOULD_ASK
                if not (any(d.get("free_minutes") or d.get("school") for d in week.values()) if f == "weekly_availability" else data.get(f))
            ],
        }

    @tool
    def get_student_profile() -> dict:
        """Get everything known so far about the student."""
        return {"profile": current(), "saved": state["student_id"] is not None}

    @tool
    def get_missing_profile_fields() -> dict:
        """List profile fields that are still needed. 'required' must be filled; for 'not_yet_discussed',
        ask once - an empty answer (e.g. no exams) is fine."""
        return missing(current()) | {"today": date.today().isoformat()}

    @tool(args_schema=ProfileUpdate)
    def update_student_profile(**updates) -> dict:
        """Save profile facts the student stated in this conversation. Pass only fields they actually said."""
        changes = ProfileUpdate(**updates).model_dump(mode="json", exclude_unset=True, exclude_none=True)
        data = current()
        if week := changes.pop("weekly_availability", None):  # merge per day: updating Monday keeps the other days
            data["weekly_availability"] = (data.get("weekly_availability") or {d: DayAvailability().model_dump() for d in DAYS}) | week
        data |= changes

        try:
            profile = ProfileIn.model_validate(data)
        except ValidationError as e:
            problems = [err for err in e.errors() if not (err["type"] == "missing" and err["loc"][0] in REQUIRED)]
            if problems:  # a real bad value: nothing is saved, the agent should re-ask
                return {"saved": False, "errors": [f"{'.'.join(map(str, p['loc']))}: {p['msg']}" for p in problems]}
            state["draft"] = data  # only required fields missing: keep as draft
            return {"saved": False, "draft_updated": True, "still_missing": missing(data)}

        with session_factory() as s:
            state["student_id"] = save_profile(s, profile, state["student_id"])
        state["draft"] = {}
        return {"saved": True, "still_missing": missing(profile.model_dump(mode="json"))}

    return [get_student_profile, get_missing_profile_fields, update_student_profile], state
