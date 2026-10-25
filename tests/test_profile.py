from datetime import date

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from northstar.db.models import Task
from northstar.db.seed import demo_profile
from northstar.db.session import init_db, make_engine
from northstar.schemas import ProfileIn
from northstar.services.profile import demo_student_id, get_profile, save_profile


@pytest.fixture
def session():
    engine = make_engine("sqlite://")  # in-memory: never touches data/northstar.db
    init_db(engine)
    with Session(engine) as s:
        yield s


def test_create_read_update(session):
    assert demo_student_id(session) is None

    student_id = save_profile(session, demo_profile(date(2026, 10, 26)))
    loaded = get_profile(session, student_id)
    assert loaded.alias == "Aarav"
    assert loaded.weekly_availability["sat"].free_minutes == 240
    assert loaded.exams[0].date == date(2026, 11, 16)  # survives the JSON round trip as a real date
    assert demo_student_id(session) == student_id

    updated = loaded.model_copy(update={"interests": ["Robotics", "AI"]})
    assert save_profile(session, updated, student_id) == student_id
    assert get_profile(session, student_id).interests == ["Robotics", "AI"]


def test_update_unknown_student_fails(session):
    with pytest.raises(ValueError):
        save_profile(session, demo_profile(), student_id=999)


@pytest.mark.parametrize("bad", [
    {"alias": "   "},
    {"age_band": "adult"},
    {"weekly_availability": {"mon": {"school": "8am-3pm", "free_minutes": 60}}},
    {"weekly_availability": {"mon": {"school": "", "free_minutes": -10}}},
    {"activities": [{"name": "Football", "day": "funday", "minutes": 60}]},
])
def test_profile_rejects_bad_input(bad):
    data = demo_profile().model_dump() | bad
    with pytest.raises(ValidationError):
        ProfileIn(**data)


def test_database_enforces_constraints(session):
    student_id = save_profile(session, demo_profile())
    day = date(2026, 10, 26)
    task = dict(student_id=student_id, title="Algebra practice", category="academics",
                week_start=day, scheduled_date=day, duration_minutes=30)

    session.add(Task(**task, status="maybe"))  # not an allowed status
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    session.add(Task(**task | {"student_id": 999}))  # foreign key to a missing student
    with pytest.raises(IntegrityError):
        session.commit()
