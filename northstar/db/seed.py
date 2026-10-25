"""Demo data so you never retype a profile.

    python -m northstar.db.seed          # create tables + demo student if the DB is empty
    python -m northstar.db.seed --reset  # wipe EVERYTHING and start fresh
"""
import sys
from datetime import date, timedelta

from northstar.db.models import Base
from northstar.db.session import engine, get_session, init_db
from northstar.schemas import ProfileIn
from northstar.services.profile import demo_student_id, save_profile


def demo_profile(today: date | None = None) -> ProfileIn:
    today = today or date.today()
    school_day = {"school": "08:00-15:00", "free_minutes": 120}
    return ProfileIn(
        alias="Aarav",
        age_band="14-16",
        grade="Grade 10",
        subjects=["Maths", "Physics", "Chemistry", "English", "Computer Science"],
        interests=["Robotics", "Electronics", "Video games"],
        strengths=["Physics", "Hands-on building"],
        challenges=["Algebra", "Staying consistent"],
        learning_preferences=["Practice problems", "Projects", "Videos"],
        weekly_availability={
            "mon": school_day, "tue": school_day, "wed": school_day, "thu": school_day, "fri": school_day,
            "sat": {"school": "", "free_minutes": 240},
            "sun": {"school": "", "free_minutes": 180},
        },
        activities=[{"name": "Football", "day": "tue", "minutes": 90}, {"name": "Football", "day": "thu", "minutes": 90}],
        exams=[{"subject": "Maths", "date": today + timedelta(days=21)}],
    )


def seed(reset: bool = False) -> int:
    if reset:
        Base.metadata.drop_all(engine)
    init_db()
    with get_session() as s:
        return demo_student_id(s) or save_profile(s, demo_profile())


if __name__ == "__main__":
    student_id = seed(reset="--reset" in sys.argv)
    print(f"Demo student ready (id={student_id})")
