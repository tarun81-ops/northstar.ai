"""Demo data so you never retype a profile.

    python -m northstar.db.seed                 # create tables + demo student if the DB is empty
    python -m northstar.db.seed --reset         # wipe EVERYTHING and start fresh
    python -m northstar.db.seed --week stressed # fill LAST week with a good / bad / stressed week of check-ins
"""
import sys
from datetime import date, timedelta

from sqlalchemy import delete
from sqlalchemy.orm import Session

from northstar.db.models import Base, ProgressLog, Task
from northstar.db.session import engine, get_session, init_db
from northstar.schemas import CheckIn, ProfileIn
from northstar.services.profile import demo_student_id, save_profile
from northstar.services.progress import log_checkin
from northstar.services.scheduler import week_start

WEEK_KINDS = ("good", "bad", "stressed")

# One task per day: (title, category, minutes, difficulty)
DEMO_WEEK = [
    ("Solve 10 factorising problems", "academics", 40, 3),
    ("Watch a video on quadratic equations", "academics", 25, 2),
    ("Wire the IR line sensors", "technical", 45, 3),
    ("Solve 8 simultaneous equations", "academics", 30, 3),
    ("Write the motor test code", "technical", 40, 3),
    ("Do a timed algebra practice set", "academics", 45, 4),
    ("Go for a 30-minute run", "wellbeing", 30, 1),
]


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


def _outcome(kind: str, i: int, category: str, minutes: int) -> dict:
    """What the check-in for task i looks like in each kind of week."""
    if kind == "good":
        return dict(completed="yes", time_spent_minutes=minutes - 5, difficulty_rating="too_easy" if i % 3 else "just_right",
                    enjoyment=5 if category == "technical" else 4, stress=2, confidence=4)
    if kind == "bad":
        if i in (0, 6):
            return dict(completed="yes", time_spent_minutes=minutes, difficulty_rating="just_right", enjoyment=3, stress=3)
        return dict(completed="no", stress=3,
                    blockers="Didn't understand factorising" if category == "academics" else "Too tired after school")
    # stressed: does the work, but it takes far longer and feels hard
    return dict(completed="yes" if i % 2 else "partial", time_spent_minutes=int(minutes * 1.5),
                difficulty_rating="too_hard" if category == "academics" else "just_right",
                enjoyment=2 if category == "academics" else 4, stress=5 if i >= 4 else 4, confidence=2)


def seed_week(session: Session, student_id: int, start: date, kind: str) -> None:
    """Fill the week starting `start` with tasks + check-ins shaped like a good / bad / stressed week.
    Replaces any tasks and check-ins already in that week."""
    if kind not in WEEK_KINDS:
        raise ValueError(f"kind must be one of {WEEK_KINDS}")
    session.execute(delete(ProgressLog).where(
        ProgressLog.student_id == student_id, ProgressLog.date.between(start, start + timedelta(days=6))
    ))
    session.execute(delete(Task).where(Task.student_id == student_id, Task.week_start == start))
    tasks = [Task(student_id=student_id, title=title, category=category, week_start=start,
                  scheduled_date=start + timedelta(days=i), duration_minutes=minutes, difficulty=difficulty)
             for i, (title, category, minutes, difficulty) in enumerate(DEMO_WEEK)]
    session.add_all(tasks)
    session.commit()
    for i, t in enumerate(tasks):
        log_checkin(session, student_id, t.scheduled_date,
                    CheckIn(task_id=t.id, **_outcome(kind, i, t.category, t.duration_minutes)))


def seed(reset: bool = False) -> int:
    if reset:
        Base.metadata.drop_all(engine)
    init_db()
    with get_session() as s:
        return demo_student_id(s) or save_profile(s, demo_profile())


if __name__ == "__main__":
    student_id = seed(reset="--reset" in sys.argv)
    print(f"Demo student ready (id={student_id})")
    if "--week" in sys.argv:
        kind = sys.argv[sys.argv.index("--week") + 1]
        last_week = week_start(date.today()) - timedelta(days=7)
        with get_session() as s:
            seed_week(s, student_id, last_week, kind)
        print(f"Filled the week of {last_week} with a {kind} week of check-ins")
