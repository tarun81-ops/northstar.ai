"""Demo data so you never retype a profile.

    python -m northstar.db.seed                 # create tables + demo student if the DB is empty
    python -m northstar.db.seed --reset         # wipe EVERYTHING and start fresh
    python -m northstar.db.seed --demo          # wipe EVERYTHING and load the full demo (goal, plan, a past week)
    python -m northstar.db.seed --week stressed # fill LAST week with a good / bad / stressed week of check-ins
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.orm import Session

from northstar.config import PROJECT_ROOT
from northstar.db.models import Base, KaggleDataset, ProgressLog, Task
from northstar.db.session import engine, get_session, init_db
from northstar.schemas import CheckIn, GoalIn, KaggleDatasetEntry, MilestoneIn, ProfileIn, TaskDraft
from northstar.services.goals import create_goal, get_active_goal, save_milestones
from northstar.services.profile import demo_student_id, save_profile
from northstar.services.progress import log_checkin
from northstar.services.scheduler import capacity, place, week_start
from northstar.services.tasks import save_week

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


DEMO_GOAL = "Score 85%+ in maths and finish my line-follower robot"
# (days from today, title, completion criteria, category) - the milestones the Roadmap Agent produced in testing
DEMO_MILESTONES = [
    (4, "Get an algebra baseline", "12 algebra problems marked and 2 error patterns noted", "academics"),
    (11, "Practise your two toughest algebra topics", "At least 7 of 10 practice problems correct", "academics"),
    (18, "Do a light maths review before the exam", "6 exam-style questions marked and corrected", "academics"),
    (28, "Build the robot chassis", "Chassis assembled, wheels turn freely, both motors tested", "technical"),
    (42, "Test the line sensors and motor control", "Sensor readings recorded; start, stop and reverse work", "technical"),
    (49, "Tune a first line-following run", "Robot follows a taped line for 1 m in 2 of 3 tries", "technical"),
    (60, "Show your maths progress and finish the robot", "85% on a practice set and a 2 m line-following run", "academics"),
]
# (title, category, minutes, difficulty, index of the milestone it serves or None, optional)
DEMO_TASKS = [
    ("Solve 12 algebra problems for your baseline", "academics", 40, 2, 0, False),
    ("Mark the 12 answers and note 2 error patterns", "academics", 25, 2, 0, False),
    ("Watch worked examples for your two toughest topics", "academics", 30, 2, 1, False),
    ("Solve 5 problems on the first tough topic", "academics", 30, 3, 1, False),
    ("Solve 5 problems on the second tough topic", "academics", 30, 3, 1, False),
    ("List the robot's next three build steps", "technical", 20, 1, None, True),
]


def load_kaggle_catalog(session: Session, catalog_path: Path | None = None) -> int:
    """Load or update Kaggle datasets from data/kaggle_catalog.json idempotently.

    Validates every entry against KaggleDatasetEntry before writing.
    Returns the count of datasets synchronized.
    """
    path = catalog_path or (PROJECT_ROOT / "data" / "kaggle_catalog.json")
    if not path.exists():
        return 0
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("datasets", [])
    synced = 0
    for item in items:
        entry = KaggleDatasetEntry.model_validate(item)
        existing = session.query(KaggleDataset).filter(KaggleDataset.kaggle_ref == entry.kaggle_ref).first()
        data = {
            "kaggle_ref": entry.kaggle_ref,
            "title": entry.title,
            "description": entry.description,
            "niche_tags": entry.niche_tags,
            "file_type": entry.file_type,
            "size_mb": entry.size_mb,
            "license": entry.license,
            "attribution_required": entry.attribution_required,
            "attribution_text": entry.attribution_text,
            "source_url": str(entry.source_url),
            "usability_score": entry.usability_score,
            "difficulty": entry.difficulty,
            "age_appropriate": entry.age_appropriate,
            "is_approved": entry.is_approved,
            "sample_questions": entry.sample_questions,
            "limitations": entry.limitations,
        }
        if existing:
            for k, v in data.items():
                setattr(existing, k, v)
        else:
            session.add(KaggleDataset(**data))
        synced += 1
    session.commit()
    return synced


def reset_demo(full: bool) -> int | None:
    """Wipe ALL data. full=True then loads a ready-to-demo student (no AI calls, so it's instant and free):
    profile, goal + milestones, a stressed week last week (ready to review) and a saved plan for this week."""
    Base.metadata.drop_all(engine)
    init_db(engine)
    if not full:
        return None
    today = date.today()
    profile = demo_profile(today)
    with get_session() as s:
        load_kaggle_catalog(s)
        sid = save_profile(s, profile)
        goal_id = create_goal(s, sid, GoalIn(title=DEMO_GOAL, category="academics", target_date=today + timedelta(days=60),
                                             description="I want to get better at algebra and actually finish a robot project"))
        save_milestones(s, goal_id, [MilestoneIn(title=t, due_date=today + timedelta(days=d), completion_criteria=c, category=cat)
                                     for d, t, c, cat in DEMO_MILESTONES])
        milestone_ids = [m.id for m in get_active_goal(s, sid).milestones]
        seed_week(s, sid, week_start(today) - timedelta(days=7), "stressed")

        drafts = [TaskDraft(title=title, category=cat, duration_minutes=minutes, difficulty=difficulty, is_optional=optional,
                            milestone_id=None if m is None else milestone_ids[m])
                  for title, cat, minutes, difficulty, m, optional in DEMO_TASKS]
        monday = week_start(today)
        placed, _ = place(drafts, capacity(profile, monday, today), {mid: i for i, mid in enumerate(milestone_ids)})
        save_week(s, sid, monday, placed)
    return sid


def seed(reset: bool = False) -> int:
    if reset:
        Base.metadata.drop_all(engine)
    init_db()
    with get_session() as s:
        load_kaggle_catalog(s)
        return demo_student_id(s) or save_profile(s, demo_profile())



if __name__ == "__main__":
    if "--demo" in sys.argv:
        print(f"Full demo loaded (student id={reset_demo(full=True)})")
        sys.exit()
    student_id = seed(reset="--reset" in sys.argv)
    print(f"Demo student ready (id={student_id})")
    if "--week" in sys.argv:
        kind = sys.argv[sys.argv.index("--week") + 1]
        last_week = week_start(date.today()) - timedelta(days=7)
        with get_session() as s:
            seed_week(s, student_id, last_week, kind)
        print(f"Filled the week of {last_week} with a {kind} week of check-ins")
