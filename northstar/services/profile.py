"""Read and write the student profile. Every caller (form, seed, Phase 3 tools) goes through here."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from northstar.db.models import Student
from northstar.schemas import ProfileIn


def save_profile(session: Session, profile: ProfileIn, student_id: int | None = None) -> int:
    """Create a student (student_id=None) or overwrite an existing one's profile. Returns the id."""
    student = session.get(Student, student_id) if student_id else Student()
    if student is None:
        raise ValueError(f"No student with id {student_id}")
    for field, value in profile.model_dump(mode="json").items():  # mode="json": dates -> ISO strings for JSON columns
        setattr(student, field, value)
    session.add(student)
    session.commit()
    return student.id


def get_profile(session: Session, student_id: int) -> ProfileIn | None:
    student = session.get(Student, student_id)
    return ProfileIn.model_validate(student, from_attributes=True) if student else None


def demo_student_id(session: Session) -> int | None:
    # ponytail: single-student demo mode; replace with the logged-in user's id when auth arrives.
    return session.scalar(select(Student.id).order_by(Student.id).limit(1))
