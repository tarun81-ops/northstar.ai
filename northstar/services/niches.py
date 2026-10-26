"""Niche discovery: interest signals -> niche suggestions -> a short curiosity experiment -> reflection.

Signals are stored rows with ids. The Niche Mentor must cite real signal ids for every niche (checked here),
and the page shows those signals itself - so "why this fits you" is grounded in the student's own data.
A finished experiment the student rated highly becomes a new signal, so the loop learns.
"""
from collections import defaultdict
from datetime import date
from statistics import mean

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from northstar.db.models import CuriosityExperiment, InterestSignal, NicheSuggestion, ProgressLog, Task
from northstar.schemas import NicheDraft, ProfileIn, Reflection

DERIVED_SOURCES = ("profile", "checkins")  # recomputed on refresh; "student" and "experiment" signals are kept
MIN_NICHES, MAX_NICHES = 3, 5
EXPERIMENT_DAYS = (3, 14)
EXPERIMENT_STEPS = (3, 6)
ENJOYED_CATEGORY = 4.0      # average enjoyment in a category to count as a signal
ENJOYED_MIN_LOGS = 2
LOVED_TASKS = 5             # most recent tasks rated 5/5 that become signals
EXPERIMENT_SIGNAL_AT = 4.0  # mean(curiosity, meaning) at which a finished experiment becomes a signal


def _own(session: Session, model, student_id: int, row_id: int):
    row = session.get(model, row_id)
    if row is None or row.student_id != student_id:
        raise ValueError(f"{model.__name__} {row_id} is not this student's")
    return row


# ---- signals ----------------------------------------------------------------------

def list_signals(session: Session, student_id: int) -> list[InterestSignal]:
    return list(session.scalars(select(InterestSignal).where(InterestSignal.student_id == student_id)
                                .order_by(InterestSignal.strength.desc(), InterestSignal.id)))


def _derived_signals(session: Session, student_id: int, profile: ProfileIn) -> dict[tuple[str, str], int]:
    """(source, signal text) -> strength, from the profile and check-in enjoyment. Texts are stable (no live numbers)."""
    derived = {("profile", f"Interested in {i}"): 3 for i in profile.interests}
    derived |= {("profile", f"Strong at {s}"): 2 for s in profile.strengths}
    rated = session.execute(
        select(Task.category, Task.title, ProgressLog.enjoyment, ProgressLog.date)
        .join(ProgressLog, ProgressLog.task_id == Task.id)
        .where(Task.student_id == student_id, ProgressLog.enjoyment.is_not(None))
    ).all()
    by_category = defaultdict(list)
    for category, _, enjoyment, _ in rated:
        by_category[category].append(enjoyment)
    for category, values in by_category.items():
        if len(values) >= ENJOYED_MIN_LOGS and mean(values) >= ENJOYED_CATEGORY:
            derived[("checkins", f"Enjoys {category} tasks")] = 4
    for r in sorted((r for r in rated if r.enjoyment == 5), key=lambda r: r.date, reverse=True)[:LOVED_TASKS]:
        derived[("checkins", f"Loved the task: {r.title}"[:200])] = 5
    return derived


def refresh_signals(session: Session, student_id: int, profile: ProfileIn) -> list[InterestSignal]:
    """Bring derived signals up to date IN PLACE, so ids stay stable while their evidence holds: signals whose
    evidence is gone are removed, new ones added. The student's own and experiment signals are never touched."""
    wanted = _derived_signals(session, student_id, profile)
    existing = {(s.source, s.signal): s for s in list_signals(session, student_id) if s.source in DERIVED_SOURCES}
    for key, signal in existing.items():
        if key in wanted:
            signal.strength = wanted[key]
        else:
            session.delete(signal)
    session.add_all(InterestSignal(student_id=student_id, source=source, signal=text, strength=strength)
                    for (source, text), strength in wanted.items() if (source, text) not in existing)
    session.commit()
    return list_signals(session, student_id)


def add_signal(session: Session, student_id: int, text: str) -> None:
    """Something the student says they're curious about - their own words count as strong evidence."""
    text = text.strip()
    if not 3 <= len(text) <= 200:
        raise ValueError("Write between 3 and 200 characters")
    session.add(InterestSignal(student_id=student_id, signal=text, source="student", strength=4))
    session.commit()


def remove_signal(session: Session, student_id: int, signal_id: int) -> None:
    signal = _own(session, InterestSignal, student_id, signal_id)
    if signal.source != "student":
        raise ValueError("Only things you added yourself can be removed; the others come from your profile and check-ins")
    session.delete(signal)
    session.commit()


# ---- niche suggestions ------------------------------------------------------------

def check_niches(niches: list[NicheDraft], signal_ids: set[int]) -> list[str]:
    """Problems with the Niche Mentor's output, in plain words (empty = OK)."""
    problems = []
    if not MIN_NICHES <= len(niches) <= MAX_NICHES:
        problems.append(f"suggest {MIN_NICHES}-{MAX_NICHES} niches, not {len(niches)}")
    if len({n.niche_name.lower() for n in niches}) != len(niches):
        problems.append("niche names must be different")
    for n in niches:
        if not n.matched_signal_ids:
            problems.append(f"'{n.niche_name}' must cite at least one signal id")
        unknown = set(n.matched_signal_ids) - signal_ids
        if unknown:
            problems.append(f"'{n.niche_name}' cites signal ids that don't exist: {sorted(unknown)}; use only {sorted(signal_ids)}")
        low, high = EXPERIMENT_DAYS
        if not low <= n.experiment.duration_days <= high:
            problems.append(f"'{n.niche_name}' experiment must last {low}-{high} days")
        low, high = EXPERIMENT_STEPS
        if not low <= len(n.experiment.steps) <= high:
            problems.append(f"'{n.niche_name}' experiment needs {low}-{high} steps")
    return problems


def suggestions(session: Session, student_id: int, status: str = "suggested") -> list[NicheSuggestion]:
    return list(session.scalars(select(NicheSuggestion).where(NicheSuggestion.student_id == student_id,
                                                              NicheSuggestion.status == status).order_by(NicheSuggestion.id)))


def save_suggestions(session: Session, student_id: int, niches: list[NicheDraft]) -> None:
    """Replace open suggestions (chosen or dismissed niches stay as history). Each niche stores a snapshot of the
    signals it cites, taken from the database - not from the LLM's text."""
    text_of = {s.id: s.signal for s in list_signals(session, student_id)}
    session.execute(delete(NicheSuggestion).where(NicheSuggestion.student_id == student_id,
                                                  NicheSuggestion.status == "suggested"))
    session.add_all(NicheSuggestion(student_id=student_id, niche_name=n.niche_name, description=n.description, why=n.why,
                                    matched_signals=[{"id": i, "signal": text_of[i]} for i in n.matched_signal_ids],
                                    skills_required=n.skills_required, experiment=n.experiment.model_dump())
                    for n in niches)
    session.commit()


def dismiss_niche(session: Session, student_id: int, niche_id: int) -> None:
    _own(session, NicheSuggestion, student_id, niche_id).status = "dismissed"
    session.commit()


# ---- curiosity experiments --------------------------------------------------------

def active_experiment(session: Session, student_id: int) -> CuriosityExperiment | None:
    return session.scalar(select(CuriosityExperiment).where(CuriosityExperiment.student_id == student_id,
                                                            CuriosityExperiment.status == "active"))


def past_experiments(session: Session, student_id: int) -> list[tuple[CuriosityExperiment, NicheSuggestion]]:
    return list(session.execute(
        select(CuriosityExperiment, NicheSuggestion).join(NicheSuggestion, CuriosityExperiment.niche_id == NicheSuggestion.id)
        .where(CuriosityExperiment.student_id == student_id, CuriosityExperiment.status != "active")
        .order_by(CuriosityExperiment.id.desc())
    ).tuples())


def start_experiment(session: Session, student_id: int, niche_id: int, today: date) -> int:
    """One experiment at a time keeps it light."""
    if active_experiment(session, student_id):
        raise ValueError("Finish or stop your current experiment first")
    niche = _own(session, NicheSuggestion, student_id, niche_id)
    niche.status = "chosen"
    exp = CuriosityExperiment(student_id=student_id, niche_id=niche.id, title=niche.experiment["title"],
                              steps=[{"text": s, "done": False} for s in niche.experiment["steps"]],
                              duration_days=niche.experiment["duration_days"], started_on=today)
    session.add(exp)
    session.commit()
    return exp.id


def set_step(session: Session, student_id: int, experiment_id: int, index: int, done: bool) -> None:
    exp = _own(session, CuriosityExperiment, student_id, experiment_id)
    steps = [dict(s) for s in exp.steps]  # new list so SQLAlchemy notices the JSON change
    steps[index]["done"] = done
    exp.steps = steps
    session.commit()


def finish_experiment(session: Session, student_id: int, experiment_id: int, reflection: Reflection) -> bool:
    """Save the reflection. Returns True if it was strong enough to become a new interest signal."""
    exp = _own(session, CuriosityExperiment, student_id, experiment_id)
    exp.status = "done"
    exp.reflection = reflection.notes
    exp.curiosity_score, exp.energy_score = reflection.curiosity, reflection.energy
    exp.confidence_score, exp.meaning_score = reflection.confidence, reflection.meaning
    score = mean([reflection.curiosity, reflection.meaning])
    if score >= EXPERIMENT_SIGNAL_AT:
        niche = session.get(NicheSuggestion, exp.niche_id)
        session.add(InterestSignal(student_id=student_id, source="experiment", strength=min(5, round(score)),
                                   signal=f"Enjoyed trying {niche.niche_name}"[:200]))
    session.commit()
    return score >= EXPERIMENT_SIGNAL_AT


def stop_experiment(session: Session, student_id: int, experiment_id: int) -> None:
    """Stopping is a fine outcome - learning what you're NOT into is useful too."""
    _own(session, CuriosityExperiment, student_id, experiment_id).status = "stopped"
    session.commit()
