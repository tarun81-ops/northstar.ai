"""Opportunity discovery from a curated list (data/opportunities.json). No scraping, and never registration.

Python decides what the student is ELIGIBLE for (age, deadline, cost) and ranks by interest overlap;
the LLM only writes "why it matches" for the top candidates. Location is never used - NorthStar doesn't store it -
so each card shows who it's open to and the student decides.
"""
import json
import re
from datetime import date
from functools import lru_cache

from sqlalchemy import select
from sqlalchemy.orm import Session

from northstar.config import PROJECT_ROOT
from northstar.db.models import Recommendation
from northstar.schemas import Opportunity, OpportunityMatch, ProfileIn

AGE_RANGE = {"11-13": (11, 13), "14-16": (14, 16), "17-19": (17, 19), "college": (18, 25)}
CANDIDATES = 8   # sent to the LLM
MAX_MATCHES = 5


@lru_cache
def load_opportunities() -> list[Opportunity]:
    data = json.loads((PROJECT_ROOT / "data" / "opportunities.json").read_text(encoding="utf-8"))
    return [Opportunity.model_validate(o) for o in data["opportunities"]]


def by_key() -> dict[str, Opportunity]:
    return {o.key: o for o in load_opportunities()}


def eligible(opportunities: list[Opportunity], profile: ProfileIn, today: date, only_free: bool = False) -> list[Opportunity]:
    """Age range overlaps the student's age band, the deadline hasn't passed (no deadline = ongoing/unchecked),
    and optionally free only (unknown cost counts as not free)."""
    low, high = AGE_RANGE[profile.age_band]
    return [o for o in opportunities
            if o.age_min <= high and o.age_max >= low
            and (o.deadline is None or o.deadline >= today)
            and (o.is_free or not only_free)]


def _words(text: str) -> set[str]:
    """Lower-case words with a simple plural trim, so 'robots' matches 'robot'."""
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in re.findall(r"[a-z0-9]+", text.lower())}


def rank(opportunities: list[Opportunity], interests: list[str]) -> list[Opportunity]:
    """Best tag overlap with the student's interests, subjects, signals and niches first; no overlap = left out."""
    words = _words(" ".join(interests))
    scored = [(-sum(1 for t in o.tags if _words(t) <= words), o.title, o) for o in opportunities]
    return [o for score, _, o in sorted(scored) if score < 0]


def check_matches(matches: list[OpportunityMatch], candidate_keys: set[str]) -> list[str]:
    problems = []
    if not 1 <= len(matches) <= MAX_MATCHES:
        problems.append(f"return 1-{MAX_MATCHES} matches, not {len(matches)}")
    keys = [m.key for m in matches]
    unknown = set(keys) - candidate_keys
    if unknown:
        problems.append(f"only use these keys: {sorted(candidate_keys)}; not {sorted(unknown)}")
    if len(set(keys)) != len(keys):
        problems.append("each opportunity only once")
    if any(re.search(r"https?://|www\.", m.why) for m in matches):
        problems.append("no links - the page shows the official link itself")
    return problems


def save_matches(session: Session, student_id: int, matches: list[OpportunityMatch]) -> None:
    """Replace open suggestions. Saved ones keep their place; dismissed ones are never suggested again."""
    existing = {r.opportunity_key: r for r in session.scalars(select(Recommendation).where(Recommendation.student_id == student_id))}
    for r in existing.values():
        if r.status == "suggested" and r.opportunity_key not in {m.key for m in matches}:
            session.delete(r)
    for m in matches:
        row = existing.get(m.key)
        if row is None:
            session.add(Recommendation(student_id=student_id, opportunity_key=m.key, reason=m.why))
        elif row.status != "dismissed":
            row.reason = m.why
    session.commit()


def recommendations(session: Session, student_id: int, status: str) -> list[Recommendation]:
    return list(session.scalars(select(Recommendation).where(Recommendation.student_id == student_id,
                                                             Recommendation.status == status).order_by(Recommendation.id)))


def dismissed_keys(session: Session, student_id: int) -> set[str]:
    return {r.opportunity_key for r in recommendations(session, student_id, "dismissed")}


def decide(session: Session, student_id: int, recommendation_id: int, status: str) -> None:
    if status not in ("saved", "dismissed", "suggested"):
        raise ValueError("status must be saved, dismissed or suggested")
    row = session.get(Recommendation, recommendation_id)
    if row is None or row.student_id != student_id:
        raise ValueError(f"Recommendation {recommendation_id} is not this student's")
    row.status = status
    session.commit()


def permission_note(opportunity: Opportunity, profile: ProfileIn) -> str | None:
    if opportunity.requires_permission:
        return "You'll need a parent's or guardian's OK to take part."
    if profile.age_band != "college":
        return "Check with a parent or teacher before signing up."
    return None
