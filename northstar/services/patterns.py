"""Spot patterns in a week of check-ins. Plain Python with named thresholds: the LLM explains patterns,
it never decides whether they exist. Tune the constants, not a prompt."""
from dataclasses import dataclass

MIN_CHECKINS = 3          # fewer check-ins than this: not enough evidence to call any pattern
BAD_WEEK_BELOW = 0.4      # completion rate
OVERLOAD_TIME_RATIO = 1.25  # actual minutes / planned minutes (checked-in tasks)
OVERLOAD_STRESS = 4.0     # average stress 1-5
STRUGGLE_SHARE = 0.4      # share of check-ins rated too hard or only partly done
EASY_SHARE = 0.5          # share of difficulty ratings that were "too easy"
EASY_MIN_COMPLETION = 0.7
SKIPS_PER_CATEGORY = 2
INTEREST_ENJOYMENT = 4.0  # average enjoyment in one category
INTEREST_MIN_RATED = 2


@dataclass
class Pattern:
    name: str          # too_easy, too_hard, skipping_subject, overloaded, losing_interest, strong_interest, bad_week
    evidence: str      # plain words, shown to the student and the LLM
    category: str | None = None


def detect(s: dict) -> list[Pattern]:
    """Patterns in a week summary (from services.progress.summarize)."""
    checked = s["done"] + s["partial"] + s["skipped"]
    if checked < MIN_CHECKINS:
        return []
    found = []

    if s["completion_rate"] < BAD_WEEK_BELOW:
        found.append(Pattern("bad_week", f"{s['done']} of {s['tasks_planned']} tasks done ({s['completion_rate']:.0%})."))

    ratio = s["actual_minutes"] / s["planned_minutes_checked_in"] if s["planned_minutes_checked_in"] else 0
    if ratio >= OVERLOAD_TIME_RATIO or (s["avg_stress"] or 0) >= OVERLOAD_STRESS:
        found.append(Pattern("overloaded", f"Tasks took {ratio:.0%} of the planned time; average stress {s['avg_stress']}/5."))

    if s["struggled"] / checked >= STRUGGLE_SHARE:
        found.append(Pattern("too_hard", f"{s['struggled']} of {checked} tasks felt too hard or were only partly done."))
    else:
        rated = sum(s["difficulty"].values())
        easy = s["difficulty"].get("too_easy", 0)
        if rated >= MIN_CHECKINS and easy / rated >= EASY_SHARE and s["completion_rate"] >= EASY_MIN_COMPLETION:
            found.append(Pattern("too_easy", f"{easy} of {rated} rated tasks felt too easy, and {s['completion_rate']:.0%} got done."))

    for cat, c in s["by_category"].items():
        if c["skipped"] >= SKIPS_PER_CATEGORY:
            found.append(Pattern("skipping_subject", f"{c['skipped']} of {c['planned']} {cat} tasks were skipped.", cat))
        elif c["rated"] >= INTEREST_MIN_RATED and (c["avg_enjoyment"] or 0) >= INTEREST_ENJOYMENT:
            found.append(Pattern("strong_interest", f"{cat} tasks averaged {c['avg_enjoyment']}/5 enjoyment.", cat))

    last3 = s["enjoyment_by_day"][-3:]
    if len(last3) == 3 and last3[0] > last3[1] > last3[2]:
        found.append(Pattern("losing_interest", f"Enjoyment dropped across the last three check-ins ({' → '.join(map(str, last3))})."))
    return found
