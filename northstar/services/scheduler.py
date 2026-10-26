"""Scheduling maths in plain Python - no LLM. The Schedule Agent proposes tasks; this module decides
how much time exists and which day each task goes on, so a week can never be overloaded."""
from collections import Counter
from datetime import date, timedelta

from northstar.schemas import DAYS, DayAvailability, Exam, ProfileIn, TaskDraft

REST_BUFFER = 0.20  # always leave 20% of free time unplanned
MAX_SAME_CATEGORY_PER_DAY = 2
EXAM_WINDOW_DAYS = 14


def week_start(d: date) -> date:
    """The Monday of d's week."""
    return d - timedelta(days=d.weekday())


def week_days(start: date) -> list[date]:
    return [start + timedelta(days=i) for i in range(7)]


def capacity(profile: ProfileIn, start: date, today: date, busy: dict[date, int] | None = None,
             load_factor: float = 1.0) -> dict[date, int]:
    """Minutes NorthStar may plan on each day of the week starting `start`:
    (free time - that day's activities) x 80% (rest buffer) x load_factor (an accepted "lighten load"), rounded down to 5,
    minus minutes already `busy` (finished tasks count against the day's limit). Days before today get 0."""
    busy = busy or {}
    activity_minutes = Counter()
    for a in profile.activities:
        activity_minutes[a.day] += a.minutes

    cap = {}
    for d in week_days(start):
        name = DAYS[d.weekday()]
        free = profile.weekly_availability.get(name, DayAvailability()).free_minutes - activity_minutes[name]
        usable = int(max(free, 0) * (1 - REST_BUFFER) * load_factor) // 5 * 5
        cap[d] = 0 if d < today else max(usable - busy.get(d, 0), 0)
    return cap


def exams_soon(profile: ProfileIn, start: date) -> list[Exam]:
    """Exams from the start of the week up to 14 days later -> the week is in 'exam mode'."""
    return [e for e in profile.exams if start <= e.date <= start + timedelta(days=EXAM_WINDOW_DAYS)]


def place(drafts: list[TaskDraft], cap: dict[date, int]) -> tuple[list[tuple[date, TaskDraft]], list[TaskDraft]]:
    """Give each task a day. Guarantees:
    - no day goes over its capacity;
    - required tasks are placed before optional ones;
    - at most MAX_SAME_CATEGORY_PER_DAY tasks of one category per day;
    - tasks for the same milestone keep the agent's order (task 2 is never before task 1);
    - tasks that don't fit are returned as overflow, never silently dropped.
    Each task goes to the allowed day with the most free time left *relative to its capacity*, earliest on ties,
    which spreads work across the week instead of piling it onto Monday or the weekend."""
    remaining = dict(cap)
    per_category = {d: Counter() for d in cap}
    not_before: dict[int, date] = {}  # milestone_id -> day of its previous task
    placed, overflow = [], []

    for task in sorted(drafts, key=lambda t: t.is_optional):  # sort is stable: agent's order kept within each group
        earliest = not_before.get(task.milestone_id, min(cap))
        options = [
            d for d in cap
            if d >= earliest
            and remaining[d] >= task.duration_minutes
            and per_category[d][task.category] < MAX_SAME_CATEGORY_PER_DAY
        ]
        if not options:
            overflow.append(task)
            continue
        day = max(options, key=lambda d: (remaining[d] / cap[d], -d.toordinal()))
        remaining[day] -= task.duration_minutes
        per_category[day][task.category] += 1
        if task.milestone_id is not None:
            not_before[task.milestone_id] = day
        placed.append((day, task))

    placed.sort(key=lambda pair: pair[0])
    return placed, overflow
