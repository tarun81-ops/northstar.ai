"""Schedule Agent: current milestones + free time -> this week's tasks.

The LLM proposes task drafts in ONE structured-output call (what to do, how long, how hard).
Python checks them, retries once with the problems, then services/scheduler.place() picks the days.
"""
import json
from dataclasses import dataclass
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from northstar.ai.context_manager import estimate_prompt_tokens, should_compact
from northstar.ai import llm_config
from northstar.ai.model_router import get_llm
from northstar.prompts import system_prompt
from northstar.schemas import Exam, MilestoneIn, ProfileIn, TaskDraft, TaskDraftList
from northstar.services.review import EASIER_MAX_DIFFICULTY, Adjustments
from northstar.services.scheduler import capacity, exams_soon, place

REQUIRED_SHARE = 0.8  # non-optional work may use at most 80% of the week's capacity
CURRENT_MILESTONES = 2  # plan for the next two unfinished milestones


def _compact_facts(facts: dict) -> dict:
    """Return a reduced-detail copy of ``facts`` for large-prompt compaction.

    Keeps all fields the scheduler needs (goal, milestones, capacity, budget,
    exams, accepted changes) but shortens verbose text fields so the prompt
    stays under the context window.
    """
    compacted = dict(facts)

    # Shorten milestone descriptions and completion criteria.
    if "current_milestones" in compacted:
        short_milestones = []
        for m in compacted["current_milestones"]:
            short_m = dict(m)
            if "description" in short_m and len(short_m["description"]) > 120:
                short_m["description"] = short_m["description"][:117] + "..."
            if "completion_criteria" in short_m and len(short_m["completion_criteria"]) > 160:
                short_m["completion_criteria"] = short_m["completion_criteria"][:157] + "..."
            short_milestones.append(short_m)
        compacted["current_milestones"] = short_milestones

    # Drop verbose student fields that aren't needed for task planning.
    if "student" in compacted and isinstance(compacted["student"], dict):
        short_student = dict(compacted["student"])
        short_student.pop("learning_preferences", None)
        short_student.pop("exams", None)
        compacted["student"] = short_student

    return compacted


class ScheduleError(Exception):
    pass


@dataclass
class WeekPlan:
    placed: list[tuple[date, TaskDraft]]
    overflow: list[TaskDraft]  # tasks that didn't fit - shown to the student, never silently dropped
    capacity: dict[date, int]
    exams: list[Exam]


def check_drafts(drafts: list[TaskDraft], cap: dict[date, int], milestone_ids: set[int],
                 adjust: Adjustments | None = None) -> list[str]:
    """Problems with the agent's tasks, in plain words (empty = OK). Accepted plan changes are enforced here."""
    adjust = adjust or Adjustments()
    problems = []
    if adjust.max_minutes:
        long = [t.title for t in drafts if t.duration_minutes > adjust.max_minutes]
        if long:
            problems.append(f"the student asked for tasks of at most {adjust.max_minutes} minutes; shorten or split: {', '.join(long)}")
    hard = [t.title for t in drafts if t.category in adjust.easier and t.difficulty > EASIER_MAX_DIFFICULTY]
    if hard:
        problems.append(f"the student asked for easier steps in {', '.join(sorted(adjust.easier))}: "
                        f"difficulty must be {EASIER_MAX_DIFFICULTY} or less for: {', '.join(hard)}")
    if not drafts:
        problems.append("there are no tasks")
    unknown = {t.milestone_id for t in drafts} - milestone_ids - {None}
    if unknown:
        problems.append(f"milestone_id must be one of {sorted(milestone_ids)} or null, not {sorted(unknown)}")
    budget = int(sum(cap.values()) * REQUIRED_SHARE)
    required = sum(t.duration_minutes for t in drafts if not t.is_optional)
    if required > budget:
        problems.append(f"non-optional tasks add up to {required} minutes but the budget is {budget}; cut or mark some optional")
    biggest_day = max(cap.values())
    too_long = [t.title for t in drafts if t.duration_minutes > biggest_day]
    if too_long:
        problems.append(f"no day has more than {biggest_day} minutes, so split: {', '.join(too_long)}")
    return problems


def plan_week(profile: ProfileIn, goal_title: str, milestones: list[MilestoneIn], start: date, today: date,
              busy: dict[date, int] | None = None, adjust: Adjustments | None = None) -> WeekPlan:
    adjust = adjust or Adjustments()
    cap = capacity(profile, start, today, busy, adjust.load_factor)
    exams = exams_soon(profile, start)
    if sum(cap.values()) == 0:
        return WeekPlan([], [], cap, exams)  # no free time left this week: nothing to plan

    current = [m for m in milestones if m.status != "done"][:CURRENT_MILESTONES]
    facts = {
        "today": today.isoformat(),
        "week": f"{start} to {max(cap)}",
        "goal": goal_title,
        "current_milestones": [m.model_dump(mode="json", exclude={"status"}) for m in current],
        "minutes_available_per_day": {f"{d:%a %d %b}": m for d, m in cap.items()},
        "required_minutes_budget": int(sum(cap.values()) * REQUIRED_SHARE),
        "exams_coming_up": [e.model_dump(mode="json") for e in exams],
        "changes_the_student_accepted": adjust.notes,
        "student": profile.model_dump(mode="json", exclude={"alias", "weekly_availability", "activities"}),
    }
    messages = [
        SystemMessage(system_prompt("schedule")),
        HumanMessage("Propose this week's tasks.\n" + json.dumps(facts, indent=1)),
    ]

    # Automatic compaction: if the prompt is too large, reduce fact detail.
    if should_compact(messages):
        logger.info(
            "plan_week: prompt estimated %d tokens (> %d threshold); compacting facts",
            estimate_prompt_tokens(messages),
            llm_config.context_compact_threshold(),
        )
        facts = _compact_facts(facts)
        messages = [
            SystemMessage(system_prompt("schedule")),
            HumanMessage("Propose this week's tasks.\n" + json.dumps(facts, indent=1)),
        ]

    llm = get_llm(
        "planning",
        required_capabilities={"structured_output"},
    ).with_structured_output(TaskDraftList)

    for _ in range(2):  # first try + one fix
        result = llm.invoke(messages)
        problems = check_drafts(result.tasks, cap, {m.id for m in current}, adjust)
        if not problems:
            placed, overflow = place(result.tasks, cap, {m.id: i for i, m in enumerate(current)})
            return WeekPlan(placed, overflow, cap, exams)
        messages += [AIMessage(result.model_dump_json()), HumanMessage("Please fix these problems: " + "; ".join(problems))]
    raise ScheduleError("The tasks still had problems: " + "; ".join(problems))
