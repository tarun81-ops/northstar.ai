"""Roadmap Agent: goal + profile -> 4-8 milestones.

Not a tool-calling loop: Python gathers the facts, the LLM makes ONE structured-output call,
and Python checks the dates. If the check fails, the problems are sent back once for a fix.
"""
import json
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from northstar.llm import get_llm
from northstar.prompts import system_prompt
from northstar.schemas import GoalIn, MilestoneDraft, MilestoneList, ProfileIn
from northstar.services.goals import check_milestones


class RoadmapError(Exception):
    pass


def generate_milestones(profile: ProfileIn, goal: GoalIn, today: date | None = None) -> list[MilestoneDraft]:
    today = today or date.today()
    facts = {
        "today": today.isoformat(),
        "goal": goal.model_dump(mode="json"),
        "weeks_until_target": round((goal.target_date - today).days / 7, 1),
        "weekly_free_hours": round(sum(d.free_minutes for d in profile.weekly_availability.values()) / 60, 1),
        "student": profile.model_dump(mode="json", exclude={"alias"}),  # the nickname isn't needed to plan
    }
    messages = [
        SystemMessage(system_prompt("roadmap")),
        HumanMessage("Create the milestones for this goal.\n" + json.dumps(facts, indent=1)),
    ]
    llm = get_llm().with_structured_output(MilestoneList)

    for _ in range(2):  # first try + one fix
        result = llm.invoke(messages)
        problems = check_milestones(result.milestones, today, goal.target_date)
        if not problems:
            return result.milestones
        messages += [AIMessage(result.model_dump_json()), HumanMessage("Please fix these problems: " + "; ".join(problems))]
    raise RoadmapError("The milestones still had problems: " + "; ".join(problems))
