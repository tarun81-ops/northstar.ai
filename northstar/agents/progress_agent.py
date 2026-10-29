"""Progress Agent: a week's numbers + detected patterns -> a kind weekly review with up to 3 proposed changes.

Python decides WHAT happened (services/progress.summarize, services/patterns.detect); the LLM explains it and
picks changes from a fixed menu in ONE structured-output call, checked by Python with one retry.
"""
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from northstar.ai.model_router import get_llm
from northstar.prompts import system_prompt
from northstar.schemas import WeeklyReviewOut
from northstar.services.patterns import Pattern
from northstar.services.review import check_review


class ReviewError(Exception):
    pass


def write_review(summary: dict, patterns: list[Pattern], goal_title: str | None) -> WeeklyReviewOut:
    facts = {"goal": goal_title, "week": summary, "patterns": [p.__dict__ for p in patterns]}
    messages = [
        SystemMessage(system_prompt("progress")),
        HumanMessage("Write this week's review.\n" + json.dumps(facts, indent=1)),
    ]
    llm = get_llm(
        "research_analysis",
        required_capabilities={"structured_output"},
    ).with_structured_output(WeeklyReviewOut)

    for _ in range(2):  # first try + one fix
        review = llm.invoke(messages)
        problems = check_review(review)
        if not problems:
            return review
        messages += [AIMessage(review.model_dump_json()), HumanMessage("Please fix these problems: " + "; ".join(problems))]
    raise ReviewError("The review still had problems: " + "; ".join(problems))
