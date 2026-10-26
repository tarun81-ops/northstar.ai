"""Niche Mentor: interest signals -> 3-5 specific niches, each with a short curiosity experiment.

ONE structured-output call. Python checks that every niche cites real signal ids (and the experiment limits),
and sends problems back once.
"""
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from northstar.db.models import InterestSignal
from northstar.llm import get_llm
from northstar.prompts import system_prompt
from northstar.schemas import NicheDraft, NicheList, ProfileIn
from northstar.services.niches import check_niches


class NicheError(Exception):
    pass


def suggest_niches(profile: ProfileIn, signals: list[InterestSignal], avoid: list[str]) -> list[NicheDraft]:
    facts = {
        "student": profile.model_dump(mode="json", include={"age_band", "grade", "subjects", "strengths", "challenges",
                                                           "learning_preferences"}),
        "signals": [{"id": s.id, "signal": s.signal, "source": s.source, "strength": s.strength} for s in signals],
        "avoid_niches": avoid,
    }
    messages = [
        SystemMessage(system_prompt("niche")),
        HumanMessage("Suggest niches for this student.\n" + json.dumps(facts, indent=1)),
    ]
    llm = get_llm().with_structured_output(NicheList)

    for _ in range(2):  # first try + one fix
        result = llm.invoke(messages)
        problems = check_niches(result.niches, {s.id for s in signals})
        if not problems:
            return result.niches
        messages += [AIMessage(result.model_dump_json()), HumanMessage("Please fix these problems: " + "; ".join(problems))]
    raise NicheError("The suggestions still had problems: " + "; ".join(problems))
