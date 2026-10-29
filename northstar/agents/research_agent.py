"""Research Coach: guidance for one step of the student's research project.

ONE structured-output call. Python checks the coach didn't include links or cite sources the project doesn't have,
and sends problems back once. The coach's reply is advice only - it never becomes part of the student's report.
"""
import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from northstar.db.models import ResearchProject
from northstar.ai.model_router import get_llm
from northstar.prompts import system_prompt
from northstar.schemas import CoachReply, ProfileIn
from northstar.services.research import CITATION, ETHICS, LINK, STEP_GUIDE


class CoachError(Exception):
    pass


def check_coach(reply: CoachReply, source_numbers: set[int]) -> list[str]:
    problems = []
    text = " ".join([reply.feedback, reply.next_action, *reply.questions, *reply.ethics_flags])
    if LINK.search(text) or re.search(r"\bwww\.", text):
        problems.append("don't include any links; point to the student's sources as [S#] or to NorthStar's library")
    unknown = {int(n) for n in CITATION.findall(text)} - source_numbers
    if unknown:
        problems.append(f"the student has no source(s) {sorted(unknown)}; only cite {sorted(source_numbers) or 'none'}")
    if not 1 <= len(reply.questions) <= 3:
        problems.append("ask 1-3 questions")
    return problems


def coach_step(profile: ProfileIn, project: ResearchProject, step_key: str) -> CoachReply:
    notes = {s.step: s.notes for s in project.steps}
    facts = {
        "student": profile.model_dump(mode="json", include={"age_band", "grade"}),
        "project": project.title,
        "step": {"name": step_key, "purpose": STEP_GUIDE[step_key][1]},
        "student_notes_for_this_step": notes[step_key],
        "earlier_steps": {k: v for k, v in notes.items() if v and k != step_key},
        "sources": [{"cite_as": f"[S{s.number}]", "title": s.title, "publisher": s.publisher,
                     "student_note": s.note, "opened": s.opened} for s in project.sources],
    }
    if step_key == "method":
        facts["ethics_checklist"] = {ETHICS[k]: bool(project.ethics.get(k)) for k in ETHICS}
    messages = [
        SystemMessage(system_prompt("research")),
        HumanMessage("Coach this step.\n" + json.dumps(facts, indent=1)),
    ]
    llm = get_llm(
        "research_analysis",
        required_capabilities={"structured_output"},
    ).with_structured_output(CoachReply)
    numbers = {s.number for s in project.sources}

    for _ in range(2):  # first try + one fix
        reply = llm.invoke(messages)
        problems = check_coach(reply, numbers)
        if not problems:
            return reply
        messages += [AIMessage(reply.model_dump_json()), HumanMessage("Please fix these problems: " + "; ".join(problems))]
    raise CoachError("The coach's reply still had problems: " + "; ".join(problems))
