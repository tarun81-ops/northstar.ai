"""Niche Mentor: interest signals -> 3-5 specific niches, each with a short curiosity experiment.

ONE structured-output call. Python checks that every niche cites real signal ids (and the experiment limits),
and sends problems back once.
"""
import json
import logging
from typing import Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from northstar.ai.context_manager import estimate_prompt_tokens, should_compact
from northstar.ai.model_router import get_llm
from northstar.ai.tool_output_utils import truncate_text
from northstar.ai import llm_config as _llm_config
from northstar.db.models import InterestSignal
from northstar.prompts import system_prompt
from northstar.schemas import NicheDraft, NicheList, ProfileIn
from northstar.services.niches import check_niches

logger = logging.getLogger(__name__)


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

    # Automatic compaction: if the prompt is too large, reduce signal detail.
    if should_compact(messages):
        logger.info(
            "suggest_niches: prompt estimated %d tokens (> %d threshold); reducing signals",
            estimate_prompt_tokens(messages),
            _llm_config.context_compact_threshold(),
        )
        facts["signals"] = facts["signals"][:10]
        for s in facts["signals"]:
            s["signal"] = truncate_text(s.get("signal", ""), max_chars=100)
        messages = [
            SystemMessage(system_prompt("niche")),
            HumanMessage("Suggest niches for this student.\n" + json.dumps(facts, indent=1)),
        ]

    llm = get_llm(
        "complex_reasoning",
        required_capabilities={"structured_output"},
    ).with_structured_output(NicheList)

    for _ in range(2):  # first try + one fix
        result = llm.invoke(messages)
        problems = check_niches(result.niches, {s.id for s in signals})
        if not problems:
            return result.niches
        messages += [AIMessage(result.model_dump_json()), HumanMessage("Please fix these problems: " + "; ".join(problems))]
    raise NicheError("The suggestions still had problems: " + "; ".join(problems))
