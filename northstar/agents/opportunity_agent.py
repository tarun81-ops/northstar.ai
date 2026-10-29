"""Opportunity Agent: eligible, pre-ranked candidates -> the 2-5 best fits, each with "why it matches".

ONE structured-output call. The LLM never sees cost, deadline or eligibility (it can't restate them wrongly) -
the page shows those from the curated data. Python checks the keys, with one retry.
"""
import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from northstar.ai.context_manager import estimate_prompt_tokens, should_compact
from northstar.ai.model_router import get_llm
from northstar.ai.tool_output_utils import truncate_text
from northstar.ai import llm_config as _llm_config
from northstar.prompts import system_prompt
from northstar.schemas import Opportunity, OpportunityMatch, OpportunityMatchList, ProfileIn
from northstar.services.opportunities import check_matches

logger = logging.getLogger(__name__)


class OpportunityError(Exception):
    pass


def match_opportunities(profile: ProfileIn, candidates: list[Opportunity], signals: list[str],
                        niches: list[str]) -> list[OpportunityMatch]:
    facts = {
        "student": profile.model_dump(mode="json", include={"age_band", "grade", "subjects", "interests", "strengths",
                                                           "learning_preferences"}),
        "interest_signals": signals,
        "niches_explored": niches,
        "candidates": [o.model_dump(mode="json", include={"key", "title", "type", "summary", "tags", "skills_required",
                                                          "format", "time_commitment"}) for o in candidates],
    }
    messages = [
        SystemMessage(system_prompt("opportunity")),
        HumanMessage("Pick the best-fitting opportunities.\n" + json.dumps(facts, indent=1)),
    ]

    # Automatic compaction: if the prompt is too large, reduce candidate detail.
    if should_compact(messages):
        logger.info(
            "match_opportunities: prompt estimated %d tokens (> %d threshold); reducing candidates",
            estimate_prompt_tokens(messages),
            _llm_config.context_compact_threshold(),
        )
        facts["candidates"] = facts["candidates"][:3]
        for c in facts["candidates"]:
            c["summary"] = truncate_text(c.get("summary", ""), max_chars=100)
            c["tags"] = c.get("tags", [])[:3]
        messages = [
            SystemMessage(system_prompt("opportunity")),
            HumanMessage("Pick the best-fitting opportunities.\n" + json.dumps(facts, indent=1)),
        ]

    llm = get_llm(
        "complex_reasoning",
        required_capabilities={"structured_output"},
    ).with_structured_output(OpportunityMatchList)
    keys = {o.key for o in candidates}

    for _ in range(2):  # first try + one fix
        result = llm.invoke(messages)
        problems = check_matches(result.matches, keys)
        if not problems:
            return result.matches
        messages += [AIMessage(result.model_dump_json()), HumanMessage("Please fix these problems: " + "; ".join(problems))]
    raise OpportunityError("The matches still had problems: " + "; ".join(problems))
