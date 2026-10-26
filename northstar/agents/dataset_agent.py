"""Dataset Agent: selects best-fitting datasets from pre-filtered catalog candidates.

Flow:
1. Python pre-filters and ranks candidates by tag overlap.
2. Filtered candidates are sent to the LLM with structured output.
3. LLM picks up to 5 datasets and provides student-friendly match reasons.
4. Python strictly validates that all returned refs exist in the candidate list.
5. If invalid ref, retries once.
6. If it still fails, falls back to top Python-ranked candidates with neutral explanation.
"""
from __future__ import annotations

import json
from typing import Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from northstar.db.models import KaggleDataset
from northstar.llm import get_llm
from northstar.prompts import system_prompt
from northstar.schemas import DatasetMatch, DatasetMatchList


def check_dataset_matches(matches: list[DatasetMatch], valid_refs: set[str]) -> list[str]:
    """Validate that every match corresponds to an approved candidate ref without duplicates."""
    problems = []
    if not matches:
        problems.append("No dataset recommendations were returned.")
    seen = set()
    for m in matches:
        if m.kaggle_ref not in valid_refs:
            problems.append(f"'{m.kaggle_ref}' is not in the approved candidate list.")
        if m.kaggle_ref in seen:
            problems.append(f"Duplicate recommendation for '{m.kaggle_ref}'.")
        seen.add(m.kaggle_ref)
    return problems


def fallback_ranked_matches(
    candidates: Sequence[KaggleDataset], limit: int = 5
) -> list[DatasetMatch]:
    """Deterministic fallback: return top Python-ranked candidates with neutral explanation."""
    matches = []
    for d in candidates[:limit]:
        reason = (
            f"Pre-ranked match based on shared research tags ({', '.join(d.niche_tags[:3])}). "
            f"Provides relevant tabular data to explore questions around {d.title.lower()}."
        )
        matches.append(DatasetMatch(kaggle_ref=d.kaggle_ref, match_reason=reason))
    return matches


def match_datasets(
    project_title: str,
    question: str,
    niche_name: str,
    candidates: Sequence[KaggleDataset],
) -> list[DatasetMatch]:
    """Select best-fitting datasets from pre-filtered candidates via LLM structured output.

    Retries once on invalid refs and falls back to top Python-ranked candidates on failure.
    """
    if not candidates:
        return []

    valid_refs = {d.kaggle_ref for d in candidates}
    facts = {
        "project_title": project_title,
        "research_question": question,
        "niche": niche_name or "General student research",
        "candidates": [
            {
                "kaggle_ref": d.kaggle_ref,
                "title": d.title,
                "description": d.description,
                "niche_tags": d.niche_tags,
                "difficulty": d.difficulty,
                "sample_questions": d.sample_questions,
                "limitations": d.limitations,
            }
            for d in candidates
        ],
    }

    messages = [
        SystemMessage(system_prompt("dataset")),
        HumanMessage(
            "Select the best-fitting datasets for this student's project:\n"
            + json.dumps(facts, indent=1)
        ),
    ]

    llm = get_llm().with_structured_output(DatasetMatchList)

    for attempt in range(2):  # First try + 1 retry
        try:
            result = llm.invoke(messages)
            if result and hasattr(result, "matches"):
                problems = check_dataset_matches(result.matches, valid_refs)
                if not problems:
                    return result.matches[:5]
                if attempt == 0:
                    messages += [
                        AIMessage(result.model_dump_json()),
                        HumanMessage(
                            f"Please fix these problems: {'; '.join(problems)}. Only use refs from the candidates list."
                        ),
                    ]
        except Exception:
            if attempt == 1:
                break

    return fallback_ranked_matches(candidates, limit=min(5, len(candidates)))
