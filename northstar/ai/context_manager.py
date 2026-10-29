"""Context management for LangGraph chat history: trimming, compaction, and context building.

Keeps prompts small by replacing older conversation turns with a structured
summary, while always preserving the system prompt, the student's active
goal/plan, and the most recent user/assistant exchange.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from northstar.ai import llm_config

logger = logging.getLogger(__name__)

# ---- token estimation -------------------------------------------------------

try:
    import tiktoken  # type: ignore

    _ENCODER = tiktoken.get_encoding("cl100k_base")

    def estimate_tokens(text: str) -> int:
        """Estimate the number of tokens in ``text`` using tiktoken when available."""
        return len(_ENCODER.encode(text))

except Exception:  # pragma: no cover - tiktoken may not be installed

    def estimate_tokens(text: str) -> int:
        """Fallback approximation: roughly 4 characters per token for English text."""
        return max(1, len(text) // 4)


# ---- structured compaction summary ------------------------------------------

@dataclass
class CompactSummary:
    """Structured summary of older conversation history.

    Fields map to the things a student mentor needs to remember without
    re-reading the full transcript.
    """

    student_goal: str | None = None
    current_milestone: str | None = None
    completed_work: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    important_decisions: list[str] = field(default_factory=list)
    deadlines: list[str] = field(default_factory=list)
    next_action: str | None = None

    def to_text(self) -> str:
        """Render the summary as a short, LLM-friendly block of text."""
        lines: list[str] = []
        if self.student_goal:
            lines.append(f"Goal: {self.student_goal}")
        if self.current_milestone:
            lines.append(f"Current milestone: {self.current_milestone}")
        if self.completed_work:
            lines.append("Completed work: " + "; ".join(self.completed_work))
        if self.blockers:
            lines.append("Blockers: " + "; ".join(self.blockers))
        if self.important_decisions:
            lines.append("Decisions: " + "; ".join(self.important_decisions))
        if self.deadlines:
            lines.append("Deadlines: " + "; ".join(self.deadlines))
        if self.next_action:
            lines.append(f"Next action: {self.next_action}")
        return "\n".join(lines) if lines else "(no prior context)"


# ---- message trimming -------------------------------------------------------

def trim_messages(messages: list, max_messages: int = 20) -> list:
    """Return ``messages`` trimmed to the most recent ``max_messages`` turns.

    System messages are always preserved at the front.  When there are more
    non-system messages than ``max_messages``, the oldest ones are dropped.
    """
    if not messages:
        return []

    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(non_system) <= max_messages:
        return list(messages)

    start = max(0, len(non_system) - max_messages)
    while start > 0:
        candidate = non_system[start:]
        call_ids = {
            call.get("id")
            for message in candidate
            if isinstance(message, AIMessage)
            for call in message.tool_calls
            if call.get("id")
        }
        orphaned_tool_result = any(
            isinstance(message, ToolMessage) and message.tool_call_id not in call_ids
            for message in candidate
        )
        if not orphaned_tool_result:
            break
        start -= 1
    kept = non_system[start:]
    logger.info(
        "trim_messages: dropped %d older message(s), kept %d (max=%d)",
        len(non_system) - len(kept),
        len(kept),
        max_messages,
    )
    return system_msgs + kept


# ---- history compaction -----------------------------------------------------

def _messages_to_text(messages: list) -> str:
    """Flatten a list of LangChain messages into a single transcript string."""
    parts: list[str] = []
    for m in messages:
        role = "system" if isinstance(m, SystemMessage) else ("assistant" if isinstance(m, AIMessage) else "user")
        parts.append(f"[{role}] {m.content}")
    return "\n".join(parts)


def compact_history(messages: list, llm, keep_recent: int = 10) -> CompactSummary:
    """Summarize older messages into a :class:`CompactSummary`.

    Everything except the most recent ``keep_recent`` non-system messages is
    considered "older" and is summarized using ``llm``.  The summary preserves
    student goal, current milestone, completed work, blockers, important
    decisions, deadlines, and next action.

    If the LLM call fails, a best-effort empty summary is returned so the
    agent can continue without history.
    """
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(non_system) <= keep_recent:
        return CompactSummary()

    older = non_system[:-keep_recent]
    transcript = _messages_to_text(older)

    summary_prompt = (
        "You are an assistant that compresses old conversation history into a "
        "structured summary for a student mentor. Read the older messages below "
        "and extract ONLY these fields, returning valid JSON:\n"
        "- student_goal: the student's stated goal (string or null)\n"
        "- current_milestone: the milestone they are working on (string or null)\n"
        "- completed_work: list of things they have finished (list of strings)\n"
        "- blockers: list of things blocking them (list of strings)\n"
        "- important_decisions: list of decisions made (list of strings)\n"
        "- deadlines: list of dates/deadlines mentioned (list of strings)\n"
        "- next_action: the single next step they should take (string or null)\n\n"
        "Older messages:\n---\n"
        + transcript
        + "\n---\nReturn only the JSON object."
    )

    try:
        response = llm.invoke([HumanMessage(summary_prompt)])
        raw = response.content if hasattr(response, "content") else str(response)
        data = json.loads(raw)
        return CompactSummary(
            student_goal=data.get("student_goal"),
            current_milestone=data.get("current_milestone"),
            completed_work=data.get("completed_work") or [],
            blockers=data.get("blockers") or [],
            important_decisions=data.get("important_decisions") or [],
            deadlines=data.get("deadlines") or [],
            next_action=data.get("next_action"),
        )
    except Exception:  # pragma: no cover - defensive; LLM may be unavailable
        logger.exception("compact_history: LLM summarization failed; returning empty summary")
        return CompactSummary()


# ---- context building -------------------------------------------------------

def build_context(
    system_prompt: str,
    recent_messages: list,
    persisted_state: dict | None = None,
    max_messages: int = 20,
) -> list:
    """Build the message list to send to the LLM.

    Always includes:
    1. The system prompt (first).
    2. Persisted state (goal, active plan, etc.) as a system message, when provided.
    3. The most recent messages, trimmed to ``max_messages``.
    """
    messages: list = [SystemMessage(system_prompt)]

    if persisted_state:
        state_lines: list[str] = []
        for key in ("goal", "active_weekly_plan", "current_milestone", "next_action"):
            value = persisted_state.get(key)
            if value is not None:
                state_lines.append(f"{key}: {value}")
        if state_lines:
            messages.append(SystemMessage("Persistent student context:\n" + "\n".join(state_lines)))

    trimmed = trim_messages(recent_messages, max_messages=max_messages)
    messages.extend(trimmed)
    return messages


# ---- automatic compaction before agent calls --------------------------------

def estimate_prompt_tokens(messages: list) -> int:
    """Estimate total tokens across all message contents."""
    return sum(estimate_tokens(getattr(m, "content", "") or "") for m in messages)


def should_compact(messages: list, threshold: int | None = None) -> bool:
    """Return True if the prompt's estimated token count exceeds the threshold."""
    if threshold is None:
        threshold = llm_config.context_compact_threshold()
    return estimate_prompt_tokens(messages) > threshold


def compact_if_needed(
    messages: list,
    llm,
    threshold: int | None = None,
    keep_recent: int = 10,
) -> tuple[list, bool]:
    """If the prompt exceeds the token threshold, compact older history.

    Returns ``(messages, compacted)`` where ``compacted`` is True when a
    compaction summary was inserted.
    """
    if threshold is None:
        threshold = llm_config.context_compact_threshold()

    if not should_compact(messages, threshold):
        return messages, False

    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(non_system) <= keep_recent:
        return trim_messages(messages, max_messages=keep_recent), True

    older = non_system[:-keep_recent]
    summary = compact_history(older, llm)

    summary_text = summary.to_text()
    logger.info(
        "compact_if_needed: prompt estimated %d tokens (> %d threshold); "
        "compacted %d older message(s) into summary",
        estimate_prompt_tokens(messages),
        threshold,
        len(older),
    )

    summary_msg = SystemMessage(f"Compressed conversation summary:\n{summary_text}")
    kept_recent = non_system[-keep_recent:]

    new_messages: list = []
    for m in system_msgs:
        new_messages.append(m)
    new_messages.append(summary_msg)
    new_messages.extend(kept_recent)
    return new_messages, True
