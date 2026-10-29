"""Tests for the context manager: trimming, compaction, and context building.

No network calls. ``compact_history`` and ``compact_if_needed`` use a fake LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from northstar.ai.context_manager import (
    CompactSummary,
    build_context,
    compact_history,
    compact_if_needed,
    estimate_prompt_tokens,
    estimate_tokens,
    should_compact,
    trim_messages,
)


# ---- estimate_tokens --------------------------------------------------------

def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_positive():
    n = estimate_tokens("hello world")
    assert n > 0


# ---- trim_messages ----------------------------------------------------------

def test_trim_messages_under_limit_keeps_all():
    msgs = [SystemMessage("sys"), HumanMessage("hi"), AIMessage("yo")]
    assert trim_messages(msgs, max_messages=20) == msgs


def test_trim_messages_drops_oldest_non_system():
    msgs = [
        SystemMessage("sys"),
        HumanMessage("m1"),
        AIMessage("r1"),
        HumanMessage("m2"),
        AIMessage("r2"),
        HumanMessage("m3"),
    ]
    result = trim_messages(msgs, max_messages=2)
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == "sys"
    non_sys = [m for m in result if not isinstance(m, SystemMessage)]
    assert len(non_sys) == 2
    assert [m.content for m in non_sys] == ["r2", "m3"]


def test_trim_messages_preserves_system_order():
    msgs = [
        SystemMessage("sys1"),
        HumanMessage("m1"),
        SystemMessage("sys2"),
        HumanMessage("m2"),
        HumanMessage("m3"),
        HumanMessage("m4"),
        HumanMessage("m5"),
    ]
    result = trim_messages(msgs, max_messages=2)
    sys_msgs = [m for m in result if isinstance(m, SystemMessage)]
    assert [m.content for m in sys_msgs] == ["sys1", "sys2"]
    non_sys = [m for m in result if not isinstance(m, SystemMessage)]
    assert len(non_sys) == 2


def test_trim_messages_keeps_tool_call_result_pairs_together():
    call = AIMessage(
        content="",
        tool_calls=[
            {"name": "lookup", "args": {}, "id": "tool-1", "type": "tool_call"}
        ],
    )
    result = ToolMessage(content="found", tool_call_id="tool-1")
    messages = [HumanMessage("old"), call, result, HumanMessage("new")]

    trimmed = trim_messages(messages, max_messages=2)

    assert trimmed == [call, result, messages[-1]]


# ---- CompactSummary ---------------------------------------------------------

def test_compact_summary_to_text():
    s = CompactSummary(
        student_goal="Score 85% in maths",
        current_milestone="Algebra baseline",
        completed_work=["Finished worksheet 1"],
        blockers=["Factorising is hard"],
        important_decisions=["Focus on algebra first"],
        deadlines=["2026-11-16"],
        next_action="Do 5 factorising problems",
    )
    text = s.to_text()
    assert "Score 85% in maths" in text
    assert "Algebra baseline" in text
    assert "Factorising is hard" in text
    assert "2026-11-16" in text


def test_compact_summary_empty():
    s = CompactSummary()
    assert s.to_text() == "(no prior context)"


# ---- compact_history --------------------------------------------------------

class FakeLLMForCompact:
    def __init__(self, summary_json: str):
        self.summary_json = summary_json
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return type("R", (), {"content": self.summary_json})()


def test_compact_history_under_keep_recent_returns_empty():
    fake = FakeLLMForCompact("{}")
    msgs = [HumanMessage("m1"), HumanMessage("m2")]
    summary = compact_history(msgs, fake, keep_recent=10)
    assert summary == CompactSummary()
    assert fake.calls == []


def test_compact_history_summarizes_older_messages():
    fake = FakeLLMForCompact(
        '{"student_goal": "Score 85% in maths", "current_milestone": "Algebra", '
        '"completed_work": ["worksheet 1"], "blockers": ["factorising"], '
        '"important_decisions": ["focus algebra"], "deadlines": ["2026-11-16"], '
        '"next_action": "5 problems"}'
    )
    msgs = [
        HumanMessage("old msg 1"),
        HumanMessage("old msg 2"),
        HumanMessage("recent msg"),
    ]
    summary = compact_history(msgs, fake, keep_recent=1)
    assert summary.student_goal == "Score 85% in maths"
    assert summary.current_milestone == "Algebra"
    assert "factorising" in summary.blockers
    assert len(fake.calls) == 1


def test_compact_history_llm_failure_returns_empty():
    class BrokenLLM:
        def invoke(self, messages):
            raise RuntimeError("down")

    msgs = [HumanMessage("old")] * 15
    summary = compact_history(msgs, BrokenLLM(), keep_recent=1)
    assert summary == CompactSummary()


# ---- build_context ----------------------------------------------------------

def test_build_context_includes_system_prompt():
    ctx = build_context("You are a mentor.", [], None)
    assert isinstance(ctx[0], SystemMessage)
    assert ctx[0].content == "You are a mentor."


def test_build_context_includes_persisted_state():
    ctx = build_context(
        "sys",
        [],
        persisted_state={"goal": "Score 85%", "active_weekly_plan": "Mon: algebra"},
    )
    state_msgs = [m for m in ctx if isinstance(m, SystemMessage) and "Persistent student context" in m.content]
    assert len(state_msgs) == 1
    assert "Score 85%" in state_msgs[0].content
    assert "Mon: algebra" in state_msgs[0].content


def test_build_context_trims_recent_messages():
    recent = [HumanMessage(f"msg {i}") for i in range(25)]
    ctx = build_context("sys", recent, None, max_messages=5)
    non_sys = [m for m in ctx if not isinstance(m, SystemMessage)]
    assert len(non_sys) == 5


# ---- should_compact / compact_if_needed -------------------------------------

def test_should_compact_below_threshold():
    msgs = [HumanMessage("short message")]
    assert not should_compact(msgs, threshold=100000)


def test_should_compact_above_threshold():
    msgs = [HumanMessage("x" * 100000)]
    assert should_compact(msgs, threshold=1000)


def test_compact_if_needed_below_threshold_unchanged():
    msgs = [HumanMessage("short")]
    result, compacted = compact_if_needed(msgs, llm=None, threshold=100000)
    assert compacted is False
    assert result == msgs


def test_compact_if_needed_above_threshold_trims():
    class FakeLLM:
        def invoke(self, messages):
            return type("R", (), {"content": "{}"})()

    msgs = [HumanMessage("x" * 50000)] * 3  # way over threshold
    result, compacted = compact_if_needed(msgs, FakeLLM(), threshold=1000, keep_recent=1)
    assert compacted is True
    assert any(
        "Compressed conversation summary" in m.content
        for m in result
        if isinstance(m, SystemMessage)
    )
