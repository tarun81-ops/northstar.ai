"""Tests for safe tool-output handling.

No network calls and no API key required.
"""

from __future__ import annotations

import pandas as pd
import pytest

from northstar.ai.tool_output_utils import (
    dataset_summary,
    dataset_summary_text,
    notebook_output_summary,
    summarize_tool_output,
    truncate_text,
)


# ---- truncate_text ----------------------------------------------------------

def test_truncate_text_under_limit():
    text = "hello world"
    assert truncate_text(text, max_chars=100) == text


def test_truncate_text_over_limit():
    text = "x" * 5000
    result = truncate_text(text, max_chars=4000)
    assert len(result) <= 4000
    assert result.endswith("\n...[truncated]")


def test_truncate_text_empty():
    assert truncate_text("") == ""
    assert truncate_text(None) == ""


def test_truncate_text_exactly_at_limit():
    text = "x" * 4000
    assert truncate_text(text, max_chars=4000) == text


# ---- dataset_summary --------------------------------------------------------

def test_dataset_summary_basic():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    summary = dataset_summary(df)
    assert summary["row_count"] == 3
    assert summary["column_count"] == 2
    assert "a" in summary["columns"]
    assert "b" in summary["columns"]
    assert summary["dtypes"]["a"] in ("int64", "int32")
    assert summary["dtypes"]["b"] == str(df["b"].dtype)
    assert summary["missing_values"]["a"] == 0
    assert summary["missing_values"]["b"] == 0


def test_dataset_summary_missing_values():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]})
    summary = dataset_summary(df)
    assert summary["missing_values"]["a"] == 1
    assert summary["missing_values"]["b"] == 1


def test_dataset_summary_head_only_5_rows():
    df = pd.DataFrame({"a": range(100)})
    summary = dataset_summary(df)
    assert len(summary["head"]) == 5


def test_dataset_summary_numeric_stats():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    summary = dataset_summary(df)
    assert "a" in summary["numeric_summary"]
    assert summary["numeric_summary"]["a"]["mean"] == 3.0


def test_dataset_summary_none():
    summary = dataset_summary(None)
    assert summary["row_count"] == 0
    assert summary["column_count"] == 0
    assert summary["head"] == []


# ---- dataset_summary_text ---------------------------------------------------

def test_dataset_summary_text_contains_key_info():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    text = dataset_summary_text(df)
    assert "Rows: 3" in text
    assert "Columns: 2" in text
    assert "a" in text
    assert "b" in text


def test_dataset_summary_text_truncates():
    df = pd.DataFrame({"a": range(1000)})
    text = dataset_summary_text(df, max_chars=100)
    assert len(text) <= 100


# ---- notebook_output_summary ------------------------------------------------

def test_notebook_output_summary_dict():
    output = {
        "success": True,
        "errors": [],
        "final_metrics": {"accuracy": 0.95},
        "important_prints": ["Model trained", "Accuracy: 0.95"],
        "logs": "Epoch 1... Epoch 2...",
    }
    result = notebook_output_summary(output)
    assert result["success"] is True
    assert result["errors"] == []
    assert result["final_metrics"] == {"accuracy": 0.95}
    assert "Model trained" in result["important_prints"]


def test_notebook_output_summary_with_errors():
    output = {
        "success": False,
        "errors": ["ValueError: bad input", "KeyError: missing column"],
        "final_metrics": {},
        "important_prints": [],
        "logs": "Traceback...",
    }
    result = notebook_output_summary(output)
    assert result["success"] is False
    assert len(result["errors"]) == 2


def test_notebook_output_summary_none():
    result = notebook_output_summary(None)
    assert result["success"] is False
    assert "No output provided" in result["errors"]


def test_notebook_output_summary_string():
    result = notebook_output_summary("some log output")
    assert result["success"] is True
    assert result["truncated_logs"] == "some log output"


def test_notebook_output_summary_truncates_logs():
    output = {"logs": "x" * 5000}
    result = notebook_output_summary(output)
    assert len(result["truncated_logs"]) <= 2000


# ---- summarize_tool_output --------------------------------------------------

class FakeLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return type("R", (), {"content": self.response_text})()


def test_summarize_tool_output_with_llm():
    llm = FakeLLM("The tool found 3 matching datasets.")
    result = summarize_tool_output("match_datasets", {"matches": 3}, llm)
    assert "matching datasets" in result
    assert len(llm.calls) == 1


def test_summarize_tool_output_without_llm():
    result = summarize_tool_output("my_tool", {"key": "value" * 500}, None)
    assert "[my_tool output (truncated)]" in result
    assert len(result) <= 4000 + 100  # some overhead for the wrapper


def test_summarize_tool_output_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = summarize_tool_output("load_dataset", df, None)
    assert "Rows: 3" in result
    assert "Columns: 2" in result


def test_summarize_tool_output_llm_failure_falls_back():
    class BrokenLLM:
        def invoke(self, messages):
            raise RuntimeError("down")

    result = summarize_tool_output("my_tool", {"key": "value"}, BrokenLLM())
    assert "[my_tool output (truncated)]" in result
