"""Safe tool-output handling for NorthStar AI agents.

Never sends an entire CSV, full notebook execution log, or large JSON blob
to the LLM. All functions truncate or summarize before the output reaches a
model prompt.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def truncate_text(text: str, max_chars: int = 4000) -> str:
    """Truncate ``text`` to at most ``max_chars`` characters.

    Adds a trailing notice when truncation occurs. Never raises on empty
    input.
    """
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    notice = "\n...[truncated]"
    kept = text[: max_chars - len(notice)]
    return kept + notice


# ---- dataset summary --------------------------------------------------------


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact, LLM-safe summary of a pandas DataFrame.

    Returns only schema-level metadata and the first 5 rows — never the full
    CSV contents.

    Keys:
        row_count, column_count, columns, dtypes, missing_values,
        numeric_summary, head (first 5 rows as list of dicts)
    """
    if df is None:
        return {
            "row_count": 0,
            "column_count": 0,
            "columns": [],
            "dtypes": {},
            "missing_values": {},
            "numeric_summary": {},
            "head": [],
        }

    dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    missing = {str(col): int(count) for col, count in df.isnull().sum().items()}

    numeric_df = df.select_dtypes(include=["number"])
    numeric_summary: dict[str, dict[str, Any]] = {}
    if not numeric_df.empty:
        desc = numeric_df.describe()
        for col in desc.columns:
            numeric_summary[str(col)] = {
                str(stat): (float(val) if pd.notna(val) else None)
                for stat, val in desc[col].items()
            }

    head_records = df.head(5).to_dict(orient="records")
    clean_head: list[dict[str, Any]] = []
    for row in head_records:
        clean_row: dict[str, Any] = {}
        for k, v in row.items():
            clean_row[str(k)] = str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
        clean_head.append(clean_row)

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": [str(c) for c in df.columns.tolist()],
        "dtypes": dtypes,
        "missing_values": missing,
        "numeric_summary": numeric_summary,
        "head": clean_head,
    }


def dataset_summary_text(df: pd.DataFrame, max_chars: int = 2000) -> str:
    """Return ``dataset_summary(df)`` as a short, LLM-friendly text block."""
    summary = dataset_summary(df)
    lines: list[str] = [
        f"Rows: {summary['row_count']}, Columns: {summary['column_count']}",
        "Columns & dtypes: " + "; ".join(f"{c} ({summary['dtypes'].get(c, '?')})" for c in summary["columns"]),
    ]
    if summary["missing_values"]:
        missing = {k: v for k, v in summary["missing_values"].items() if v > 0}
        if missing:
            lines.append("Missing values: " + "; ".join(f"{k}={v}" for k, v in missing.items()))
    if summary["numeric_summary"]:
        stats_parts = []
        for col, stats in summary["numeric_summary"].items():
            mean = stats.get("mean")
            std = stats.get("std")
            stats_parts.append(f"{col}: mean={mean}, std={std}")
        lines.append("Numeric summary: " + "; ".join(stats_parts))
    if summary["head"]:
        lines.append("First 5 rows (sample):")
        for i, row in enumerate(summary["head"], 1):
            lines.append(f"  {i}. " + "; ".join(f"{k}={v}" for k, v in row.items()))
    text = "\n".join(lines)
    return truncate_text(text, max_chars=max_chars)


# ---- notebook output summary ------------------------------------------------


def notebook_output_summary(output: Any) -> dict[str, Any]:
    """Summarize notebook / code-execution output without sending full logs.

    Accepts a dict-like object with keys such as ``success``, ``errors``,
    ``metrics``, ``prints``, ``logs``. Returns a dict with:
        success, errors, final_metrics, important_prints, truncated_logs
    """
    if output is None:
        return {
            "success": False,
            "errors": ["No output provided"],
            "final_metrics": {},
            "important_prints": [],
            "truncated_logs": "",
        }

    if isinstance(output, str):
        # Treat a raw string as a log.
        return {
            "success": True,
            "errors": [],
            "final_metrics": {},
            "important_prints": [],
            "truncated_logs": truncate_text(output, max_chars=2000),
        }

    if not isinstance(output, dict):
        output = {"raw": output}

    success = bool(output.get("success", True))
    errors = output.get("errors", [])
    if isinstance(errors, list):
        errors = [str(e) for e in errors]
    else:
        errors = [str(errors)]

    final_metrics = output.get("final_metrics", {})
    if not isinstance(final_metrics, dict):
        final_metrics = {"value": str(final_metrics)}

    important_prints = output.get("important_prints", [])
    if isinstance(important_prints, str):
        important_prints = [important_prints]
    elif not isinstance(important_prints, list):
        important_prints = [str(important_prints)]

    logs = output.get("logs", "")
    truncated_logs = truncate_text(str(logs), max_chars=2000)

    return {
        "success": success,
        "errors": errors,
        "final_metrics": final_metrics,
        "important_prints": important_prints[:10],  # cap at 10
        "truncated_logs": truncated_logs,
    }


# ---- generic tool-output summarization --------------------------------------


def summarize_tool_output(tool_name: str, output: Any, llm) -> str:
    """Produce a short, LLM-safe summary of a tool's output.

    For structured outputs (dict/list) the function first truncates the
    serialized form. If an LLM is provided, it asks the model to summarize
    the tool output into 2-3 sentences; on any failure it falls back to
    truncation. Never sends raw CSV rows or full execution logs to the LLM.

    Args:
        tool_name: name of the tool that produced the output (for logging only).
        output: raw tool output (dict, list, DataFrame, str, etc.).
        llm: a LangChain chat model (optional). If None, truncation is used.

    Returns:
        A short summary string safe to include in an LLM prompt.
    """
    # Convert common types to a compact, safe representation first.
    if isinstance(output, pd.DataFrame):
        safe_repr = dataset_summary_text(output, max_chars=1500)
    elif isinstance(output, (dict, list)):
        safe_repr = truncate_text(json.dumps(output, default=str, indent=1), max_chars=2000)
    else:
        safe_repr = truncate_text(str(output), max_chars=2000)

    if llm is None:
        return f"[{tool_name} output (truncated)]:\n{safe_repr}"

    summary_prompt = (
        f"Summarize the following output from the tool '{tool_name}' into 2-3 "
        f"concise sentences. Focus on the key findings, numbers, and any errors. "
        f"Do not reproduce raw data rows or full logs.\n\n---\n{safe_repr}\n---"
    )
    try:
        response = llm.invoke([{"role": "user", "content": summary_prompt}])
        content = getattr(response, "content", str(response))
        return truncate_text(str(content).strip(), max_chars=1000)
    except Exception:  # pragma: no cover - defensive; LLM may be unavailable
        logger.exception("summarize_tool_output: LLM summarization failed for %s", tool_name)
        return f"[{tool_name} output (truncated)]:\n{safe_repr}"
