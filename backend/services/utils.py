from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from sts2_core.paths import PROJECT_ROOT


def clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(parsed, high))


def positive_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0, parsed)


def at_least_one_int(value: Any, default: int = 1) -> int:
    return max(1, positive_int(value, default))


def maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def now_ms() -> int:
    return int(time.time() * 1000)


def normalize_language(value: Any) -> str:
    language = str(value or "").strip().lower()
    if language in {"en", "eng", "english", "en-us", "en_us"}:
        return "en"
    return "zh"


def stream_event(event_type: str, data: dict[str, Any] | None = None) -> str:
    payload = {"type": event_type, "ts": now_ms(), **(data or {})}
    return json.dumps(payload, ensure_ascii=False) + "\n"


def content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value or "")


def response_reasoning_content(response: Any) -> str:
    additional_kwargs = getattr(response, "additional_kwargs", {})
    if isinstance(additional_kwargs, dict):
        value = additional_kwargs.get("reasoning_content") or additional_kwargs.get("reasoning")
        if value:
            return str(value)
    return ""


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def safe_log_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        text = "anonymous"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return text[:120] or f"anonymous-{int(time.time() * 1000)}"


def recent_chat_messages(raw_messages: list[dict[str, Any]], limit: int = 0) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    for message in raw_messages:
        role = str(message.get("role") or "").lower()
        if role not in {"user", "assistant", "system"}:
            continue
        content = str(message.get("content") or "").strip()
        if content:
            item: dict[str, Any] = {"role": role, "content": content}
            if role == "assistant":
                reasoning_content = str(
                    message.get("reasoning_content")
                    or message.get("reasoningContent")
                    or ""
                )
                if reasoning_content:
                    item["reasoning_content"] = reasoning_content
            clean.append(item)
    return clean[-limit:] if limit else clean


def workspace_relative_path(path: Any) -> str:
    raw_path = str(path or "").strip()
    if not raw_path:
        return ""
    candidate = Path(raw_path)
    try:
        resolved = candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
        if resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents:
            return resolved.relative_to(PROJECT_ROOT).as_posix()
    except (OSError, ValueError):
        pass
    return raw_path


def trace_write_target(trace: dict[str, Any]) -> str:
    tool = str(trace.get("tool") or "")
    if tool not in {"local_file_write", "local_file_replace", "local_file_copy_tree", "local_file_create_dir"}:
        return ""
    result = trace.get("result", {})
    args = trace.get("arguments", {})
    if not isinstance(result, dict):
        result = {}
    if not isinstance(args, dict):
        args = {}
    if tool == "local_file_copy_tree":
        raw_path = result.get("target_path") or args.get("target_path")
    else:
        raw_path = result.get("path") or args.get("path")
    return workspace_relative_path(raw_path).replace("\\", "/").lower()


def mcp_trace_count(traces) -> int:
    return sum(
        1
        for trace in traces
        if str(trace.get("tool", "")).startswith("local_file_") or str(trace.get("tool", "")) == "rag_query"
    )


def trim_tool_message_history(messages: list, *, keep_recent: int = 8, summary_chars: int = 400) -> list:
    """Sliding-window trim of ToolMessage history.

    Replaces the *content* of ToolMessages older than the most recent ``keep_recent``
    with a short stub so the LLM still sees the call happened (and gets the
    declared path / a hint of the result) without re-paying for full file
    contents on every turn. The list is returned as a NEW list; AIMessage and
    HumanMessage / SystemMessage entries are passed through unchanged.

    Heuristic stub format::

        [stale tool result; first {summary_chars} chars retained]
        <truncated payload>

    Streaming generators must keep ToolMessages aligned with their preceding
    AIMessage tool_calls; we never drop messages, only shrink content.
    """
    try:
        from langchain_core.messages import ToolMessage
    except ImportError:
        return list(messages)

    keep_recent = max(1, keep_recent)
    summary_chars = max(80, summary_chars)
    tool_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    if len(tool_indices) <= keep_recent:
        return list(messages)
    stale_cutoff = set(tool_indices[:-keep_recent])

    result = []
    for idx, message in enumerate(messages):
        if idx in stale_cutoff and isinstance(message, ToolMessage):
            raw = str(getattr(message, "content", "") or "")
            if len(raw) <= summary_chars + 64:
                result.append(message)
                continue
            stub = (
                "[stale tool result trimmed; only the first "
                f"{summary_chars} chars kept to save context]\n"
                + raw[:summary_chars]
            )
            result.append(
                ToolMessage(
                    content=stub,
                    tool_call_id=getattr(message, "tool_call_id", None) or "",
                )
            )
        else:
            result.append(message)
    return result


def rag_query_summaries_from_traces(traces) -> tuple[int, list[str], list[str]]:
    context_count = 0
    queries: list[str] = []
    query_parts: list[str] = []
    for trace in traces:
        if trace.get("tool") != "rag_query":
            continue
        result = trace.get("result", {})
        if not isinstance(result, dict) or not result.get("ok"):
            continue
        context_count += int(result.get("context_count") or 0)
        query = str(result.get("query") or "").strip()
        if query:
            queries.append(query)
        for part in result.get("query_parts", []):
            text = str(part or "").strip()
            if text:
                query_parts.append(text)
    return context_count, queries, query_parts
