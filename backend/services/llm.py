from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

try:
    from langchain_openai import ChatOpenAI
    import langchain_openai.chat_models.base as langchain_openai_base
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing package: langchain-openai. Install it in conda env sts2_agent.") from exc


_ORIGINAL_CONVERT_DICT_TO_MESSAGE = langchain_openai_base._convert_dict_to_message
_ORIGINAL_CONVERT_DELTA_TO_MESSAGE_CHUNK = langchain_openai_base._convert_delta_to_message_chunk


def _patch_langchain_openai_reasoning_content() -> None:
    """Preserve third-party OpenAI-compatible reasoning_content fields."""
    if getattr(langchain_openai_base, "_sts2_reasoning_content_patched", False):
        return

    def convert_dict_to_message_with_reasoning(message_dict: Any) -> Any:
        message = _ORIGINAL_CONVERT_DICT_TO_MESSAGE(message_dict)
        if isinstance(message, AIMessage):
            reasoning_content = ""
            if isinstance(message_dict, dict):
                reasoning_content = str(message_dict.get("reasoning_content") or "")
            if reasoning_content:
                message.additional_kwargs["reasoning_content"] = reasoning_content
        return message

    def convert_delta_to_message_chunk_with_reasoning(message_dict: Any, default_class: Any) -> Any:
        chunk = _ORIGINAL_CONVERT_DELTA_TO_MESSAGE_CHUNK(message_dict, default_class)
        reasoning_content = ""
        if isinstance(message_dict, dict):
            reasoning_content = str(message_dict.get("reasoning_content") or "")
        if reasoning_content and hasattr(chunk, "additional_kwargs"):
            chunk.additional_kwargs["reasoning_content"] = reasoning_content
        return chunk

    langchain_openai_base._convert_dict_to_message = convert_dict_to_message_with_reasoning
    langchain_openai_base._convert_delta_to_message_chunk = convert_delta_to_message_chunk_with_reasoning
    langchain_openai_base._sts2_reasoning_content_patched = True


_patch_langchain_openai_reasoning_content()


def _approx_token_count(message: dict[str, Any]) -> int:
    """Rough token estimate for an OpenAI-format chat message.

    Uses ~3 chars/token (conservative for mixed CJK + code). Includes content,
    tool_calls arguments, and tool_call_id so structural fields are counted.
    """
    total_chars = 0
    content = message.get("content")
    if isinstance(content, str):
        total_chars += len(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    total_chars += len(text)
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            if isinstance(function, dict):
                total_chars += len(str(function.get("name") or ""))
                total_chars += len(str(function.get("arguments") or ""))
    tool_call_id = message.get("tool_call_id")
    if isinstance(tool_call_id, str):
        total_chars += len(tool_call_id)
    # +4 fudge for role/structural overhead per message.
    return total_chars // 3 + 4


def _trim_messages_to_budget(
    payload_messages: list[dict[str, Any]],
    context_length: int,
) -> list[dict[str, Any]]:
    """Drop oldest messages until the running token estimate fits the budget.

    Preserves: system messages, the final user message, and tool_call /
    tool-result pairs (an assistant message with tool_calls is removed
    together with all of its tool responses, or kept together).
    """
    if context_length <= 0 or not payload_messages:
        return payload_messages

    # Reserve ~1024 tokens for completion headroom.
    budget = max(context_length - 1024, context_length // 2)
    totals = [_approx_token_count(message) for message in payload_messages]
    if sum(totals) <= budget:
        return payload_messages

    # Identify protected indices: all system messages + the last user message.
    protected: set[int] = set()
    for index, message in enumerate(payload_messages):
        if message.get("role") == "system":
            protected.add(index)
    for index in range(len(payload_messages) - 1, -1, -1):
        if payload_messages[index].get("role") == "user":
            protected.add(index)
            break

    # Group assistant(tool_calls) with their tool-result followers so we drop
    # them atomically.
    groups: list[list[int]] = []
    index = 0
    while index < len(payload_messages):
        message = payload_messages[index]
        if (
            message.get("role") == "assistant"
            and isinstance(message.get("tool_calls"), list)
            and message.get("tool_calls")
        ):
            group = [index]
            j = index + 1
            while j < len(payload_messages) and payload_messages[j].get("role") == "tool":
                group.append(j)
                j += 1
            groups.append(group)
            index = j
        else:
            groups.append([index])
            index += 1

    # Drop groups from oldest to newest, skipping any group that contains a
    # protected index, until under budget.
    kept = [True] * len(payload_messages)
    running = sum(totals)
    for group in groups:
        if running <= budget:
            break
        if any(idx in protected for idx in group):
            continue
        for idx in group:
            kept[idx] = False
            running -= totals[idx]

    if running > budget:
        # Could not trim enough without touching protected messages; return as-is
        # rather than mangling tool_call pairs.
        return payload_messages

    return [message for index, message in enumerate(payload_messages) if kept[index]]


class ReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant that passes reasoning_content back for thinking models."""

    sts2_context_length: int = 0

    def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        payload_messages = payload.get("messages")
        if not isinstance(payload_messages, list):
            return payload

        assistant_messages = [message for message in messages if isinstance(message, AIMessage)]
        assistant_payloads = [
            message
            for message in payload_messages
            if isinstance(message, dict) and message.get("role") == "assistant"
        ]
        for source, target in zip(assistant_messages, assistant_payloads):
            reasoning_content = str(source.additional_kwargs.get("reasoning_content") or "")
            if reasoning_content:
                target["reasoning_content"] = reasoning_content
            if target.get("tool_calls") and target.get("content") is None:
                target["content"] = ""

        context_length = int(getattr(self, "sts2_context_length", 0) or 0)
        if context_length > 0:
            payload["messages"] = _trim_messages_to_budget(payload_messages, context_length)
        return payload


def create_llm(
    provider: dict[str, Any],
    *,
    model_override: str | None = None,
    temperature_override: float | None = None,
) -> ChatOpenAI:
    api_key = str(provider.get("api_key") or "").strip()
    base_url = str(provider.get("base_url") or "").strip()
    model = str(model_override or provider.get("model") or "").strip()
    if not api_key:
        raise ValueError("当前 AI 没有保存 API Key，请先在设置里填写并保存。")
    if not base_url:
        raise ValueError("当前 AI 没有 Base URL，请先在设置里填写并保存。")
    if not model:
        raise ValueError("当前 AI 没有模型名，请先在设置里填写并保存。")
    temperature = temperature_override if temperature_override is not None else float(provider.get("temperature", 0.2))
    try:
        context_length = int(provider.get("context_length") or 0)
    except (TypeError, ValueError):
        context_length = 0
    return ReasoningChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        sts2_context_length=max(0, context_length),
    )
