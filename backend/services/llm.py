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


class ReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant that passes reasoning_content back for thinking models."""

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
    return ReasoningChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=temperature)
