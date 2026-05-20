from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent import to_langchain_messages
from mcp.local_files import LocalFileMCP
from sts2_core.paths import PROJECT_ROOT

from services.rag import public_context_groups
from services.utils import (
    json_safe,
    normalize_language,
    positive_int,
    recent_chat_messages,
    safe_log_name,
    workspace_relative_path,
)


CHAT_LOG_ROOT = PROJECT_ROOT / "data" / "logs"
_CHAT_LOG_LOCK = Lock()


def conversation_log_path(conversation_id: Any) -> Path:
    return CHAT_LOG_ROOT / f"conversation_{safe_log_name(conversation_id)}.jsonl"


def append_ai_chat_log(entry: dict[str, Any]) -> None:
    log_path = conversation_log_path(entry.get("conversation_id"))
    record = {
        "schema_version": 1,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(json_safe(record), ensure_ascii=False)
        with _CHAT_LOG_LOCK:
            with log_path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
    except OSError as exc:
        print(f"failed to write AI chat log: {exc}", file=sys.stderr)


def append_ai_chat_event(
    *,
    payload: dict[str, Any],
    request_id: str,
    event: str,
    data: dict[str, Any] | None = None,
) -> None:
    append_ai_chat_log(
        {
            "event": event,
            "request_id": request_id,
            "conversation_id": str(payload.get("conversation_id") or ""),
            "data": data or {},
        }
    )


def build_ai_chat_log_entry(
    *,
    endpoint: str,
    status: str,
    payload: dict[str, Any],
    raw_messages: list[dict[str, Any]],
    last_user: str,
    provider: dict[str, Any],
    model: str,
    language: str,
    use_rag: bool,
    use_agent: bool,
    desc_top_k: int,
    context_n: int,
    search_query: str,
    search_query_parts: list[str],
    context_groups: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    answer: str = "",
    reasoning_content: str = "",
    memory_summary_before: str = "",
    memory_summary_after: str = "",
    duration_ms: int = 0,
    error: str = "",
) -> dict[str, Any]:
    contexts = [
        context
        for group in context_groups
        for context in group.get("contexts", [])
        if isinstance(group.get("contexts"), list)
    ]
    return {
        "event": "request_summary",
        "endpoint": endpoint,
        "status": status,
        "conversation_id": str(payload.get("conversation_id") or ""),
        "provider_id": provider.get("provider_id"),
        "model": model,
        "language": language,
        "settings": {
            "use_rag": use_rag,
            "use_agent": use_agent,
            "desc_top_k": desc_top_k,
            "context_n": context_n,
            "agent_max_steps": positive_int(payload.get("agent_max_steps"), 0),
            "selected_files": payload.get("selected_files") if isinstance(payload.get("selected_files"), list) else [],
        },
        "input": {
            "last_user": last_user,
            "messages": recent_chat_messages(raw_messages),
            "memory_summary": memory_summary_before,
        },
        "retrieval": {
            "search_query": search_query,
            "search_query_parts": search_query_parts,
            "context_count": len(contexts),
            "groups": public_context_groups(context_groups),
        },
        "assistant": {
            "answer": answer,
            "reasoning_content": reasoning_content,
            "memory_summary": memory_summary_after,
        },
        "agent_traces": traces,
        "error": error,
        "duration_ms": duration_ms,
    }


def build_grouped_prompt_context(groups: list[dict[str, Any]]) -> str:
    if not groups:
        return "No code context found."

    parts: list[str] = []
    for group_idx, group in enumerate(groups, start=1):
        sub_query = str(group.get("query", ""))
        source_db = str(group.get("source_db", "descriptions"))
        contexts = group.get("contexts", [])
        parts.append(f"######## Retrieval Group {group_idx} ########")
        parts.append(f"SplitQuery: {sub_query}")
        parts.append(f"VectorDomain: {source_db}")
        if not isinstance(contexts, list) or not contexts:
            parts.append("No code context found for this query part.")
            parts.append(f"######## End Retrieval Group {group_idx} ########")
            continue

        for item_idx, context in enumerate(contexts, start=1):
            domain = str(context.get("domain", ""))
            if domain == "powers":
                reference_scope = "POWER_CODE_ONLY"
            elif domain == "cards":
                reference_scope = "CARD_CODE_ONLY"
            else:
                reference_scope = "GAME_TEXT_REFERENCE"
            parts.append(
                "\n".join(
                    [
                        f"[Context {item_idx}]",
                        f"ID: {context.get('id', '')}",
                        f"Domain: {domain}",
                        f"ReferenceScope: {reference_scope}",
                        f"Entity: {context.get('entity_name', '')}",
                        f"ModelFolder: {context.get('model_folder', '')}",
                        f"CodeRole: {context.get('member_name', '')}",
                        f"DescriptionScore: {context.get('desc_score', '')}",
                        f"File: {context.get('file_path', '')}",
                        f"MCPReadPath: {context.get('mcp_file_path') or context.get('file_path', '')}",
                        f"DescriptionKeyPath: {context.get('description_key_path', '')}",
                        f"Title:\n{context.get('title', '')}",
                        f"Description:\n{context.get('description', '')}",
                        f"Code:\n{context.get('code', '')}",
                    ]
                )
            )
        parts.append(f"######## End Retrieval Group {group_idx} ########")
    return "\n\n".join(parts)


def extract_using_namespaces(code: str) -> list[str]:
    namespaces: list[str] = []
    for match in re.finditer(r"^\s*using\s+(?!static\b)([A-Za-z_][A-Za-z0-9_.]*)\s*;", code, re.MULTILINE):
        namespace = match.group(1)
        if namespace.startswith("System"):
            continue
        if namespace not in namespaces:
            namespaces.append(namespace)
    return namespaces


def extract_referenced_type_names(code: str) -> list[str]:
    names: set[str] = set()
    patterns = [
        r"\b([A-Z][A-Za-z0-9_]*)\s*\.",
        r"\bnew\s+([A-Z][A-Za-z0-9_]*)\b",
        r"<\s*([A-Z][A-Za-z0-9_]*)\s*>",
        r":\s*([A-Z][A-Za-z0-9_]*)\b",
    ]
    for pattern in patterns:
        names.update(re.findall(pattern, code))
    blocked = {"Task", "IEnumerable", "List", "Dictionary", "String", "Int32", "Boolean"}
    return sorted(name for name in names if name not in blocked)


def build_namespace_audit(
    *,
    local_files: LocalFileMCP,
    context_groups: list[dict[str, Any]],
    max_namespaces: int = 0,
    max_type_reads: int = 0,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    namespaces: list[str] = []
    referenced_types: set[str] = set()
    for group in context_groups:
        contexts = group.get("contexts", [])
        if not isinstance(contexts, list):
            continue
        for context in contexts:
            if not isinstance(context, dict):
                continue
            code = str(context.get("code") or "")
            for namespace in extract_using_namespaces(code):
                if namespace not in namespaces:
                    namespaces.append(namespace)
            referenced_types.update(extract_referenced_type_names(code))

    traces: list[dict[str, Any]] = []
    text_parts: list[str] = []
    viewed_files: list[str] = []
    listed_namespaces: list[str] = []
    missing_types: list[str] = []
    namespace_roots: list[str] = []

    for namespace in (namespaces[:max_namespaces] if max_namespaces else namespaces):
        rel_dir = f"data/libs/sts2_decompiled/{namespace}"
        listed_namespaces.append(namespace)
        namespace_roots.append(rel_dir)
        list_args = {"path": rel_dir, "pattern": "*.cs", "limit": 0}
        list_result = local_files.call("local_file_list", list_args)
        traces.append({"tool": "local_file_list", "arguments": list_args, "result": list_result})
        items = list_result.get("items", []) if list_result.get("ok") else []
        file_by_stem = {
            str(item.get("name", ""))[:-3]: str(item.get("path", ""))
            for item in items
            if str(item.get("type")) == "file" and str(item.get("name", "")).endswith(".cs")
        }
        text_parts.append(
            f"### Namespace: {namespace}\n"
            f"Directory: {rel_dir}\n"
            f"Files: {', '.join(sorted(file_by_stem)) or 'not found'}"
        )

        for type_name in sorted(referenced_types):
            if max_type_reads and len(viewed_files) >= max_type_reads:
                break
            path = file_by_stem.get(type_name)
            if not path:
                continue
            read_args = {"path": path, "max_chars": 0}
            read_result = local_files.call("local_file_read", read_args)
            traces.append({"tool": "local_file_read", "arguments": read_args, "result": read_result})
            if read_result.get("ok"):
                viewed_files.append(str(read_result.get("path") or path))
                text_parts.append(f"### API File: {read_result.get('path', path)}\n{read_result.get('content', '')}")

    available_files = {
        str(trace.get("arguments", {}).get("path", ""))
        for trace in traces
        if trace.get("tool") == "local_file_read" and trace.get("result", {}).get("ok")
    }
    for type_name in sorted(referenced_types):
        if any(path.endswith(f"/{type_name}.cs") or path.endswith(f"\\{type_name}.cs") for path in available_files):
            continue
        if type_name.endswith("Power") or type_name.endswith("Cmd") or type_name.endswith("Action"):
            missing_types.append(type_name)

    summary = {
        "namespaces": listed_namespaces,
        "namespace_roots": namespace_roots,
        "viewed_files": viewed_files,
        "missing_referenced_types": missing_types,
        "trace_count": len(traces),
    }
    return "\n\n".join(text_parts), traces, summary


def build_system_prompt(language: str) -> str:
    if normalize_language(language) == "en":
        return (
            "=== STS2 Code Assistant ===\n"
            "- Prioritize RAG context + local-file MCP results; default to English\n"
            "- Cards and same-name Powers can be explained together, but do NOT mix their code patterns\n"
            "- Domain=cards / CARD_CODE_ONLY: card file structure/constructors/lifecycles/APIs only\n"
            "- Domain=powers / POWER_CODE_ONLY: Power file structure/constructors/lifecycles/APIs only\n"
            "- Investigation order: rag_query FIRST -> use returned MCPReadPath/file_path directly with local_file_read[_many]\n"
            "- Known multi-path reads: ONE local_file_read_many call with all paths; do NOT broad local_file_search across data/libs/sts2_decompiled or data/\n"
            "- Use local_file_search only when RAG context is insufficient AND you have a specific symbol to grep\n"
            "- rag_query rules and code-writing rules are in two separate system messages — follow them strictly"
        )
    return (
        "=== STS2 代码助手 ===\n"
        "- 优先使用 RAG 上下文 + 本地文件 MCP 结果；默认中文回答\n"
        "- Cards 与同名 Powers 可一起解释，但禁止混用两类代码模式\n"
        "- Domain=cards / CARD_CODE_ONLY 只参考卡牌结构/构造/生命周期/API\n"
        "- Domain=powers / POWER_CODE_ONLY 只参考 Power 结构/构造/生命周期/API\n"
        "- 调查顺序：先 rag_query -> 用返回的 MCPReadPath/file_path 直接 local_file_read[_many]\n"
        "- 已知多路径一次性 local_file_read_many；禁止在 data/libs/sts2_decompiled 或 data/ 大范围 local_file_search\n"
        "- 仅当 RAG 上下文不足且有明确符号要 grep 时才用 local_file_search\n"
        "- rag_query 与代码写入规则在另两条独立系统消息中，严格遵守"
    )


def build_memory_prompt(summary: str, language: str) -> SystemMessage | None:
    summary = summary.strip()
    if not summary:
        return None
    if normalize_language(language) == "en":
        return SystemMessage(
            content=(
                "Conversation summary memory follows. Use it only to complete the context for the user's current question; "
                "if it conflicts with the latest messages or RAG evidence, prefer the latest messages and RAG evidence.\n"
                f"{summary}"
            )
        )
    return SystemMessage(
        content=(
            "会话摘要记忆如下。它只用于补全用户当前问题的上下文；"
            "如果摘要与最新消息或 RAG 证据冲突，以最新消息和 RAG 证据为准。\n"
            f"{summary}"
        )
    )


def build_query_prompt(language: str) -> SystemMessage:
    """Dedicated rules for how to call the rag_query tool.

    Kept separate from the general system prompt so the model treats it as a
    standalone procedural contract rather than one bullet among many.
    """
    if normalize_language(language) == "en":
        return SystemMessage(
            content=(
                "========================================\n"
                "!!! rag_query CALLING RULES - VIOLATION = ERROR !!!\n"
                "========================================\n"
                "1. SPLIT FIRST: if `query` text contains 。，；、！？ . , ; ! ? \\n \\r, split on EVERY occurrence "
                "and call rag_query once per segment. NO splitting punctuation may remain inside `query`.\n"
                "2. DOMAIN(S): valid values are cards / powers / relics / potions / orbs / enchantments / "
                "afflictions / rest_site_ui / events. Default to a single-element list with your best routing guess. "
                "Pass multiple domains in ONE call only when the user explicitly tagged the segment with multiple domains.\n"
                "3. ROUTE BY EFFECT TARGET, not parent object: a relic clause 'upgrade all cards in deck' -> `cards`, "
                "'channel 1 Lightning' -> `orbs`. Fall back to the parent's domain only when the clause describes the "
                "object itself (trigger / passive / right-click action).\n"
                "4. USER ASSIGNMENT IS THE WHOLE PLAN: if the user message contains a '用户自定义 Domain 分配' / "
                "'User-supplied domain assignments' section, that section IS the complete rag_query plan. Issue exactly "
                "ONE rag_query per `- [tag] [domain(s)] text` line — `query` = that text verbatim, `domains` = those listed "
                "domain(s). Do NOT split those lines further, do NOT change their domain, and do NOT issue any other "
                "rag_query call (no exploratory lookups for rarity / name / parent object).\n"
                "5. EVERY segment after splitting must be queried — no omissions.\n"
                "6. `query` is verbatim text (no keyword extraction), same language as source, NEVER translate.\n"
                "7. Do NOT merge descriptions from multiple objects into one query.\n"
                "8. When the user did NOT pre-assign domains and you're unsure, commit to your single best guess. "
                "Do NOT fan one segment out to multiple domains on a hunch.\n"
                "9. TOP-3 AUTO-LOADED: each rag_query result already includes a `top_files` array with the full source "
                "of the first three matches. Treat them as already read — DO NOT call local_file_read on those paths. "
                "Use them as your primary code evidence; only reach further (results 4+, search) when the top-3 do not "
                "contain the symbol/pattern you actually need.\n"
                "\n"
                "✗ WRONG: rag_query(query='XXX,XX.XX', ...) — splitting punctuation remains\n"
                "✓ RIGHT: split into 'XXX' / 'XX' / 'XX' first, then three separate rag_query calls\n"
                "========================================\n"
            )
        )
    return SystemMessage(
        content=(
            "========================================\n"
            "!!! rag_query 调用规则 - 违反即错误 !!!\n"
            "========================================\n"
            "1. 切分优先：`query` 文本中含 。，；、！？ . , ; ! ? \\n \\r 的，必须按每个标点切开，每段单独一次 rag_query；"
            "`query` 内不得残留分割标点。\n"
            "2. domain 取值：cards / powers / relics / potions / orbs / enchantments / afflictions / rest_site_ui / events。"
            "默认 `domains` 是单元素列表，你自己挑最匹配的一个；只有当用户显式给同一段标了多个 domain 时，才在一次调用里传多个。\n"
            "3. 自行判断按子句作用对象选 domain，不按父级物体。\n"
            "4. 用户分配即完整计划：如果用户消息里有「用户自定义 Domain 分配」段，那一段就是 rag_query 的全部计划。"
            "对每条 `- [tag] [domain(s)] 文本` 严格一次 rag_query，`query` 就是该文本原文，`domains` 就是该行列出的 domain(s)。"
            "不再切分、不改 domain，也不要为稀有度 / 名称 / 父物体之类的额外信息发任何其他 rag_query。\n"
            "5. 切分后所有段都必须查询，不许遗漏。\n"
            "6. `query` 用原文（不提取关键词），与原文同语言，绝不翻译。\n"
            "7. 禁止把多个物体的描述拼成一条 query。\n"
            "8. 用户没分配 domain 且你拿不准时，挑最像的提交一个，禁止凭猜测撒网到多个 domain。\n"
            "9. 前 3 已自动读：rag_query 返回里的 `top_files` 数组已经直接附带了前三条结果的完整源码，"
            "视为已读取，禁止再对这些路径调 local_file_read。优先用这三个文件作为代码证据；只有当前 3 条都没有所需符号或模式时，"
            "再去 local_file_read 第 4 条之后或调用 local_file_search。\n"
            "\n"
            "✗ 错误：rag_query(query='XXX，XX。XX', ...) —— 仍残留分割标点\n"
            "✓ 正确：先切成 'XXX' / 'XX' / 'XX'，再分别 rag_query\n"
            "========================================\n"
        )
    )


def build_write_prompt(language: str) -> SystemMessage:
    """Dedicated rules for generating and saving code/files.

    Kept separate from the general system prompt so the model treats it as a
    standalone procedural contract while it is in the writing phase.
    """
    if normalize_language(language) == "en":
        return SystemMessage(
            content=(
                "========================================\n"
                "!!! CODE-WRITING RULES - VIOLATION = ERROR !!!\n"
                "========================================\n"
                "1. Before writing, check using/import namespaces at the top of retrieved code; verify every method you call exists. "
                "Inventing methods/types/namespaces = ERROR.\n"
                "2. If no available API can implement the effect, tell the user 'cannot be implemented' — NEVER fabricate.\n"
                "3. When context is insufficient, state exactly what is missing. Prefer reusing existing code over writing new.\n"
                "4. PER-OBJECT ISOLATION: each object's code only references code retrieved from its own query; do NOT mix references across objects.\n"
                "5. LOCALIZATION: after each object's code, append `description` and `title` to the matching-language JSON under `localization/`. "
                "Keys use UPPER_SNAKE_CASE of the class name. Batch ALL entries in ONE write — do not re-read the JSON between artifacts.\n"
                "6. WRITE FLOW: `.cs` files may be written in parallel or one-by-one. Do NOT re-read a file already read this turn unless you have written to it since.\n"
                "7. AUTONOMOUS WRITES: once given the task, write ALL files and finish. Do NOT end with 'should I continue?' / 'want me to write?'. "
                "For unverifiable APIs, pick a verified alternative + leave a brief code comment, then keep going.\n"
                "8. Only stop and ask when the user's request itself is genuinely ambiguous in a way that would lead to a wrong implementation.\n"
                "========================================\n"
            )
        )
    return SystemMessage(
        content=(
            "========================================\n"
            "!!! 代码写入规则 - 违反即错误 !!!\n"
            "========================================\n"
            "1. 写代码前先看 query 到的代码顶部 using/import 命名空间，逐一确认要调用的方法存在。"
            "编造方法/类型/命名空间 = 错误。\n"
            "2. 没有可用 API 能实现需求时，明确告诉用户『无法实现』，绝不要编造。\n"
            "3. 上下文不足时，明确说缺什么。有现成代码优先复用。\n"
            "4. 物体隔离：每个物体的代码只参考其自身 query 到的代码，多物体之间禁止混合参考。\n"
            "5. 本地化：每个物体代码完成后，去 `localization/` 对应语言 JSON 追加 `description` 和 `title`。"
            "key 用类名 UPPER_SNAKE_CASE。所有条目一次性批量追加，禁止在物体之间反复重读 JSON。\n"
            "6. 写入流程：`.cs` 文件可并行可串行。本轮已读过的文件，没写过就不要重读。\n"
            "7. 自主写入：拿到任务直接把所有文件写完，禁止在结尾问『是否继续』『要不要我写入』。"
            "无法确认的 API 自行选已验证替代方案 + 代码注释简短说明，然后继续写完。\n"
            "8. 仅在用户需求本身存在会导致错误方向的歧义时，才停下提问。\n"
            "========================================\n"
        )
    )


def build_one_by_one_prompt(language: str) -> SystemMessage:
    """Force the assistant to process a JSON list one object at a time.

    Combines three reinforcement techniques:
      1. Explicit pacing (per-object turn boundary).
      4. Explicit prohibitions (negative constraints).
      5. TodoWrite as the visible mechanism for serial progress.
    """
    if normalize_language(language) == "en":
        return SystemMessage(
            content=(
                "========================================\n"
                "!!! ONE-BY-ONE RULES - VIOLATION = ERROR !!!\n"
                "========================================\n"
                "1. Use TodoWrite to create one pending todo per object (title = object name).\n"
                "2. Each turn: process exactly 1 object — mark in_progress -> execute -> mark completed "
                "-> output 'Object N done, reply \"continue\" for next' -> STOP immediately.\n"
                "3. At most 1 todo in_progress at any time.\n"
                "4. NEVER start the next object before receiving 'continue'.\n"
                "5. After all objects are done, output 'All done'.\n"
                "========================================\n"
            )
        )
    return SystemMessage(
        content=(
            "========================================\n"
            "!!! 逐个处理规则 - 违反即错误 !!!\n"
            "========================================\n"
            "1. 用 TodoWrite 为每个对象建一条 todo（标题=对象名称，状态=pending）。\n"
            "2. 每轮只处理 1 个：标 in_progress -> 执行 -> 标 completed "
            "-> 输出『第 N 个完毕，回复\"继续\"处理下一个』-> 立即停。\n"
            "3. 任何时刻最多 1 个 todo 处于 in_progress。\n"
            "4. 未收到『继续』绝对不能开始下一个。\n"
            "5. 全部完成后输出『全部完成』。\n"
            "========================================\n"
        )
    )


def build_summary_prompt(
    *,
    previous_summary: str,
    messages: list[dict[str, Any]],
    answer: str,
    language: str,
) -> list[Any]:
    history = "\n".join(
        f"{item['role']}: {item['content']}" for item in recent_chat_messages(messages)
    )
    if normalize_language(language) == "en":
        return [
            SystemMessage(
                content=(
                    "Maintain the conversation summary memory for the STS2 RAG assistant. "
                    "Output at most 6 bullet points, preserving the user's goal, key code names, confirmed conclusions, "
                    "and unresolved questions. Do not record API keys, unrelated small talk, or full long code blocks."
                )
            ),
            HumanMessage(
                content=(
                    f"Previous summary:\n{previous_summary or 'None'}\n\n"
                    f"Recent conversation:\n{history or 'None'}\n\n"
                    f"Assistant's latest answer:\n{answer[:5000]}\n\n"
                    "Output the updated summary."
                )
            ),
        ]
    return [
        SystemMessage(
            content=(
                "你负责维护 STS2 RAG 助手的会话摘要记忆。"
                "输出 6 条以内的要点，保留用户目标、关键代码名、已确认结论和未解决问题。"
                "不要记录 API Key、无关闲聊或完整长代码。"
            )
        ),
        HumanMessage(
            content=(
                f"旧摘要:\n{previous_summary or '无'}\n\n"
                f"最近对话:\n{history or '无'}\n\n"
                f"助手最新回答:\n{answer[:5000]}\n\n"
                "请输出更新后的摘要。"
            )
        ),
    ]


def build_prompt_messages(
    *,
    raw_messages: list[dict[str, Any]],
    memory_summary: str,
    search_query: str,
    context_text: str,
    selected_file_context: str,
    namespace_audit_context: str = "",
    language: str,
    one_by_one: bool = False,
) -> list[Any]:
    prompt_messages: list[Any] = [SystemMessage(content=build_system_prompt(language))]
    prompt_messages.append(build_query_prompt(language))
    prompt_messages.append(build_write_prompt(language))
    if one_by_one:
        prompt_messages.append(build_one_by_one_prompt(language))
    memory_prompt = build_memory_prompt(memory_summary, language)
    if memory_prompt is not None:
        prompt_messages.append(memory_prompt)
    prompt_messages.extend(to_langchain_messages(recent_chat_messages(raw_messages)))
    if normalize_language(language) == "en":
        namespace_context = namespace_audit_context or "No namespace audit."
        file_context = selected_file_context or "No selected local files."
        prompt_messages.append(
            HumanMessage(
                content=(
                    f"RAG retrieval description: {search_query}\n\n"
                    f"RAG context:\n{context_text}\n\n"
                    f"Namespace MCP audit context:\n{namespace_context}\n\n"
                    f"User-selected local-file MCP context:\n{file_context}\n\n"
                    "The RAG retrieval description may be imprecise and should be treated as a hint, not ground truth. "
                    "When additional retrieval is needed, rewrite the query yourself in the text style of Slay the Spire / STS2 game text before searching, but keep the query in the SAME language as the user's question — do not translate. "
                    "Answer the user's actual question. When generating code, only use APIs that can be confirmed from the RAG context, "
                    "namespace MCP audit context, or local files. If a method, type, or lifecycle hook cannot be found in these contexts, "
                    "clearly warn the user that it does not exist or cannot be confirmed; do not invent it."
                )
            )
        )
        return prompt_messages
    namespace_context = namespace_audit_context or "无命名空间审计上下文。"
    file_context = selected_file_context or "无用户选择的本地文件。"
    prompt_messages.append(
        HumanMessage(
            content=(
                f"RAG 检索描述: {search_query}\n\n"
                f"RAG 上下文:\n{context_text}\n\n"
                f"命名空间 MCP 审计上下文:\n{namespace_context}\n\n"
                f"用户选择的本地文件 MCP 上下文:\n{file_context}\n\n"
                "Query 必须和用户提问保持同一种语言，不要翻译。"
                "请回答用户的实际问题。生成代码时只能使用 RAG 上下文、命名空间 MCP 审计上下文或本地文件中能确认存在的 API。"
                "如果某个方法、类型或生命周期钩子在这些上下文中找不到，请明确提醒用户它不存在或无法确认，不要编造。"
            )
        )
    )
    return prompt_messages


def read_selected_file_context(
    local_files: LocalFileMCP,
    selected_files: Any,
) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(selected_files, list) or not selected_files:
        return "", []

    parts: list[str] = []
    traces: list[dict[str, Any]] = []
    for raw_path in selected_files:
        path = str(raw_path or "").strip()
        if not path:
            continue
        result = local_files.read_file(path, max_chars=0)
        traces.append({"tool": "local_file_read", "arguments": {"path": path, "max_chars": 0}, "result": result})
        if result.get("ok"):
            parts.append(f"### Local File: {result.get('path', path)}\n{result.get('content', '')}")
        else:
            parts.append(f"### Local File: {path}\nRead failed: {result.get('error', 'unknown error')}")

    return ("\n\n".join(parts), traces) if parts else ("", traces)


def read_rag_context_files_via_mcp(
    local_files: LocalFileMCP,
    context_groups: list[dict[str, Any]],
    *,
    max_files: int = 0,
) -> tuple[str, list[dict[str, Any]]]:
    paths: list[str] = []
    seen: set[str] = set()
    for group in context_groups:
        contexts = group.get("contexts", [])
        if not isinstance(contexts, list):
            continue
        for context in contexts:
            if not isinstance(context, dict):
                continue
            rel_path = workspace_relative_path(context.get("file_path"))
            if not rel_path or rel_path in seen:
                continue
            seen.add(rel_path)
            paths.append(rel_path)
            if max_files and len(paths) >= max_files:
                break
        if max_files and len(paths) >= max_files:
            break

    parts: list[str] = []
    traces: list[dict[str, Any]] = []
    for path in paths:
        result = local_files.read_file(path, max_chars=0)
        trace = {"tool": "local_file_read", "arguments": {"path": path, "max_chars": 0}, "result": result}
        traces.append(trace)
        if result.get("ok"):
            parts.append(f"### Agent MCP Read: {result.get('path', path)}\n{result.get('content', '')}")
        else:
            parts.append(f"### Agent MCP Read: {path}\nRead failed: {result.get('error', 'unknown error')}")

    return ("\n\n".join(parts), traces) if parts else ("", traces)
