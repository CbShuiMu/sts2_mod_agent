from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mcp.local_files import LocalFileMCP


WRITE_TOOLS = {"local_file_write", "local_file_replace", "local_file_copy_tree", "local_file_create_dir"}
KNOWN_EXTERNAL_TYPES = {
    "Action",
    "Array",
    "Bool",
    "Boolean",
    "Decimal",
    "Dictionary",
    "Func",
    "HashSet",
    "Harmony",
    "IEnumerable",
    "IReadOnlyList",
    "Int32",
    "List",
    "Math",
    "Object",
    "Path",
    "String",
    "Task",
    "Type",
    "ValueTask",
    "Callable",
    "Color",
    "Control",
    "Engine",
    "Node",
    "ResourceLoader",
    "Texture2D",
    "Tween",
    "Vector2",
    "Void",
}
_REFERENCE_INDEX_CACHE: dict[str, dict[str, Any]] = {}
_REFERENCE_FILES_BY_TYPE_CACHE: dict[str, list[Path]] = {}
SOURCE_AUDIT_BLOCKED_PARTS = {".godot", ".idea", "bin", "obj"}
ROOT_MODEL_FOLDERS = {"Cards", "Powers", "Potions", "Relics", "Orbs", "Enchantments", "Afflictions"}
TEMPLATE_REQUIRED_PATHS = {"STSCustomCard.csproj", "project.godot", "src/Core/Models", "libs", "localization"}


AGENT_TOOL_SELECTION_PROMPT = (
    "If the user asks to create, copy, edit, or modify files/projects, you MUST actually call "
    "local_file_copy_tree / local_file_write / local_file_replace / local_file_create_dir. Do not only return code. "
    "When creating a new mod project, first copy the entire mods/template directory with local_file_copy_tree, then edit files only inside the copied project structure. "
    "When an empty directory itself is the requested artifact, use local_file_create_dir; do not use local_file_list as a substitute for directory creation. "
    "Do not create model folders such as Cards, Powers, Potions, or Relics at the mod root; model files belong under src/Core/Models/Cards, src/Core/Models/Powers, src/Core/Models/Potions, src/Core/Models/Relics, etc. "
    "You are in the MCP work phase, not the final-answer phase. "
    "For file/project creation tasks, the task is not complete until the target files under mods/ have been created or modified by MCP write tools. "
    "For mod generation, every requested model artifact must have its own concrete .cs file under the appropriate src/Core/Models subfolder, for example src/Core/Models/Cards/, src/Core/Models/Powers/, src/Core/Models/Potions/, src/Core/Models/Relics/. "
    "If ModInitializer registers or references a model type, verify that the corresponding .cs file exists in the mod; if it does not exist, create it before finalizing. "
    "Do not leave a registered type such as GiftPotion without creating Potions/GiftPotion.cs. "
    "Before writing ModHelper.AddModelToPool, verify the pool type exists as a concrete class in data/Models, data/libs/sts2_decompiled, or the mod source. "
    "Do not invent pool types such as PotionPool; query/list the appropriate namespace and use an existing class such as SharedPotionPool, IroncladPotionPool, or another verified pool. "
    "For any external type, static method, property, enum, namespace, lifecycle hook, or command API used in generated code, verify it exists in RAG context or local-file MCP context first. "
    "If a referenced API cannot be found after searching the corresponding namespace, do not invent it; either choose a verified alternative or state the missing API. "
    "Do not stop after only reading APIs, and do not provide code snippets as the final result instead of editing files. "
    "Use rag_query FIRST when reference code is needed. The detailed rag_query rules — splitting each description on punctuation YOURSELF, issuing one rag_query per segment, and how to set `domains` — are documented in the dedicated query system message; follow it strictly. If the user message contains a '用户自定义 Domain 分配' section, treat its `- [tag] [domains] text` lines as the complete rag_query plan and issue no other rag_query calls. The returned contexts already include MCPReadPath/file_path; use those exact paths with local_file_read or local_file_read_many. "
    "Avoid running broad local_file_search across data/libs/sts2_decompiled or data/ — that tree is large and slow. Only use local_file_search when rag_query did not yield enough context AND you have a specific symbol/identifier to grep for. "
    "When you already know several paths (for example zhs/eng localization JSONs, or multiple reference .cs in one namespace), call local_file_read_many ONCE with all paths instead of issuing many local_file_read calls. "
    "When this turn requires writing or replacing several files, emit them as PARALLEL tool_calls inside the same assistant message — do not serialize across turns. "
    "CRITICAL: every tool_call inside a parallel batch must have UNIQUE (name, args). Never emit two tool_calls in the same assistant message with the same name and identical arguments — duplicates are auto-rejected and waste a recovery turn. If you need to insert N relic registrations into ModInitializer.cs, that is ONE local_file_replace call where new_text contains all N lines, NOT N separate replace calls with the same old_text. "
    "If a tool result you got earlier is sufficient, do not call the same tool with the same args again to 're-verify' — the system will reject duplicates and the prior result is still in this conversation. "
    "If the user asks for several artifacts of the same kind (e.g. four relics), draft the full code for ALL of them in a single reasoning pass, then write every .cs and append every localization entry in one turn; do not loop 'design one → write one → re-read JSON' per artifact. "
    "Avoid repeating the same query once it has already returned enough evidence; move from investigation to editing. "
    "Once a file has been read in this conversation, do not re-read it unless you have written to it since. "
    "Write access is restricted to mods/. Always target paths inside mods/. "
    "If an API truly cannot be verified after searching, write only the parts that can be implemented safely and explain the missing API in the final answer. "
    "If no MCP work is needed, reply only with NO_MCP_NEEDED: followed by one short reason."
)

FINAL_ANSWER_PROMPT = (
    "MCP work phase may continue if the user asked for file changes and the target files have not been written yet. "
    "If file changes are still needed, call local_file_write/local_file_replace/local_file_create_dir now instead of returning code text. "
    "Before final answer, confirm that any new mod project was copied from mods/template and keeps model files under src/Core/Models, not root-level Cards/Powers/Potions/Relics folders. "
    "Before final answer, confirm that all requested mod artifacts and all model types registered in ModInitializer have corresponding source files. "
    "Also confirm that every ModHelper.AddModelToPool pool type is a verified concrete class, not an invented or abstract type. "
    "Also confirm that external namespaces, types, and static method calls used by generated mod code are verified from reference source or local files. "
    "Only provide the final answer after the needed mods/ files have been created or modified, or after you can clearly explain which required API could not be verified. "
    "Base the final answer on rag_query context, selected-file context, and MCP results. Use only verified types and methods."
)

AGENT_CONTINUE_PROMPT = (
        "Continue the task. If the user requested file/project changes and no successful local_file_write/local_file_replace/local_file_create_dir has happened yet, "
        "inspect the target mods/ files if needed, then call local_file_write, local_file_replace, or local_file_create_dir. "
    "If a registered model type does not have a matching .cs source file in the mod, create that missing file now. "
    "If model files or folders were created at the mod root, move/recreate them under src/Core/Models and clean up the wrong registrations/paths. "
    "If a registered pool type does not exist, search/list the pool namespace and replace it with a verified concrete pool class. "
    "If generated source contains unverified external namespaces, types, or Type.Method calls, search/read the corresponding namespace source and fix or report the missing API. "
    "Do not output textual tool-call markup as prose. Do not repeat the same search unless a different target path or query is necessary."
)


def message_reasoning_content(message: Any) -> str:
    value = getattr(message, "additional_kwargs", {}).get("reasoning_content", "")
    if value is None:
        return ""
    return str(value)


def serialize_messages(messages: Iterable[Any]) -> list[dict[str, str]]:
    serialized: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, ToolMessage):
            role = "tool"
        else:
            role = "user"
        item = {"role": role, "content": str(message.content)}
        if isinstance(message, AIMessage):
            reasoning_content = message_reasoning_content(message)
            if reasoning_content:
                item["reasoning_content"] = reasoning_content
        serialized.append(item)
    return serialized


def to_langchain_messages(messages: list[dict[str, Any]]) -> list[Any]:
    converted: list[Any] = []
    for message in messages:
        role = str(message.get("role", "")).lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "assistant":
            reasoning_content = str(
                message.get("reasoning_content")
                or message.get("reasoningContent")
                or ""
            )
            additional_kwargs = (
                {"reasoning_content": reasoning_content}
                if reasoning_content
                else {}
            )
            converted.append(AIMessage(content=content, additional_kwargs=additional_kwargs))
        elif role == "system":
            converted.append(SystemMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


def compute_agent_followup_prompt(
    *,
    local_files: "LocalFileMCP",
    original_messages: list[Any],
    seen_keys: set[str],
) -> str | None:
    """When the LLM stops emitting tool calls, decide whether to nudge it.

    Returns a single combined HumanMessage content to inject, or None if the
    agent is truly done. ``seen_keys`` is updated with whatever issue keys we
    surfaced so we never repeat the same nudge twice.
    """
    nudges: list[str] = []

    structure_issues = find_mod_template_structure_issues(local_files)
    structure_key = "structure:" + "\n".join(structure_issues)
    if structure_issues and structure_key not in seen_keys:
        seen_keys.add(structure_key)
        nudges.append(
            "The generated mod project does not follow the copied template structure:\n"
            + "\n".join(structure_issues)
            + "\nA new mod must be based on a full copy of mods/template. Put model source files under "
            "src/Core/Models/<Domain>/, not at the mod root. Fix the structure with MCP tools before final answer."
        )

    missing_artifacts = find_missing_requested_model_artifacts(local_files, original_messages)
    missing_key = "missing_artifact:" + "\n".join(missing_artifacts)
    if missing_artifacts and missing_key not in seen_keys:
        seen_keys.add(missing_key)
        nudges.append(
            "The user requested multiple mod artifacts, but the copied mod project is missing some "
            "required source files under src/Core/Models:\n"
            + "\n".join(missing_artifacts)
            + "\nContinue MCP work and create the missing requested files in the correct template structure. "
            "Do not repeat an already-written file."
        )

    unverified_refs = find_unverified_mod_code_references(local_files)
    unverified_key = "unverified:" + "\n".join(unverified_refs)
    if unverified_refs and unverified_key not in seen_keys:
        seen_keys.add(unverified_key)
        nudges.append(
            "The generated mod source contains external namespaces, types, or static API calls "
            "that are not verified in project/reference source:\n"
            + "\n".join(unverified_refs)
            + "\nUse local_file_read_many or local_file_search/local_file_list/local_file_read to inspect "
            "the corresponding namespace source. Replace invented APIs with verified ones, or explain the "
            "missing API if no verified equivalent exists."
        )

    invalid_pools = find_invalid_registered_pool_types(local_files)
    invalid_pool_key = "invalid_pool:" + "\n".join(invalid_pools)
    if invalid_pools and invalid_pool_key not in seen_keys:
        seen_keys.add(invalid_pool_key)
        nudges.append(
            "The mod is not complete. These ModInitializer pool types are not verified concrete "
            "classes in the project source or decompiled STS2 source:\n"
            + "\n".join(invalid_pools)
            + "\nUse local_file_search/local_file_list to inspect the appropriate pool namespace, then "
            "replace the invalid pool type with an existing concrete pool class before final answer."
        )

    missing_sources = find_missing_registered_mod_sources(local_files)
    missing_src_key = "missing_source:" + "\n".join(missing_sources)
    if missing_sources and missing_src_key not in seen_keys:
        seen_keys.add(missing_src_key)
        nudges.append(
            "The mod is not complete. These model types are registered in ModInitializer but "
            "do not have corresponding source classes in the mod:\n"
            + "\n".join(missing_sources)
            + "\nCreate the missing .cs files now with local_file_write before giving a final answer."
        )

    if not nudges:
        return None
    return "\n\n".join(nudges)


def prepare_local_file_agent_messages(
    *,
    llm: Any,
    messages: list[Any],
    local_files: LocalFileMCP,
    max_steps: int = 48,
) -> tuple[list[Any], list[dict[str, Any]], str | None]:
    """Run MCP-style tool calls and return messages for a streamed final answer."""
    traces: list[dict[str, Any]] = []
    executed_tool_keys: set[str] = set()
    written_targets: set[str] = set()
    missing_source_prompts: set[str] = set()
    no_progress_rounds = 0
    tool_llm = llm.bind_tools(local_files.tool_specs())
    working = list(messages) + [HumanMessage(content=AGENT_TOOL_SELECTION_PROMPT)]

    while True:
        response = tool_llm.invoke(working)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            decision = str(response.content).strip()
            text_tool_calls = parse_text_tool_calls(decision)
            if text_tool_calls:
                working.append(SystemMessage(content=f"Model emitted textual MCP calls; executing them instead of showing them to the user:\n{decision[:1200]}"))
                executed_any_tool = False
                for call in text_tool_calls:
                    name = str(call.get("name") or "")
                    args = normalize_tool_args(name, call.get("args") or {})
                    if not isinstance(args, dict):
                        args = {}
                    invalid_reason = invalid_tool_call_reason(name, args)
                    if invalid_reason:
                        executed_tool_keys.add(tool_command_key(name, args))
                        working.append(SystemMessage(content=invalid_reason))
                        continue
                    result = local_files.call(name, args)
                    executed_any_tool = True
                    mark_tool_execution(
                        name=name,
                        args=args,
                        result=result,
                        executed_tool_keys=executed_tool_keys,
                        written_targets=written_targets,
                    )
                    traces.append({"tool": name, "arguments": args, "result": trace_result(name, result)})
                    working.append(
                        SystemMessage(
                            content=(
                                f"MCP tool result for {name}:\n"
                                f"{json.dumps(result, ensure_ascii=False)}"
                            )
                        )
                    )
                if executed_any_tool:
                    no_progress_rounds = 0
                    working.append(HumanMessage(content=AGENT_CONTINUE_PROMPT))
                else:
                    no_progress_rounds += 1
                    working.append(
                        HumanMessage(
                            content=(
                                "The last MCP calls were invalid and made no progress. "
                                "Use a non-empty search query only if needed; otherwise write the missing requested files."
                            )
                        )
                    )
                continue

            structure_issues = find_mod_template_structure_issues(local_files)
            structure_issue_key = "\n".join(structure_issues)
            if structure_issues and structure_issue_key not in missing_source_prompts:
                missing_source_prompts.add(structure_issue_key)
                working.append(
                    HumanMessage(
                        content=(
                            "The generated mod project does not follow the copied template structure:\n"
                            f"{structure_issue_key}\n"
                            "A new mod must be based on a full copy of mods/template. Put model source files under "
                            "src/Core/Models/<Domain>/, not at the mod root. Fix the structure with MCP tools before final answer."
                        )
                    )
                )
                continue

            missing_requested_artifacts = find_missing_requested_model_artifacts(local_files, messages)
            missing_requested_key = "\n".join(missing_requested_artifacts)
            if missing_requested_artifacts and missing_requested_key not in missing_source_prompts:
                missing_source_prompts.add(missing_requested_key)
                working.append(
                    HumanMessage(
                        content=(
                            "The user requested multiple mod artifacts, but the copied mod project is missing some "
                            "required source files under src/Core/Models:\n"
                            f"{missing_requested_key}\n"
                            "Continue MCP work and create the missing requested files in the correct template structure. "
                            "Do not repeat an already-written file."
                        )
                    )
                )
                continue

            unverified_refs = find_unverified_mod_code_references(local_files)
            unverified_ref_key = "\n".join(unverified_refs)
            if unverified_refs and unverified_ref_key not in missing_source_prompts:
                missing_source_prompts.add(unverified_ref_key)
                working.append(
                    HumanMessage(
                        content=(
                            "The generated mod source contains external namespaces, types, or static API calls "
                            "that are not verified in project/reference source:\n"
                            f"{unverified_ref_key}\n"
                            "Use local_file_search/local_file_list/local_file_read to inspect the corresponding "
                            "namespace source. Replace invented APIs with verified ones, or explain the missing API "
                            "if no verified equivalent exists."
                        )
                    )
                )
                continue

            invalid_pools = find_invalid_registered_pool_types(local_files)
            invalid_pool_key = "\n".join(invalid_pools)
            if invalid_pools and invalid_pool_key not in missing_source_prompts:
                missing_source_prompts.add(invalid_pool_key)
                working.append(
                    HumanMessage(
                        content=(
                            "The mod is not complete. These ModInitializer pool types are not verified concrete "
                            "classes in the project source or decompiled STS2 source:\n"
                            f"{invalid_pool_key}\n"
                            "Use local_file_search/local_file_list to inspect the appropriate pool namespace, then "
                            "replace the invalid pool type with an existing concrete pool class before final answer."
                        )
                    )
                )
                continue

            missing_sources = find_missing_registered_mod_sources(local_files)
            missing_key = "\n".join(missing_sources)
            if missing_sources and missing_key not in missing_source_prompts:
                missing_source_prompts.add(missing_key)
                working.append(
                    HumanMessage(
                        content=(
                            "The mod is not complete. These model types are registered in ModInitializer but "
                            "do not have corresponding source classes in the mod:\n"
                            f"{missing_key}\n"
                            "Create the missing .cs files now with local_file_write before giving a final answer."
                        )
                    )
                )
                continue

            traces.append(
                {
                    "tool": "agent_tool_decision",
                    "arguments": {},
                    "result": {"ok": True, "decision": decision[:800]},
                }
            )
            working.append(
                SystemMessage(
                    content=f"Agent MCP 工具决策: {decision[:800] or '未调用 MCP 工具。'}"
                )
            )
            working.append(HumanMessage(content=FINAL_ANSWER_PROMPT))
            return working, traces, None

        working.append(response)
        for call in tool_calls:
            name = str(call.get("name") or "")
            args = normalize_tool_args(name, call.get("args") or {})
            if not isinstance(args, dict):
                args = {}
            invalid_reason = invalid_tool_call_reason(name, args)
            if invalid_reason:
                result = {"ok": False, "skipped": True, "reason": invalid_reason}
                executed_tool_keys.add(tool_command_key(name, args))
                working.append(
                    ToolMessage(
                        content=json.dumps(result, ensure_ascii=False),
                        tool_call_id=str(call.get("id") or name),
                    )
                )
                continue
            result = local_files.call(name, args)
            mark_tool_execution(
                name=name,
                args=args,
                result=result,
                executed_tool_keys=executed_tool_keys,
                written_targets=written_targets,
            )
            traces.append({"tool": name, "arguments": args, "result": trace_result(name, result)})
            working.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=str(call.get("id") or name),
                )
            )

    working.append(HumanMessage(content=FINAL_ANSWER_PROMPT))
    return working, traces, None


def find_mod_template_structure_issues(local_files: LocalFileMCP) -> list[str]:
    root = getattr(local_files, "root", None)
    if not isinstance(root, Path):
        return []
    mods_root = root / "mods"
    template_root = mods_root / "template"
    if not mods_root.exists() or not template_root.exists():
        return []

    issues: list[str] = []
    for mod_root in sorted(path for path in mods_root.iterdir() if path.is_dir() and path.name != "template"):
        if not (mod_root / "ModInitializer.cs").exists():
            continue
        rel_mod = mod_root.relative_to(root).as_posix()
        for rel_path in sorted(TEMPLATE_REQUIRED_PATHS):
            required = mod_root / Path(rel_path)
            if not required.exists():
                issues.append(f"- {rel_mod}: missing template path {rel_path}; copy mods/template first")
        for folder_name in sorted(ROOT_MODEL_FOLDERS):
            wrong_folder = mod_root / folder_name
            if wrong_folder.exists() and wrong_folder.is_dir():
                correct_folder = f"src/Core/Models/{folder_name}"
                issues.append(
                    f"- {rel_mod}: root-level {folder_name}/ is invalid; model files must be under {correct_folder}/"
                )
        models_root = mod_root / "src" / "Core" / "Models"
        if models_root.exists():
            for source in sorted(models_root.rglob("*.cs")):
                if source.name.endswith(".uid"):
                    continue
                try:
                    rel_parts = source.relative_to(models_root).parts
                except ValueError:
                    continue
                if len(rel_parts) < 2 or rel_parts[0] not in ROOT_MODEL_FOLDERS:
                    issues.append(
                        f"- {source.relative_to(root).as_posix()}: model source should be inside a domain folder under src/Core/Models"
                    )
    return dedupe_strings(issues)


def find_missing_requested_model_artifacts(local_files: LocalFileMCP, messages: list[Any]) -> list[str]:
    root = getattr(local_files, "root", None)
    if not isinstance(root, Path):
        return []
    requested_domains = requested_model_domains(messages)
    if not requested_domains:
        return []

    mods_root = root / "mods"
    if not mods_root.exists():
        return []
    target_mods = target_mod_roots(root, messages)
    if not target_mods:
        target_mods = [
            path
            for path in sorted(mods_root.iterdir())
            if path.is_dir() and path.name != "template" and (path / "ModInitializer.cs").exists()
        ]

    missing: list[str] = []
    for mod_root in target_mods:
        if not mod_root.exists() or mod_root.name == "template":
            continue
        rel_mod = mod_root.relative_to(root).as_posix()
        for domain in sorted(requested_domains):
            if not has_non_template_model_source(root, mod_root, domain):
                missing.append(
                    f"- {rel_mod}: requested {domain} artifact is missing; create src/Core/Models/{domain}/<Name>.cs"
                )
    return dedupe_strings(missing)


def requested_model_domains(messages: list[Any]) -> set[str]:
    text = "\n".join(str(getattr(message, "content", "")) for message in messages if isinstance(message, HumanMessage)).lower()
    domains: set[str] = set()
    if any(token in text for token in ["card", "cards", "卡牌", "一张牌", "牌效果"]):
        domains.add("Cards")
    if any(token in text for token in ["potion", "potions", "药水"]):
        domains.add("Potions")
    if any(token in text for token in ["relic", "relics", "遗物"]):
        domains.add("Relics")
    if any(token in text for token in ["orb", "orbs", "充能球"]):
        domains.add("Orbs")
    if any(token in text for token in ["enchantment", "enchantments", "附魔"]):
        domains.add("Enchantments")
    if any(token in text for token in ["affliction", "afflictions", "负面牌", "苦痛"]):
        domains.add("Afflictions")
    return domains


def target_mod_roots(root: Path, messages: list[Any]) -> list[Path]:
    text = "\n".join(str(getattr(message, "content", "")) for message in messages if isinstance(message, HumanMessage))
    names: list[str] = []
    for match in re.findall(r"mods[\\/]+([A-Za-z_][A-Za-z0-9_-]*)", text):
        if match not in names:
            names.append(match)
    for match in re.findall(r"(?:叫|名为|named|called)\s*([A-Za-z_][A-Za-z0-9_-]*)", text, flags=re.IGNORECASE):
        if match not in names:
            names.append(match)
    for match in re.findall(r"\b([A-Za-z_][A-Za-z0-9_-]*Mod)\b", text):
        if match not in names:
            names.append(match)
    return [root / "mods" / name for name in names]


def has_non_template_model_source(root: Path, mod_root: Path, domain: str) -> bool:
    model_dir = mod_root / "src" / "Core" / "Models" / domain
    if not model_dir.exists():
        return False
    template_root = root / "mods" / "template"
    for source in model_dir.glob("*.cs"):
        if source.name.endswith(".uid"):
            continue
        rel = source.relative_to(mod_root)
        template_source = template_root / rel
        if not template_source.exists():
            return True
        try:
            if source.read_text(encoding="utf-8", errors="ignore") != template_source.read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            return True
    return False


def find_unverified_mod_code_references(local_files: LocalFileMCP) -> list[str]:
    root = getattr(local_files, "root", None)
    if not isinstance(root, Path):
        return []
    mods_root = root / "mods"
    if not mods_root.exists():
        return []

    problems: list[str] = []
    for mod_root in sorted(path for path in mods_root.iterdir() if path.is_dir() and (path / "ModInitializer.cs").exists()):
        local_types = collect_declared_types(mod_root)
        for source in sorted(iter_audited_sources(mod_root)):
            if source.name.endswith(".uid"):
                continue
            try:
                code = source.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            rel_source = source.relative_to(root).as_posix()
            for namespace in extract_using_namespaces(code):
                if namespace.startswith("MegaCrit.") and not namespace_exists(root, namespace):
                    problems.append(f"- {rel_source}: using namespace '{namespace}' was not found in reference source")

            for type_name in sorted(extract_type_reference_candidates(code)):
                if should_skip_type_reference(type_name, local_types):
                    continue
                status = source_type_status(root, type_name)
                if status == "missing":
                    problems.append(f"- {rel_source}: type '{type_name}' was not found in reference source or mod source")

            for type_name, member_name in sorted(extract_static_member_calls(code)):
                if should_skip_type_reference(type_name, local_types):
                    continue
                status = source_type_status(root, type_name)
                if status == "missing":
                    problems.append(f"- {rel_source}: type '{type_name}' for call '{type_name}.{member_name}' was not found")
                    continue
                if not type_member_exists(root, type_name, member_name):
                    problems.append(f"- {rel_source}: member call '{type_name}.{member_name}(...)' was not found on verified type '{type_name}'")
    return dedupe_strings(problems)


def collect_declared_types(root: Path) -> set[str]:
    types: set[str] = set()
    for source in iter_audited_sources(root):
        try:
            content = source.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        types.update(re.findall(r"\b(?:class|enum|interface|struct)\s+([A-Za-z_][A-Za-z0-9_]*)\b", content))
    return types


def iter_audited_sources(root: Path) -> list[Path]:
    return [
        source
        for source in root.rglob("*.cs")
        if not source.name.endswith(".uid") and not any(part in SOURCE_AUDIT_BLOCKED_PARTS for part in source.parts)
    ]


def extract_using_namespaces(code: str) -> list[str]:
    return re.findall(r"^\s*using\s+(?!static\b)([A-Za-z_][A-Za-z0-9_.]*)\s*;", code, flags=re.MULTILINE)


def extract_type_reference_candidates(code: str) -> set[str]:
    stripped = strip_comments_and_strings(code)
    stripped = re.sub(r"^\s*using\s+[^;]+;", "", stripped, flags=re.MULTILINE)
    stripped = re.sub(r"^\s*namespace\s+[A-Za-z_][A-Za-z0-9_.]*\s*[{;]?", "", stripped, flags=re.MULTILINE)
    candidates: set[str] = set()
    patterns = [
        r"\btypeof\s*\(\s*([A-Z][A-Za-z0-9_]*)\s*\)",
        r"\bnew\s+([A-Z][A-Za-z0-9_]*)\b",
        r"<\s*([A-Z][A-Za-z0-9_]*)\s*(?:>|,)",
        r":\s*([A-Z][A-Za-z0-9_]*)\b",
        r"\b([A-Z][A-Za-z0-9_]*)\??\s+[a-z_][A-Za-z0-9_]*\b",
    ]
    for pattern in patterns:
        candidates.update(re.findall(pattern, stripped))
    return candidates


def extract_static_member_calls(code: str) -> set[tuple[str, str]]:
    stripped = strip_comments_and_strings(code)
    calls = set()
    for type_name, member_name in re.findall(
        r"\b([A-Z][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^;{}()]*>)?\s*\(",
        stripped,
    ):
        calls.add((type_name, member_name))
    return calls


def strip_comments_and_strings(code: str) -> str:
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r'@"(?:[^"]|"")*"', '""', code)
    code = re.sub(r'"(?:\\.|[^"\\])*"', '""', code)
    code = re.sub(r"'(?:\\.|[^'\\])'", "''", code)
    return code


def should_skip_type_reference(type_name: str, local_types: set[str]) -> bool:
    if not type_name or type_name in local_types or type_name in KNOWN_EXTERNAL_TYPES:
        return True
    if type_name.startswith("_") or type_name.startswith("I") and len(type_name) == 1:
        return True
    return False


def namespace_exists(root: Path, namespace: str) -> bool:
    if namespace.startswith("MegaCrit.Sts2.Core."):
        if (root / "data" / "libs" / "sts2_decompiled" / namespace).exists():
            return True
    namespace_pattern = re.compile(rf"\bnamespace\s+{re.escape(namespace)}\b")
    mods_root = root / "mods"
    if mods_root.exists():
        for source in iter_audited_sources(mods_root):
            try:
                if namespace_pattern.search(source.read_text(encoding="utf-8", errors="ignore")):
                    return True
            except OSError:
                continue
    return False


def source_type_status(root: Path, type_name: str) -> str:
    if not type_name:
        return "missing"
    mod_status = mod_type_status(root, type_name)
    if mod_status != "missing":
        return mod_status
    type_pattern = type_declaration_pattern(type_name)
    for source in reference_files_for_type(root, type_name):
        try:
            match = type_pattern.search(source.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if match:
            if match.group("kind") == "class":
                return "abstract" if match.group("abstract") else "concrete"
            return match.group("kind")
    return "missing"


def type_member_exists(root: Path, type_name: str, member_name: str) -> bool:
    if mod_type_member_exists(root, type_name, member_name):
        return True
    member_pattern = re.compile(rf"\b{re.escape(member_name)}\s*(?:<|\(|=>|{{)")
    for source in reference_files_for_type(root, type_name):
        try:
            if member_pattern.search(source.read_text(encoding="utf-8", errors="ignore")):
                return True
        except OSError:
            continue
    return False


def source_files_for_type(root: Path, type_name: str) -> list[Path]:
    matches: list[Path] = []
    for base in source_search_roots(root):
        if not base.exists():
            continue
        matches.extend(base.rglob(f"{type_name}.cs"))
    return matches


def mod_type_status(root: Path, type_name: str) -> str:
    type_pattern = type_declaration_pattern(type_name)
    mods_root = root / "mods"
    if not mods_root.exists():
        return "missing"
    for source in iter_audited_sources(mods_root):
        try:
            match = type_pattern.search(source.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if match:
            if match.group("kind") == "class":
                return "abstract" if match.group("abstract") else "concrete"
            return match.group("kind")
    return "missing"


def mod_type_member_exists(root: Path, type_name: str, member_name: str) -> bool:
    type_pattern = type_declaration_pattern(type_name)
    member_pattern = re.compile(rf"\b{re.escape(member_name)}\s*(?:<|\(|=>|{{)")
    mods_root = root / "mods"
    if not mods_root.exists():
        return False
    for source in iter_audited_sources(mods_root):
        try:
            content = source.read_text(encoding="utf-8", errors="ignore")
            if type_pattern.search(content) and member_pattern.search(content):
                return True
        except OSError:
            continue
    return False


def reference_index(root: Path) -> dict[str, Any]:
    cache_key = str(root.resolve())
    if cache_key in _REFERENCE_INDEX_CACHE:
        return _REFERENCE_INDEX_CACHE[cache_key]

    type_pattern = re.compile(
        r"\b(?P<abstract>abstract\s+)?(?:(?:sealed|static|partial|readonly)\s+)*"
        r"(?P<kind>class|enum|interface|struct)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"
    )
    namespace_pattern = re.compile(r"\bnamespace\s+([A-Za-z_][A-Za-z0-9_.]*)\b")
    member_pattern = re.compile(r"\b(?:public|protected|internal|private)\s+(?:static\s+)?(?:async\s+)?[A-Za-z0-9_<>,.?[\]\s]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:<|\(|=>|{)")
    types: dict[str, str] = {}
    members: dict[str, set[str]] = {}
    namespaces: set[str] = set()
    for base in reference_search_roots(root):
        if not base.exists():
            continue
        for source in base.rglob("*.cs"):
            try:
                content = source.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            namespaces.update(namespace_pattern.findall(content))
            matches = list(type_pattern.finditer(content))
            for idx, match in enumerate(matches):
                name = match.group("name")
                if match.group("kind") == "class":
                    status = "abstract" if match.group("abstract") else "concrete"
                else:
                    status = match.group("kind")
                types.setdefault(name, status)
                start = match.end()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
                members.setdefault(name, set()).update(member_pattern.findall(content[start:end]))

    index = {"types": types, "members": members, "namespaces": namespaces}
    _REFERENCE_INDEX_CACHE[cache_key] = index
    return index


def reference_files_for_type(root: Path, type_name: str) -> list[Path]:
    cache_key = f"{root.resolve()}::{type_name}"
    if cache_key in _REFERENCE_FILES_BY_TYPE_CACHE:
        return _REFERENCE_FILES_BY_TYPE_CACHE[cache_key]
    files: list[Path] = []
    for base in reference_search_roots(root):
        if base.exists():
            files.extend(base.rglob(f"{type_name}.cs"))
    _REFERENCE_FILES_BY_TYPE_CACHE[cache_key] = files
    return files


def type_declaration_pattern(type_name: str) -> re.Pattern[str]:
    return re.compile(
        rf"\b(?P<abstract>abstract\s+)?(?:(?:sealed|static|partial|readonly)\s+)*"
        rf"(?P<kind>class|enum|interface|struct)\s+{re.escape(type_name)}\b"
    )


def source_search_roots(root: Path) -> list[Path]:
    return [root / "mods", root / "data" / "Models", root / "data" / "libs" / "sts2_decompiled"]


def reference_search_roots(root: Path) -> list[Path]:
    return [root / "data" / "Models", root / "data" / "libs" / "sts2_decompiled"]


def dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def find_invalid_registered_pool_types(local_files: LocalFileMCP) -> list[str]:
    root = getattr(local_files, "root", None)
    if not isinstance(root, Path):
        return []
    mods_root = root / "mods"
    if not mods_root.exists():
        return []

    invalid: list[str] = []
    for initializer in sorted(mods_root.glob("*/ModInitializer.cs")):
        try:
            content = initializer.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pool_type, model_type in registered_pool_calls(content):
            pool_name = type_name_only(pool_type)
            model_name = type_name_only(model_type)
            status = concrete_class_status(root, pool_name)
            if status == "concrete":
                continue
            rel_initializer = initializer.relative_to(root).as_posix()
            suggestions = ", ".join(suggest_pool_types(root, pool_name, model_name)[:12])
            suffix = f" Valid options include: {suggestions}." if suggestions else ""
            invalid.append(
                f"- {pool_name}: used for {model_name} in {rel_initializer}, but it is {status}.{suffix}"
            )
    return invalid


def find_missing_registered_mod_sources(local_files: LocalFileMCP) -> list[str]:
    root = getattr(local_files, "root", None)
    if not isinstance(root, Path):
        return []
    mods_root = root / "mods"
    if not mods_root.exists():
        return []

    missing: list[str] = []
    for initializer in sorted(mods_root.glob("*/ModInitializer.cs")):
        try:
            content = initializer.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pool_type, model_type in registered_pool_calls(content):
            model_name = type_name_only(model_type)
            if not model_name or source_class_exists(initializer.parent, model_name):
                continue
            folder = registered_model_folder(pool_type, model_name)
            rel_initializer = initializer.relative_to(root).as_posix()
            rel_hint = (initializer.parent / "src" / "Core" / "Models" / folder / f"{model_name}.cs").relative_to(root).as_posix()
            missing.append(f"- {model_name}: registered in {rel_initializer}; create {rel_hint}")
    return missing


def registered_pool_calls(content: str) -> list[tuple[str, str]]:
    return re.findall(
        r"ModHelper\.AddModelToPool\s*\(\s*typeof\(([^)]+)\)\s*,\s*typeof\(([^)]+)\)",
        content,
    )


def type_name_only(raw_type: str) -> str:
    return raw_type.split("<", 1)[0].split(".")[-1].strip()


def concrete_class_status(root: Path, class_name: str) -> str:
    status = source_type_status(root, class_name)
    return status if status in {"concrete", "abstract"} else "missing"


def suggest_pool_types(root: Path, pool_name: str, model_name: str) -> list[str]:
    domain = pool_domain(pool_name, model_name)
    folders = {
        "Potion": ["data/Models/PotionPools", "data/libs/sts2_decompiled/MegaCrit.Sts2.Core.Models.PotionPools"],
        "Card": ["data/Models/CardPools", "data/libs/sts2_decompiled/MegaCrit.Sts2.Core.Models.CardPools"],
        "Relic": ["data/Models/RelicPools", "data/libs/sts2_decompiled/MegaCrit.Sts2.Core.Models.RelicPools"],
    }.get(domain, [])
    suggestions: list[str] = []
    for folder in folders:
        base = root / folder
        if not base.exists():
            continue
        for source in sorted(base.glob("*Pool.cs")):
            name = source.stem
            if concrete_class_status(root, name) == "concrete" and name not in suggestions:
                suggestions.append(name)
    return suggestions


def pool_domain(pool_name: str, model_name: str) -> str:
    combined = f"{pool_name} {model_name}".lower()
    if "potion" in combined:
        return "Potion"
    if "relic" in combined:
        return "Relic"
    return "Card"


def source_class_exists(mod_root: Path, class_name: str) -> bool:
    class_pattern = re.compile(rf"\bclass\s+{re.escape(class_name)}\b")
    for source in mod_root.rglob("*.cs"):
        if source.name.endswith(".uid"):
            continue
        try:
            if class_pattern.search(source.read_text(encoding="utf-8", errors="ignore")):
                return True
        except OSError:
            continue
    return False


def registered_model_folder(pool_type: str, model_name: str) -> str:
    lowered_pool = pool_type.lower()
    if "potionpool" in lowered_pool or model_name.endswith("Potion"):
        return "Potions"
    if "relicpool" in lowered_pool or model_name.endswith("Relic"):
        return "Relics"
    if "cardpool" in lowered_pool:
        return "Cards"
    if model_name.endswith("Power"):
        return "Powers"
    return "Cards"


def invoke_with_local_file_agent(
    *,
    llm: Any,
    messages: list[Any],
    local_files: LocalFileMCP,
    max_steps: int = 48,
) -> tuple[str, list[dict[str, Any]]]:
    """Run the local-file agent and return a complete answer."""
    working, traces, direct_answer = prepare_local_file_agent_messages(
        llm=llm,
        messages=messages,
        local_files=local_files,
        max_steps=max_steps,
    )
    if direct_answer is not None:
        return direct_answer, traces

    final = llm.invoke(working)
    return str(final.content), traces


def execute_tool_call(local_files: LocalFileMCP, call: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    name = str(call.get("name") or "")
    args = call.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    result = local_files.call(name, args)
    return name, args, result


def normalize_tool_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").lower()


def normalize_tool_args(name: str, args: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        return {}
    normalized = dict(args)
    if name == "local_file_search":
        normalized["query"] = str(normalized.get("query") or "").strip()
        normalized["path"] = normalize_tool_path(normalized.get("path") or ".") or "."
        normalized["limit"] = int(normalized.get("limit") or 0)
    elif name == "local_file_list":
        normalized["path"] = normalize_tool_path(normalized.get("path") or ".") or "."
        normalized["pattern"] = str(normalized.get("pattern") or "*").strip() or "*"
        normalized["limit"] = int(normalized.get("limit") or 0)
    elif name == "local_file_read":
        normalized["path"] = normalize_tool_path(normalized.get("path") or "")
        normalized["max_chars"] = int(normalized.get("max_chars") or 0)
    elif name in WRITE_TOOLS:
        for key in ["path", "source_path", "target_path"]:
            if key in normalized:
                normalized[key] = normalize_tool_path(normalized.get(key))
    return normalized


def invalid_tool_call_reason(name: str, args: dict[str, Any]) -> str:
    if name == "local_file_search" and not str(args.get("query") or "").strip():
        return "Skipped invalid local_file_search: query is required and must be non-empty."
    if name in {"local_file_read", "local_file_write", "local_file_replace", "local_file_create_dir"} and not str(args.get("path") or "").strip():
        return f"Skipped invalid {name}: path is required."
    if name == "local_file_copy_tree" and (
        not str(args.get("source_path") or "").strip() or not str(args.get("target_path") or "").strip()
    ):
        return "Skipped invalid local_file_copy_tree: source_path and target_path are required."
    return ""


def write_target_key(name: str, args: dict[str, Any]) -> str:
    if name == "local_file_copy_tree":
        return normalize_tool_path(args.get("target_path"))
    if name in {"local_file_write", "local_file_replace", "local_file_create_dir"}:
        return normalize_tool_path(args.get("path"))
    return ""


def tool_command_key(name: str, args: dict[str, Any]) -> str:
    args = normalize_tool_args(name, args)
    if name in WRITE_TOOLS:
        target = write_target_key(name, args)
        return f"{name}:{target}"
    try:
        payload = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        payload = str(sorted(args.items()))
    return f"{name}:{payload}"


def mark_tool_execution(
    *,
    name: str,
    args: dict[str, Any],
    result: dict[str, Any],
    executed_tool_keys: set[str],
    written_targets: set[str],
) -> None:
    executed_tool_keys.add(tool_command_key(name, args))
    if result.get("ok"):
        target = write_target_key(name, args)
        if target:
            written_targets.add(target)


def parse_text_tool_calls(content: str) -> list[dict[str, Any]]:
    normalized = normalize_text_tool_markup(content)
    if "<tool_calls" not in normalized and "<invoke" not in normalized:
        return []

    calls: list[dict[str, Any]] = []
    for match in re.finditer(r"<invoke\s+name=[\"']([^\"']+)[\"'][^>]*>(.*?)</invoke>", normalized, re.DOTALL | re.IGNORECASE):
        name = match.group(1).strip()
        body = match.group(2)
        args: dict[str, Any] = {}
        for param in re.finditer(
            r"<parameter\s+name=[\"']([^\"']+)[\"']([^>]*)>(.*?)</parameter>",
            body,
            re.DOTALL | re.IGNORECASE,
        ):
            key = param.group(1).strip()
            attrs = param.group(2) or ""
            value = param.group(3).strip()
            if not value:
                attr_value = re.search(
                    r"(?:string|value|number|integer|boolean)=[\"']([^\"']*)[\"']",
                    attrs,
                    re.IGNORECASE,
                )
                if attr_value and attr_value.group(1).strip().lower() not in {"true", "false"}:
                    value = attr_value.group(1).strip()
            args[key] = coerce_tool_arg(value)
        if name:
            calls.append({"name": name, "args": args})
    return calls


def normalize_text_tool_markup(content: str) -> str:
    text = content.replace("｜", "|")
    text = re.sub(r"[\|?]+\s*DSML\s*[\|?]+", "", text, flags=re.IGNORECASE)
    text = text.replace("< /", "</").replace("< / ", "</")
    text = re.sub(r"<\s+", "<", text)
    text = re.sub(r"</\s+", "</", text)
    text = re.sub(r"\s+>", ">", text)
    return text


def strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value)


def coerce_tool_arg(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"-?\d+", value.strip()):
        try:
            return int(value.strip())
        except ValueError:
            return value
    return value


def trace_result(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Keep UI traces readable while preserving full tool output for the model."""
    if tool_name == "local_file_read":
        return {key: value for key, value in result.items() if key != "content"}
    if tool_name == "local_file_read_many":
        cleaned = {key: value for key, value in result.items() if key != "items"}
        items = result.get("items") or []
        cleaned["items"] = [
            {key: value for key, value in item.items() if key != "content"}
            for item in items
            if isinstance(item, dict)
        ]
        return cleaned
    if tool_name == "rag_query":
        cleaned = {key: value for key, value in result.items() if key not in {"context_text", "groups"}}
        cleaned["groups"] = result.get("public_groups", [])
        return cleaned
    if tool_name == "local_file_copy_tree":
        # The full result contains every created_dir / copied_file path which can balloon
        # to ~175 KB on a template copy. Strip the heavy arrays for the UI trace —
        # the counts are enough; the full result still goes to the model via ToolMessage.
        heavy_keys = {"created_dirs", "copied_files", "skipped_files"}
        return {key: value for key, value in result.items() if key not in heavy_keys}
    if tool_name in {"local_file_write", "local_file_replace", "local_file_create_dir"}:
        return result
    return result
