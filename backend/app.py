#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from langchain_core.messages import HumanMessage, ToolMessage

from agent import (
    AGENT_TOOL_SELECTION_PROMPT,
    compute_agent_followup_prompt,
    invoke_with_local_file_agent,
    invalid_tool_call_reason,
    normalize_tool_args,
    serialize_messages,
    tool_command_key,
    trace_result,
)
from mcp.local_files import LocalFileMCP
from settings_store import SettingsStore
from sts2_core.embeddings import DEFAULT_EMBEDDING_MODEL, load_env_file
from sts2_core.localization_vector_utils import (
    localization_collection_name,
    localization_persist_path,
)
from sts2_core.milvus import has_milvus_lite, is_local_lite_uri, resolve_milvus_uri
from sts2_core.paths import (
    DEFAULT_DESC_COLLECTION_NAME,
    DEFAULT_DESC_PERSIST_DIR,
    DEFAULT_ENV_PATH,
    DEFAULT_VECTOR_DB_ROOT,
    DEFAULT_MODELS_ROOT,
    DEFAULT_SETTINGS_PATH,
    FRONTEND_ROOT,
    PROJECT_ROOT,
)
from sts2_core.retrieval import retrieve_code_context_groups

from services.llm import create_llm
from services.prompts import (
    append_ai_chat_event,
    append_ai_chat_log,
    build_ai_chat_log_entry,
    build_grouped_prompt_context,
    build_prompt_messages,
    build_summary_prompt,
    read_selected_file_context,
)
from services.rag import (
    AgentMCP,
    LOCALIZATION_RAG_DOMAINS,
    MAIN_RAG_DOMAINS,
    RAG_DOMAIN_LABELS,
    RAG_DOMAINS,
    RagService,
    filter_groups_by_domains,
    normalize_rag_domains,
    public_context_groups,
)
from services.utils import (
    at_least_one_int,
    content_text,
    maybe_float,
    mcp_trace_count,
    normalize_language,
    positive_int,
    rag_query_summaries_from_traces,
    response_reasoning_content,
    safe_log_name,
    stream_event,
    trace_write_target,
    trim_tool_message_history,
)


def is_mod_scoped_tool_call(name: str, args: dict[str, Any]) -> bool:
    """Do not duplicate-throttle tools operating inside mods/.

    Generated mod files are active work products: the agent may need to read,
    overwrite, replace, search, or list the same paths repeatedly while fixing
    code. Keep duplicate protection for large reference trees such as data/.
    """
    def is_mod_path(value: Any) -> bool:
        path = str(value or "").replace("\\", "/").lower().strip("/")
        return path == "mods" or path.startswith("mods/")

    if name in {
        "local_file_read",
        "local_file_write",
        "local_file_replace",
        "local_file_create_dir",
        "local_file_search",
        "local_file_list",
    }:
        return is_mod_path(args.get("path"))
    if name == "local_file_read_many":
        paths = args.get("paths")
        if not isinstance(paths, list) or not paths:
            return False
        return all(is_mod_path(path) for path in paths)
    if name == "local_file_copy_tree":
        return is_mod_path(args.get("target_path"))
    return False


def raw_tool_call_ids(message: Any) -> list[str]:
    """Return every tool_call id declared on an AI message, including unparsed calls."""
    ids: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "")
        if text and text not in ids:
            ids.append(text)

    for call in getattr(message, "tool_calls", None) or []:
        if isinstance(call, dict):
            add(call.get("id"))
    for call in getattr(message, "invalid_tool_calls", None) or []:
        if isinstance(call, dict):
            add(call.get("id"))
    for call in getattr(message, "tool_call_chunks", None) or []:
        if isinstance(call, dict):
            add(call.get("id"))

    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    if isinstance(additional_kwargs, dict):
        for call in additional_kwargs.get("tool_calls") or []:
            if isinstance(call, dict):
                add(call.get("id"))
    return ids


_SENTENCE_SPLIT_RE = re.compile(r"[\.!?,;。！？，；、\n]+")
_DESCRIPTION_KEY_RE = re.compile(r"description", re.IGNORECASE)
_COLOR_TAG_RE = re.compile(r"\[/?[^\]\n]+\]")


def _strip_color_tags(text: str) -> str:
    if not text:
        return ""
    cleaned = _COLOR_TAG_RE.sub("", text)
    return re.sub(r"[ \t]+", " ", cleaned).strip()


def _split_description_text(text: str) -> list[str]:
    text = _strip_color_tags(text)
    if not text:
        return []
    pieces = [piece.strip() for piece in _SENTENCE_SPLIT_RE.split(text)]
    return [piece for piece in pieces if piece]


def _walk_descriptions(node: Any, key_path: list[str], out: list[tuple[str, str]]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            key_str = str(key)
            next_path = key_path + [key_str]
            if isinstance(value, str):
                if _DESCRIPTION_KEY_RE.search(key_str):
                    out.append((".".join(next_path), value))
            else:
                _walk_descriptions(value, next_path, out)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _walk_descriptions(item, key_path + [f"[{index}]"], out)


def extract_description_segments(sources: list[tuple[str, str]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    counter = 0
    group_counter = 0
    for source, content in sources:
        pairs: list[tuple[str, str]] = []
        parsed = None
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            parsed = None
        if parsed is not None:
            _walk_descriptions(parsed, [], pairs)
        if not pairs:
            for match in re.finditer(
                r'"([^"\n]*description[^"\n]*)"\s*:\s*"((?:\\.|[^"\\])*)"',
                content,
                flags=re.IGNORECASE,
            ):
                key = match.group(1)
                raw = match.group(2)
                try:
                    decoded = json.loads(f'"{raw}"')
                except ValueError:
                    decoded = raw
                pairs.append((key, decoded))
        for key, value in pairs:
            pieces = _split_description_text(value)
            if not pieces:
                continue
            group_counter += 1
            group_id = f"grp-{group_counter}"
            cleaned_original = _strip_color_tags(value)
            for piece in pieces:
                counter += 1
                segments.append({
                    "id": f"seg-{counter}",
                    "source": source,
                    "key": key,
                    "text": piece,
                    "group_id": group_id,
                    "group_label": f"{source} :: {key}",
                    "original": cleaned_original,
                })
    return segments


def invalid_unparsed_tool_message(tool_call_id: str) -> ToolMessage:
    payload = {
        "ok": False,
        "error": (
            "invalid or unparsed tool call returned by the model; the backend "
            "could not recover a valid tool name and arguments for this tool_call_id"
        ),
    }
    return ToolMessage(content=json.dumps(payload, ensure_ascii=False), tool_call_id=tool_call_id)


def build_app(
    *,
    desc_persist_dir: Path,
    desc_collection_name: str,
    models_root: Path,
    embedding_model: str,
    settings_path: Path,
    code_chars: int,
    default_context_n: int,
    default_desc_top_k: int,
) -> Flask:
    app = Flask(__name__, static_folder=str(FRONTEND_ROOT), static_url_path="")
    settings = SettingsStore(settings_path)
    local_files = LocalFileMCP(PROJECT_ROOT, writable_root=PROJECT_ROOT / "mods")
    rag = RagService(
        desc_persist_dir=desc_persist_dir,
        desc_collection_name=desc_collection_name,
        embedding_model=embedding_model,
        models_root=models_root,
        code_chars=code_chars,
    )
    milvus_uri = rag.milvus_uri

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        if not request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.route("/api/<path:_path>", methods=["OPTIONS"])
    def options_api(_path: str):
        return ("", 204)

    @app.get("/")
    def index():
        return send_from_directory(FRONTEND_ROOT, "index.html")

    @app.get("/make")
    def make_page():
        return send_from_directory(FRONTEND_ROOT, "make.html")

    @app.get("/query")
    def query_page():
        return send_from_directory(FRONTEND_ROOT, "query.html")

    @app.get("/mod")
    def mod_page():
        return send_from_directory(FRONTEND_ROOT, "mod.html")

    @app.get("/settings")
    def settings_page():
        return send_from_directory(FRONTEND_ROOT, "settings.html")

    @app.get("/api/mods")
    def list_mods():
        mods_root = PROJECT_ROOT / "mods"
        mods = []
        if mods_root.exists():
            for item in sorted(mods_root.iterdir()):
                if item.is_dir() and not item.name.startswith("."):
                    mods.append({"name": item.name})
        return jsonify({"ok": True, "mods": mods})

    @app.get("/api/mods/<mod_name>/models")
    def list_mod_models(mod_name: str):
        if ".." in mod_name or "/" in mod_name or "\\" in mod_name:
            return jsonify({"ok": False, "error": "invalid mod name"}), 400
        mod_root = PROJECT_ROOT / "mods" / mod_name
        models_path = mod_root / "src" / "Core" / "Models"

        rules_by_cat: dict[str, list[tuple[str, str]]] = {}
        rules_file = PROJECT_ROOT / "data" / "settings" / "rules.json"
        try:
            raw = rules_file.read_text(encoding="utf-8")
            raw = re.sub(r",(\s*[}\]])", r"\1", raw)
            rules_doc = json.loads(raw)
            for rule in rules_doc.get("rules", []) or []:
                cat = str(rule.get("name") or "").strip()
                if not cat:
                    continue
                paths = []
                for k, v in rule.items():
                    if k.startswith("file_path") and isinstance(v, str) and v:
                        paths.append((k, v))
                rules_by_cat[cat.lower()] = paths
        except (OSError, ValueError):
            pass

        asset_exts = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".bmp", ".gif", ".tres"}
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

        # Load localization JSON files lazily, keyed by (lang_dir, category_lower).
        loc_root = mod_root / "localization"
        loc_cache: dict[tuple[str, str], dict[str, Any]] = {}

        def load_localization(lang_dir: str, category_lower: str) -> dict[str, Any]:
            cache_key = (lang_dir, category_lower)
            if cache_key in loc_cache:
                return loc_cache[cache_key]
            data: dict[str, Any] = {}
            if category_lower:
                f = loc_root / lang_dir / f"{category_lower}.json"
                if f.exists():
                    try:
                        raw = f.read_text(encoding="utf-8")
                        raw = re.sub(r",(\s*[}\]])", r"\1", raw)
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            data = parsed
                    except (OSError, ValueError):
                        data = {}
            loc_cache[cache_key] = data
            return data

        # zh UI → zhs folder, en UI → eng folder.
        ui_lang_to_dir = {"zh": "zhs", "en": "eng"}

        def to_snake(name: str) -> str:
            s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
            s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
            return s.lower()

        def find_assets(category: str, snake: str):
            found: list[dict[str, str]] = []
            missing: list[dict[str, str]] = []
            for key, rel_path in rules_by_cat.get((category or "").lower(), []):
                base = mod_root / rel_path
                hit = False
                if base.exists():
                    for f in base.rglob(snake + ".*"):
                        if f.is_file() and f.suffix.lower() in asset_exts:
                            found.append({
                                "key": key,
                                "path": str(f.relative_to(mod_root)).replace("\\", "/"),
                            })
                            hit = True
                if not hit:
                    missing.append({"key": key, "path": rel_path})
            return found, missing

        files = []
        class_re = re.compile(r"\bclass\s+([A-Za-z_]\w*)")
        if models_path.exists():
            for cs_file in sorted(models_path.rglob("*.cs")):
                rel = cs_file.relative_to(models_path)
                rel_posix = str(rel).replace("\\", "/")
                folder = str(rel.parent).replace("\\", "/")
                if folder == ".":
                    folder = ""
                category = folder.split("/", 1)[0] if folder else ""
                class_name = ""
                try:
                    text = cs_file.read_text(encoding="utf-8", errors="ignore")
                    m = class_re.search(text)
                    if m:
                        class_name = m.group(1)
                except OSError:
                    pass
                resolved_class = class_name or cs_file.stem
                snake = to_snake(resolved_class)
                assets_found, missing_list = find_assets(category, snake)
                assets_missing = missing_list

                localization: dict[str, dict[str, str]] = {}
                image_rel: str = ""
                if assets_found:
                    for a in assets_found:
                        p = a.get("path") or ""
                        if Path(p).suffix.lower() in image_exts:
                            image_rel = p
                            break
                cat_lower = (category or "").lower()
                key_upper = snake.upper()
                for ui_lang, lang_dir in ui_lang_to_dir.items():
                    loc = load_localization(lang_dir, cat_lower)
                    title = loc.get(f"{key_upper}.title")
                    desc = loc.get(f"{key_upper}.description")
                    if title or desc:
                        localization[ui_lang] = {
                            "title": title or "",
                            "description": desc or "",
                        }

                files.append({
                    "name": cs_file.name,
                    "rel_path": rel_posix,
                    "folder": folder,
                    "category": category,
                    "class_name": resolved_class,
                    "snake_name": snake,
                    "assets_found": assets_found,
                    "assets_missing": assets_missing,
                    "assets_expected": [k for k, _ in rules_by_cat.get((category or "").lower(), [])],
                    "image_path": image_rel,
                    "localization": localization,
                })
        return jsonify({"ok": True, "mod": mod_name, "files": files})

    @app.delete("/api/mods/<mod_name>")
    def delete_mod(mod_name: str):
        if ".." in mod_name or "/" in mod_name or "\\" in mod_name or not mod_name.strip():
            return jsonify({"ok": False, "error": "invalid mod name"}), 400
        mods_root = (PROJECT_ROOT / "mods").resolve()
        target = (mods_root / mod_name).resolve()
        try:
            target.relative_to(mods_root)
        except ValueError:
            return jsonify({"ok": False, "error": "path escapes mods root"}), 400
        if target == mods_root:
            return jsonify({"ok": False, "error": "refusing to delete mods root"}), 400
        if not target.exists():
            return jsonify({"ok": False, "error": f"mod not found: {mod_name}"}), 404
        if not target.is_dir():
            return jsonify({"ok": False, "error": "target is not a directory"}), 400
        import shutil
        def _on_error(func, path, exc_info):
            # Windows often marks files read-only; retry after clearing the bit.
            try:
                import stat
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except OSError:
                raise
        try:
            shutil.rmtree(target, onerror=_on_error)
        except OSError as exc:
            return jsonify({"ok": False, "error": f"delete failed: {exc}"}), 500
        return jsonify({"ok": True, "mod": mod_name})

    @app.post("/api/mods/<mod_name>/open-folder")
    def open_mod_folder(mod_name: str):
        if ".." in mod_name or "/" in mod_name or "\\" in mod_name:
            return jsonify({"ok": False, "error": "invalid mod name"}), 400
        payload = request.get_json(silent=True) or {}
        rel = str(payload.get("path") or "").strip().replace("\\", "/")
        if not rel or rel.startswith("/") or ".." in rel.split("/"):
            return jsonify({"ok": False, "error": "invalid path"}), 400
        mod_root = (PROJECT_ROOT / "mods" / mod_name).resolve()
        target = (mod_root / rel).resolve()
        try:
            target.relative_to(mod_root)
        except ValueError:
            return jsonify({"ok": False, "error": "path escapes mod root"}), 400
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return jsonify({"ok": False, "error": f"mkdir failed: {exc}"}), 500
        import subprocess
        try:
            if sys.platform.startswith("win"):
                import os as _os
                _os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except OSError as exc:
            return jsonify({"ok": False, "error": f"open failed: {exc}"}), 500
        return jsonify({"ok": True, "path": str(target)})

    @app.post("/api/mods/<mod_name>/export")
    def export_mod(mod_name: str):
        if ".." in mod_name or "/" in mod_name or "\\" in mod_name:
            return jsonify({"ok": False, "error": "invalid mod name"}), 400
        mod_root = (PROJECT_ROOT / "mods" / mod_name).resolve()
        if not mod_root.is_dir():
            return jsonify({"ok": False, "error": f"mod not found: {mod_name}"}), 404

        raw = (os.environ.get("EXPORT_TOOL_PATH") or "").strip().strip('"').strip("'")
        if not raw:
            return jsonify({"ok": False, "error": "EXPORT_TOOL_PATH is not configured"}), 400
        cmd_path = Path(raw)
        if not cmd_path.is_file():
            return jsonify({"ok": False, "error": f"export tool not found: {cmd_path}"}), 404

        text = cmd_path.read_text(encoding="utf-8")
        line_sep = "\r\n" if "\r\n" in text else ("\n" if "\n" in text else "\r\n")
        lines = text.splitlines()
        mod_path_value = str(mod_root)
        set_re = re.compile(
            r"^(?P<prefix>\s*set\s+)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=(?P<value>.*)$",
            re.IGNORECASE,
        )
        replaced = False
        for i, raw_line in enumerate(lines):
            m = set_re.match(raw_line)
            if m and m.group("key").upper() == "MOD_PATH":
                lines[i] = f"set MOD_PATH={mod_path_value}"
                replaced = True
                break
        if not replaced:
            insert_at = 0
            for i, raw_line in enumerate(lines):
                if raw_line.strip():
                    insert_at = i
                    break
            lines.insert(insert_at, f"set MOD_PATH={mod_path_value}")
        new_text = line_sep.join(lines)
        if not new_text.endswith(line_sep):
            new_text += line_sep
        cmd_path.write_text(new_text, encoding="utf-8")

        import subprocess
        try:
            if sys.platform.startswith("win"):
                CREATE_NEW_CONSOLE = 0x00000010
                subprocess.Popen(
                    ["cmd.exe", "/c", str(cmd_path)],
                    cwd=str(cmd_path.parent),
                    creationflags=CREATE_NEW_CONSOLE,
                    close_fds=True,
                )
            else:
                subprocess.Popen(
                    ["/bin/sh", str(cmd_path)],
                    cwd=str(cmd_path.parent),
                    close_fds=True,
                )
        except OSError as exc:
            return jsonify({"ok": False, "error": f"launch failed: {exc}"}), 500

        return jsonify({
            "ok": True,
            "mod": mod_name,
            "mod_path": mod_path_value,
            "cmd_path": str(cmd_path),
        })

    @app.get("/api/mods/<mod_name>/file")
    def serve_mod_file(mod_name: str):
        if ".." in mod_name or "/" in mod_name or "\\" in mod_name:
            return jsonify({"ok": False, "error": "invalid mod name"}), 400
        rel = str(request.args.get("path") or "").strip().replace("\\", "/")
        if not rel or rel.startswith("/") or ".." in rel.split("/"):
            return jsonify({"ok": False, "error": "invalid path"}), 400
        mod_root = (PROJECT_ROOT / "mods" / mod_name).resolve()
        target = (mod_root / rel).resolve()
        try:
            target.relative_to(mod_root)
        except ValueError:
            return jsonify({"ok": False, "error": "path escapes mod root"}), 400
        if not target.is_file():
            return jsonify({"ok": False, "error": "not found"}), 404
        return send_from_directory(target.parent, target.name)

    @app.get("/api/health")
    def health():
        local_lite = is_local_lite_uri(milvus_uri)
        description_db_exists = (not local_lite) or desc_persist_dir.exists()
        vector_domains = []
        for domain in RAG_DOMAINS:
            if domain in MAIN_RAG_DOMAINS:
                exists = description_db_exists
                collection = desc_collection_name
                persist_path = desc_persist_dir
            else:
                persist_path = localization_persist_path(DEFAULT_VECTOR_DB_ROOT, domain)
                exists = (not is_local_lite_uri(resolve_milvus_uri(persist_path))) or persist_path.exists()
                collection = localization_collection_name(domain)
            vector_domains.append(
                {
                    "id": domain,
                    "label": RAG_DOMAIN_LABELS.get(domain, domain.replace("_", " ").title()),
                    "available": exists,
                    "collection": collection,
                    "persist_path": str(persist_path),
                }
            )
        return jsonify(
            {
                "ok": True,
                "project_root": str(PROJECT_ROOT),
                "backend_root": str(BACKEND_ROOT),
                "frontend_root": str(FRONTEND_ROOT),
                "models_root_exists": models_root.exists(),
                "description_db_exists": description_db_exists,
                "milvus_uri": milvus_uri,
                "milvus_lite_available": has_milvus_lite(),
                "default_context_n": default_context_n,
                "default_desc_top_k": default_desc_top_k,
                "vector_domains": vector_domains,
            }
        )

    # ─── .env editor (settings modal) ──────────────────────────────────
    # Editable subset of the .env file. Tokens / API keys are returned
    # masked-only; the frontend never sees the cleartext value. To clear a
    # sensitive value, the frontend can POST the literal empty string —
    # the modal only sends sensitive keys when the user typed something.
    ENV_EDITABLE_KEYS = [
        "deepseek_api_key",
        "HF_TOKEN",
        "MILVUS_TOKEN",
        "EMBEDDING_MODEL",
        "EMBEDDING_BATCH_SIZE",
        "MILVUS_URI",
        "MILVUS_DB_NAME",
        "DESC_COLLECTION_NAME",
        "APP_HOST",
        "APP_PORT",
        "DESC_TOP_K",
        "CONTEXT_N",
        "CODE_CHARS",
        "GAME_ROOT",
        "EXPORT_TOOL_PATH",
        "GODOT_TOOL_PATH",
    ]
    ENV_SENSITIVE_KEYS = {"deepseek_api_key", "HF_TOKEN", "MILVUS_TOKEN"}

    def _parse_env_file() -> tuple[list[str], dict[str, int]]:
        """Return (lines, key→line_index) for the current .env file.

        Preserves comments, ordering, and unknown keys. The map points at the
        line that defines each known key so an update rewrites it in place.
        """
        path = DEFAULT_ENV_PATH
        lines: list[str] = []
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
        index: dict[str, int] = {}
        for i, raw in enumerate(lines):
            stripped = raw.lstrip()
            # Skip blank lines and full-line comments.
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key = stripped.split("=", 1)[0].strip()
            if key and key not in index:
                index[key] = i
        return lines, index

    def _read_env_values() -> dict[str, str]:
        _, index = _parse_env_file()
        lines = DEFAULT_ENV_PATH.read_text(encoding="utf-8").splitlines() if DEFAULT_ENV_PATH.exists() else []
        out: dict[str, str] = {}
        for key, line_no in index.items():
            raw = lines[line_no]
            _, _, value = raw.partition("=")
            out[key] = value.strip()
        return out

    def _mask_secret(value: str) -> dict[str, Any]:
        if not value:
            return {"masked": True, "preview": "", "present": False}
        # Show up to 4 leading characters + asterisks; never reveal more.
        head = value[:4]
        return {
            "masked": True,
            "preview": f"{head}{'•' * max(4, min(12, len(value) - len(head)))}",
            "present": True,
        }

    @app.get("/api/env")
    def get_env_values():
        raw = _read_env_values()
        payload: dict[str, Any] = {}
        for key in ENV_EDITABLE_KEYS:
            value = raw.get(key, "")
            if key in ENV_SENSITIVE_KEYS:
                payload[key] = _mask_secret(value)
            else:
                payload[key] = {"value": value}
        return jsonify({"ok": True, "values": payload})

    @app.post("/api/env")
    def update_env_values():
        body = request.get_json(silent=True) or {}
        updates = body.get("updates")
        if not isinstance(updates, dict):
            return jsonify({"ok": False, "error": "updates must be an object"}), 400

        lines, index = _parse_env_file()
        # Validate every key first; reject unknown keys outright.
        for key in updates.keys():
            if key not in ENV_EDITABLE_KEYS:
                return jsonify({"ok": False, "error": f"unknown env key: {key}"}), 400

        def quote_if_needed(value: str) -> str:
            if value == "":
                return ""
            # Quote values containing whitespace or '#' so the parser doesn't
            # truncate them at an inline comment.
            if any(ch.isspace() for ch in value) or "#" in value:
                escaped = value.replace("\\", "\\\\").replace('"', '\\"')
                return f'"{escaped}"'
            return value

        for key, raw_value in updates.items():
            if raw_value is None:
                continue
            value = str(raw_value)
            new_line = f"{key}={quote_if_needed(value)}"
            if key in index:
                lines[index[key]] = new_line
            else:
                if lines and lines[-1].strip() != "":
                    lines.append("")
                lines.append(new_line)
                index[key] = len(lines) - 1

        text = "\n".join(lines)
        if not text.endswith("\n"):
            text += "\n"
        DEFAULT_ENV_PATH.write_text(text, encoding="utf-8")

        # Re-read and return masked snapshot so the modal can refresh state.
        raw = _read_env_values()
        payload: dict[str, Any] = {}
        for key in ENV_EDITABLE_KEYS:
            value = raw.get(key, "")
            if key in ENV_SENSITIVE_KEYS:
                payload[key] = _mask_secret(value)
            else:
                payload[key] = {"value": value}
        return jsonify({"ok": True, "values": payload})

    # ─── ExportMod.cmd editor ─────────────────────────────────────────
    # The four `set NAME=VALUE` lines at the top of the export script.
    # We rewrite those lines in place, preserving everything else.
    EXPORT_CMD_KEYS = ["MOD_PATH", "GODOT_PATH", "GODOT_EXPORT_PATH", "STS2_PATH"]
    _EXPORT_SET_RE = re.compile(
        r"^(?P<prefix>\s*set\s+)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=(?P<value>.*)$",
        re.IGNORECASE,
    )

    def _export_cmd_path() -> Path | None:
        raw = (os.environ.get("EXPORT_TOOL_PATH") or "").strip().strip('"').strip("'")
        if not raw:
            return None
        try:
            return Path(raw)
        except Exception:
            return None

    def _parse_export_cmd(path: Path) -> tuple[list[str], str, dict[str, int]]:
        """Return (lines, line_sep, key→line_index) for ExportMod.cmd."""
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        # Detect line separator (.cmd files are typically CRLF on Windows).
        line_sep = "\r\n" if "\r\n" in text else ("\n" if "\n" in text else "\r\n")
        lines = text.splitlines() if text else []
        index: dict[str, int] = {}
        for i, raw in enumerate(lines):
            m = _EXPORT_SET_RE.match(raw)
            if m:
                key = m.group("key").upper()
                if key not in index:
                    index[key] = i
        return lines, line_sep, index

    def _export_cmd_payload(lines: list[str], index: dict[str, int]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in EXPORT_CMD_KEYS:
            if key in index:
                m = _EXPORT_SET_RE.match(lines[index[key]])
                value = m.group("value").strip() if m else ""
            else:
                value = ""
            payload[key] = {"value": value}
        return payload

    @app.get("/api/export-config")
    def get_export_config():
        path = _export_cmd_path()
        if path is None:
            return jsonify({"ok": False, "error": "EXPORT_TOOL_PATH is not configured"}), 404
        if not path.exists():
            return jsonify({"ok": False, "error": f"file not found: {path}"}), 404
        lines, _sep, index = _parse_export_cmd(path)
        return jsonify({"ok": True, "values": _export_cmd_payload(lines, index), "path": str(path)})

    @app.post("/api/export-config")
    def update_export_config():
        body = request.get_json(silent=True) or {}
        updates = body.get("updates")
        if not isinstance(updates, dict):
            return jsonify({"ok": False, "error": "updates must be an object"}), 400

        path = _export_cmd_path()
        if path is None:
            return jsonify({"ok": False, "error": "EXPORT_TOOL_PATH is not configured"}), 400
        if not path.exists():
            return jsonify({"ok": False, "error": f"file not found: {path}"}), 404

        normalized = {}
        for key, value in updates.items():
            ukey = str(key).upper()
            if ukey not in EXPORT_CMD_KEYS:
                return jsonify({"ok": False, "error": f"unknown key: {key}"}), 400
            if value is None:
                continue
            normalized[ukey] = str(value)

        lines, line_sep, index = _parse_export_cmd(path)
        for key, value in normalized.items():
            new_line = f"set {key}={value}"
            if key in index:
                lines[index[key]] = new_line
            else:
                if lines and lines[-1].strip() != "":
                    lines.append("")
                lines.append(new_line)
                index[key] = len(lines) - 1

        text = line_sep.join(lines)
        if not text.endswith(line_sep):
            text += line_sep
        path.write_text(text, encoding="utf-8")

        lines, _sep, index = _parse_export_cmd(path)
        return jsonify({"ok": True, "values": _export_cmd_payload(lines, index), "path": str(path)})

    @app.get("/api/config")
    def get_config():
        return jsonify(settings.public_config())

    @app.post("/api/config/providers/<provider_id>")
    def save_provider(provider_id: str):
        payload = request.get_json(silent=True) or {}
        settings.update_provider(provider_id, payload)
        return jsonify(settings.public_config())

    @app.post("/api/config/default")
    def save_default_provider():
        payload = request.get_json(silent=True) or {}
        provider_id = str(payload.get("provider_id") or "deepseek").strip()
        settings.set_default(provider_id)
        return jsonify(settings.public_config())

    @app.get("/api/agent/tools")
    def agent_tools():
        tools = AgentMCP(
            local_files=local_files,
            rag_query_handler=rag.make_rag_query_handler(
                desc_top_k_default=default_desc_top_k,
                context_n_default=default_context_n,
                enabled=True,
                local_files=local_files,
            ),
        )
        return jsonify({"ok": True, "tools": tools.tool_specs()})

    @app.get("/api/mcp/files/list")
    def list_local_files():
        path = str(request.args.get("path") or ".")
        pattern = str(request.args.get("pattern") or "*")
        limit = positive_int(request.args.get("limit"), 0)
        return jsonify(local_files.list_files(path=path, pattern=pattern, limit=limit))

    @app.post("/api/mcp/files/read")
    def read_local_file():
        payload = request.get_json(silent=True) or {}
        max_chars = positive_int(payload.get("max_chars"), 0)
        return jsonify(local_files.read_file(str(payload.get("path") or ""), max_chars=max_chars))

    @app.post("/api/descriptions/split")
    def api_descriptions_split():
        payload = request.get_json(silent=True) or {}

        sources: list[tuple[str, str]] = []  # (label, content)
        raw_files = payload.get("selected_files")
        if isinstance(raw_files, list):
            for raw_path in raw_files:
                path = str(raw_path or "").strip()
                if not path:
                    continue
                result = local_files.read_file(path, max_chars=0)
                if result.get("ok"):
                    sources.append((str(result.get("path") or path), str(result.get("content") or "")))

        raw_attachments = payload.get("attachments")
        if isinstance(raw_attachments, list):
            for item in raw_attachments:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "attachment").strip() or "attachment"
                content = str(item.get("content") or "")
                if content:
                    sources.append((name, content))

        segments = extract_description_segments(sources)
        domains = [
            {"id": domain, "label": RAG_DOMAIN_LABELS.get(domain, domain)}
            for domain in RAG_DOMAINS
        ]
        return jsonify({"segments": segments, "domains": domains})

    @app.post("/api/query")
    def api_query():
        payload = request.get_json(silent=True) or {}
        query = str(payload.get("query") or "").strip()
        if not query:
            return jsonify({"error": "query is required"}), 400

        desc_top_k = at_least_one_int(payload.get("desc_top_k"), default_desc_top_k)
        context_n = at_least_one_int(payload.get("context_n"), default_context_n)
        domains_were_provided = "domains" in payload
        requested_domains = normalize_rag_domains(payload.get("domains"))
        if domains_were_provided and not requested_domains:
            return jsonify({"error": "at least one valid vector domain is required"}), 400
        if not requested_domains:
            requested_domains = list(RAG_DOMAINS)
        main_domains = [domain for domain in requested_domains if domain in MAIN_RAG_DOMAINS]
        localization_domains = [domain for domain in requested_domains if domain in LOCALIZATION_RAG_DOMAINS]
        if main_domains and rag.description_db_missing():
            return jsonify({"error": f"Milvus description DB not found: {desc_persist_dir}. Build it first with python backend\\scripts\\build.py --rebuild"}), 400
        started = time.perf_counter()
        groups: list[dict[str, Any]] = []
        searched_domains: list[str] = []
        if main_domains:
            main_groups = retrieve_code_context_groups(
                desc_db=rag.get_desc_db(),
                models_root=models_root,
                query=query,
                desc_top_k=desc_top_k,
                code_chars=code_chars,
                context_n=context_n,
                domains=main_domains,
            )
            main_source = main_domains[0] if len(main_domains) == 1 else "descriptions"
            for group in main_groups:
                group["source_db"] = main_source
                for context in group.get("contexts", []):
                    if isinstance(context, dict):
                        context["source_db"] = context.get("domain") or main_source
            groups.extend(main_groups)
            searched_domains.extend(main_domains)
        for domain, db in rag.get_localization_dbs(localization_domains):
            domain_groups = retrieve_code_context_groups(
                desc_db=db,
                models_root=models_root,
                query=query,
                desc_top_k=desc_top_k,
                code_chars=code_chars,
                context_n=context_n,
            )
            for group in domain_groups:
                group["source_db"] = domain
                for context in group.get("contexts", []):
                    if isinstance(context, dict):
                        context["source_db"] = domain
            if domain_groups:
                groups.extend(domain_groups)
                searched_domains.append(domain)
        groups = filter_groups_by_domains(groups, requested_domains)
        contexts = [
            context
            for group in groups
            for context in group.get("contexts", [])
            if isinstance(group.get("contexts"), list)
        ]
        return jsonify(
            {
                "query": query,
                "groups": groups,
                "public_groups": public_context_groups(groups),
                "requested_domains": requested_domains,
                "searched_domains": searched_domains,
                "search_plan": [
                    {
                        "domain": domain,
                        "vector_db": (
                            "descriptions"
                            if domain in MAIN_RAG_DOMAINS
                            else domain
                        ),
                    }
                    for domain in requested_domains
                ],
                "context_count": len(contexts),
                "desc_top_k": desc_top_k,
                "context_n": context_n,
                "duration_ms": int((time.perf_counter() - started) * 1000),
            }
        )

    @app.post("/api/chat/stream")
    def api_chat_stream():
        payload = request.get_json(silent=True) or {}
        raw_messages = payload.get("messages") or []
        if not isinstance(raw_messages, list) or not raw_messages:
            return jsonify({"error": "messages is required"}), 400

        last_user = next(
            (
                str(message.get("content") or "").strip()
                for message in reversed(raw_messages)
                if str(message.get("role") or "").lower() == "user"
            ),
            "",
        )
        if not last_user:
            return jsonify({"error": "no user message found"}), 400

        provider_id = str(payload.get("provider_id") or "").strip() or None
        provider = settings.resolve_provider(provider_id)
        model_override = str(payload.get("model") or "").strip() or None
        temperature_override = maybe_float(payload.get("temperature"))
        try:
            llm = create_llm(provider, model_override=model_override, temperature_override=temperature_override)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        def generate():
            started = time.perf_counter()
            context_groups: list[dict[str, Any]] = []
            traces: list[dict[str, Any]] = []
            answer_parts: list[str] = []
            reasoning_parts: list[str] = []
            memory_summary = str(payload.get("memory_summary") or "").strip()
            search_query = str(payload.get("search_query") or "").strip()
            use_rag = bool(payload.get("use_rag", True))
            use_agent = bool(payload.get("use_agent", False))
            one_by_one = bool(payload.get("one_by_one", False))
            language = normalize_language(payload.get("language") or payload.get("lang"))
            desc_top_k = default_desc_top_k
            context_n = default_context_n
            updated_summary = memory_summary
            request_id = f"{int(time.time() * 1000)}-{safe_log_name(last_user)[:40]}"
            # Completion tracking — populated by the agent loop, defaulted for the
            # non-agent path so the post-loop summary block can read them safely.
            written_targets: set[str] = set()
            final_answer_emitted = True  # non-agent path is always "clean"

            def log_event(event: str, data: dict[str, Any] | None = None) -> None:
                append_ai_chat_event(
                    payload=payload,
                    request_id=request_id,
                    event=event,
                    data={
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                        **(data or {}),
                    },
                )

            def emit(event_type: str, data: dict[str, Any] | None = None) -> str:
                log_event(event_type, data)
                return stream_event(event_type, data)

            try:
                log_event(
                    "request_start",
                    {
                        "endpoint": "/api/chat/stream",
                        "provider_id": provider.get("provider_id"),
                        "model": model_override or provider.get("model"),
                        "language": language,
                        "use_rag": use_rag,
                        "use_agent": use_agent,
                        "input": {
                            "last_user": last_user,
                            "memory_summary": memory_summary,
                        },
                        "selected_files": payload.get("selected_files") if isinstance(payload.get("selected_files"), list) else [],
                    },
                )
                yield emit(
                    "start",
                    {
                        "provider_id": provider.get("provider_id"),
                        "model": model_override or provider.get("model"),
                        "use_rag": use_rag,
                        "use_agent": use_agent,
                    },
                )
                if memory_summary:
                    yield emit("memory_loaded", {"summary": memory_summary})

                if use_agent:
                    search_query = (
                        "Agent will extract retrieval queries and call rag_query MCP."
                        if language == "en"
                        else "Agent 会自行提取检索查询并调用 rag_query MCP。"
                    )
                elif not search_query:
                    search_query = last_user

                desc_top_k = at_least_one_int(payload.get("desc_top_k"), default_desc_top_k)
                context_n = at_least_one_int(payload.get("context_n"), default_context_n)
                if not use_rag:
                    context_text = "RAG disabled." if language == "en" else "RAG 已禁用。"
                elif language == "en":
                    context_text = "No RAG context has been preloaded. In Agent mode, call rag_query MCP when reference code is needed."
                else:
                    context_text = "尚未预加载 RAG 上下文。Agent 模式下需要参考代码时，请调用 rag_query MCP。"
                if (not use_agent) and use_rag:
                    if rag.description_db_missing():
                        raise ValueError(
                            f"Milvus description DB not found: {desc_persist_dir}. "
                            "Build it first with python backend\\scripts\\build.py --rebuild"
                        )
                    retrieval_started = time.perf_counter()
                    yield emit(
                        "retrieval_start",
                        {"query": search_query, "desc_top_k": desc_top_k, "context_n": context_n},
                    )
                    context_groups = retrieve_code_context_groups(
                        desc_db=rag.get_desc_db(),
                        models_root=models_root,
                        query=search_query,
                        desc_top_k=desc_top_k,
                        code_chars=code_chars,
                        context_n=context_n,
                    )
                    context_text = build_grouped_prompt_context(context_groups)
                    contexts = [
                        context
                        for group in context_groups
                        for context in group.get("contexts", [])
                        if isinstance(group.get("contexts"), list)
                    ]
                    yield emit(
                        "retrieval_done",
                        {
                            "query": search_query,
                            "groups": public_context_groups(context_groups),
                            "context_count": len(contexts),
                            "duration_ms": int((time.perf_counter() - retrieval_started) * 1000),
                        },
                    )

                selected_file_context, selected_file_traces = read_selected_file_context(
                    local_files,
                    payload.get("selected_files"),
                )
                traces.extend(selected_file_traces)
                if selected_file_traces:
                    yield emit("local_files_done", {"traces": selected_file_traces})

                prompt_messages = build_prompt_messages(
                    raw_messages=raw_messages,
                    memory_summary=memory_summary,
                    search_query=search_query,
                    context_text=context_text,
                    selected_file_context=selected_file_context,
                    language=language,
                    one_by_one=one_by_one,
                )
                yield emit(
                    "prompt_ready",
                    {
                        "message_count": len(prompt_messages),
                        "context_chars": len(context_text),
                        "selected_file_chars": len(selected_file_context),
                    },
                )

                if use_agent:
                    agent_max_steps = positive_int(payload.get("agent_max_steps"), 0)
                    yield emit(
                        "agent_start",
                        {"max_steps": agent_max_steps},
                    )
                    agent_tools = AgentMCP(
                        local_files=local_files,
                        rag_query_handler=rag.make_rag_query_handler(
                            desc_top_k_default=desc_top_k,
                            context_n_default=context_n,
                            enabled=use_rag,
                            domain_hint_text=last_user,
                            local_files=local_files,
                        ),
                    )
                    current_messages: list[Any] = list(prompt_messages) + [
                        HumanMessage(content=AGENT_TOOL_SELECTION_PROMPT)
                    ]
                    written_targets: set[str] = set()
                    executed_tool_keys: set[str] = set()
                    # Cache the public trace_result for each (tool, args) so the
                    # dedup rejection can return the prior result inline. This
                    # prevents the agent from wasting an extra turn re-reading
                    # state to "verify what happened".
                    executed_tool_results: dict[str, Any] = {}
                    seen_followup_keys: set[str] = set()
                    stream_tool_llm = llm.bind_tools(agent_tools.tool_specs())
                    stream_max_steps = max(1, agent_max_steps) if agent_max_steps else 48
                    READ_ONLY_TOOLS = {
                        "rag_query",
                        "local_file_read",
                        "local_file_read_many",
                        "local_file_search",
                        "local_file_list",
                    }
                    final_answer_emitted = False
                    consecutive_readonly_steps = 0
                    for step_index in range(stream_max_steps):
                        # B: sliding-window trim of stale ToolMessages so the prompt
                        # does not balloon as the loop progresses.
                        current_messages = trim_tool_message_history(current_messages, keep_recent=8)
                        yield emit("generation_start", {"step": step_index + 1, "max_steps": stream_max_steps})
                        accumulated = None
                        for chunk in stream_tool_llm.stream(current_messages):
                            accumulated = chunk if accumulated is None else accumulated + chunk
                            reasoning = response_reasoning_content(chunk)
                            if reasoning:
                                reasoning_parts.append(reasoning)
                                yield emit("reasoning_content", {"text": reasoning})
                            text = content_text(getattr(chunk, "content", ""))
                            if not text:
                                continue
                            answer_parts.append(text)
                            yield emit("token", {"text": text})

                        structured_calls = list(getattr(accumulated, "tool_calls", []) or []) if accumulated is not None else []
                        raw_call_ids = raw_tool_call_ids(accumulated) if accumulated is not None else []
                        if not structured_calls and not raw_call_ids:
                            # No tool calls this turn — either done, or completion check fails.
                            followup = compute_agent_followup_prompt(
                                local_files=local_files,
                                original_messages=prompt_messages,
                                seen_keys=seen_followup_keys,
                            )
                            if followup is None:
                                final_answer_emitted = True
                                break
                            current_messages.append(HumanMessage(content=followup))
                            consecutive_readonly_steps = 0
                            continue

                        current_messages.append(accumulated)
                        executed_any_tool = False
                        any_write_this_step = False
                        any_useful_this_step = False
                        duplicate_only = True
                        handled_tool_call_ids: set[str] = set()
                        for call_index, call in enumerate(structured_calls):
                            name = str(call.get("name") or "")
                            raw_args = call.get("args") or {}
                            if not isinstance(raw_args, dict):
                                raw_args = {}
                            args = normalize_tool_args(name, raw_args)
                            if not isinstance(args, dict):
                                args = {}
                            call_id = str(call.get("id") or "")
                            if not call_id and call_index < len(raw_call_ids):
                                call_id = raw_call_ids[call_index]
                            invalid_reason = invalid_tool_call_reason(name, args)
                            if invalid_reason:
                                executed_tool_keys.add(tool_command_key(name, args))
                                current_messages.append(
                                    ToolMessage(content=invalid_reason, tool_call_id=call_id)
                                )
                                if call_id:
                                    handled_tool_call_ids.add(call_id)
                                continue
                            command_key = tool_command_key(name, args)
                            # Hard-stop duplicate (tool, args) calls — they previously
                            # burned whole turns repeating useless searches. The
                            # rejection now carries the PRIOR result so the agent
                            # doesn't need to re-read or re-search to recover; it can
                            # just continue with whatever it actually planned next.
                            if command_key in executed_tool_keys and not is_mod_scoped_tool_call(name, args):
                                prior_result = executed_tool_results.get(command_key)
                                rejection_payload = {
                                    "ok": False,
                                    "skipped": True,
                                    "reason": (
                                        "duplicate tool call — this exact (tool, args) was already "
                                        "executed earlier in this conversation. The prior result is "
                                        "embedded below as 'prior_result'; treat it as authoritative "
                                        "and do NOT re-read or re-search to verify. Continue with "
                                        "the next planned action (write files, append localization, "
                                        "or finalize the answer)."
                                    ),
                                    "prior_result": prior_result,
                                }
                                rejection_text = json.dumps(rejection_payload, ensure_ascii=False)
                                current_messages.append(
                                    ToolMessage(content=rejection_text, tool_call_id=call_id)
                                )
                                if call_id:
                                    handled_tool_call_ids.add(call_id)
                                # Surface the rejection in the UI too, so the user
                                # can see it was deliberately skipped (not silently lost).
                                yield emit(
                                    "agent_trace",
                                    {
                                        "tool": name,
                                        "arguments": args,
                                        "result": {
                                            "ok": False,
                                            "skipped": True,
                                            "reason": "duplicate",
                                        },
                                        "skipped_duplicate": True,
                                    },
                                )
                                continue
                            duplicate_only = False
                            pending_trace = {"tool": name, "arguments": args, "result": {}}
                            result = agent_tools.call(name, args)
                            executed_tool_keys.add(command_key)
                            executed_any_tool = True
                            pending_trace["result"] = result
                            if result.get("ok") and (target := trace_write_target(pending_trace)):
                                written_targets.add(target)
                                any_write_this_step = True
                            if name not in READ_ONLY_TOOLS and result.get("ok"):
                                any_write_this_step = True
                            if result.get("ok"):
                                any_useful_this_step = True
                            trace = {"tool": name, "arguments": args, "result": trace_result(name, result)}
                            traces.append(trace)
                            executed_tool_results[command_key] = trace["result"]
                            yield emit("agent_trace", trace)
                            current_messages.append(
                                ToolMessage(
                                    content=json.dumps(result, ensure_ascii=False),
                                    tool_call_id=call_id,
                                )
                            )
                            if call_id:
                                handled_tool_call_ids.add(call_id)
                        for call_id in raw_call_ids:
                            if call_id not in handled_tool_call_ids:
                                current_messages.append(invalid_unparsed_tool_message(call_id))
                        # Track read-only-only streaks; if the agent has been
                        # investigating without writing for too long, force it
                        # to either write or finalize.
                        if any_write_this_step:
                            consecutive_readonly_steps = 0
                        elif executed_any_tool:
                            consecutive_readonly_steps += 1
                        nudge_messages: list[str] = []
                        if duplicate_only and structured_calls:
                            followup = compute_agent_followup_prompt(
                                local_files=local_files,
                                original_messages=prompt_messages,
                                seen_keys=seen_followup_keys,
                            )
                            if followup is None:
                                # All calls were dedup'd AND completion checks pass —
                                # the agent is genuinely done, just looping. Force
                                # the next turn to produce the final answer.
                                nudge_messages.append(
                                    "Every tool call you just emitted was a duplicate of a previous one. "
                                    "The prior results are still in this conversation, and all completion "
                                    "checks pass. STOP calling tools. In your next reply, output the final "
                                    "answer in plain text only — no more tool_calls. Briefly list the files "
                                    "you wrote and any caveats."
                                )
                            else:
                                nudge_messages.append(
                                    "Every tool call you just emitted was a duplicate of a previous one. "
                                    "Do not retry them. The remaining gaps are listed below; address each "
                                    "with NEW arguments (different paths or contents):\n\n" + followup
                                )
                            consecutive_readonly_steps = 0
                        elif not executed_any_tool:
                            followup = compute_agent_followup_prompt(
                                local_files=local_files,
                                original_messages=prompt_messages,
                                seen_keys=seen_followup_keys,
                            ) or (
                                "You only emitted MCP tool calls that have already been executed or were invalid. "
                                "Do not repeat those commands. Provide the final answer if all requested artifacts are complete."
                            )
                            nudge_messages.append(followup)
                            consecutive_readonly_steps += 1
                        elif consecutive_readonly_steps >= 3:
                            nudge_messages.append(
                                "You have spent the last several turns only on read-only investigation tools. "
                                "STOP investigating. With the evidence you already have, write the requested .cs files "
                                "now using local_file_write (parallel tool_calls in a single assistant message), then "
                                "append the localization entries with local_file_replace, then provide the final answer. "
                                "If a specific API still cannot be verified, pick a verified alternative or say so in the "
                                "final answer — do not loop on more searches."
                            )
                            consecutive_readonly_steps = 0
                        else:
                            nudge_messages.append(
                                "If the needed files were already written, do not call MCP write tools again. "
                                "Provide the final answer now. Only call a tool for a different missing file or a failed write."
                            )
                        # Encourage termination as the budget runs out.
                        steps_left = stream_max_steps - (step_index + 1)
                        if steps_left <= 3:
                            nudge_messages.append(
                                f"Only {steps_left} agent step(s) remain. Finish the work in this budget: write any "
                                "remaining required files and produce the final answer. Stop investigating."
                            )
                        current_messages.append(HumanMessage(content="\n\n".join(nudge_messages)))

                    # Fix 1: step budget exhausted without a final answer — force one
                    # tool-less wrap-up so the user actually receives a reply.
                    if not final_answer_emitted:
                        yield emit(
                            "warning",
                            {
                                "stage": "agent_loop",
                                "message": (
                                    "Agent step budget reached without a finalized answer; "
                                    "forcing a tool-less wrap-up turn."
                                ),
                                "writes_done": len(written_targets),
                            },
                        )
                        current_messages.append(
                            HumanMessage(
                                content=(
                                    "The agent step budget is exhausted. Do NOT call any more tools. "
                                    "Reply now in plain text: summarize what you actually wrote (file paths), what you "
                                    "could not finish, and any verified-API limitations. If nothing was written, explain "
                                    "concretely why and what the user should do next."
                                )
                            )
                        )
                        yield emit("generation_start", {"step": "wrap_up"})
                        for chunk in llm.stream(current_messages):
                            reasoning = response_reasoning_content(chunk)
                            if reasoning:
                                reasoning_parts.append(reasoning)
                                yield emit("reasoning_content", {"text": reasoning})
                            text = content_text(getattr(chunk, "content", ""))
                            if not text:
                                continue
                            answer_parts.append(text)
                            yield emit("token", {"text": text})
                else:
                    yield emit("generation_start")
                    for chunk in llm.stream(prompt_messages):
                        reasoning = response_reasoning_content(chunk)
                        if reasoning:
                            reasoning_parts.append(reasoning)
                            yield emit("reasoning_content", {"text": reasoning})
                        text = content_text(getattr(chunk, "content", ""))
                        if not text:
                            continue
                        answer_parts.append(text)
                        yield emit("token", {"text": text})

                answer = "".join(answer_parts)
                if answer.strip():
                    summary_started = time.perf_counter()
                    yield emit("summary_start")
                    try:
                        updated_summary = content_text(
                            llm.invoke(
                                build_summary_prompt(
                                    previous_summary=memory_summary,
                                    messages=raw_messages,
                                    answer=answer,
                                    language=language,
                                )
                            ).content
                        ).strip()
                        yield emit(
                            "memory_updated",
                            {
                                "summary": updated_summary,
                                "duration_ms": int((time.perf_counter() - summary_started) * 1000),
                            },
                        )
                    except Exception as exc:
                        yield emit(
                            "warning",
                            {"stage": "summary", "message": f"summary update failed: {exc}"},
                        )

                contexts = [
                    context
                    for group in context_groups
                    for context in group.get("contexts", [])
                    if isinstance(group.get("contexts"), list)
                ]
                rag_trace_context_count, rag_trace_queries, rag_trace_query_parts = rag_query_summaries_from_traces(traces)
                search_query_for_log = " | ".join(rag_trace_queries) or search_query
                search_query_parts_for_log = rag_trace_query_parts or [str(group.get("query", "")) for group in context_groups]
                duration_ms = int((time.perf_counter() - started) * 1000)
                # Build completion status the frontend can render as a clear badge.
                writes_count = sum(
                    1
                    for trace in traces
                    if str(trace.get("tool", "")) in {
                        "local_file_write",
                        "local_file_replace",
                        "local_file_copy_tree",
                        "local_file_create_dir",
                    }
                    and trace.get("result", {}).get("ok")
                )
                if not use_agent:
                    completion = "ok"
                elif final_answer_emitted:
                    completion = "ok"
                else:
                    completion = "budget_exhausted"
                done_payload = {
                    "answer": answer,
                    "provider_id": provider.get("provider_id"),
                    "model": model_override or provider.get("model"),
                    "context_count": len(contexts) + rag_trace_context_count,
                    "search_query": search_query_for_log,
                    "search_query_parts": search_query_parts_for_log,
                    "agent_trace_count": mcp_trace_count(traces),
                    "memory_summary": updated_summary,
                    "reasoning_content": "".join(reasoning_parts),
                    "duration_ms": duration_ms,
                    "completion": completion,
                    "writes_count": writes_count,
                    "written_files": sorted(written_targets)[:12],
                    "use_agent": use_agent,
                }
                append_ai_chat_log(
                    build_ai_chat_log_entry(
                        endpoint="/api/chat/stream",
                        status="ok",
                        payload=payload,
                        raw_messages=raw_messages,
                        last_user=last_user,
                        provider=provider,
                        model=str(model_override or provider.get("model") or ""),
                        language=language,
                        use_rag=use_rag,
                        use_agent=use_agent,
                        desc_top_k=desc_top_k,
                        context_n=context_n,
                        search_query=search_query_for_log,
                        search_query_parts=search_query_parts_for_log,
                        context_groups=context_groups,
                        traces=traces,
                        answer=answer,
                        reasoning_content="".join(reasoning_parts),
                        memory_summary_before=memory_summary,
                        memory_summary_after=updated_summary,
                        duration_ms=duration_ms,
                    )
                )
                yield emit("done", done_payload)
            except GeneratorExit:
                append_ai_chat_log(
                    build_ai_chat_log_entry(
                        endpoint="/api/chat/stream",
                        status="aborted",
                        payload=payload,
                        raw_messages=raw_messages,
                        last_user=last_user,
                        provider=provider,
                        model=str(model_override or provider.get("model") or ""),
                        language=language,
                        use_rag=use_rag,
                        use_agent=use_agent,
                        desc_top_k=desc_top_k,
                        context_n=context_n,
                        search_query=search_query,
                        search_query_parts=[str(group.get("query", "")) for group in context_groups],
                        context_groups=context_groups,
                        traces=traces,
                        answer="".join(answer_parts),
                        reasoning_content="".join(reasoning_parts),
                        memory_summary_before=memory_summary,
                        memory_summary_after=updated_summary,
                        duration_ms=int((time.perf_counter() - started) * 1000),
                    )
                )
                raise
            except Exception as exc:
                append_ai_chat_log(
                    build_ai_chat_log_entry(
                        endpoint="/api/chat/stream",
                        status="error",
                        payload=payload,
                        raw_messages=raw_messages,
                        last_user=last_user,
                        provider=provider,
                        model=str(model_override or provider.get("model") or ""),
                        language=language,
                        use_rag=use_rag,
                        use_agent=use_agent,
                        desc_top_k=desc_top_k,
                        context_n=context_n,
                        search_query=search_query,
                        search_query_parts=[str(group.get("query", "")) for group in context_groups],
                        context_groups=context_groups,
                        traces=traces,
                        answer="".join(answer_parts),
                        reasoning_content="".join(reasoning_parts),
                        memory_summary_before=memory_summary,
                        memory_summary_after=updated_summary,
                        duration_ms=int((time.perf_counter() - started) * 1000),
                        error=str(exc),
                    )
                )
                yield emit("error", {"message": str(exc), "completion": "error"})

        return Response(
            stream_with_context(generate()),
            mimetype="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/chat")
    def api_chat():
        payload = request.get_json(silent=True) or {}
        raw_messages = payload.get("messages") or []
        if not isinstance(raw_messages, list) or not raw_messages:
            return jsonify({"error": "messages is required"}), 400

        last_user = next(
            (
                str(message.get("content") or "").strip()
                for message in reversed(raw_messages)
                if str(message.get("role") or "").lower() == "user"
            ),
            "",
        )
        if not last_user:
            return jsonify({"error": "no user message found"}), 400

        provider_id = str(payload.get("provider_id") or "").strip() or None
        provider = settings.resolve_provider(provider_id)
        model_override = str(payload.get("model") or "").strip() or None
        temperature_override = maybe_float(payload.get("temperature"))
        try:
            llm = create_llm(provider, model_override=model_override, temperature_override=temperature_override)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        use_rag = bool(payload.get("use_rag", True))
        use_agent = bool(payload.get("use_agent", False))
        one_by_one = bool(payload.get("one_by_one", False))
        language = normalize_language(payload.get("language") or payload.get("lang"))
        desc_top_k = at_least_one_int(payload.get("desc_top_k"), default_desc_top_k)
        context_n = at_least_one_int(payload.get("context_n"), default_context_n)
        search_query = str(payload.get("search_query") or "").strip() or last_user
        memory_summary = str(payload.get("memory_summary") or "").strip()

        context_groups: list[dict[str, Any]] = []
        context_text = "RAG disabled." if language == "en" else "RAG 已禁用。"
        if use_rag:
            if rag.description_db_missing():
                return jsonify({"error": f"Milvus description DB not found: {desc_persist_dir}. Build it first with python backend\\scripts\\build.py --rebuild"}), 400
            context_groups = retrieve_code_context_groups(
                desc_db=rag.get_desc_db(),
                models_root=models_root,
                query=search_query,
                desc_top_k=desc_top_k,
                code_chars=code_chars,
                context_n=context_n,
            )
            context_text = build_grouped_prompt_context(context_groups)

        selected_file_context, selected_file_traces = read_selected_file_context(local_files, payload.get("selected_files"))
        prompt_messages = build_prompt_messages(
            raw_messages=raw_messages,
            memory_summary=memory_summary,
            search_query=search_query,
            context_text=context_text,
            selected_file_context=selected_file_context,
            language=language,
            one_by_one=one_by_one,
        )

        answer = ""
        reasoning_content = ""
        traces = selected_file_traces
        started = time.perf_counter()
        try:
            if use_agent:
                agent_tools = AgentMCP(
                    local_files=local_files,
                    rag_query_handler=rag.make_rag_query_handler(
                        desc_top_k_default=desc_top_k,
                        context_n_default=context_n,
                        enabled=use_rag,
                        domain_hint_text=last_user,
                        local_files=local_files,
                    ),
                )
                answer, traces = invoke_with_local_file_agent(
                    llm=llm,
                    messages=prompt_messages,
                    local_files=agent_tools,
                    max_steps=positive_int(payload.get("agent_max_steps"), 0),
                )
                traces = selected_file_traces + traces
            else:
                response = llm.invoke(prompt_messages)
                answer = str(response.content)
                reasoning_content = response_reasoning_content(response)
                traces = selected_file_traces
        except Exception as exc:
            append_ai_chat_log(
                build_ai_chat_log_entry(
                    endpoint="/api/chat",
                    status="error",
                    payload=payload,
                    raw_messages=raw_messages,
                    last_user=last_user,
                    provider=provider,
                    model=str(model_override or provider.get("model") or ""),
                    language=language,
                    use_rag=use_rag,
                    use_agent=use_agent,
                    desc_top_k=desc_top_k,
                    context_n=context_n,
                    search_query=search_query,
                    search_query_parts=[str(group.get("query", "")) for group in context_groups],
                    context_groups=context_groups,
                    traces=traces,
                    answer=answer,
                    reasoning_content=reasoning_content,
                    memory_summary_before=memory_summary,
                    memory_summary_after=memory_summary,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    error=str(exc),
                )
            )
            return jsonify({"error": f"LLM request failed: {exc}"}), 502

        contexts = [
            context
            for group in context_groups
            for context in group.get("contexts", [])
            if isinstance(group.get("contexts"), list)
        ]
        rag_trace_context_count, rag_trace_queries, rag_trace_query_parts = rag_query_summaries_from_traces(traces)
        search_query_for_log = " | ".join(rag_trace_queries) or search_query
        search_query_parts_for_log = rag_trace_query_parts or [str(group.get("query", "")) for group in context_groups]
        append_ai_chat_log(
            build_ai_chat_log_entry(
                endpoint="/api/chat",
                status="ok",
                payload=payload,
                raw_messages=raw_messages,
                last_user=last_user,
                provider=provider,
                model=str(model_override or provider.get("model") or ""),
                language=language,
                use_rag=use_rag,
                use_agent=use_agent,
                desc_top_k=desc_top_k,
                context_n=context_n,
                search_query=search_query_for_log,
                search_query_parts=search_query_parts_for_log,
                context_groups=context_groups,
                traces=traces,
                answer=answer,
                reasoning_content=reasoning_content if not use_agent else "",
                memory_summary_before=memory_summary,
                memory_summary_after=memory_summary,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        )
        return jsonify(
            {
                "answer": answer,
                "provider_id": provider.get("provider_id"),
                "model": model_override or provider.get("model"),
                "context_count": len(contexts) + rag_trace_context_count,
                "search_query": search_query_for_log,
                "search_query_parts": search_query_parts_for_log,
                "agent_traces": traces,
                "reasoning_content": reasoning_content if not use_agent else "",
                "debug_prompt_messages": serialize_messages(prompt_messages),
            }
        )

    return app


def _env_str(key: str, default: str) -> str:
    value = os.environ.get(key, "").strip()
    return value if value else default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STS2 RAG chat API and frontend server.")
    parser.add_argument(
        "--desc-persist-dir",
        type=Path,
        default=DEFAULT_DESC_PERSIST_DIR,
        help="Milvus Lite db path. Set MILVUS_URI to use a running Milvus server instead.",
    )
    parser.add_argument(
        "--desc-collection-name",
        type=str,
        default=_env_str("DESC_COLLECTION_NAME", DEFAULT_DESC_COLLECTION_NAME),
    )
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=_env_str("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
    parser.add_argument("--settings-path", type=Path, default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--desc-top-k", type=int, default=_env_int("DESC_TOP_K", 4))
    parser.add_argument("--code-chars", type=int, default=_env_int("CODE_CHARS", 2200))
    parser.add_argument("--default-context-n", type=int, default=_env_int("CONTEXT_N", 3))
    parser.add_argument("--host", type=str, default=_env_str("APP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=_env_int("APP_PORT", 7870))
    return parser.parse_args()


def main() -> None:
    load_env_file(DEFAULT_ENV_PATH)
    args = parse_args()
    if not FRONTEND_ROOT.exists():
        raise SystemExit(f"Frontend folder not found: {FRONTEND_ROOT}")
    if not args.models_root.exists():
        raise SystemExit(f"Models root not found: {args.models_root}")

    app = build_app(
        desc_persist_dir=args.desc_persist_dir,
        desc_collection_name=args.desc_collection_name,
        models_root=args.models_root,
        embedding_model=args.embedding_model,
        settings_path=args.settings_path,
        code_chars=args.code_chars,
        default_context_n=args.default_context_n,
        default_desc_top_k=args.desc_top_k,
    )
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
