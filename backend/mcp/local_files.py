from __future__ import annotations

import fnmatch
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BLOCKED_NAMES = {".env", "ai_settings.json"}
BLOCKED_PARTS = {"__pycache__", ".git", "vector_db"}
DEFAULT_IGNORES = {"*.pyc", "*.sqlite3", "*.dll", "*.exe", "*.bin"}
COPY_TREE_IGNORES = {"*.pyc", "*.sqlite3", "*.exe", "*.bin"}
TEXT_EXTENSIONS = {
    ".cs",
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
}
WRITE_EXTENSIONS = TEXT_EXTENSIONS | {".csproj", ".sln"}


READ_CACHE_MAX_ENTRIES = 64
READ_CACHE_MAX_BYTES = 16 * 1024 * 1024  # 16MB total cached content
READ_MANY_MAX_FILES = 16


@dataclass
class LocalFileMCP:
    """Small MCP-style local file tool with a strict project-root sandbox."""

    root: Path
    writable_root: Path | None = None

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        if self.writable_root is not None:
            self.writable_root = self.writable_root.resolve()
        # (path, mtime_ns) -> read_file payload dict; LRU via insertion order
        self._read_cache: "dict[Path, tuple[int, dict[str, Any]]]" = {}
        self._read_cache_bytes = 0

    def _cache_get(self, target: Path) -> dict[str, Any] | None:
        entry = self._read_cache.get(target)
        if entry is None:
            return None
        cached_mtime, payload = entry
        try:
            current_mtime = target.stat().st_mtime_ns
        except OSError:
            self._cache_drop(target)
            return None
        if current_mtime != cached_mtime:
            self._cache_drop(target)
            return None
        # bump LRU
        self._read_cache.pop(target, None)
        self._read_cache[target] = entry
        return payload

    def _cache_put(self, target: Path, payload: dict[str, Any]) -> None:
        size = len(payload.get("content", "") or "")
        if size > READ_CACHE_MAX_BYTES:
            return
        self._cache_drop(target)
        try:
            mtime = target.stat().st_mtime_ns
        except OSError:
            return
        self._read_cache[target] = (mtime, payload)
        self._read_cache_bytes += size
        while (
            self._read_cache
            and (
                len(self._read_cache) > READ_CACHE_MAX_ENTRIES
                or self._read_cache_bytes > READ_CACHE_MAX_BYTES
            )
        ):
            oldest, (_, oldest_payload) = next(iter(self._read_cache.items()))
            self._read_cache.pop(oldest, None)
            self._read_cache_bytes -= len(oldest_payload.get("content", "") or "")

    def _cache_drop(self, target: Path) -> None:
        entry = self._read_cache.pop(target, None)
        if entry is not None:
            self._read_cache_bytes -= len(entry[1].get("content", "") or "")

    def tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "local_file_list",
                    "description": "List files inside the local project workspace. Use this to inspect namespace folders before relying on an API.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Workspace-relative directory path."},
                            "pattern": {"type": "string", "description": "Optional glob pattern, for example *.py."},
                            "limit": {"type": "integer", "minimum": 0, "description": "Maximum items to return. Use 0 or omit for no limit."},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "local_file_search",
                    "description": "Search text in files inside the local project workspace. Use this for C# namespace, type, method, property, field, constructor, or using-directive verification.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Text to search for."},
                            "path": {"type": "string", "description": "Workspace-relative directory path."},
                            "limit": {"type": "integer", "minimum": 0, "description": "Maximum matches to return. Use 0 or omit for no limit."},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "local_file_read",
                    "description": "Read a text file inside the local project workspace. Use this to confirm whether a referenced API actually exists before generating code. If you need several files, prefer local_file_read_many in a single call.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Workspace-relative file path."},
                            "max_chars": {"type": "integer", "minimum": 0, "description": "Maximum characters to read. Use 0 or omit for the full file."},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "local_file_read_many",
                    "description": (
                        "Read up to "
                        f"{READ_MANY_MAX_FILES}"
                        " text files in a single tool call. ALWAYS prefer this over multiple local_file_read calls when "
                        "you already know the paths (for example, the four localization JSONs zhs/eng × cards/relics/powers/potions, "
                        "or several .cs files in the same namespace). Returns a list of per-file results; each item has the same shape as local_file_read."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": f"Up to {READ_MANY_MAX_FILES} workspace-relative file paths.",
                            },
                            "max_chars": {"type": "integer", "minimum": 0, "description": "Maximum characters per file. Use 0 or omit for the full file."},
                        },
                        "required": ["paths"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "local_file_write",
                    "description": (
                        "Create or overwrite a UTF-8 text file. Write access is restricted to mods/ only. "
                        "Use this for generated mod source files after deciding the exact target path."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Workspace-relative path under mods/, or absolute path inside the mods folder."},
                            "content": {"type": "string", "description": "Full file content to write. The path must include a concrete filename and a writable text extension such as .cs, .json, .csproj, .md, .txt, .yaml, .yml, .toml, .html, .css, .js, or .py."},
                            "create_dirs": {"type": "boolean", "description": "Create parent folders under mods/ if needed."},
                            "overwrite": {"type": "boolean", "description": "Allow overwriting an existing file. Defaults to true."},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "local_file_copy_tree",
                    "description": (
                        "Copy a directory tree inside mods/. Both source and destination must stay inside mods/. "
                        "Use this to duplicate the template project before editing generated mod files."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_path": {"type": "string", "description": "Workspace-relative source directory under mods/, for example mods/template."},
                            "target_path": {"type": "string", "description": "Workspace-relative target directory under mods/, for example mods/curseMod."},
                            "overwrite": {"type": "boolean", "description": "Allow overwriting files in an existing target directory."},
                            "max_files": {"type": "integer", "minimum": 0, "description": "Maximum files to copy. Use 0 or omit for no limit."},
                        },
                        "required": ["source_path", "target_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "local_file_create_dir",
                    "description": (
                        "Create a directory inside mods/. Use this only when the task explicitly needs an empty directory; "
                        "use local_file_write for files and local_file_copy_tree for project/template copies."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Workspace-relative directory path under mods/, for example mods/examplemod/src/Core/Models/Cards."},
                            "parents": {"type": "boolean", "description": "Create parent directories if needed. Defaults to true."},
                            "exist_ok": {"type": "boolean", "description": "Treat an existing directory as success. Defaults to true."},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "local_file_replace",
                    "description": (
                        "Replace text in an existing UTF-8 text file. Write access is restricted to mods/ only. "
                        "Use exact old_text so accidental broad edits are avoided."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Workspace-relative path under mods/, or absolute path inside the mods folder."},
                            "old_text": {"type": "string", "description": "Exact text to replace."},
                            "new_text": {"type": "string", "description": "Replacement text."},
                            "replace_all": {"type": "boolean", "description": "Replace all occurrences instead of one."},
                        },
                        "required": ["path", "old_text", "new_text"],
                    },
                },
            },
        ]

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "local_file_list":
            return self.list_files(
                path=str(args.get("path") or "."),
                pattern=str(args.get("pattern") or "*"),
                limit=int(args.get("limit") or 0),
            )
        if name == "local_file_read":
            return self.read_file(
                path=str(args.get("path") or ""),
                max_chars=int(args.get("max_chars") or 0),
            )
        if name == "local_file_read_many":
            raw_paths = args.get("paths") or []
            if not isinstance(raw_paths, list):
                return {"ok": False, "error": "paths must be an array of strings", "items": []}
            return self.read_files(
                paths=[str(item or "") for item in raw_paths],
                max_chars=int(args.get("max_chars") or 0),
            )
        if name == "local_file_search":
            return self.search_files(
                query=str(args.get("query") or ""),
                path=str(args.get("path") or "."),
                limit=int(args.get("limit") or 0),
            )
        if name == "local_file_write":
            return self.write_file(
                path=str(args.get("path") or ""),
                content=str(args.get("content") or ""),
                create_dirs=bool(args.get("create_dirs", True)),
                overwrite=bool(args.get("overwrite", True)),
            )
        if name == "local_file_replace":
            return self.replace_in_file(
                path=str(args.get("path") or ""),
                old_text=str(args.get("old_text") or ""),
                new_text=str(args.get("new_text") or ""),
                replace_all=bool(args.get("replace_all", False)),
            )
        if name == "local_file_copy_tree":
            return self.copy_tree(
                source_path=str(args.get("source_path") or ""),
                target_path=str(args.get("target_path") or ""),
                overwrite=bool(args.get("overwrite", False)),
                max_files=int(args.get("max_files") or 0),
            )
        if name == "local_file_create_dir":
            return self.create_dir(
                path=str(args.get("path") or ""),
                parents=bool(args.get("parents", True)),
                exist_ok=bool(args.get("exist_ok", True)),
            )
        return {"ok": False, "error": f"unknown MCP tool: {name}"}

    def list_files(self, path: str = ".", pattern: str = "*", limit: int = 0) -> dict[str, Any]:
        try:
            base = self._resolve(path)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "items": []}
        if not base.exists():
            suggestions = self._suggest_sibling_paths(base, limit=8)
            payload: dict[str, Any] = {"ok": False, "error": "path does not exist", "items": []}
            if suggestions:
                payload["suggestions"] = suggestions
                payload["hint"] = (
                    "The exact directory does not exist. The closest existing siblings are listed in 'suggestions'; "
                    "retry with one of those paths instead of the missing one."
                )
            return payload
        if not base.is_dir():
            return {"ok": False, "error": "path is not a directory", "items": []}

        items: list[dict[str, Any]] = []
        limit = max(0, limit)
        for child in sorted(base.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if self._blocked(child) or self._ignored(child):
                continue
            rel = child.relative_to(self.root).as_posix()
            if child.is_file() and not fnmatch.fnmatch(child.name, pattern):
                continue
            items.append(
                {
                    "name": child.name,
                    "path": rel,
                    "type": "directory" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
            if limit and len(items) >= limit:
                break
        return {"ok": True, "root": str(self.root), "path": base.relative_to(self.root).as_posix() or ".", "items": items}

    def read_file(self, path: str, max_chars: int = 0) -> dict[str, Any]:
        try:
            target = self._resolve(path)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        if self._blocked(target) or self._ignored(target):
            return {"ok": False, "error": "file is blocked by the local-file MCP safety policy"}
        if not target.exists() or not target.is_file():
            return {"ok": False, "error": "file does not exist"}
        max_chars = max(0, max_chars)
        cached = self._cache_get(target)
        if cached is not None:
            full_content = cached.get("content", "") or ""
            truncated = bool(max_chars and len(full_content) > max_chars)
            content = (full_content[:max_chars] + "\n\n... [truncated]") if truncated else full_content
            return {
                "ok": True,
                "path": cached["path"],
                "chars": len(content),
                "truncated": truncated,
                "content": content,
                "cache_hit": True,
            }
        try:
            content = target.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        full_payload = {
            "ok": True,
            "path": target.relative_to(self.root).as_posix(),
            "chars": len(content),
            "truncated": False,
            "content": content,
        }
        self._cache_put(target, full_payload)
        truncated = bool(max_chars and len(content) > max_chars)
        if truncated:
            return {
                "ok": True,
                "path": full_payload["path"],
                "chars": max_chars + len("\n\n... [truncated]"),
                "truncated": True,
                "content": content[:max_chars] + "\n\n... [truncated]",
            }
        return dict(full_payload)

    def read_files(self, paths: list[str], max_chars: int = 0) -> dict[str, Any]:
        if not paths:
            return {"ok": False, "error": "paths is required", "items": []}
        if len(paths) > READ_MANY_MAX_FILES:
            return {
                "ok": False,
                "error": f"too many paths; local_file_read_many accepts at most {READ_MANY_MAX_FILES}",
                "items": [],
            }
        items: list[dict[str, Any]] = []
        ok_count = 0
        for raw_path in paths:
            result = self.read_file(raw_path, max_chars=max_chars)
            result["request_path"] = str(raw_path or "")
            if result.get("ok"):
                ok_count += 1
            items.append(result)
        return {
            "ok": ok_count > 0,
            "count": len(items),
            "ok_count": ok_count,
            "items": items,
        }

    def search_files(self, query: str, path: str = ".", limit: int = 0) -> dict[str, Any]:
        needle = query.strip()
        if not needle:
            return {"ok": False, "error": "query is required", "items": []}
        try:
            base = self._resolve(path)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "items": []}
        if not base.exists():
            suggestions = self._suggest_sibling_paths(base, limit=8)
            payload: dict[str, Any] = {"ok": False, "error": "path does not exist", "items": []}
            if suggestions:
                payload["suggestions"] = suggestions
                payload["hint"] = (
                    "Search path does not exist; try one of the sibling paths listed in 'suggestions', "
                    "or omit the path arg to search the whole workspace."
                )
            return payload
        limit = max(0, limit)
        items = self._search_base(base, needle, limit)
        fallback_path = None
        if not items and (base != self.root):
            fallback_path = "."
            items = self._search_base(self.root, needle, limit)
        result = {
            "ok": True,
            "root": str(self.root),
            "path": base.relative_to(self.root).as_posix() or ".",
            "items": items,
        }
        if fallback_path is not None:
            result["fallback_path"] = fallback_path
        return result

    def _search_base(self, base: Path, needle: str, limit: int) -> list[dict[str, Any]]:
        files = [base] if base.is_file() else sorted(base.rglob("*"), key=self._search_order)
        exact_items: list[dict[str, Any]] = []
        lower_needle = needle.lower()
        for file_path in files:
            if limit and len(exact_items) >= limit:
                break
            if not file_path.is_file() or self._blocked(file_path) or self._ignored(file_path):
                continue
            if file_path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for line_no, line in enumerate(lines, start=1):
                if lower_needle in line.lower():
                    exact_items.append(
                        {
                            "path": file_path.relative_to(self.root).as_posix(),
                            "line": line_no,
                            "preview": line.strip()[:260],
                        }
                    )
                    break
        if exact_items:
            return exact_items

        terms = self._search_terms(needle)
        if not terms:
            return []
        fuzzy_items: list[dict[str, Any]] = []
        for file_path in files:
            if limit and len(fuzzy_items) >= limit:
                break
            if not file_path.is_file() or self._blocked(file_path) or self._ignored(file_path):
                continue
            if file_path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for line_no, line in enumerate(lines, start=1):
                lower_line = line.lower()
                matched_terms = [term for term in terms if term in lower_line]
                if not matched_terms:
                    continue
                fuzzy_items.append(
                    {
                        "path": file_path.relative_to(self.root).as_posix(),
                        "line": line_no,
                        "preview": line.strip()[:260],
                        "match_terms": matched_terms,
                    }
                )
                break
        return fuzzy_items

    @staticmethod
    def _search_terms(query: str) -> list[str]:
        raw_terms = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*|[\u4e00-\u9fff]{2,}", query)
        blocked = {
            "a",
            "an",
            "and",
            "class",
            "method",
            "public",
            "static",
            "task",
            "void",
            "with",
        }
        terms: list[str] = []
        seen = set()
        for term in raw_terms:
            lowered = term.lower().strip(".")
            if not lowered or lowered in blocked:
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            terms.append(lowered)
        return terms

    def _search_order(self, path: Path) -> tuple[int, int, str]:
        rel = path.relative_to(self.root).as_posix() if path.is_absolute() else path.as_posix()
        is_csharp = path.suffix.lower() == ".cs"
        is_game_code = rel.startswith(("data/libs/", "data/Models/", "mods/"))
        return (0 if is_game_code else 1, 0 if is_csharp else 1, rel.lower())

    def write_file(
        self,
        *,
        path: str,
        content: str,
        create_dirs: bool = True,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        try:
            target = self._resolve_writable(path)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        if not content:
            return {"ok": False, "error": "content is required for local_file_write"}
        if target.exists() and not target.is_file():
            return {"ok": False, "error": "target exists and is not a file"}
        if self._blocked(target):
            return {"ok": False, "error": "file is blocked by the local-file MCP write safety policy"}
        if target.suffix.lower() not in WRITE_EXTENSIONS:
            allowed = ", ".join(sorted(WRITE_EXTENSIONS))
            return {
                "ok": False,
                "error": (
                    "write path must include a concrete text filename with an allowed extension "
                    f"({allowed}); directories or extensionless paths are not writable"
                ),
                "path": target.relative_to(self.root).as_posix(),
            }
        if target.exists() and not overwrite:
            return {"ok": False, "error": "file exists and overwrite is false"}
        if not target.parent.exists():
            if not create_dirs:
                return {"ok": False, "error": "parent directory does not exist"}
            target.parent.mkdir(parents=True, exist_ok=True)
        previous_chars = target.stat().st_size if target.exists() else 0
        try:
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        self._cache_drop(target)
        return {
            "ok": True,
            "path": target.relative_to(self.root).as_posix(),
            "writable_root": str(self.writable_root),
            "created": previous_chars == 0 and target.exists(),
            "chars": len(content),
            "previous_bytes": previous_chars,
        }

    def replace_in_file(
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        if not old_text:
            return {"ok": False, "error": "old_text is required"}
        try:
            target = self._resolve_writable(path)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        if not target.exists() or not target.is_file():
            return {"ok": False, "error": "file does not exist"}
        if self._blocked(target):
            return {"ok": False, "error": "file is blocked by the local-file MCP write safety policy"}
        if target.suffix.lower() not in WRITE_EXTENSIONS:
            allowed = ", ".join(sorted(WRITE_EXTENSIONS))
            return {
                "ok": False,
                "error": (
                    "replace path must be a concrete text filename with an allowed extension "
                    f"({allowed}); directories or extensionless paths are not writable"
                ),
                "path": target.relative_to(self.root).as_posix(),
            }
        try:
            content = target.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        occurrences = content.count(old_text)
        if occurrences == 0:
            return {"ok": False, "error": "old_text not found", "path": target.relative_to(self.root).as_posix()}
        if occurrences > 1 and not replace_all:
            return {
                "ok": False,
                "error": "old_text appears multiple times; set replace_all=true or provide a more specific old_text",
                "occurrences": occurrences,
                "path": target.relative_to(self.root).as_posix(),
            }
        updated = content.replace(old_text, new_text) if replace_all else content.replace(old_text, new_text, 1)
        try:
            target.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        self._cache_drop(target)
        return {
            "ok": True,
            "path": target.relative_to(self.root).as_posix(),
            "writable_root": str(self.writable_root),
            "occurrences_replaced": occurrences if replace_all else 1,
            "chars": len(updated),
        }

    def copy_tree(
        self,
        *,
        source_path: str,
        target_path: str,
        overwrite: bool = False,
        max_files: int = 0,
    ) -> dict[str, Any]:
        try:
            source = self._resolve_writable(source_path)
            target = self._resolve_writable(target_path)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        if not source.exists() or not source.is_dir():
            return {"ok": False, "error": "source directory does not exist"}
        if self._blocked(source) or self._blocked(target):
            return {"ok": False, "error": "path is blocked by the local-file MCP write safety policy"}
        directories = [
            item
            for item in sorted(source.rglob("*"))
            if item.is_dir() and not self._blocked(item)
        ]
        files = [
            item
            for item in sorted(source.rglob("*"))
            if item.is_file() and not self._blocked(item) and not self._ignored_for_copy(item)
        ]
        max_files = max(0, max_files)
        if max_files:
            files = files[:max_files]
        created_dirs: list[str] = []
        if not target.exists():
            created_dirs.append(target.relative_to(self.root).as_posix())
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return {"ok": False, "error": str(exc), "path": target.relative_to(self.root).as_posix()}
        for directory in directories:
            rel = directory.relative_to(source)
            dest_dir = target / rel
            if not dest_dir.exists():
                created_dirs.append(dest_dir.relative_to(self.root).as_posix())
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                return {"ok": False, "error": str(exc), "path": dest_dir.relative_to(self.root).as_posix()}
        copied: list[str] = []
        skipped: list[str] = []
        for file_path in files:
            rel = file_path.relative_to(source)
            dest = target / rel
            if dest.exists() and not overwrite:
                skipped.append(dest.relative_to(self.root).as_posix())
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(file_path, dest)
            except OSError as exc:
                return {"ok": False, "error": str(exc), "path": dest.relative_to(self.root).as_posix()}
            self._cache_drop(dest)
            copied.append(dest.relative_to(self.root).as_posix())
        return {
            "ok": True,
            "source_path": source.relative_to(self.root).as_posix(),
            "target_path": target.relative_to(self.root).as_posix(),
            "writable_root": str(self.writable_root),
            "copied_count": len(copied),
            "skipped_count": len(skipped),
            "created_dir_count": len(created_dirs),
            "created_dirs": created_dirs,
            "copied": copied,
            "skipped": skipped,
        }

    def create_dir(
        self,
        *,
        path: str,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> dict[str, Any]:
        try:
            target = self._resolve_writable(path)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        if self._blocked(target):
            return {"ok": False, "error": "path is blocked by the local-file MCP write safety policy"}
        if target.exists() and not target.is_dir():
            return {"ok": False, "error": "target exists and is not a directory", "path": target.relative_to(self.root).as_posix()}
        existed = target.exists()
        try:
            target.mkdir(parents=parents, exist_ok=exist_ok)
        except OSError as exc:
            return {"ok": False, "error": str(exc), "path": target.relative_to(self.root).as_posix()}
        return {
            "ok": True,
            "path": target.relative_to(self.root).as_posix(),
            "writable_root": str(self.writable_root),
            "created": not existed,
            "existed": existed,
        }

    def _resolve(self, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ValueError("path is required")
        candidate = Path(raw_path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError("path escapes the project workspace")
        return resolved

    def _resolve_writable(self, raw_path: str) -> Path:
        if self.writable_root is None:
            raise ValueError("write tools are disabled")
        if not raw_path.strip():
            raise ValueError("path is required")
        candidate = Path(raw_path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.root / candidate).resolve()
        if resolved != self.writable_root and self.writable_root not in resolved.parents:
            raise ValueError(f"write path must stay inside {self.writable_root}")
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError("path escapes the project workspace")
        return resolved

    def _suggest_sibling_paths(self, missing: Path, *, limit: int = 8) -> list[str]:
        """When a requested path doesn't exist, suggest existing siblings whose
        names share a prefix with the missing basename.

        E.g. agent searches data/libs/sts2_decompiled/megacrit.sts2.core (no such
        dir); we suggest .../megacrit.sts2.core.commands, .../megacrit.sts2.core.entities.players,
        etc. — the namespaces it actually wanted.
        """
        try:
            parent = missing.parent
            target_name = missing.name
        except (OSError, ValueError):
            return []
        if not parent.exists() or not parent.is_dir():
            return []
        try:
            children = sorted(parent.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            return []
        target_lower = target_name.lower()
        prefixed: list[Path] = []
        contained: list[Path] = []
        for child in children:
            if self._blocked(child):
                continue
            name_lower = child.name.lower()
            if name_lower == target_lower:
                continue
            if name_lower.startswith(target_lower):
                prefixed.append(child)
            elif target_lower and target_lower in name_lower:
                contained.append(child)
        merged = prefixed + contained
        out: list[str] = []
        for child in merged[:max(1, limit)]:
            try:
                out.append(child.relative_to(self.root).as_posix())
            except ValueError:
                continue
        return out

    @staticmethod
    def _ignored(path: Path) -> bool:
        return any(fnmatch.fnmatch(path.name, pattern) for pattern in DEFAULT_IGNORES)

    @staticmethod
    def _ignored_for_copy(path: Path) -> bool:
        return any(fnmatch.fnmatch(path.name, pattern) for pattern in COPY_TREE_IGNORES)

    @staticmethod
    def _blocked(path: Path) -> bool:
        if path.name in BLOCKED_NAMES:
            return True
        return any(part in BLOCKED_PARTS for part in path.parts)


def main() -> None:
    """Tiny JSON-lines MCP runner for manual stdio experiments."""
    import sys

    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    mcp = LocalFileMCP(root)
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            result = mcp.call(str(payload.get("tool")), payload.get("arguments") or {})
        except Exception as exc:  # pragma: no cover - manual utility path
            result = {"ok": False, "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
