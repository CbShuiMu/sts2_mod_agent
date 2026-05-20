from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Iterable

from mcp.local_files import LocalFileMCP
from sts2_core.localization_vector_utils import (
    LOCALIZATION_VECTOR_DOMAINS,
    localization_collection_name,
    localization_persist_path,
)
from sts2_core.milvus import is_local_lite_uri, open_description_db, resolve_milvus_uri
from sts2_core.paths import DEFAULT_VECTOR_DB_ROOT
from sts2_core.retrieval import retrieve_code_context_groups

from services.utils import at_least_one_int


MAIN_RAG_DOMAINS = ("cards", "powers")
LOCALIZATION_RAG_DOMAINS = LOCALIZATION_VECTOR_DOMAINS
RAG_DOMAINS = MAIN_RAG_DOMAINS + LOCALIZATION_RAG_DOMAINS

RAG_DOMAIN_ALIASES = {
    "card": "cards",
    "cards": "cards",
    "cardmodel": "cards",
    "power": "powers",
    "powers": "powers",
    "powermodel": "powers",
    "relic": "relics",
    "relics": "relics",
    "relicmodel": "relics",
    "potion": "potions",
    "potions": "potions",
    "potionmodel": "potions",
    "orb": "orbs",
    "orbs": "orbs",
    "orbmodel": "orbs",
    "enchantment": "enchantments",
    "enchantments": "enchantments",
    "enchantmentmodel": "enchantments",
    "affliction": "afflictions",
    "afflictions": "afflictions",
    "afflictionmodel": "afflictions",
    "restsite": "rest_site_ui",
    "restsiteui": "rest_site_ui",
    "rest_site": "rest_site_ui",
    "rest_site_ui": "rest_site_ui",
    "rest": "rest_site_ui",
    "restui": "rest_site_ui",
    "event": "events",
    "events": "events",
    "eventmodel": "events",
}

RAG_DOMAIN_KEYWORDS = {
    "cards": ("card", "cards", "cardmodel", "cardtype", "targettype", "colorless", "curse", "draw", "exhaust", "卡牌", "牌", "无色", "诅咒", "抽到", "消耗"),
    "powers": ("power", "powers", "powermodel", "buff", "debuff", "能力", "状态"),
    "relics": ("relic", "relics", "relicmodel", "遗物"),
    "potions": ("potion", "potions", "potionmodel", "onuse", "药水"),
    "orbs": ("orb", "orbs", "orbmodel", "充能球", "球"),
    "enchantments": ("enchantment", "enchantments", "enchantmentmodel", "附魔"),
    "afflictions": ("affliction", "afflictions", "afflictionmodel", "负面牌", "苦痛"),
    "rest_site_ui": ("rest_site", "rest_site_ui", "restsite", "rest", "smith", "heal", "dig", "cook", "mend", "lift", "clone", "hatch", "篝火", "休息", "锻造", "挖掘", "烹饪", "愈合", "训练", "克隆", "孵化"),
    "events": ("event", "events", "eventmodel", "option", "page", "loss", "事件", "选项", "遭遇"),
}

RAG_DOMAIN_LABELS = {
    "cards": "Cards",
    "powers": "Powers",
    "relics": "Relics",
    "potions": "Potions",
    "orbs": "Orbs",
    "enchantments": "Enchantments",
    "afflictions": "Afflictions",
    "rest_site_ui": "Rest Site UI",
    "events": "Events",
}


def normalize_rag_domains(value: Any) -> list[str]:
    raw_values: list[str] = []
    if isinstance(value, str):
        raw_values = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[一-鿿]+", value)
    elif isinstance(value, list):
        raw_values = [str(item) for item in value]

    domains: list[str] = []
    for raw in raw_values:
        key = re.sub(r"[^A-Za-z0-9_]+", "", raw).strip().lower()
        domain = RAG_DOMAIN_ALIASES.get(key)
        if domain and domain not in domains:
            domains.append(domain)
    return domains


def infer_rag_domains(*texts: str) -> list[str]:
    combined = "\n".join(text for text in texts if text).lower()
    domains: list[str] = []
    for domain, keywords in RAG_DOMAIN_KEYWORDS.items():
        if any(keyword.lower() in combined for keyword in keywords):
            domains.append(domain)
    return domains


def rag_query_fanout_key(query: str) -> str:
    return re.sub(r"\s+", "", query).strip().lower()


def public_context_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public_groups: list[dict[str, Any]] = []
    for group in groups:
        contexts = []
        for context in group.get("contexts", []):
            if not isinstance(context, dict):
                continue
            contexts.append(
                {
                    "id": context.get("id", ""),
                    "domain": context.get("domain", ""),
                    "source_db": context.get("source_db", group.get("source_db", "descriptions")),
                    "entity_name": context.get("entity_name", ""),
                    "member_name": context.get("member_name", ""),
                    "desc_score": context.get("desc_score", ""),
                    "model_folder": context.get("model_folder", ""),
                    "file_path": context.get("file_path", ""),
                    "mcp_file_path": context.get("mcp_file_path", context.get("file_path", "")),
                    "absolute_file_path": context.get("absolute_file_path", ""),
                    "description_key_path": context.get("description_key_path", ""),
                    "description_key_resolved_path": context.get("description_key_resolved_path", ""),
                    "title": context.get("title", ""),
                    "description": context.get("description", ""),
                }
            )
        public_groups.append(
            {
                "query": group.get("query", ""),
                "source_db": group.get("source_db", "descriptions"),
                "contexts": contexts,
            }
        )
    return public_groups


def filter_groups_by_domains(groups: list[dict[str, Any]], domains: Iterable[str]) -> list[dict[str, Any]]:
    allowed = {str(domain or "").strip().lower() for domain in domains}
    if not allowed:
        return groups

    filtered_groups: list[dict[str, Any]] = []
    for group in groups:
        group_source = str(group.get("source_db") or "").strip().lower()
        kept_contexts: list[dict[str, Any]] = []
        for context in group.get("contexts", []):
            if not isinstance(context, dict):
                continue
            context_source = str(context.get("source_db") or "").strip().lower()
            context_domain = str(context.get("domain") or "").strip().lower()
            if context_source in allowed or context_domain in allowed or group_source in allowed:
                kept_contexts.append(context)
        if kept_contexts:
            filtered_group = dict(group)
            filtered_group["contexts"] = kept_contexts
            filtered_groups.append(filtered_group)
    return filtered_groups


class RagService:
    """Encapsulates description / localization Milvus DBs and the rag_query handler factory."""

    def __init__(
        self,
        *,
        desc_persist_dir: Path,
        desc_collection_name: str,
        embedding_model: str,
        models_root: Path,
        code_chars: int,
    ) -> None:
        self.desc_persist_dir = desc_persist_dir
        self.desc_collection_name = desc_collection_name
        self.embedding_model = embedding_model
        self.models_root = models_root
        self.code_chars = code_chars
        self.milvus_uri = resolve_milvus_uri(desc_persist_dir)
        self._desc_db_cache: dict[str, Any] = {}
        self._localization_db_cache: dict[str, Any] = {}

    def get_desc_db(self) -> Any:
        if "db" not in self._desc_db_cache:
            self._desc_db_cache["db"] = open_description_db(
                persist_dir=self.desc_persist_dir,
                collection_name=self.desc_collection_name,
                model_name=self.embedding_model,
            )
        return self._desc_db_cache["db"]

    def get_localization_dbs(self, domains: Iterable[str] | None = None) -> list[tuple[str, Any]]:
        if domains is None:
            requested = {domain.strip().lower() for domain in LOCALIZATION_VECTOR_DOMAINS}
        else:
            requested = {domain.strip().lower() for domain in domains}
        dbs: list[tuple[str, Any]] = []
        for domain in LOCALIZATION_VECTOR_DOMAINS:
            if domain not in requested:
                continue
            persist_dir = localization_persist_path(DEFAULT_VECTOR_DB_ROOT, domain)
            if is_local_lite_uri(resolve_milvus_uri(persist_dir)) and not persist_dir.exists():
                continue
            if domain not in self._localization_db_cache:
                self._localization_db_cache[domain] = open_description_db(
                    persist_dir=persist_dir,
                    collection_name=localization_collection_name(domain),
                    model_name=self.embedding_model,
                )
            dbs.append((domain, self._localization_db_cache[domain]))
        return dbs

    def description_db_missing(self) -> bool:
        return is_local_lite_uri(self.milvus_uri) and not self.desc_persist_dir.exists()

    def make_rag_query_handler(
        self,
        *,
        desc_top_k_default: int,
        context_n_default: int,
        enabled: bool,
        domain_hint_text: str = "",
        local_files: Any | None = None,
        auto_read_top_n: int = 3,
        auto_read_max_chars: int = 8000,
    ):
        routed_query_domains: dict[str, set[str]] = {}
        auto_read_seen: set[str] = set()

        def rag_query(args: dict[str, Any]) -> dict[str, Any]:
            query = str(args.get("query") or "").strip()
            if not enabled:
                return {"ok": False, "error": "RAG is disabled for this request"}
            if not query:
                return {"ok": False, "error": "query is required"}
            if self.description_db_missing():
                return {
                    "ok": False,
                    "error": (
                        f"Milvus description DB not found: {self.desc_persist_dir}. "
                        "Build it first with python backend\\scripts\\build.py --rebuild"
                    ),
                }
            desc_top_k = at_least_one_int(args.get("desc_top_k"), desc_top_k_default)
            context_n = at_least_one_int(args.get("context_n"), context_n_default)
            raw_domains = normalize_rag_domains(args.get("domains"))
            if not raw_domains:
                raw_domains = infer_rag_domains(query, domain_hint_text)[:1]
            if not raw_domains:
                raw_domains = [RAG_DOMAINS[0]]
            requested_domains = raw_domains
            query_fanout_key = rag_query_fanout_key(query)
            if query_fanout_key:
                prior_domains = routed_query_domains.get(query_fanout_key)
                current_set = set(requested_domains)
                if prior_domains and not (current_set & prior_domains):
                    return {
                        "ok": False,
                        "skipped": True,
                        "error": (
                            "rag_query fan-out blocked: this exact query segment was already "
                            f"routed to domain(s) {sorted(prior_domains)}, so it must not be re-queried "
                            f"against disjoint domain(s) {sorted(current_set)}. Pass all needed domains "
                            "in the original call."
                        ),
                        "query": query,
                        "requested_domains": requested_domains,
                        "prior_domains": sorted(prior_domains),
                    }
                routed_query_domains.setdefault(query_fanout_key, current_set)
            main_domains = [domain for domain in requested_domains if domain in MAIN_RAG_DOMAINS]
            localization_domains = [domain for domain in requested_domains if domain in LOCALIZATION_RAG_DOMAINS]
            started = time.perf_counter()
            groups: list[dict[str, Any]] = []
            searched_domains: list[str] = []
            if main_domains:
                main_groups = retrieve_code_context_groups(
                    desc_db=self.get_desc_db(),
                    models_root=self.models_root,
                    query=query,
                    desc_top_k=desc_top_k,
                    code_chars=self.code_chars,
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
            for domain, db in self.get_localization_dbs(localization_domains):
                domain_groups = retrieve_code_context_groups(
                    desc_db=db,
                    models_root=self.models_root,
                    query=query,
                    desc_top_k=desc_top_k,
                    code_chars=self.code_chars,
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
            contexts = [
                context
                for group in groups
                for context in group.get("contexts", [])
                if isinstance(group.get("contexts"), list)
            ]
            from services.prompts import build_grouped_prompt_context  # local import to avoid cycle
            result: dict[str, Any] = {
                "ok": True,
                "query": query,
                "query_parts": [str(group.get("query", "")) for group in groups],
                "desc_top_k": desc_top_k,
                "context_n": context_n,
                "context_count": len(contexts),
                "requested_domains": requested_domains,
                "searched_domains": searched_domains,
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "public_groups": public_context_groups(groups),
                "context_text": build_grouped_prompt_context(groups),
            }
            if local_files is not None and auto_read_top_n > 0:
                top_paths: list[str] = []
                for group in groups:
                    for ctx in group.get("contexts", []):
                        if not isinstance(ctx, dict):
                            continue
                        candidate = str(
                            ctx.get("mcp_file_path") or ctx.get("file_path") or ""
                        ).strip()
                        if not candidate or candidate in top_paths:
                            continue
                        top_paths.append(candidate)
                        if len(top_paths) >= auto_read_top_n:
                            break
                    if len(top_paths) >= auto_read_top_n:
                        break

                top_files: list[dict[str, Any]] = []
                for path in top_paths:
                    if path in auto_read_seen:
                        top_files.append({
                            "path": path,
                            "ok": True,
                            "cached_in_session": True,
                            "note": "Already auto-loaded in an earlier rag_query this turn; reuse that content instead of re-reading.",
                        })
                        continue
                    read_result = local_files.read_file(path, max_chars=auto_read_max_chars)
                    auto_read_seen.add(path)
                    entry: dict[str, Any] = {"path": path, "ok": bool(read_result.get("ok"))}
                    if read_result.get("ok"):
                        entry["content"] = read_result.get("content", "")
                        entry["chars"] = read_result.get("chars", len(entry["content"]))
                        if read_result.get("truncated"):
                            entry["truncated"] = True
                    else:
                        entry["error"] = read_result.get("error", "read failed")
                    top_files.append(entry)

                result["top_files"] = top_files
                result["top_files_note"] = (
                    f"The top {len(top_files)} result file(s) have been auto-loaded above. "
                    "Treat them as already read — DO NOT call local_file_read on these paths again. "
                    "Only call local_file_read[_many] for additional files beyond the top-3 if needed."
                )

            return result

        return rag_query


class AgentMCP:
    def __init__(
        self,
        *,
        local_files: LocalFileMCP,
        rag_query_handler: Any | None = None,
    ) -> None:
        self.local_files = local_files
        self.rag_query_handler = rag_query_handler

    def tool_specs(self) -> list[dict[str, Any]]:
        specs = list(self.local_files.tool_specs())
        if self.rag_query_handler is not None:
            specs.insert(
                0,
                {
                    "type": "function",
                    "function": {
                        "name": "rag_query",
                        "description": (
                            "MUST SPLIT BEFORE CALLING — `query` MUST NOT contain any of these splitting punctuation marks: "
                            "Chinese 。，；、！？ / ASCII . , ; ! ? / newline \\n \\r. If the source description has any of "
                            "these marks, split it on EVERY occurrence FIRST, then issue ONE rag_query per resulting segment. "
                            "Sending a query that still contains these punctuation marks is the #1 forbidden mistake. "
                            "If the description has no such punctuation, send it as one query unchanged. "
                            "ONE SEGMENT → ONE DOMAIN: `domains` is ALWAYS a single-element list, e.g. ['cards']. Never query "
                            "the same segment against multiple domains. If unsure which domain fits, commit to your best single guess. "
                            "EXAMPLE — description '拾起时，升级你的所有打击和防御。将1张永恒的凡庸加入你的牌组' must produce EXACTLY THREE calls: "
                            "rag_query('拾起时', ['relics']); rag_query('升级你的所有打击和防御', ['cards']); rag_query('将1张永恒的凡庸加入你的牌组', ['cards']). "
                            "WRONG examples (never do these): "
                            "(a) rag_query('拾起时，升级你的所有打击和防御。将1张永恒的凡庸加入你的牌组', ['relics']) — query still contains 。， punctuation, FORBIDDEN. "
                            "(b) rag_query('升级你的所有打击和防御', ['cards']); rag_query('升级你的所有打击和防御', ['relics']) — same segment, multiple domains, FORBIDDEN. "
                            "\n\n"
                            "Beyond the split rule above, this tool queries the STS2 vector database for reference code and game text. "
                            "ALWAYS use this BEFORE broad local_file_search on data/ or data/libs/sts2_decompiled — that tree is huge and slow. "
                            "Available domains: cards, powers, relics, potions, orbs, enchantments, afflictions, rest_site_ui, events. "
                            "If a segment itself names a category (card, relic, potion, power, etc.), let that hint pick the domain. "
                            "The user's wording may be inaccurate — rewrite the segment in Slay the Spire / STS2 game-text style before querying, "
                            "but keep the query in the SAME language as the user's question (if the user wrote Chinese, query MUST be Chinese; never translate). "
                            "Returned contexts already include MCPReadPath/file_path; feed those exact paths into local_file_read_many "
                            "(preferred when you have several paths) or local_file_read for precise source inspection."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": (
                                        "MUST be a single segment with NO splitting punctuation inside — no Chinese 。，；、！？ , "
                                        "no ASCII . , ; ! ? , no newline. If the source description has any of these, you must split "
                                        "FIRST and call rag_query once per segment. Same language as the user's question; never translate. "
                                        "Rewrite in Slay the Spire / STS2 game-text style if needed."
                                    ),
                                },
                                "desc_top_k": {"type": "integer", "minimum": 1},
                                "context_n": {"type": "integer", "minimum": 1},
                                "domains": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": list(RAG_DOMAINS),
                                    },
                                    "minItems": 1,
                                    "maxItems": len(RAG_DOMAINS),
                                    "description": "Domain(s) to search, e.g. ['cards']. Default is a single-element list (one domain per segment). Pass multiple domains ONLY when the user message explicitly tagged the segment with multiple domains; in that case put all of them in this ONE call rather than fanning out across calls.",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                },
            )
        return specs

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "rag_query":
            if self.rag_query_handler is None:
                return {"ok": False, "error": "rag_query is disabled for this request"}
            return self.rag_query_handler(args)
        return self.local_files.call(name, args)
