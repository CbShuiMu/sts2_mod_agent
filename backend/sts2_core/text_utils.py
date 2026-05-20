from __future__ import annotations

import re
from typing import List


def normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def id_to_code_name(base_id: str) -> str:
    return "".join(part[:1].upper() + part[1:].lower() for part in base_id.split("_") if part)


def pascal_to_upper_snake(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    if not cleaned:
        return ""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", cleaned)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.upper()


def make_code_candidates(base_id: str) -> str:
    compact = base_id.replace("_", "")
    pascal = id_to_code_name(base_id)
    return "|".join([base_id, compact, pascal])


def parse_candidates(raw: str) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split("|") if part.strip()]


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n... [truncated]"


def query_terms(query: str) -> List[str]:
    terms = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", query.lower())
    quoted = re.findall(r"""["'“”‘’]([^"'“”‘’]+)["'“”‘’]""", query.lower())
    terms.extend(term.strip().lower() for term in quoted if term.strip())
    return sorted(set(terms), key=len, reverse=True)


def split_search_queries(query: str) -> List[str]:
    stripped = query.strip()
    return [stripped] if stripped else []
