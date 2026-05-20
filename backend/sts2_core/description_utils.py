from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple

from sts2_core.paths import DEFAULT_LOCALIZATION_ROOT, PROJECT_ROOT


DESCRIPTION_FILES = {
    ("eng", "cards"): ("eng", "cards.json"),
    ("eng", "powers"): ("eng", "powers.json"),
    ("zhs", "cards"): ("zhs", "cards.json"),
    ("zhs", "powers"): ("zhs", "powers.json"),
}


def read_json(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items()}


def split_id_and_kind(key: str) -> Tuple[str, str]:
    if "." not in key:
        return key, "unknown"
    base_id, kind = key.rsplit(".", 1)
    return base_id, kind


def normalize_description_text(text: str) -> str:
    cleaned = text.replace("\\n", "\n")
    cleaned = re.sub(r"\{([^}:]+)(?::[^}]*)?\}", r" \1 ", cleaned)
    cleaned = re.sub(r"\[/?[^\]]+\]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def description_search_text(*parts: str) -> str:
    text = "\n".join(part for part in parts if part)
    normalized = normalize_description_text(text)
    return "\n".join([text, normalized, text, normalized]).strip()


def load_description_groups(desc_root: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    grouped: Dict[Tuple[str, str], Dict[str, str]] = {}
    for (lang, domain), parts in DESCRIPTION_FILES.items():
        data = read_json(desc_root.joinpath(*parts))
        for key, value in data.items():
            base_id, kind = split_id_and_kind(key)
            if kind not in {"title", "description", "smartDescription"}:
                continue
            grouped.setdefault((domain, base_id), {})[f"{lang}_{kind}"] = value
    return grouped


def has_description_files(desc_root: Path) -> bool:
    return any(desc_root.joinpath(*parts).exists() for parts in DESCRIPTION_FILES.values())


def discover_description_root(desc_root: Path, models_root: Path) -> Path:
    candidates = [
        desc_root,
        PROJECT_ROOT / "description",
        DEFAULT_LOCALIZATION_ROOT,
    ]
    for parent in [models_root, *models_root.parents]:
        candidates.extend(
            [
                parent / "localization",
                parent / "Slay the Spire 2" / "localization",
            ]
        )

    seen = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if has_description_files(candidate):
            return candidate
    return desc_root
