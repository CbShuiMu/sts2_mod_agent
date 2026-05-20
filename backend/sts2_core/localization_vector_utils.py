from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping

from langchain_core.documents import Document

from sts2_core.description_utils import (
    description_search_text,
    normalize_description_text,
    read_json,
    split_id_and_kind,
)
from sts2_core.cs_utils import domain_folder, find_model_file_from_description_key
from sts2_core.paths import DEFAULT_MODELS_ROOT, PROJECT_ROOT


LOCALIZATION_VECTOR_DOMAINS = (
    "relics",
    "potions",
    "orbs",
    "enchantments",
    "afflictions",
    "rest_site_ui",
    "events",
)
LOCALIZATION_DOMAIN_ALIASES = {
    "relic": "relics",
    "relics": "relics",
    "potion": "potions",
    "potions": "potions",
    "orb": "orbs",
    "orbs": "orbs",
    "enchantment": "enchantments",
    "enchantments": "enchantments",
    "affliction": "afflictions",
    "afflictions": "afflictions",
    "rest_site": "rest_site_ui",
    "rest_site_ui": "rest_site_ui",
    "restsite": "rest_site_ui",
    "restsiteui": "rest_site_ui",
    "event": "events",
    "events": "events",
}
SEARCH_FIELD_ORDER = (
    "description",
    "smartDescription",
    "title",
)
TITLE_FIELD_ALIASES = {
    "rest_site_ui": "name",
}


def normalize_localization_domain(domain: str) -> str:
    key = domain.strip().lower().removesuffix(".json")
    normalized = LOCALIZATION_DOMAIN_ALIASES.get(key)
    if not normalized:
        valid = ", ".join(LOCALIZATION_VECTOR_DOMAINS)
        raise ValueError(f"Unknown localization vector domain '{domain}'. Expected one of: {valid}")
    return normalized


def localization_filename(domain: str) -> str:
    return f"{normalize_localization_domain(domain)}.json"


def localization_collection_name(domain: str) -> str:
    return f"sts2_{normalize_localization_domain(domain)}"


def localization_persist_path(vector_root: Path, domain: str) -> Path:
    return vector_root / f"{normalize_localization_domain(domain)}_milvus.db"


def _source_field_to_canonical_field(domain: str, field: str) -> str | None:
    normalized_domain = normalize_localization_domain(domain)
    if field in {"description", "smartDescription", "title"}:
        return field
    if field == TITLE_FIELD_ALIASES.get(normalized_domain):
        return "title"
    return None


def _workspace_relative(path: Path | None) -> str:
    if path is None:
        return ""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def localization_key_candidate_path(models_root: Path, domain: str, base_id: str) -> str:
    normalized_domain = normalize_localization_domain(domain)
    folder = normalized_domain
    filename = f"{base_id.replace('_', '').lower()}.cs"
    candidate = models_root / folder / filename
    try:
        return candidate.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(candidate)


def load_localization_groups(
    localization_root: Path,
    domain: str,
    languages: Iterable[str] = ("eng", "zhs"),
) -> Dict[str, Dict[str, str]]:
    normalized_domain = normalize_localization_domain(domain)
    filename = localization_filename(normalized_domain)
    grouped: Dict[str, Dict[str, str]] = {}
    for lang in languages:
        data = read_json(localization_root / lang / filename)
        for key, value in data.items():
            base_id, field = split_id_and_kind(key)
            canonical_field = _source_field_to_canonical_field(normalized_domain, field)
            if canonical_field is None:
                continue
            grouped.setdefault(base_id, {})[f"{lang}_{canonical_field}"] = value
    return grouped


def _ordered_field_values(fields: Mapping[str, str], lang: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen = set()
    for field in SEARCH_FIELD_ORDER:
        key = f"{lang}_{field}"
        value = fields.get(key, "")
        if value:
            pairs.append((field, value))
            seen.add(key)
    for key, value in sorted(fields.items()):
        if key in seen or not key.startswith(f"{lang}_") or not value:
            continue
        pairs.append((key.removeprefix(f"{lang}_"), value))
    return pairs


def _format_language_block(lang: str, fields: Mapping[str, str]) -> str:
    values = _ordered_field_values(fields, lang)
    if not values:
        return ""
    lines = [f"{field}: {value}" for field, value in values]
    return f"{lang.upper()}:\n" + "\n".join(lines)


def _search_parts(fields: Mapping[str, str]) -> list[str]:
    parts: list[str] = []
    for lang in ("zhs", "eng"):
        parts.extend(value for _, value in _ordered_field_values(fields, lang))
    return parts


def build_localization_document(
    domain: str,
    base_id: str,
    fields: Mapping[str, str],
    models_root: Path = DEFAULT_MODELS_ROOT,
) -> Document:
    normalized_domain = normalize_localization_domain(domain)
    source_file = find_model_file_from_description_key(models_root, normalized_domain, base_id)
    source_file_path = _workspace_relative(source_file)
    source_absolute_path = str(source_file.resolve()) if source_file else ""
    description_key_path = localization_key_candidate_path(models_root, normalized_domain, base_id)
    model_folder = domain_folder(normalized_domain)
    zhs_title = fields.get("zhs_title", "")
    eng_title = fields.get("eng_title", "")
    zh_block = _format_language_block("zhs", fields)
    en_block = _format_language_block("eng", fields)
    raw_search = "\n".join(_search_parts(fields))
    search_text = description_search_text(raw_search)

    page = (
        "LocalizationSearchText:\n"
        f"{search_text}\n\n"
        "LocalizationFields:\n"
        f"{zh_block}\n\n"
        f"{en_block}\n\n"
        "LowPriorityIdentity:\n"
        f"ZH_Title: {zhs_title}\n"
        f"EN_Title: {eng_title}\n"
        f"Domain: {normalized_domain}\n"
        f"ID: {base_id}\n"
        f"ModelFolder: {model_folder}\n"
        f"DescriptionKeyPath: {description_key_path}\n"
        f"SourceFilePath: {source_file_path}\n"
    )
    field_names = sorted({key.split("_", 1)[1] for key, value in fields.items() if "_" in key and value})

    return Document(
        page_content=page.strip(),
        metadata={
            "domain": normalized_domain,
            "id": base_id,
            "id_compact": base_id.replace("_", ""),
            "model_folder": model_folder,
            "description_key_path": description_key_path,
            "description_key_resolved_path": source_file_path,
            "source_file_path": source_file_path,
            "source_absolute_path": source_absolute_path,
            "zh_title": zhs_title or "unknown",
            "en_title": eng_title or "unknown",
            "zh_description": fields.get("zhs_description", "") or fields.get("zhs_smartDescription", ""),
            "en_description": fields.get("eng_description", "") or fields.get("eng_smartDescription", ""),
            "field_names": "|".join(field_names),
            "normalized_text": normalize_description_text(raw_search),
            "language": "bilingual",
            "source": localization_filename(normalized_domain),
        },
    )


def build_localization_documents(
    localization_root: Path,
    domain: str,
    models_root: Path = DEFAULT_MODELS_ROOT,
) -> list[Document]:
    groups = load_localization_groups(localization_root, domain)
    normalized_domain = normalize_localization_domain(domain)
    return [
        build_localization_document(normalized_domain, base_id, fields, models_root=models_root)
        for base_id, fields in sorted(groups.items())
    ]
