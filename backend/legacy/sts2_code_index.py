from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sts2_core.cs_utils import (
    extract_card_constructor_metadata,
    find_matching,
    split_top_level_args,
)
from sts2_core.description_utils import read_json, split_id_and_kind
from sts2_core.text_utils import pascal_to_upper_snake

try:
    from langchain_core.documents import Document
except ImportError:
    @dataclass
    class Document:  # type: ignore[no-redef]
        page_content: str
        metadata: Dict[str, object]


DESCRIPTION_KINDS = {"title", "description", "smartDescription"}
MODEL_DIRS = {"Cards": "cards", "Powers": "powers"}


def load_description_records(desc_root: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    datasets = {
        ("eng", "cards"): read_json(desc_root / "eng" / "cards.json"),
        ("eng", "powers"): read_json(desc_root / "eng" / "powers.json"),
        ("zhs", "cards"): read_json(desc_root / "zhs" / "cards.json"),
        ("zhs", "powers"): read_json(desc_root / "zhs" / "powers.json"),
    }
    grouped: Dict[Tuple[str, str], Dict[str, str]] = {}
    for (lang, domain), data in datasets.items():
        for key, value in data.items():
            base_id, kind = split_id_and_kind(key)
            if kind not in DESCRIPTION_KINDS:
                continue
            grouped.setdefault((domain, base_id), {})[f"{lang}_{kind}"] = value
    return grouped


def collect_model_files(models_root: Path, include_mocks: bool = False) -> List[Path]:
    files: List[Path] = []
    for dirname in MODEL_DIRS:
        root = models_root / dirname
        if not root.exists():
            continue
        for path in root.rglob("*.cs"):
            if not include_mocks and "Mocks" in path.parts:
                continue
            files.append(path)
    return sorted(files)


def domain_for_file(path: Path) -> str:
    for part in path.parts:
        if part in MODEL_DIRS:
            return MODEL_DIRS[part]
    return "unknown"


def extract_first(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    if not match:
        return default
    return match.group(1).strip()


def extract_canonical_vars(content: str) -> List[str]:
    block = ""
    match = re.search(r"\bCanonicalVars\b\s*=>", content)
    if match:
        semi = content.find(";", match.end())
        if semi != -1:
            block = content[match.end() : semi]
    if not block:
        return []
    names: List[str] = []
    for var_type, arg_text in re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*(?:Var))\s*\((.*?)\)",
        block,
        flags=re.DOTALL,
    ):
        args = split_top_level_args(arg_text)
        key = args[0].strip().strip('"') if args else var_type
        if not key or re.match(r"^[0-9.]+m?$", key):
            key = var_type
        names.append(f"{var_type}:{key}")
    return names


def extract_references(content: str) -> List[str]:
    patterns = [
        r"\bPowerCmd\.Apply<([A-Za-z_][A-Za-z0-9_]*)>",
        r"\bHoverTipFactory\.FromPower<([A-Za-z_][A-Za-z0-9_]*)>",
        r"\bHoverTipFactory\.FromCard<([A-Za-z_][A-Za-z0-9_]*)>",
        r"\bModelDb\.Card<([A-Za-z_][A-Za-z0-9_]*)>",
        r"\bGetPower<([A-Za-z_][A-Za-z0-9_]*)>",
        r"\bHasPower<([A-Za-z_][A-Za-z0-9_]*)>",
        r"\bnew\s+([A-Za-z_][A-Za-z0-9_]*(?:Power|Card))\s*\(",
    ]
    refs = set()
    for pattern in patterns:
        refs.update(re.findall(pattern, content))
    return sorted(refs)


def extract_effect_terms(content: str) -> List[str]:
    lower = content.lower()
    rules = [
        ("damage", ["damagecmd", "attack(", "damage"]),
        ("block", ["gainblock", "block"]),
        ("draw", ["draw(", "drawcard", "draw"]),
        ("discard", ["discard"]),
        ("energy", ["gainenergy", "energy"]),
        ("heal", ["heal", "restorehp"]),
        ("hp_loss", ["losehp", "hploss"]),
        ("apply_power", ["powercmd.apply"]),
        ("exhaust", ["exhaust"]),
        ("upgrade", ["upgrade"]),
        ("generate_card", ["cardfactory", "addcard", "cardpilecmd.add"]),
    ]
    return [label for label, needles in rules if any(n in lower for n in needles)]


def member_header(
    entity_name: str,
    domain: str,
    entity_id: str,
    title: str,
    description: str,
) -> str:
    lines = [
        f"Entity: {entity_name}",
        f"Domain: {domain}",
        f"ID: {entity_id}",
    ]
    if title:
        lines.append(f"Title: {title}")
    if description:
        lines.append(f"Description: {description}")
    return "\n".join(lines)


def find_member_chunks(content: str, member_kind: str) -> List[Tuple[str, str]]:
    chunks: List[Tuple[str, str]] = []
    if member_kind == "method":
        pattern = re.compile(
            r"(?m)^\s*(?:public|protected|private|internal)\s+"
            r"(?:(?:override|virtual|static|async|sealed|new)\s+)*"
            r"[A-Za-z0-9_<>,\?\[\]\.\s]+\s+"
            r"([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*(?:=>|{)"
        )
    else:
        pattern = re.compile(
            r"(?m)^\s*(?:public|protected|private|internal)\s+"
            r"(?:(?:override|virtual|static|sealed|new)\s+)*"
            r"[A-Za-z0-9_<>,\?\[\]\.\s]+\s+"
            r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=>|{)"
        )
    for match in pattern.finditer(content):
        name = match.group(1)
        after_name = content[match.end() - 1]
        if member_kind == "property":
            name_end = match.start(1) + len(name)
            if content[name_end:].lstrip().startswith("("):
                continue
        start = match.start()
        body_start = match.end() - 1
        if after_name == "{" or content[body_start] == "{":
            end = find_matching(content, body_start, "{", "}") + 1
        else:
            end = content.find(";", body_start) + 1
            if end <= 0:
                continue
        chunks.append((name, content[start:end].strip()))
    return chunks


def build_entity_summary(
    content: str,
    entity_name: str,
    domain: str,
    entity_id: str,
    descriptions: Dict[str, str],
    metadata: Dict[str, str],
    method_names: Iterable[str],
    property_names: Iterable[str],
    references: Iterable[str],
    canonical_vars: Iterable[str],
) -> str:
    lines = [
        f"Entity: {entity_name}",
        f"Domain: {domain}",
        f"ID: {entity_id}",
        f"EN_Title: {descriptions.get('eng_title', '')}",
        f"ZH_Title: {descriptions.get('zhs_title', '')}",
        f"EN_Description: {descriptions.get('eng_description', '')}",
        f"EN_SmartDescription: {descriptions.get('eng_smartDescription', '')}",
        f"ZH_Description: {descriptions.get('zhs_description', '')}",
        f"ZH_SmartDescription: {descriptions.get('zhs_smartDescription', '')}",
    ]
    for key in ["energy_cost", "card_type", "rarity", "target_type"]:
        if metadata.get(key):
            lines.append(f"{key}: {metadata[key]}")
    lines.extend(
        [
            f"CanonicalVars: {' | '.join(canonical_vars) or 'none'}",
            f"Effects: {' | '.join(extract_effect_terms(content)) or 'none'}",
            f"References: {' | '.join(references) or 'none'}",
            f"Methods: {' | '.join(method_names) or 'none'}",
            f"Properties: {' | '.join(property_names) or 'none'}",
        ]
    )
    return "\n".join(lines)


def build_code_documents(
    models_root: Path,
    desc_root: Path,
    include_mocks: bool = False,
) -> List[Document]:
    descriptions = load_description_records(desc_root)
    docs: List[Document] = []
    for path in collect_model_files(models_root, include_mocks=include_mocks):
        content = path.read_text(encoding="utf-8", errors="ignore")
        entity_name = extract_first(
            r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b", content, path.stem
        )
        domain = domain_for_file(path)
        entity_id = pascal_to_upper_snake(entity_name)
        desc = descriptions.get((domain, entity_id), {})
        refs = extract_references(content)
        ref_ids = [pascal_to_upper_snake(ref) for ref in refs]
        canonical_vars = extract_canonical_vars(content)
        card_meta = (
            extract_card_constructor_metadata(entity_name, content)
            if domain == "cards"
            else {}
        )
        methods = find_member_chunks(content, "method")
        properties = find_member_chunks(content, "property")
        rel_path = str(path.relative_to(models_root))
        base_metadata = {
            "source": str(path),
            "relative_path": rel_path,
            "file_name": path.name,
            "entity_name": entity_name,
            "entity_kind": "card" if domain == "cards" else "power",
            "domain": domain,
            "id": entity_id,
            "id_compact": entity_id.replace("_", ""),
            "title_en": desc.get("eng_title", "") or "unknown",
            "title_zh": desc.get("zhs_title", "") or "unknown",
            "references": "|".join(refs),
            "reference_ids": "|".join(ref_ids),
            "effects": "|".join(extract_effect_terms(content)),
            "language": "csharp",
            "ext": path.suffix,
        }
        for key, value in card_meta.items():
            base_metadata[key] = value

        summary = build_entity_summary(
            content=content,
            entity_name=entity_name,
            domain=domain,
            entity_id=entity_id,
            descriptions=desc,
            metadata=card_meta,
            method_names=[name for name, _ in methods],
            property_names=[name for name, _ in properties],
            references=refs,
            canonical_vars=canonical_vars,
        )
        docs.append(
            Document(
                page_content=summary,
                metadata={
                    **base_metadata,
                    "chunk_type": "entity_summary",
                    "member_name": entity_name,
                    "member_kind": "entity",
                },
            )
        )

        header = member_header(
            entity_name=entity_name,
            domain=domain,
            entity_id=entity_id,
            title=desc.get("eng_title", ""),
            description=desc.get("eng_description", "")
            or desc.get("eng_smartDescription", ""),
        )
        for name, code in methods:
            docs.append(
                Document(
                    page_content=(
                        f"{header}\nMemberKind: method\nMemberName: {name}\n\nCode:\n{code}"
                    ),
                    metadata={
                        **base_metadata,
                        "chunk_type": "method",
                        "member_name": name,
                        "member_kind": "method",
                    },
                )
            )
        for name, code in properties:
            docs.append(
                Document(
                    page_content=(
                        f"{header}\nMemberKind: property\nMemberName: {name}\n\nCode:\n{code}"
                    ),
                    metadata={
                        **base_metadata,
                        "chunk_type": "property",
                        "member_name": name,
                        "member_kind": "property",
                    },
                )
            )
    return docs
