from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from sts2_core.text_utils import id_to_code_name, normalize_name, truncate_text


def domain_folder(domain: str) -> str:
    folders = {
        "cards": "Cards",
        "powers": "Powers",
        "potions": "Potions",
        "relics": "Relics",
        "orbs": "Orbs",
        "enchantments": "Enchantments",
        "afflictions": "Afflictions",
        "rest_site_ui": "RestSiteUi",
        "events": "Events",
    }
    return folders.get(domain.strip().lower(), domain[:1].upper() + domain[1:])


def find_matching(text: str, start_idx: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    in_string = False
    in_char = False
    escaped = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and (in_string or in_char):
            escaped = True
            continue
        if ch == '"' and not in_char:
            in_string = not in_string
            continue
        if ch == "'" and not in_string:
            in_char = not in_char
            continue
        if in_string or in_char:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return len(text) - 1


def split_top_level_args(arg_text: str) -> List[str]:
    args: List[str] = []
    start = 0
    depth = 0
    for i, ch in enumerate(arg_text):
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(depth - 1, 0)
        elif ch == "," and depth == 0:
            args.append(arg_text[start:i].strip())
            start = i + 1
    tail = arg_text[start:].strip()
    if tail:
        args.append(tail)
    return args


def extract_parenthesized(text: str, start_idx: int) -> str:
    paren_start = text.find("(", start_idx)
    if paren_start == -1:
        return ""
    end = find_matching(text, paren_start, "(", ")")
    return text[paren_start + 1 : end]


def extract_class_name(content: str, fallback: str) -> str:
    match = re.search(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b", content)
    return match.group(1) if match else fallback


def extract_card_constructor_metadata(class_name: str, content: str) -> dict[str, str]:
    pattern = rf"\bpublic\s+{re.escape(class_name)}\s*\([^)]*\)\s*:\s*base\s*\("
    match = re.search(pattern, content, flags=re.MULTILINE)
    if not match:
        return {}
    args = split_top_level_args(extract_parenthesized(content, match.end() - 1))
    meta: dict[str, str] = {}
    if len(args) > 0:
        meta["energy_cost"] = args[0]
    if len(args) > 1:
        meta["card_type"] = args[1].replace("CardType.", "")
    if len(args) > 2:
        meta["rarity"] = args[2].replace("CardRarity.", "")
    if len(args) > 3:
        meta["target_type"] = args[3].replace("TargetType.", "")
    return meta


def extract_power_metadata(content: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    power_type = re.search(r"\bPowerType\b\s*=>\s*PowerType\.([A-Za-z_][A-Za-z0-9_]*)", content)
    stack_type = re.search(
        r"\bPowerStackType\b\s*=>\s*PowerStackType\.([A-Za-z_][A-Za-z0-9_]*)",
        content,
    )
    if power_type:
        meta["power_type"] = power_type.group(1)
    if stack_type:
        meta["stack_type"] = stack_type.group(1)
    return meta


def extract_canonical_vars_block(content: str) -> str:
    match = re.search(r"\bCanonicalVars\b\s*=>", content)
    if not match:
        return ""
    semi = content.find(";", match.end())
    if semi == -1:
        return ""
    return content[match.end() : semi].strip()


def extract_canonical_vars(content: str) -> Tuple[List[str], List[str]]:
    block = extract_canonical_vars_block(content)
    if not block:
        return [], []
    vars_out: List[str] = []
    var_types: List[str] = []
    for match in re.finditer(
        r"\bnew\s+([A-Za-z_][A-Za-z0-9_]*Var)(?:<([^>]+)>)?\s*\((.*?)\)",
        block,
        flags=re.DOTALL,
    ):
        var_type = match.group(1)
        generic = match.group(2) or ""
        args = re.sub(r"\s+", " ", match.group(3)).strip()
        typed = f"{var_type}<{generic}>" if generic else var_type
        var_types.append(typed)
        vars_out.append(f"{typed}({args})")
    return vars_out, sorted(set(var_types))


def extract_referenced_powers(content: str) -> List[str]:
    refs = set()
    patterns = [
        r"\bPowerCmd\.Apply<([A-Za-z_][A-Za-z0-9_]*Power)>",
        r"\bPowerVar<([A-Za-z_][A-Za-z0-9_]*Power)>",
        r"\bHoverTipFactory\.FromPower<([A-Za-z_][A-Za-z0-9_]*Power)>",
        r"\bGetPower<([A-Za-z_][A-Za-z0-9_]*Power)>",
        r"\bHasPower<([A-Za-z_][A-Za-z0-9_]*Power)>",
    ]
    for pattern in patterns:
        refs.update(re.findall(pattern, content))
    return sorted(refs)


def infer_effect_tags(var_types: Iterable[str], content: str) -> List[str]:
    text = " ".join(var_types).lower() + "\n" + content.lower()
    rules = [
        ("damage", ["damagevar", "damagecmd.attack"]),
        ("block", ["blockvar", "gainblock"]),
        ("weak", ["weakpower"]),
        ("vulnerable", ["vulnerablepower"]),
        ("poison", ["poisonpower"]),
        ("strength", ["strengthpower"]),
        ("dexterity", ["dexteritypower"]),
        ("draw", ["cardsvar", "cardpilecmd.draw"]),
        ("energy", ["energyvar", "gainenergy"]),
        ("hp_loss", ["hplossvar", "losehp"]),
        ("power", ["powervar", "powercmd.apply"]),
    ]
    return [name for name, needles in rules if any(needle in text for needle in needles)]


def find_model_file(
    models_root: Path,
    domain: str,
    base_id: str,
    candidates: Sequence[str] = (),
) -> Path | None:
    keyed = find_model_file_from_description_key(models_root, domain, base_id)
    if keyed is not None:
        return keyed

    folder = models_root / domain_folder(domain)
    if not folder.exists():
        return None

    expected = {normalize_name(base_id.replace("_", "")), normalize_name(id_to_code_name(base_id))}
    expected.update(normalize_name(candidate) for candidate in candidates if candidate)

    exact: List[Path] = []
    fuzzy: List[Path] = []
    for path in folder.rglob("*.cs"):
        stem = normalize_name(path.stem)
        if stem in expected:
            exact.append(path)
            continue
        if any(item and (item in stem or stem in item) for item in expected):
            fuzzy.append(path)
    if exact:
        return sorted(exact)[0]
    if fuzzy:
        return sorted(fuzzy)[0]
    return None


def find_model_file_from_description_key(models_root: Path, domain: str, base_id: str) -> Path | None:
    folder = models_root / domain_folder(domain)
    if not folder.exists():
        return None
    compact_key = normalize_name(base_id.replace("_", ""))
    for path in sorted(folder.rglob("*.cs")):
        if normalize_name(path.stem) == compact_key:
            return path
    return None


def read_model_code(path: Path, max_chars: int) -> str:
    return truncate_text(path.read_text(encoding="utf-8", errors="ignore"), max_chars=max_chars)


def find_power_file(models_root: Path, power_name: str) -> Path | None:
    folder = models_root / "Powers"
    if not folder.exists():
        return None
    expected = normalize_name(power_name)
    for path in folder.rglob("*.cs"):
        if normalize_name(path.stem) == expected:
            return path
    return None


def find_card_file_by_name(models_root: Path, card_name: str) -> Path | None:
    folder = models_root / "Cards"
    if not folder.exists():
        return None
    expected = normalize_name(card_name)
    for path in folder.rglob("*.cs"):
        if normalize_name(path.stem) == expected:
            return path
    return None


def find_cards_referencing_power(models_root: Path, power_name: str, limit: int = 3) -> List[Path]:
    folder = models_root / "Cards"
    if not folder.exists():
        return []
    pattern = re.compile(rf"\b{re.escape(power_name)}\b")
    hits: List[Path] = []
    for path in sorted(folder.rglob("*.cs")):
        content = path.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(content):
            hits.append(path)
            if len(hits) >= limit:
                break
    return hits
