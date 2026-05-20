#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from langchain_core.documents import Document

from sts2_core.milvus import build_description_db, open_description_db, resolve_milvus_uri
from sts2_core.cs_utils import (
    extract_canonical_vars,
    extract_card_constructor_metadata,
    extract_class_name,
    extract_power_metadata,
    extract_referenced_powers,
    find_model_file_from_description_key,
    find_model_file,
    infer_effect_tags,
)
from sts2_core.description_utils import (
    DESCRIPTION_FILES,
    description_search_text,
    discover_description_root,
    load_description_groups,
)
from sts2_core.embeddings import DEFAULT_EMBEDDING_MODEL, load_env_file
from sts2_core.localization_vector_utils import (
    LOCALIZATION_VECTOR_DOMAINS,
    build_localization_documents,
    localization_collection_name,
    localization_filename,
    localization_persist_path,
    normalize_localization_domain,
)
from sts2_core.paths import (
    DEFAULT_DESC_COLLECTION_NAME,
    DEFAULT_DESC_PERSIST_DIR,
    DEFAULT_ENV_PATH,
    DEFAULT_LOCALIZATION_ROOT,
    DEFAULT_MODELS_ROOT,
    DEFAULT_VECTOR_DB_ROOT,
    PROJECT_ROOT,
)
from sts2_core.text_utils import id_to_code_name, make_code_candidates, pascal_to_upper_snake


def workspace_relative(path: Path | None) -> str:
    if path is None:
        return ""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def description_key_candidate_path(models_root: Path, domain: str, base_id: str) -> str:
    folder = domain
    filename = f"{base_id.replace('_', '').lower()}.cs"
    candidate = models_root / folder / filename
    try:
        return candidate.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(candidate)


def collect_model_groups(models_root: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    grouped: Dict[Tuple[str, str], Dict[str, str]] = {}
    for domain, folder_name in [("cards", "Cards"), ("powers", "Powers")]:
        folder = models_root / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*.cs")):
            if "Mocks" in path.parts:
                continue
            class_name = path.stem
            base_id = pascal_to_upper_snake(class_name)
            if not base_id:
                continue
            grouped[(domain, base_id)] = {
                "eng_title": class_name,
                "zhs_title": class_name,
                "eng_description": "",
                "zhs_description": "",
            }
    return grouped


def build_document(
    models_root: Path,
    domain: str,
    base_id: str,
    fields: Dict[str, str],
) -> Document:
    zhs_title = fields.get("zhs_title", "")
    eng_title = fields.get("eng_title", "")
    zhs_desc = fields.get("zhs_description", "")
    eng_desc = fields.get("eng_description", "")
    zhs_smart = fields.get("zhs_smartDescription", "")
    eng_smart = fields.get("eng_smartDescription", "")

    description_key_file = find_model_file_from_description_key(models_root, domain, base_id)
    model_file = description_key_file or find_model_file(models_root, domain, base_id)
    source_file_path = workspace_relative(model_file)
    source_absolute_path = str(model_file.resolve()) if model_file else ""
    description_key_path = description_key_candidate_path(models_root, domain, base_id)
    description_key_resolved_path = workspace_relative(description_key_file)
    code = model_file.read_text(encoding="utf-8", errors="ignore") if model_file else ""
    class_name = extract_class_name(code, id_to_code_name(base_id)) if code else id_to_code_name(base_id)
    model_meta = (
        extract_card_constructor_metadata(class_name, code)
        if domain == "cards"
        else extract_power_metadata(code)
    )
    canonical_vars, var_types = extract_canonical_vars(code)
    referenced_powers = extract_referenced_powers(code)
    effect_tags = infer_effect_tags(var_types, code)

    desc_search = description_search_text(zhs_desc, eng_desc, zhs_smart, eng_smart)
    feature_search = (
        f"CardType: {model_meta.get('card_type', '')}\n"
        f"CardRarity: {model_meta.get('rarity', '')}\n"
        f"TargetType: {model_meta.get('target_type', '')}\n"
        f"PowerType: {model_meta.get('power_type', '')}\n"
        f"PowerStackType: {model_meta.get('stack_type', '')}\n"
        f"CanonicalVarTypes: {' | '.join(var_types)}\n"
        f"ReferencedPowers: {' | '.join(referenced_powers)}\n"
        f"EffectTags: {' | '.join(effect_tags)}\n"
    )
    page = (
        "DescriptionSearchText:\n"
        f"{desc_search}\n\n"
        "CodeDerivedFeatures:\n"
        f"{feature_search}\n"
        "LowPriorityIdentity:\n"
        f"ZH_Title: {zhs_title}\n"
        f"EN_Title: {eng_title}\n"
        f"Domain: {domain}\n"
        f"ID: {base_id}\n"
        f"CodeName: {class_name}\n"
        f"DescriptionKeyPath: {description_key_path}\n"
        f"SourceFilePath: {source_file_path}\n"
    )

    return Document(
        page_content=page,
        metadata={
            "domain": domain,
            "id": base_id,
            "id_compact": base_id.replace("_", ""),
            "code_name": class_name,
            "code_candidates": f"{make_code_candidates(base_id)}|{class_name}",
            "source": str(model_file or ""),
            "description_key_path": description_key_path,
            "description_key_resolved_path": description_key_resolved_path,
            "source_file_path": source_file_path,
            "source_absolute_path": source_absolute_path,
            "zh_title": zhs_title or "unknown",
            "en_title": eng_title or "unknown",
            "zh_description": zhs_desc or zhs_smart,
            "en_description": eng_desc or eng_smart,
            "card_type": model_meta.get("card_type", ""),
            "rarity": model_meta.get("rarity", ""),
            "target_type": model_meta.get("target_type", ""),
            "power_type": model_meta.get("power_type", ""),
            "stack_type": model_meta.get("stack_type", ""),
            "canonical_var_types": "|".join(var_types),
            "referenced_powers": "|".join(referenced_powers),
            "effect_tags": "|".join(effect_tags),
            "language": "bilingual",
        },
    )


def build_documents(desc_root: Path, models_root: Path) -> List[Document]:
    desc_root = discover_description_root(desc_root, models_root)
    groups = load_description_groups(desc_root)
    if not groups:
        groups = collect_model_groups(models_root)
    docs = [
        build_document(models_root=models_root, domain=domain, base_id=base_id, fields=fields)
        for (domain, base_id), fields in sorted(groups.items())
    ]
    return docs


def build_db(
    desc_root: Path,
    models_root: Path,
    persist_dir: Path,
    collection_name: str,
    model_name: str,
    rebuild: bool,
) -> None:
    resolved_desc_root = discover_description_root(desc_root, models_root)
    docs = build_documents(desc_root=resolved_desc_root, models_root=models_root)
    if not docs:
        expected = [resolved_desc_root.joinpath(*parts) for parts in DESCRIPTION_FILES.values()]
        missing = "\n".join(f"- {path}" for path in expected if not path.exists())
        raise RuntimeError(
            f"No description documents found under: {resolved_desc_root}\n"
            f"Missing expected files:\n{missing}\n"
            f"Also found no C# model files under: {models_root}\\Cards or {models_root}\\Powers"
        )

    build_description_db(
        documents=docs,
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=model_name,
        rebuild=rebuild,
    )

    print(f"Description root: {resolved_desc_root}")
    print(f"Models root: {models_root}")
    print(f"Description chunks indexed: {len(docs)}")
    print(f"Milvus URI/path: {resolve_milvus_uri(persist_dir)}")
    print(f"Collection: {collection_name}")


def build_localization_db(
    *,
    localization_root: Path,
    models_root: Path,
    domain: str,
    vector_root: Path,
    persist_dir: Path | None,
    collection_name: str | None,
    model_name: str,
    rebuild: bool,
) -> None:
    normalized_domain = normalize_localization_domain(domain)
    docs = build_localization_documents(
        localization_root=localization_root,
        domain=normalized_domain,
        models_root=models_root,
    )
    if not docs:
        filename = localization_filename(normalized_domain)
        expected = [localization_root / lang / filename for lang in ("eng", "zhs")]
        missing = "\n".join(f"- {path}" for path in expected if not path.exists())
        raise RuntimeError(
            f"No localization documents found for {normalized_domain} under: {localization_root}\n"
            f"Missing expected files:\n{missing}"
        )

    resolved_persist_dir = persist_dir or localization_persist_path(vector_root, normalized_domain)
    resolved_collection_name = collection_name or localization_collection_name(normalized_domain)
    build_description_db(
        documents=docs,
        persist_dir=resolved_persist_dir,
        collection_name=resolved_collection_name,
        model_name=model_name,
        rebuild=rebuild,
    )

    print(f"\nLocalization domain: {normalized_domain}")
    print(f"Localization root: {localization_root}")
    print(f"Models root: {models_root}")
    print(f"Localization chunks indexed: {len(docs)}")
    print(f"Milvus URI/path: {resolve_milvus_uri(resolved_persist_dir)}")
    print(f"Collection: {resolved_collection_name}")


def build_localization_dbs(
    *,
    localization_root: Path,
    models_root: Path,
    domains: list[str],
    vector_root: Path,
    model_name: str,
    rebuild: bool,
) -> None:
    for domain in domains:
        build_localization_db(
            localization_root=localization_root,
            models_root=models_root,
            domain=domain,
            vector_root=vector_root,
            persist_dir=None,
            collection_name=None,
            model_name=model_name,
            rebuild=rebuild,
        )


def preview_localization_query(
    *,
    persist_dir: Path,
    collection_name: str,
    model_name: str,
    query: str,
    k: int,
) -> None:
    db = open_description_db(
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=model_name,
    )
    hits = db.similarity_search_with_score(query, k=k)
    print(f"\nQuery: {query}")
    print(f"Top {k} localization matches:")
    for i, (doc, score) in enumerate(hits, start=1):
        print(
            f"{i}. score={score:.4f} id={doc.metadata.get('id')} "
            f"domain={doc.metadata.get('domain')} "
            f"title={doc.metadata.get('en_title') or doc.metadata.get('zh_title')}"
        )


def preview_query(
    persist_dir: Path,
    collection_name: str,
    model_name: str,
    query: str,
    k: int,
) -> None:
    db = open_description_db(
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=model_name,
    )
    hits = db.similarity_search_with_score(query, k=k)
    print(f"\nQuery: {query}")
    print(f"Top {k} description matches:")
    for i, (doc, score) in enumerate(hits, start=1):
        print(
            f"{i}. score={score:.4f} id={doc.metadata.get('id')} "
            f"domain={doc.metadata.get('domain')} "
            f"type={doc.metadata.get('card_type') or doc.metadata.get('power_type')} "
            f"target={doc.metadata.get('target_type')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build STS2 Milvus DBs for card/power descriptions and localization files."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        help=(
            "What to build: all, descriptions, relics, potions, orbs, enchantments, "
            "afflictions, rest_site_ui, events. Singular aliases such as enchantment are accepted."
        ),
    )
    parser.add_argument(
        "--description-root",
        type=Path,
        default=DEFAULT_LOCALIZATION_ROOT,
        help="Folder containing eng/zhs localization JSON files.",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=DEFAULT_MODELS_ROOT,
        help="Root folder containing Models/Cards and Models/Powers.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_DESC_PERSIST_DIR,
        help="Description Milvus Lite db path. Set MILVUS_URI to use a remote Milvus server instead.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=os.environ.get("DESC_COLLECTION_NAME", "").strip() or DEFAULT_DESC_COLLECTION_NAME,
    )
    parser.add_argument(
        "--vector-root",
        type=Path,
        default=DEFAULT_VECTOR_DB_ROOT,
        help="Folder for per-localization Milvus Lite db files.",
    )
    parser.add_argument(
        "--localization-persist-dir",
        type=Path,
        default=None,
        help="Optional Milvus Lite db path when building one localization target.",
    )
    parser.add_argument(
        "--localization-collection-name",
        type=str,
        default=None,
        help="Optional collection name when building one localization target.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=os.environ.get("EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL,
    )
    parser.add_argument("--test-query", type=str, default="attack all enemies and apply weak")
    parser.add_argument(
        "--localization-test-query",
        type=str,
        default=None,
        help="Optional preview query when building one localization target.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-preview", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_env_file(DEFAULT_ENV_PATH)
    args = parse_args()
    raw_target = args.target.strip().lower()
    build_descriptions = raw_target in {"all", "descriptions", "description"}
    build_all_localizations = raw_target == "all"
    single_localization_target = None
    if not build_descriptions and not build_all_localizations:
        single_localization_target = normalize_localization_domain(raw_target)

    if build_descriptions:
        build_db(
            desc_root=args.description_root,
            models_root=args.models_root,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            model_name=args.embedding_model,
            rebuild=args.rebuild,
        )

    if build_all_localizations:
        build_localization_dbs(
            localization_root=args.description_root,
            models_root=args.models_root,
            domains=list(LOCALIZATION_VECTOR_DOMAINS),
            vector_root=args.vector_root,
            model_name=args.embedding_model,
            rebuild=args.rebuild,
        )
    elif single_localization_target:
        build_localization_db(
            localization_root=args.description_root,
            models_root=args.models_root,
            domain=single_localization_target,
            vector_root=args.vector_root,
            persist_dir=args.localization_persist_dir,
            collection_name=args.localization_collection_name,
            model_name=args.embedding_model,
            rebuild=args.rebuild,
        )

    if not args.skip_preview and build_descriptions:
        preview_query(
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            model_name=args.embedding_model,
            query=args.test_query,
            k=args.top_k,
        )
    if not args.skip_preview and single_localization_target and args.localization_test_query:
        preview_localization_query(
            persist_dir=args.localization_persist_dir
            or localization_persist_path(args.vector_root, single_localization_target),
            collection_name=args.localization_collection_name
            or localization_collection_name(single_localization_target),
            model_name=args.embedding_model,
            query=args.localization_test_query,
            k=args.top_k,
        )


if __name__ == "__main__":
    main()
