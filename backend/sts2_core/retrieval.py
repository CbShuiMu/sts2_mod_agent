from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from langchain_core.documents import Document

from sts2_core.cs_utils import (
    domain_folder,
    extract_referenced_powers,
    find_card_file_by_name,
    find_cards_referencing_power,
    find_model_file,
    find_power_file,
    read_model_code,
)
from sts2_core.paths import PROJECT_ROOT
from sts2_core.text_utils import parse_candidates, split_search_queries


BM25_CANDIDATE_K = 1000


def domain_filter_expr(domains: Sequence[str] | None) -> str | None:
    allowed = [
        domain.strip().lower()
        for domain in domains or []
        if domain.strip() and domain.strip().lower().replace("_", "").isalnum()
    ]
    if not allowed:
        return None
    quoted = ", ".join(f'"{domain}"' for domain in sorted(set(allowed)))
    return f"domain in [{quoted}]"


def description_text(doc: Document) -> str:
    return "\n".join(
        str(doc.metadata.get(key, ""))
        for key in ["zh_description", "en_description"]
        if doc.metadata.get(key)
    ).lower()


def title_text(doc: Document) -> str:
    return "\n".join(
        str(doc.metadata.get(key, ""))
        for key in ["zh_title", "en_title", "code_name"]
        if doc.metadata.get(key)
    ).lower()


def bm25_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw):
            tokens.append(raw)
            if len(raw) > 1:
                tokens.extend(raw[i : i + 2] for i in range(len(raw) - 1))
            continue
        tokens.append(raw)
        if "_" in raw:
            tokens.extend(part for part in raw.split("_") if part)
    return tokens


def bm25_field_scores(
    docs: Sequence[Document],
    query: str,
    field_texts: Sequence[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    query_counter = Counter(bm25_tokens(query))
    if not docs or not query_counter:
        return [0.0 for _ in docs]

    tokenized_docs = [bm25_tokens(text) for text in field_texts]
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    if avg_doc_len <= 0:
        return [0.0 for _ in docs]

    doc_freqs: Counter[str] = Counter()
    for tokens in tokenized_docs:
        doc_freqs.update(set(tokens))

    total_docs = len(docs)
    scores: List[float] = []
    for tokens, doc_len in zip(tokenized_docs, doc_lengths):
        term_freqs = Counter(tokens)
        score = 0.0
        for term, query_freq in query_counter.items():
            freq = term_freqs.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (total_docs - doc_freqs[term] + 0.5) / (doc_freqs[term] + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / avg_doc_len)
            score += query_freq * idf * (freq * (k1 + 1)) / denom
        scores.append(score)
    return scores


def rerank_title_description_bm25(
    hits: Iterable[Tuple[Document, float]],
    query: str,
) -> List[Tuple[Document, float]]:
    hit_list = list(hits)
    docs = [doc for doc, _ in hit_list]
    title_scores = bm25_field_scores(docs, query, [title_text(doc) for doc in docs])
    description_scores = bm25_field_scores(docs, query, [description_text(doc) for doc in docs])
    has_title_match = any(score > 0 for score in title_scores)

    def rank_key(indexed_hit: Tuple[int, Tuple[Document, float]]) -> Tuple[float, float, float, float, int]:
        index, (_, score) = indexed_hit
        # BM25 is a relevance score where higher is better. Milvus returns a
        # distance-like score for this vector store, so lower is better. When
        # any title matches, title hits form the first tier; description only
        # decides ranking after title candidates are exhausted.
        if has_title_match:
            title_score = title_scores[index]
            title_tier = 0 if title_score > 0 else 1
            return title_tier, -title_score, -description_scores[index], float(score), index
        return 0, -description_scores[index], 0.0, float(score), index

    return [hit for _, hit in sorted(enumerate(hit_list), key=rank_key)]


def rerank_description_first(
    hits: Iterable[Tuple[Document, float]],
    query: str,
) -> List[Tuple[Document, float]]:
    return rerank_title_description_bm25(hits, query)


def dedupe_contexts(contexts: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for item in contexts:
        key = item.get("file_path", "") or "|".join(
            [
                item.get("domain", ""),
                item.get("id", ""),
                item.get("member_name", ""),
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def workspace_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def metadata_source_path(doc: Document, models_root: Path) -> Path | None:
    for key in (
        "description_key_resolved_path",
        "source_file_path",
        "description_key_path",
        "source",
        "source_absolute_path",
    ):
        raw = str(doc.metadata.get(key, "") or "").strip()
        if not raw:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidates = [
                (PROJECT_ROOT / candidate).resolve(),
                (models_root / candidate).resolve(),
            ]
        else:
            candidates = [candidate.resolve()]
        for path in candidates:
            if path.exists() and path.is_file():
                return path
    return None


def make_context_item(
    *,
    doc: Document,
    score: float,
    path: Path,
    model_folder: str,
    role: str,
    code_chars: int,
    include_description: bool = False,
) -> Dict[str, str]:
    title = str(doc.metadata.get("zh_title") or doc.metadata.get("en_title") or "")
    en_title = str(doc.metadata.get("en_title") or "")
    zh_title = str(doc.metadata.get("zh_title") or "")
    description = ""
    if include_description:
        description = extract_public_description(doc)
    return {
        "id": str(doc.metadata.get("id", "")),
        "domain": str(doc.metadata.get("domain", "")),
        "entity_name": path.stem,
        "chunk_type": "full_file",
        "member_name": role,
        "desc_score": f"{float(score):.4f}",
        "code_score": "",
        "model_folder": model_folder,
        "file_path": workspace_relative(path),
        "absolute_file_path": str(path.resolve()),
        "mcp_file_path": workspace_relative(path),
        "description_key_path": str(doc.metadata.get("description_key_path", "")),
        "description_key_resolved_path": str(doc.metadata.get("description_key_resolved_path", "")),
        "code": read_model_code(path, max_chars=code_chars),
        "title": title,
        "zh_title": zh_title,
        "en_title": en_title,
        "description": description,
    }


def make_metadata_context_item(
    *,
    doc: Document,
    score: float,
    model_folder: str,
    role: str,
    include_description: bool = False,
) -> Dict[str, str]:
    title = str(doc.metadata.get("zh_title") or doc.metadata.get("en_title") or "")
    en_title = str(doc.metadata.get("en_title") or "")
    zh_title = str(doc.metadata.get("zh_title") or "")
    description = extract_public_description(doc) if include_description else ""
    return {
        "id": str(doc.metadata.get("id", "")),
        "domain": str(doc.metadata.get("domain", "")),
        "entity_name": str(doc.metadata.get("id", "")),
        "chunk_type": "metadata",
        "member_name": role,
        "desc_score": f"{float(score):.4f}",
        "code_score": "",
        "model_folder": model_folder,
        "file_path": "",
        "absolute_file_path": "",
        "mcp_file_path": "",
        "description_key_path": str(doc.metadata.get("description_key_path", "")),
        "description_key_resolved_path": str(doc.metadata.get("description_key_resolved_path", "")),
        "code": "",
        "title": title,
        "zh_title": zh_title,
        "en_title": en_title,
        "description": description,
    }


def extract_public_description(doc: Document) -> str:
    metadata_lines = []
    for key, label in [
        ("zh_title", "ZH_Title"),
        ("en_title", "EN_Title"),
        ("zh_description", "ZH_Description"),
        ("en_description", "EN_Description"),
    ]:
        value = str(doc.metadata.get(key, "")).strip()
        if value and value.lower() != "unknown":
            metadata_lines.append(f"{label}: {value}")
    if metadata_lines:
        return "\n".join(metadata_lines)

    keep_prefixes = (
        "ZH_Title:",
        "EN_Title:",
        "ZH_Description:",
        "EN_Description:",
        "ZH_SmartDescription:",
        "EN_SmartDescription:",
    )
    lines = []
    for line in doc.page_content.splitlines():
        if line.startswith(keep_prefixes):
            key, value = line.split(":", 1)
            value = value.strip()
            if value and value.lower() != "unknown":
                lines.append(f"{key}: {value}")
    return "\n".join(lines)


def expand_full_code_context(
    *,
    doc: Document,
    score: float,
    models_root: Path,
    code_chars: int,
) -> List[Dict[str, str]]:
    domain = str(doc.metadata.get("domain", ""))
    base_id = str(doc.metadata.get("id", ""))
    candidates = parse_candidates(str(doc.metadata.get("code_candidates", "")))
    primary = metadata_source_path(doc, models_root)
    if primary is None:
        primary = find_model_file(
            models_root=models_root,
            domain=domain,
            base_id=base_id,
            candidates=candidates,
        )
    if primary is None:
        return [
            make_metadata_context_item(
                doc=doc,
                score=score,
                model_folder=domain_folder(domain),
                role="localization",
                include_description=True,
            )
        ]

    folder = domain_folder(domain)
    contexts = [
        make_context_item(
            doc=doc,
            score=score,
            path=primary,
            model_folder=folder,
            role="primary",
            code_chars=code_chars,
            include_description=True,
        )
    ]
    primary_code = primary.read_text(encoding="utf-8", errors="ignore")

    if domain == "cards":
        paired_power = find_power_file(models_root, f"{primary.stem}Power")
        if paired_power is not None:
            contexts.append(
                make_context_item(
                    doc=doc,
                    score=score,
                    path=paired_power,
                    model_folder="Powers",
                    role=f"paired_power:{paired_power.stem}",
                    code_chars=code_chars,
                )
            )
        for power_name in extract_referenced_powers(primary_code):
            power_file = find_power_file(models_root, power_name)
            if power_file is None:
                continue
            contexts.append(
                make_context_item(
                    doc=doc,
                    score=score,
                    path=power_file,
                    model_folder="Powers",
                    role=f"referenced_power:{power_name}",
                    code_chars=code_chars,
                )
            )
    elif domain == "powers":
        paired_card_name = primary.stem[:-5] if primary.stem.endswith("Power") else primary.stem
        paired_card = find_card_file_by_name(models_root, paired_card_name)
        if paired_card is not None:
            contexts.append(
                make_context_item(
                    doc=doc,
                    score=score,
                    path=paired_card,
                    model_folder="Cards",
                    role=f"paired_card:{paired_card.stem}",
                    code_chars=code_chars,
                )
            )
        for card_file in find_cards_referencing_power(models_root, primary.stem):
            contexts.append(
                make_context_item(
                    doc=doc,
                    score=score,
                    path=card_file,
                    model_folder="Cards",
                    role=f"referencing_card:{card_file.stem}",
                    code_chars=code_chars,
                )
            )

    return dedupe_contexts(contexts)


def retrieve_code_context(
    desc_db: object,
    models_root: Path,
    query: str,
    desc_top_k: int,
    code_chars: int,
    context_n: int | None = None,
    domains: Sequence[str] | None = None,
    **_: object,
) -> List[Dict[str, str]]:
    allowed_domains = {domain.strip().lower() for domain in domains or [] if domain.strip()}
    raw_k = max(desc_top_k * 5, desc_top_k, BM25_CANDIDATE_K)
    expr = domain_filter_expr(domains)
    if expr:
        hits: List[Tuple[Document, float]] = desc_db.similarity_search_with_score(query, k=raw_k, expr=expr)
    else:
        hits = desc_db.similarity_search_with_score(query, k=raw_k)
    if allowed_domains:
        # Integrity guard for old collections or vector stores that return paired
        # context from mixed metadata despite the Milvus expression filter above.
        hits = [
            (doc, score)
            for doc, score in hits
            if str(doc.metadata.get("domain", "")).strip().lower() in allowed_domains
        ]
    hits = rerank_description_first(hits, query)[:desc_top_k]
    contexts: List[Dict[str, str]] = []
    primary_count = 0
    for doc, score in hits:
        if context_n is not None and primary_count >= context_n:
            break
        expanded = expand_full_code_context(
            doc=doc,
            score=score,
            models_root=models_root,
            code_chars=code_chars,
        )
        if expanded:
            primary_count += 1
            contexts.extend(expanded)
    return dedupe_contexts(contexts)


def retrieve_code_context_groups(
    desc_db: object,
    models_root: Path,
    query: str,
    desc_top_k: int,
    code_chars: int,
    context_n: int | None = None,
    domains: Sequence[str] | None = None,
) -> List[Dict[str, object]]:
    groups: List[Dict[str, object]] = []
    for sub_query in split_search_queries(query):
        contexts = retrieve_code_context(
            desc_db=desc_db,
            models_root=models_root,
            query=sub_query,
            desc_top_k=desc_top_k,
            code_chars=code_chars,
            context_n=context_n,
            domains=domains,
        )
        groups.append({"query": sub_query, "contexts": contexts})
    return groups
