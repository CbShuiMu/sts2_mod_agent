#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sts2_core.milvus import open_description_db
from sts2_core.embeddings import DEFAULT_EMBEDDING_MODEL, load_env_file
from sts2_core.paths import (
    DEFAULT_DESC_COLLECTION_NAME,
    DEFAULT_DESC_PERSIST_DIR,
    DEFAULT_ENV_PATH,
    DEFAULT_MODELS_ROOT,
)
from sts2_core.retrieval import retrieve_code_context


def main() -> None:
    load_env_file(DEFAULT_ENV_PATH)
    parser = argparse.ArgumentParser(description="Two-step query: description -> code")
    parser.add_argument("--query", type=str, required=True, help="Natural language query.")
    parser.add_argument(
        "--desc-persist-dir",
        type=Path,
        default=DEFAULT_DESC_PERSIST_DIR,
    )
    parser.add_argument(
        "--desc-collection-name",
        type=str,
        default=DEFAULT_DESC_COLLECTION_NAME,
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=DEFAULT_MODELS_ROOT,
    )
    parser.add_argument(
        "--show-code",
        action="store_true",
        help="Print matched C# code chunks.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
    )
    parser.add_argument("--desc-top-k", type=int, default=3)
    parser.add_argument(
        "--context-n",
        type=int,
        default=3,
        help="Number of description hits to expand into full C# file context.",
    )
    parser.add_argument(
        "--code-preview-chars",
        type=int,
        default=2000,
        help="Max characters printed for each matched code chunk.",
    )
    args = parser.parse_args()

    desc_db = open_description_db(
        persist_dir=args.desc_persist_dir,
        collection_name=args.desc_collection_name,
        model_name=args.embedding_model,
    )
    desc_hits = desc_db.similarity_search_with_score(args.query, k=args.desc_top_k)
    print(f"Description hits for query: {args.query}")
    for i, (doc, score) in enumerate(desc_hits, start=1):
        print(
            f"{i}. score={score:.4f} id={doc.metadata.get('id')} "
            f"domain={doc.metadata.get('domain')} "
            f"code_candidates={doc.metadata.get('code_candidates')}"
        )

    contexts = retrieve_code_context(
        desc_db=desc_db,
        models_root=args.models_root,
        query=args.query,
        desc_top_k=args.desc_top_k,
        code_chars=args.code_preview_chars,
        context_n=args.context_n,
    )
    if not contexts:
        print("\nNo matching code context found.")
        return

    print("\nCode context (full C# files):")
    for i, item in enumerate(contexts, start=1):
        print(
            f"{i}. id={item['id']} domain={item['domain']} "
            f"folder={item.get('model_folder', '')} role={item['member_name']} "
            f"desc_score={item['desc_score']}"
        )
        print(f"   file={item['file_path']}")
        snippet = item["code"].replace("\n", " ")[:180]
        print(f"   {snippet}...")

    if args.show_code:
        print("\nCode chunks:\n")
        for i, item in enumerate(contexts, start=1):
            print(f"[Code Context {i}] {item['file_path']} :: {item['member_name']}")
            if item.get("description"):
                print("\nTitle and description:\n")
                print(item["description"])
            print(item["code"])
            print()


if __name__ == "__main__":
    main()
