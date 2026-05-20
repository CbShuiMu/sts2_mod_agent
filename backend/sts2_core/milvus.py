from __future__ import annotations

import os
import importlib.util
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from langchain_core.documents import Document

from sts2_core.embeddings import create_embeddings

try:
    from langchain_community.vectorstores import Milvus
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing Milvus dependencies. Install in conda env sts2_agent:\n"
        "pip install langchain-community pymilvus"
    ) from exc


def resolve_milvus_uri(persist_path: Path) -> str:
    """Use MILVUS_URI when set, otherwise use a local Milvus Lite db file."""
    env_uri = os.getenv("MILVUS_URI", "").strip()
    if env_uri:
        return env_uri
    return str(persist_path)


def is_local_lite_uri(uri: str) -> bool:
    lowered = uri.lower()
    return not (
        lowered.startswith("http://")
        or lowered.startswith("https://")
        or lowered.startswith("tcp://")
        or lowered.startswith("grpc://")
    )


def has_milvus_lite() -> bool:
    return importlib.util.find_spec("milvus_lite") is not None


def validate_milvus_runtime(persist_path: Path) -> None:
    uri = resolve_milvus_uri(persist_path)
    if is_local_lite_uri(uri) and not has_milvus_lite():
        raise RuntimeError(
            "Milvus Lite is required for local .db storage, but it is not available in this Python environment. "
            "Install pymilvus[milvus_lite] on a supported platform, or set MILVUS_URI to a running Milvus server, "
            "for example MILVUS_URI=http://127.0.0.1:19530."
        )


def milvus_connection_args(persist_path: Path) -> dict[str, str]:
    uri = resolve_milvus_uri(persist_path)
    if is_local_lite_uri(uri):
        args = {"uri": uri}
    else:
        parsed = urlparse(uri)
        args = {
            "host": parsed.hostname or "127.0.0.1",
            "port": str(parsed.port or 19530),
        }
    token = os.getenv("MILVUS_TOKEN", "").strip()
    db_name = os.getenv("MILVUS_DB_NAME", "").strip()
    if token:
        args["token"] = token
    if db_name:
        args["db_name"] = db_name
    return args


def milvus_index_params() -> dict[str, str]:
    return {
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {"M": 8, "efConstruction": 64},
    }


def open_description_db(
    *,
    persist_dir: Path,
    collection_name: str,
    model_name: str,
) -> Milvus:
    validate_milvus_runtime(persist_dir)
    embeddings = create_embeddings(model_name=model_name)
    return Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args=milvus_connection_args(persist_dir),
        index_params=milvus_index_params(),
        consistency_level="Strong",
    )


def build_description_db(
    *,
    documents: Iterable[Document],
    persist_dir: Path,
    collection_name: str,
    model_name: str,
    rebuild: bool,
) -> Milvus:
    validate_milvus_runtime(persist_dir)
    embeddings = create_embeddings(model_name=model_name)
    if not os.getenv("MILVUS_URI", "").strip():
        persist_dir.parent.mkdir(parents=True, exist_ok=True)
    db = Milvus.from_documents(
        documents=list(documents),
        embedding=embeddings,
        collection_name=collection_name,
        connection_args=milvus_connection_args(persist_dir),
        index_params=milvus_index_params(),
        consistency_level="Strong",
        drop_old=rebuild,
    )
    db.col.flush()
    return db
