from __future__ import annotations

import os
from threading import RLock
from pathlib import Path
from typing import List, Sequence

try:
    from langchain_core.embeddings import Embeddings
    from transformers import pipeline
except ImportError as exc:
    raise SystemExit(
        "Missing required packages. Install:\n"
        "pip install langchain langchain-community pymilvus transformers torch"
    ) from exc


DEFAULT_EMBEDDING_MODEL = "codefuse-ai/F2LLM-v2-0.6B"
_EMBEDDING_CACHE: dict[str, "PipelineEmbeddings"] = {}
_EMBEDDING_CACHE_LOCK = RLock()


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class PipelineEmbeddings(Embeddings):
    """LangChain embedding adapter around transformers feature-extraction pipeline."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.pipe = pipeline("feature-extraction", model=model_name)
        self._lock = RLock()

    @staticmethod
    def _mean_pool(token_vectors: Sequence[Sequence[float]]) -> List[float]:
        if not token_vectors:
            return []
        dim = len(token_vectors[0])
        sums = [0.0] * dim
        for vec in token_vectors:
            for i, value in enumerate(vec):
                sums[i] += float(value)
        count = float(len(token_vectors))
        return [value / count for value in sums]

    @staticmethod
    def resolved_batch_size() -> int:
        # Read EMBEDDING_BATCH_SIZE per-call so .env loaded after import still wins.
        raw = os.environ.get("EMBEDDING_BATCH_SIZE", "").strip()
        try:
            value = int(raw) if raw else 16
        except ValueError:
            value = 16
        return max(1, value)

    DEFAULT_BATCH_SIZE = 16  # legacy attribute; embed_documents now calls resolved_batch_size()

    @staticmethod
    def _normalize_token_vectors(item) -> Sequence[Sequence[float]]:
        # feature-extraction usually returns [tokens, hidden], but some versions
        # return [[tokens, hidden]] for a single input.
        if not item:
            return []
        first = item[0]
        if first and isinstance(first, list) and first and isinstance(first[0], list):
            return first
        return item

    def _embed_one(self, text: str) -> List[float]:
        with self._lock:
            raw = self.pipe(text, truncation=True, max_length=512)
        return self._mean_pool(self._normalize_token_vectors(raw))

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        with self._lock:
            raw_batch = self.pipe(
                texts,
                truncation=True,
                max_length=512,
                batch_size=len(texts),
            )
        # Pipeline with a list input returns a list of per-text outputs.
        return [self._mean_pool(self._normalize_token_vectors(item)) for item in raw_batch]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        batch_size = self.resolved_batch_size()
        results: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            results.extend(self._embed_batch(texts[start:start + batch_size]))
        return results

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)


def create_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Embeddings:
    normalized = (model_name or DEFAULT_EMBEDDING_MODEL).strip()
    with _EMBEDDING_CACHE_LOCK:
        if normalized not in _EMBEDDING_CACHE:
            _EMBEDDING_CACHE[normalized] = PipelineEmbeddings(model_name=normalized)
        return _EMBEDDING_CACHE[normalized]
