from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROVIDER_CATALOG: list[dict[str, Any]] = [
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "key_envs": ["DEEPSEEK_API_KEY", "DEEPSEEK_KEY", "deepseek_api_key"],
    },
    {
        "id": "openai",
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"],
        "key_envs": ["OPENAI_API_KEY", "openai_api_key"],
    },
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "models": ["openai/gpt-4.1-mini", "deepseek/deepseek-chat-v3-0324"],
        "key_envs": ["OPENROUTER_API_KEY"],
    },
    {
        "id": "custom",
        "name": "OpenAI Compatible",
        "base_url": "",
        "models": ["custom-model"],
        "key_envs": ["CUSTOM_OPENAI_API_KEY"],
    },
]


def provider_defaults(provider_id: str) -> dict[str, Any]:
    provider = next((item for item in PROVIDER_CATALOG if item["id"] == provider_id), None)
    if provider is None:
        provider = PROVIDER_CATALOG[-1]
    return {
        "provider_id": provider["id"],
        "name": provider["name"],
        "base_url": provider.get("base_url", ""),
        "model": provider.get("models", [""])[0],
        "temperature": 0.2,
        "context_length": 524288,
        "api_key": "",
    }


class SettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"default_provider_id": "deepseek", "providers": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"default_provider_id": "deepseek", "providers": {}}
        if not isinstance(data, dict):
            return {"default_provider_id": "deepseek", "providers": {}}
        data.setdefault("default_provider_id", "deepseek")
        data.setdefault("providers", {})
        return data

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def public_config(self) -> dict[str, Any]:
        data = self.load()
        providers: dict[str, dict[str, Any]] = {}
        for item in PROVIDER_CATALOG:
            provider_id = item["id"]
            merged = provider_defaults(provider_id)
            merged.update(data.get("providers", {}).get(provider_id, {}))
            env_key = self._env_key(item)
            has_key = bool(merged.get("api_key") or env_key)
            providers[provider_id] = {
                "provider_id": provider_id,
                "name": item["name"],
                "base_url": merged.get("base_url", ""),
                "model": merged.get("model", ""),
                "temperature": merged.get("temperature", 0.2),
                "context_length": int(merged.get("context_length", 524288) or 0),
                "has_key": has_key,
                "key_source": "saved" if merged.get("api_key") else ("env" if env_key else ""),
            }
        return {
            "catalog": PROVIDER_CATALOG,
            "default_provider_id": data.get("default_provider_id", "deepseek"),
            "providers": providers,
        }

    def update_provider(self, provider_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        data = self.load()
        provider = provider_defaults(provider_id)
        provider.update(data.get("providers", {}).get(provider_id, {}))

        for key in ["base_url", "model"]:
            if key in updates:
                provider[key] = str(updates.get(key) or "").strip()
        if "temperature" in updates:
            try:
                provider["temperature"] = float(updates["temperature"])
            except (TypeError, ValueError):
                provider["temperature"] = 0.2
        if "context_length" in updates:
            try:
                value = int(updates["context_length"])
                provider["context_length"] = max(0, value)
            except (TypeError, ValueError):
                provider["context_length"] = 524288
        if "api_key" in updates:
            raw_key = str(updates.get("api_key") or "").strip()
            if raw_key:
                provider["api_key"] = raw_key
        if updates.get("clear_key"):
            provider["api_key"] = ""

        data.setdefault("providers", {})[provider_id] = provider
        data["default_provider_id"] = str(updates.get("default_provider_id") or data.get("default_provider_id") or provider_id)
        self.save(data)
        return self.resolve_provider(provider_id)

    def set_default(self, provider_id: str) -> None:
        data = self.load()
        data["default_provider_id"] = provider_id
        self.save(data)

    def resolve_provider(self, provider_id: str | None) -> dict[str, Any]:
        data = self.load()
        resolved_id = provider_id or data.get("default_provider_id") or "deepseek"
        merged = provider_defaults(resolved_id)
        merged.update(data.get("providers", {}).get(resolved_id, {}))
        catalog = next((item for item in PROVIDER_CATALOG if item["id"] == resolved_id), None)
        env_key = self._env_key(catalog or {})
        if not merged.get("api_key") and env_key:
            merged["api_key"] = env_key
        return merged

    @staticmethod
    def _env_key(provider: dict[str, Any]) -> str:
        for name in provider.get("key_envs", []):
            value = os.getenv(name)
            if value:
                return value
        return ""
