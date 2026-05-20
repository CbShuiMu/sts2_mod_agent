from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
DEFAULT_MODELS_ROOT = PROJECT_ROOT / "data" / "Models"
DEFAULT_LOCALIZATION_ROOT = PROJECT_ROOT / "data" / "localization"
DEFAULT_VECTOR_DB_ROOT = PROJECT_ROOT / "data" / "vector_db"
DEFAULT_DESC_PERSIST_DIR = DEFAULT_VECTOR_DB_ROOT / "description_milvus.db"
DEFAULT_DESC_COLLECTION_NAME = "sts2_descriptions"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "data" / "settings" / "ai_settings.json"
