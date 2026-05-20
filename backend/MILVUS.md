# Milvus vector store

This project now uses Milvus through LangChain's Milvus vector store and `pymilvus`.

## Local Milvus Lite

The default path is:

```text
data/vector_db/description_milvus.db
```

Build all default vector DBs with:

```powershell
conda activate sts2_agent
python backend\scripts\build.py --rebuild
```

This builds the card/power description DB plus one separate localization DB for each of:

```text
data/vector_db/relics_milvus.db
data/vector_db/potions_milvus.db
data/vector_db/orbs_milvus.db
data/vector_db/enchantments_milvus.db
data/vector_db/afflictions_milvus.db
data/vector_db/rest_site_ui_milvus.db
data/vector_db/events_milvus.db
```

To build only one localization DB:

```powershell
python backend\scripts\build.py --target relics --rebuild --skip-preview
python backend\scripts\build.py --target enchantment --rebuild --skip-preview
python backend\scripts\build.py --target rest_site_ui --rebuild --skip-preview
python backend\scripts\build.py --target events --rebuild --skip-preview
```

Milvus Lite requires the `milvus_lite` package. If your platform does not provide it, use a Milvus server instead.

## Milvus server

Set `MILVUS_URI` before building or running:

```powershell
$env:MILVUS_URI = "http://127.0.0.1:19530"
python backend\scripts\build.py --rebuild
python backend\app.py --port 7870
```

This repository includes a local Milvus Standalone Docker Compose file:

```powershell
docker compose -f docker-compose.milvus.yml up -d
$env:MILVUS_URI = "http://127.0.0.1:19530"
python backend\scripts\build.py --rebuild
python backend\app.py --port 7870
```

Milvus service ports:

```text
19530 - Milvus gRPC/API
9091  - Milvus health and web UI
9001  - MinIO console
```

Optional environment variables:

```text
MILVUS_TOKEN
MILVUS_DB_NAME
```
