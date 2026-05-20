# STS2 Mod Agent

> 一个面向 **Slay the Spire 2** 的 Mod 智能体：基于 Flask + LangChain + Milvus 的 RAG 检索后端，配合可视化前端，由 LLM Agent 直接读写 `mods/` 目录下的 C#/Godot 源码与本地化文件。

**语言 / Language**: **中文**（当前） | [English](./README.en.md)

---

## 目录

- [项目简介](#项目简介)
- [仓库结构](#仓库结构)
- [安装依赖](#安装依赖)
- [运行](#运行)
- [使用指南](#使用指南)
- [项目特点](#项目特点)

---

## 项目简介

STS2 Mod Agent 把游戏源码、本地化文本（卡牌 / 遗物 / 药水 / 残响 / 附魔 / 异常 / 休息点 UI / 事件）切分成语义片段，写入 Milvus 向量库；前端发起对话或检索请求后，后端通过 RAG 召回相关上下文，并交给 LLM Agent 调用 MCP 工具完成 Mod 的代码生成、文件落地与资产校验。

| 层级 | 技术栈 |
| --- | --- |
| 后端 | Python 3.14 · Flask 3 · LangChain 1.2 · `langchain-milvus` · `pymilvus` |
| 向量库 | Milvus 2.6（默认 Milvus Lite 本地 `.db`，也可走 Docker Standalone） |
| 嵌入模型 | `codefuse-ai/F2LLM-v2-0.6B`（`transformers` + `torch`） |
| LLM | DeepSeek / OpenAI 兼容协议（在前端 `/api/config` 配置） |
| 前端 | 原生 HTML/CSS/JS（`/`、`/make`、`/query`、`/mod` 多页面） |
| MCP | `LocalFileMCP` 本地文件读写 + `rag_query` 检索工具 |

---

## 仓库结构

```
sts2_mod_agent/
  backend/                   后端：Flask API、RAG 检索、Agent 主循环
    app.py                   Flask 入口、/api/* 路由、流式对话
    agent.py                 Agent 主循环、工具调度、去重 / 兜底
    sts2_core/               检索核心：embedding、BM25 rerank、Milvus 封装
    services/                LLM 客户端、prompts、rag_query 工具实现
    mcp/                     LocalFileMCP 本地文件读写沙箱
    scripts/                 离线脚本（build.py 等向量库构建）
    settings_store.py        /api/config 配置持久化
    legacy/                  旧版索引代码（保留参考）

  frontend/                  原生 HTML/CSS/JS：/、/make、/query、/mod、/settings

  data/                      本地数据（多数已 .gitignore）
    localization/            游戏本体本地化 JSON（zhs / eng × cards / powers …）
    Models/                  反编译后的游戏 C# 源码（参考）
    libs/                    反编译依赖 dll
    settings/                rules.json 等运行时配置
    milvus/                  docker-compose Milvus 数据卷（含已构建向量库）
    logs/                    ai_chat_log 等请求级日志

  mods/                      Mod 工程根目录（Agent 写入沙箱）
    template/                空白模板，第一轮对话从这里 copy_tree
    <your_mod>/              生成的 mod：src/Core/Models/*.cs、localization/、resources/

  tools/                     外部工具脚本（多数已 .gitignore，按需下载）
    ExportMod.cmd            打包导出 mod 给游戏加载（dotnet build）
    lookup_symbol.py         Type.Member 反查反编译源码位置

  docker-compose.milvus.yml  独立 Milvus 容器（不走 Milvus Lite 时启用）
  .env.example               环境变量样板
```

---

## 安装依赖

### 前置要求

- **Python 3.14**（在 3.11 / 3.12 上同样能跑，但 `requirements.lock.txt` 未在更低版本测试）
- 包管理器二选一：**[uv](https://docs.astral.sh/uv/)（推荐）** 或 conda
- 可选：Docker（仅当不使用 Milvus Lite 时需要）
- **外部工具**（用于「导出 / 反编译 / 编译 Mod」工作流，下载后把绝对路径填到 `.env` 的 Tool paths 段，见步骤 1）：
  - **[Godot 4.5.1 Mono (win64)](https://godotengine.org/download/archive)** — 打开 / 编辑 mod 的 `.tres` 资源与场景，必须用 **Mono** 版本才能跑 C# 脚本。
  - **[GDRE Tools v2.5.0-beta.5 (windows)](https://github.com/GDRETools/gdsdecomp/releases)** — Godot 资源反编译工具，用于从游戏本体提取 `.pck` 内容做参考。
  - **[.NET 9 SDK](https://dotnet.microsoft.com/en-us/download/dotnet)** — `tools/ExportMod.cmd` 走 `dotnet build` 编译 mod 的 C# 工程，必须装 SDK（不是 Runtime）。

### 步骤 1：克隆并配置环境变量

```bash
git clone <this-repo>
cd sts2_mod_agent
cp .env.example .env
```

编辑 `.env`。最少只需要填 LLM 的 key 就能跑，其它都是可调：

```env
# 必填（或在 UI /api/config 里填）
deepseek_api_key=

# 嵌入模型 & 向量库
EMBEDDING_MODEL=codefuse-ai/F2LLM-v2-0.6B   # 任意 HuggingFace feature-extraction 模型 id
EMBEDDING_BATCH_SIZE=16
MILVUS_URI=http://127.0.0.1:19530           # 留空走 Milvus Lite
# MILVUS_TOKEN=
MILVUS_DB_NAME=                             # 留空用默认库
DESC_COLLECTION_NAME=                       # 留空用默认 collection 名

# 服务端口
APP_HOST=127.0.0.1
APP_PORT=7870

# RAG 默认值（CLI flag 仍可覆盖）
DESC_TOP_K=4
CONTEXT_N=3
CODE_CHARS=2200

# 可选：HF token / 镜像
HF_TOKEN=
# HF_ENDPOINT=https://hf-mirror.com         # 国内网络解开注释

# 工具路径（用于「导出 / Godot」工作流，也可在前端「设置 → 工具路径」里改）
GAME_ROOT=D:\SteamLibrary\steamapps\common\Slay the Spire 2
EXPORT_TOOL_PATH=E:\JellyProject\sts2_mod_agent\tools\ExportMod.cmd
GODOT_TOOL_PATH=E:\path\to\Godot_v4.5.1-stable_mono_win64.exe
```

> `GAME_ROOT` 指向 STS2 本体安装目录；`EXPORT_TOOL_PATH` 是仓库自带的 `tools/ExportMod.cmd`（把 mod 打包导出给游戏加载）；`GODOT_TOOL_PATH` 指向你本地的 Godot Mono 可执行文件。三者都是绝对路径，不填则相关功能在前端会标灰不可用。

完整的 key 列表见 [.env.example](.env.example)。CLI flag（如 `--port 8000`）始终优先于 `.env`，`.env` 优先于代码里的硬编码默认值。

### 步骤 2：安装 Python 依赖

任选一种方式（**推荐 uv**：单一可执行文件、零配置、装得快）。

#### 方式 A · uv（推荐）

如果还没装 uv：

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或通过 pipx / pip
pipx install uv      # 或: pip install uv
```

然后在仓库根目录：

```bash
uv python install 3.14            # 一次性下载 Python 3.14
uv venv --python 3.14              # 创建 .venv/
# 激活：
#   Windows PowerShell: .venv\Scripts\Activate.ps1
#   Windows cmd:        .venv\Scripts\activate.bat
#   macOS / Linux:      source .venv/bin/activate

# 常规安装
uv pip install -r backend/requirements.txt

# 或：完全复现作者环境
uv pip install -r backend/requirements.lock.txt
```

#### 方式 B · conda + pip

```bash
conda create -n sts2_agent python=3.14 -y
conda activate sts2_agent

# 常规安装
pip install -r backend/requirements.txt

# 或：完全复现作者环境
pip install -r backend/requirements.lock.txt
```

> **CUDA 版 torch（两种方式通用）**：先按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 装好对应 CUDA 的 `torch==2.11.0`，再执行 `pip install -r backend/requirements.lock.txt`（或 `uv pip install ...`），pip / uv 会跳过已满足的 torch 版本。

### 步骤 3（可选）：启动独立 Milvus

如果 `milvus_lite` 在你的 Python 版本下没有预编译 wheel：

```bash
docker compose -f docker-compose.milvus.yml up -d
```

服务端口：`19530`（gRPC）、`9091`（health/UI）、`9001`（MinIO Console）。

### 步骤 4：构建向量库

```bash
# uv 用户：先激活 .venv（见步骤 2），或在命令前加 `uv run`
# conda 用户：conda activate sts2_agent
python backend/scripts/build.py --rebuild
```

会生成卡牌/技能主库 + 7 个本地化分库（relics / potions / orbs / enchantments / afflictions / rest_site_ui / events），输出到 `data/vector_db/`。

只想重建某一个：

```bash
python backend/scripts/build.py --target relics --rebuild --skip-preview
```

### 步骤 5：验证

```bash
python -c "import flask, langchain, langchain_openai, pymilvus, transformers, torch; print('OK')"
```

更详细的故障排查见 [backend/INSTALL.md](./backend/INSTALL.md) 与 [backend/MILVUS.md](./backend/MILVUS.md)。

---

## 运行

```bash
# uv 用户
.venv\Scripts\Activate.ps1            # Windows，macOS/Linux 用 source .venv/bin/activate
python backend/app.py --port 7870
# 或不激活：
uv run python backend/app.py --port 7870

# conda 用户
conda activate sts2_agent
python backend/app.py --port 7870
```

打开浏览器访问 <http://127.0.0.1:7870>。

| 路由 | 用途 |
| --- | --- |
| `/` | 项目首页 |
| `/make` | 对话式 Mod 生成（启用 Agent + RAG，自动写入 `mods/`） |
| `/query` | 纯检索：在向量库里搜原版资产/代码 |
| `/mod` | 浏览已有 Mod 的 C# 类、资产匹配情况、缺失资源 |

后端命令行参数（部分）：

```bash
python backend/app.py \
  --port 7870 \
  --desc-top-k 4 \
  --default-context-n 3 \
  --code-chars 2200
```

---

## 使用指南

1. **配置 LLM 提供商**：首页右上角设置里填入 DeepSeek（或任意 OpenAI 兼容端点）的 `base_url`、`api_key`、`model`，保存为默认。
2. **生成 Mod**：在 `/make` 输入需求（例如「做一张抽 2 张牌、回 4 血的卡」），勾选 **Agent + RAG**，Agent 会：
   - 调用 `rag_query` MCP 检索游戏内同类卡牌的实现与本地化；
   - 用 `local_file_read` / `local_file_write` / `local_file_replace` 写入 `mods/<your_mod>/` 下的 `.cs`、`.tres`、`localization/*.csv`；
   - 在前端实时回传 `agent_trace` 事件，可看到每一步工具调用、参数、返回值。
3. **检索原版资料**：在 `/query` 选择领域（卡牌描述 / 遗物 / 药水 …）、调整 `desc_top_k`、`context_n`，直接搜索。
4. **检查 Mod 状态**：`/mod` 列出每个 mod 的 C# 类与按 `data/settings/rules.json` 计算的资产路径匹配情况，红色为缺失。

### 推荐的对话提示词

第一轮直接用：

```
在mods中复制template并创建一个新项目叫<项目名>，
mod包含x个遗物,x个卡牌
```

### 用可视化工具补充细节（推荐）

1. 打开 **<https://sts2custom.shuimu.co.nz/>**，可视化填卡牌 / 遗物的数值、效果、本地化文本；
2. 「导出 JSON」下载配置文件；
3. 把 JSON **直接拖进 `/make` 聊天框**，弹出对话框为每段描述指派目标领域（cards / relics / powers / …）；
4. 先发上面的模板创建项目，再拖 JSON 补细节。

---

## 项目特点

### RAG 检索

围绕「自然语言描述 ⇄ 游戏源码」这条特殊映射手写的一套混合召回（[retrieval.py](backend/sts2_core/retrieval.py)），不直接用 LangChain 的现成 retriever。

- **描述当索引，代码当答案** — 索引文本是 `data/localization/{eng,zhs}/{cards,powers}.json` 抽出的 title + description + smartDescription，命中后通过 metadata 反向解析到对应 `.cs` 文件。
- **多领域分库** — cards / powers 主库 + relics / potions / orbs / enchantments / afflictions / rest_site_ui / events 各自独立 Milvus collection，按需过滤。
- **自实现 BM25 rerank** — 向量召回拉 1000 候选送 rerank，按「title 命中 > description 命中」分两档排序，向量距离仅做 tie-break，避免弱命中盖过强命中。
- **中英混合分词** — 英文 snake_case 拆词，中文走字符级 bigram，不依赖任何中文分词库。
- **配对引用扩展** — 命中卡时同时返回配对 Power + 所有引用 Power，反向亦然，给模型一个完整代码闭环防止瞎编 API。
- **Embedding 文本工程化** — 短描述按 `[原文, normalized, 原文, normalized]` 拼四份，补偿短文本在 vector space 的稀疏。
- **本地推理** — `codefuse-ai/F2LLM-v2-0.6B` + transformers feature-extraction，手写 mean pooling，RLock 保护并发。

### 描述拆分 + 领域路由

针对「STS2 的整段描述常跨多领域」「模型把整句送检索就塌陷」反复打磨出的工程化手段。

- **JSON 上传 → 按句切 → 指派 domain** — `POST /api/descriptions/split` 把上传 JSON 里所有 `*description*` 字段按中英文标点切成原子片段，前端弹框让用户为每段勾选目标领域，最终拼成 `[Relics] 拾起时` / `[Cards] 升级所有打击` 喂给 Agent，把一次模糊的全域检索拆成 N 次精准的「单段 × 单域」查询。
- **`rag_query` 强约束 prompt** — 工具描述明确禁止 `query` 含 `。，；、！？.,;!?\n\r`，含标点必须先切，每段一次调用、配一个 domain，并给了正反例。
- **fan-out 拦截** — 同一段 query 反复换 domain 时直接 skip，强制模型一次性传入所有需要的 domain，把 3-4 轮收敛到 1 轮。
- **领域别名 + 关键词推断兜底** — `card / cards / cardModel / CardModel` 都映射到 `cards`；模型忘传 `domains` 时按 query 关键词推断，避免落到 9 域全扫。
- **Top-3 命中自动回读** — `rag_query` 返回前同步读取前 3 个 `.cs` 源文件（截 8 KB）塞进结果，并告诉 Agent「别再 `read_file` 重复一遍」，把「先检索再读文件」两轮压成一轮。

### Agent 主循环稳定性

[backend/app.py](backend/app.py) 中的通用稳定性优化。

- **重复调用去重** — `(tool, args)` 精确去重，命中时把首次结果作为 `prior_result` 回灌给 LLM，避免「验证刚才的结果」。
- **`mods/` 子树豁免去重** — 生成的 mod 文件是活的，Agent 经常写完读、读完覆盖，对 `mods/` 内调用不去重；`data/` 等参考树仍严格去重。
- **滑动窗口裁剪** — 每步循环开头裁掉过期 `ToolMessage`（保留最近 8 条），防止 prompt 随步数线性膨胀。
- **只读熔断** — 连续 3 步只读工具时注入提示，强制写文件或给出最终答复。
- **预算兜底** — 达到 `stream_max_steps` 仍无最终答时追加一轮禁工具的 wrap-up，保证用户拿到回复。
- **完成度检测** — 检查文件是否落地、本地化是否追加，未完成时生成针对性追问而非泛泛「继续」。
- **未解析 tool_call 回灌** — 给每个未消费的 `tool_call_id` 补一条带错误说明的 `ToolMessage`，保证消息序列对 OpenAI/DeepSeek API 合法。

### 工具与基础设施

- **Agent + MCP 工具链** — LLM 调用 `local_file_read / write / replace / search / list / read_many / copy_tree / create_dir / rag_query`，写入沙箱约束在 `mods/` 内。
- **流式可观测** — `/api/chat/stream` NDJSON 实时下发 `retrieval_start / retrieval_done / generation_start / token / reasoning_content / agent_trace / memory_updated / done`，前端逐 token 渲染并展示工具调用面板。
- **会话记忆** — 每轮生成 `memory_summary` 随下次请求回传，无需手动复述上下文。
- **资产规则校验** — `data/settings/rules.json` 描述每类对象的资产路径，`/mod` 页面按 snake_case 类名自动核对缺失项并标红。
- **请求级日志** — `ai_chat_log` 记录每次会话的 traces / reasoning / memory before-after / duration，方便回放调参。
- **本地优先 + 一体化托管** — 默认 Milvus Lite + 本地 embedding 推理；Flask 同时托管 `frontend/` 与 `/api/*`，无需另起前端进程。

---

## License

仓库内 Mod 子项目分别遵循各自的 license；后端代码采用 MIT。
