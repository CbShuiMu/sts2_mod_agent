# sts2_agent 后端环境安装指南

## 环境要求

- **Python**: 3.14（已在该版本下验证；3.11+ 也大概率可用，但 lock 文件未在更低版本测试过）
- **包管理器**: 推荐 conda + pip 组合（与本仓库使用的 `sts2_agent` conda 环境一致）
- **OS**: 在 Windows 11 上开发；Linux/macOS 同样支持

## 两种安装方式

### 方式 A：常规安装（推荐日常开发）

只装直接依赖，pip 会自动解析传递依赖到当前可用的最新兼容版。适合首次搭建或 Python 版本不完全一致的情况。

```bash
conda create -n sts2_agent python=3.14 -y
conda activate sts2_agent
pip install -r backend/requirements.txt
```

### 方式 B：锁定安装（推荐 CI / 复现 bug）

使用 `requirements.lock.txt` 一比一复现作者本机的依赖版本。

```bash
conda create -n sts2_agent python=3.14 -y
conda activate sts2_agent
pip install -r backend/requirements.lock.txt
```

## 平台相关注意事项

### 1. PyTorch (`torch==2.11.0`)

`requirements*.txt` 默认从 PyPI 安装 CPU 版 torch。如需 GPU/CUDA 版本，**先**按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选好对应 CUDA 版本安装：

```bash
# 例：CUDA 12.4
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu124
# 然后再执行
pip install -r backend/requirements.lock.txt
```

pip 会跳过已满足版本的 torch，只补齐其它依赖。

### 2. Milvus / `pymilvus[milvus_lite]`

代码默认走 Milvus Lite（本地 `.db` 文件），但 `milvus_lite` 不一定有适配 Python 3.14 的预编译 wheel。运行时如果出现：

```
Milvus Lite is required for local .db storage, but it is not available...
```

两种解决方法二选一：

- **降到 Python 3.11/3.12** 重新创建 conda 环境（`milvus_lite` 在这些版本上有官方 wheel）；
- **使用独立 Milvus 服务**：仓库根目录已有 `docker-compose.milvus.yml`，运行后在 `.env` 写入：
  ```
  MILVUS_URI=http://127.0.0.1:19530
  ```

### 3. `transformers` + 模型权重

首次运行 embedding 时，`transformers` 会从 HuggingFace 拉取默认模型 `codefuse-ai/F2LLM-v2-0.6B`。国内网络可设置镜像：

```bash
# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
# bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 更新 lock 文件

修改了 `requirements.txt` 后，重新生成锁定文件：

```bash
conda activate sts2_agent
pip install -r backend/requirements.txt --upgrade
pip freeze > backend/requirements.lock.txt
```

记得手动把文件顶部的注释（Python 版本、生成时间）改回来。

## 验证安装

```bash
conda activate sts2_agent
python -c "import flask, langchain, langchain_openai, pymilvus, transformers, torch; print('OK')"
```

## 启动后端

```bash
conda activate sts2_agent
python backend/app.py
```
