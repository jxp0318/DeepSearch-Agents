"""
自建 RAG 配置模块

集中管理知识库根目录、索引存储目录与 embedding 端点配置。
embedding 是可插拔的：未配置 EMBEDDING_* 时，检索自动降级为纯 BM25，
系统保持可用（当前 LLM 端点没有 /v1/embeddings，实测 404）。
"""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# 当前文件位于 app/rag/config.py，parents[2] 即项目根目录
PROJECT_ROOT = Path(__file__).parents[2].resolve()

# 知识库根目录：每个子目录 = 一个知识库（如 电商行业/、金融行业/）
KNOWLEDGE_BASE_ROOT = PROJECT_ROOT / "docs" / "knowledge_base"

# 索引持久化目录：每个知识库一个子目录，含 chunks.jsonl / vectors.npy / meta.json
INDEX_ROOT = PROJECT_ROOT / "app" / "rag" / "indexes"

# 切分参数：面向中文报告类文档的经验值
CHUNK_SIZE = 600          # 目标块长度（字符），中文约 300~400 token
CHUNK_OVERLAP = 100        # 相邻块重叠字符数，保证跨块语义不断裂

# 检索参数
DEFAULT_TOP_K = 5          # 工具默认返回片段数

# embedding 端点（OpenAI 兼容 /embeddings 协议）。三项齐全才启用向量检索；
# 缺任一项时检索层自动降级为纯 BM25，并通过 meta.json 记录当前索引模式。
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "").rstrip("/")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")
EMBEDDING_ENABLED = bool(EMBEDDING_BASE_URL and EMBEDDING_API_KEY and EMBEDDING_MODEL)

# 每次 embedding 请求的最大文本条数：分批调用，避免单请求过大被网关拒绝
EMBEDDING_BATCH_SIZE = 16


def list_knowledge_base_names() -> list[str]:
    """
    列出所有可用知识库名称（KNOWLEDGE_BASE_ROOT 下的直接子目录）

    目录即知识库，不引入额外的注册表：新增知识库 = 新建目录 + 放入文档 + 重新摄入。
    """
    if not KNOWLEDGE_BASE_ROOT.exists():
        return []
    return sorted(p.name for p in KNOWLEDGE_BASE_ROOT.iterdir() if p.is_dir())
