"""
自建 RAG 配置模块

集中管理知识库根目录、索引存储目录、embedding 端点、向量库（Qdrant）与
关键词库（Elasticsearch）配置。
embedding / Qdrant / ES 都是可插拔的：未配置 EMBEDDING_*、Qdrant 不可达或
ES 不可达时，检索按 ADR-006 / ADR-007 / ADR-008 的降级链回退，系统保持可用。
（本项目 LLM 端点没有 /v1/embeddings，实测 404，故 embedding 必须独立配置。）
"""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# 当前文件位于 app/rag/config.py，parents[2] 即项目根目录
PROJECT_ROOT = Path(__file__).parents[2].resolve()

# 知识库根目录：每个子目录 = 一个知识库（如 电商行业/、金融行业/）
KNOWLEDGE_BASE_ROOT = PROJECT_ROOT / "docs" / "knowledge_base"

# 原文索引目录：每个知识库一个子目录，含 chunks.jsonl（chunk 原文）与 meta.json（索引模式）
# 向量不在这里，存放在 Qdrant（见 ADR-007）
INDEX_ROOT = PROJECT_ROOT / "app" / "rag" / "indexes"

# 切分参数：面向中文报告类文档的经验值
# 上限由 embedding 模型决定：bge-large-zh-v1.5 的最大输入是 512 token，中文约 1 token/字，
# 因此目标块长度压到 400 字符以内留足余量（600 字符实测会被 TEI 以 413 拒绝）
CHUNK_SIZE = 400          # 目标块长度（字符），中文约 200~300 token
CHUNK_OVERLAP = 80         # 相邻块重叠字符数，保证跨块语义不断裂

# embedding 单条输入字符数硬上限：防御性裁剪，防止将来调整 CHUNK_SIZE 或遇到
# 英文/表格密集内容时超出模型 token 上限（TEI 超限直接返回 413 而非截断）
EMBEDDING_MAX_CHARS = 480

# 检索参数
DEFAULT_TOP_K = 5          # 工具默认返回片段数

# embedding 端点（OpenAI 兼容 /embeddings 协议）。三项齐全才启用向量检索；
# 缺任一项时检索层自动降级为纯 BM25，并通过 meta.json 记录当前索引模式。
# 推荐后端：本地 TEI 服务（HuggingFace text-embeddings-inference）加载 bge-large-zh-v1.5，
# 它同时提供原生 /embed 与 OpenAI 兼容的 /v1/embeddings，后者可直接对接本模块。
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "").rstrip("/")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")
EMBEDDING_ENABLED = bool(EMBEDDING_BASE_URL and EMBEDDING_API_KEY and EMBEDDING_MODEL)

# 每次 embedding 请求的最大文本条数：分批调用，避免单请求过大被网关拒绝。
# CPU 推理下批越小单请求越快——8 条是本机实测的折中（16 条时单批会顶到超时线）
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))

# 单次 embedding 请求超时（秒）：CPU 推理本身较慢，加上开发机可能存在 HTTP 代理，
# 批请求耗时波动大，30 秒会误判超时导致摄入半途失败（实测踩到过）
EMBEDDING_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT", "120"))

# 向量库（Qdrant）：见 ADR-007
# 只负责 chunk 向量的持久化与相似度检索；原文仍保存在本地 chunks.jsonl（BM25 路依赖它），
# 两路通过 chunk 在 jsonl 中的行号对齐。
# 未配置 embedding 或 Qdrant 不可达时，检索层降级为纯 BM25，不影响任务执行。
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
# 单 collection + payload 的 kb 字段过滤：新增知识库不必建新表
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "deepsearch_kb")
# 连接与查询超时（秒）：向量库不可达时快速失败并降级，不让检索卡住
QDRANT_TIMEOUT = float(os.getenv("QDRANT_TIMEOUT", "5"))

# 关键词检索（Elasticsearch）：见 ADR-008
# 承接双路检索里的「关键词路」——IK 中文分词 + 倒排索引 + BM25 打分（ES 7 起默认打分器）。
# 与 Qdrant 对称：单索引 + kb 字段过滤，seq 与 chunks.jsonl 行号对齐。
# ES 不可达时检索层回退到进程内的 rank_bm25 实现，知识库能力不消失（三级降级）。
ES_URL = os.getenv("ES_URL", "http://localhost:9200").rstrip("/")
ES_USERNAME = os.getenv("ES_USERNAME", "")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")
# 单索引 + kb 字段过滤（与 Qdrant 的建模方式保持一致）
ES_INDEX = os.getenv("ES_INDEX", "deepsearch_kb")
# 连接与查询超时（秒）：ES 不可达时快速失败并回退本地 BM25
ES_TIMEOUT = float(os.getenv("ES_TIMEOUT", "5"))
# 是否启用 ES 关键词路：置 0 可强制走进程内 rank_bm25（离线演示 / 评测对照用）
ES_ENABLED = os.getenv("ES_ENABLED", "1").strip().lower() not in {"0", "false", "no"}


def list_knowledge_base_names() -> list[str]:
    """
    列出所有可用知识库名称（KNOWLEDGE_BASE_ROOT 下的直接子目录）

    目录即知识库，不引入额外的注册表：新增知识库 = 新建目录 + 放入文档 + 重新摄入。
    """
    if not KNOWLEDGE_BASE_ROOT.exists():
        return []
    return sorted(p.name for p in KNOWLEDGE_BASE_ROOT.iterdir() if p.is_dir())
