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
#
# 可被环境变量 RAG_INDEX_ROOT 覆盖：做**解析器 A/B 对照**时把对照臂的索引写到独立目录，
# 生产索引保持不动（对照臂用不同的解析器与切块结果，混进同一目录会互相覆盖）。
INDEX_ROOT = Path(os.getenv("RAG_INDEX_ROOT") or (PROJECT_ROOT / "app" / "rag" / "indexes"))

# 后端（Qdrant / ES）的库名后缀：Qdrant 是「单 collection + payload.kb 过滤」、
# ES 是「单索引 + 字段 kb 过滤」，两侧都靠 kb 值隔离库。对照臂若沿用同一 kb 值，
# 两臂的向量与关键词点会混在同一个库里互相污染，故用后缀把它们分开。
KB_BACKEND_SUFFIX = os.getenv("RAG_KB_SUFFIX", "").strip()

# 目标块长度：报告类中文文档的经验值。原为适配 bge-large-zh-v1.5 的 512 token 上限而定
# （中文约 1 token/字，600 字符实测会被 TEI 以 413 拒绝）；现役 bge-m3 上限 8192、已非约束，
# 但保持 400 不变，以免改动既有 chunk 切分与评测题库的 gold 口径
CHUNK_SIZE = 400          # 目标块长度（字符），中文约 200~300 token
CHUNK_OVERLAP = 80         # 相邻块重叠字符数，保证跨块语义不断裂

# embedding 单条输入字符数硬上限：防御性裁剪，防止将来调整 CHUNK_SIZE 或遇到
# 英文/表格密集内容时超出模型 token 上限（TEI 超限直接返回 413 而非截断）
EMBEDDING_MAX_CHARS = 480

# 检索参数
DEFAULT_TOP_K = 5          # 工具默认返回片段数

# 索引 schema 版本：chunk 字段结构变更时递增，旧索引会被自动转为全量重建（见 ingest）。
# v2 新增 lang / cjk_ratio / zh_text 三个字段（字段级语言路由，见 ADR-010）
# v3 新增 page_index / page_label / section_path / chunk_type / table / content_hash /
#    parser_version / needs_ocr；同时 PDF 的 block_index 语义从「页码」升级为「节序号」
#    ——这是一次不兼容变更，v2 索引必须整库重建（见 docs/pdf-parsing-plan.md §3）
INDEX_SCHEMA_VERSION = 3

# PDF 解析器选择（⑦ 切换开关）：
#   layout  —— 自研版面解析链路（app/rag/pdf_parser.py：分栏/页眉页脚/页码双轨/标题多信号/
#              表格独立成块/按节切块），产出 schema v3 chunk，是**当前生产默认**
#   chunker —— 旧链路（pypdf 按页提取 + chunker.chunk_text），保留作回退与 A/B 对照
# 只影响 PDF。
PDF_PARSER = os.getenv("PDF_PARSER", "layout").strip().lower()

# 非 PDF 文档（docx / md / txt / pptx / html）解析器选择：
#   struct  —— 统一结构化解析链路（app/rag/struct_parser.py）：各格式先转成统一中间
#              表示（标题 / 正文 / 表格），再按与 PDF 完全相同的契约产出 schema v3
#              chunk——section_path / chunk_type / table 都是**真值**。**当前默认**。
#   chunker —— 旧链路（chunker.extract_document + chunk_text）：非 PDF 的结构字段只能
#              补中性默认值（section_path=[] / chunk_type 恒为 body / table=None），
#              且 docx 的表格（document.tables）**不会被读取**——表格内容会整块丢失。
#              保留作回退与 A/B 对照。
STRUCT_PARSER = os.getenv("STRUCT_PARSER", "struct").strip().lower()

# 自研 PDF 解析器版本号：写进每个 chunk 的 parser_version，便于回溯与 A/B 对比
# 0.4.0 新增无框线多列表的「列网格重建」（pdf_layout.table_grid_lines，带三重门禁）
PARSER_VERSION = "layout-0.4.0"

# 统一结构化解析器版本号（非 PDF 文档）：同样是 chunk 级 parser_version，
# 与 PDF 的 PARSER_VERSION 区分开，便于观测「这份块是哪条链路产的」
STRUCT_PARSER_VERSION = "struct-0.1.0"

# 语言判定阈值（见 app/rag/language.py 与 ADR-010）。均为**先验值、未做校准实验**，
# 设计目标是「不误杀」而非最优：真正的隔离由 zh_text 字段完成，阈值只管路由决策。
LANG_DOC_ZH_MIN_RATIO = 0.20      # 文档级：全文汉字占比 ≥ 此值判中文
LANG_QUERY_ZH_MIN_RATIO = 0.30    # 查询级：≥ 此值判中文，其余判中英混排

# embedding 端点（OpenAI 兼容 /embeddings 协议）。三项齐全才启用向量检索；
# 缺任一项时检索层自动降级为纯 BM25，并通过 meta.json 记录当前索引模式。
# 推荐后端：本地 TEI 服务（HuggingFace text-embeddings-inference，**GPU 版镜像**）加载 bge-m3，
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

# 精排（cross-encoder）——见 ADR-012（接入生产）/ ADR-013（候选深度定为 25）
# 把 RRF 融合后的候选池用 cross-encoder 逐对打分重排，是检索侧最有效的杠杆：
# 端到端实测 Recall@5 59.3% → 87.0%（每路候选 25 条），跨语言组提升最大。
#
# 推理走**进程内 ONNX + onnxruntime**（不走 HTTP、不走容器、不用 TEI）——这一步同时
# 消掉了 TEI 的两个遗留问题：queue_time 排队与 candle 后端的 SIGSEGV 崩溃。
# 模型目录里的 onnx/model.onnx 由 optimum-cli 从 model.safetensors 导出。
#
# 三态开关：auto（默认，仅当 CUDA 执行提供器真正可用时启用）/ 1 强制启用（允许 CPU，
# 慢但可用）/ 0 关闭。auto 之所以要求 CUDA，是为了避免在纯 CPU 机器上把检索拖到秒级。
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "auto").strip().lower()
RERANK_DISABLED = RERANK_ENABLED in {"0", "false", "no", "off"}
RERANK_FORCED = RERANK_ENABLED in {"1", "true", "yes", "on"}

# 模型目录（含 model.safetensors 与 onnx/ 子目录）。缺省在项目内，约 2.2GB，不入 git。
RERANK_MODEL_DIR = Path(
    os.getenv("RERANK_MODEL_DIR") or (PROJECT_ROOT / "docker" / "reranker" / "bge-reranker-v2-m3")
)
# 执行提供器偏好：auto（CUDA 优先、CPU 兜底）| cuda（只要 CUDA）| cpu
RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "auto").strip().lower()
# ONNX 权重子目录：auto（默认，有 onnx_fp16/ 就用它、否则回退 onnx/）| onnx | onnx_fp16
#
# fp16 权重由 docker/reranker/export_onnx_fp16.py 从 fp32 的 onnx/ 转换而来（权重减半、
# Ada 上 fp16 吞吐翻倍）。2026-10-13 端到端 A/B（60 查询，K=25 每路）：精排 P50
# 592ms → 295ms、端到端 P50 707ms → 402ms，而 ★Rec@5 / NDCG@5 / AP@5 / MRR@5
# 与 ▲逐条 60 条查询**完全相同**（0/60 变化），logits 最大绝对差 0.0057。
# 因此默认走 auto：有 fp16 就用（快一倍），没有就静默用 fp32（不报错、不降级）。
# 想强制某一种时显式设 RERANK_ONNX_SUBDIR=onnx 或 onnx_fp16。
RERANK_ONNX_SUBDIR = os.getenv("RERANK_ONNX_SUBDIR", "auto").strip() or "auto"
# 单次前向的候选批大小
#
# ⚠️ 反直觉但已实测（2026-10-13，RTX 4060 Laptop，app/eval/bench_rerank.py）：**批越小越快**。
# K=25 时候选中位耗时 bs=4 → 330ms、bs=8 → 363ms、bs=16 → 447ms、bs=25 → 493ms。
# 原因是 cross-encoder 是 transformer，单序列注意力 O(L²)，而 `_encode` 要把一批里
# 所有序列补齐到**该批最长**——长表格块会把同批的短正文块一起抬到 260 token。
# 因此 (1) 批不宜大，(2) 更要紧的是配合 RERANK_LENGTH_SORTED 按长度排序切批。
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "8"))
# 切批前是否按序列长度升序排序（见 reranker._batch_plan）。这是上面那条实测的直接产物：
# 排序后每批内长度接近，padding 浪费最小。置 False 可复现「原序切批」的对照臂。
# 它**不改变分数与排名**（逐对打分，与计算顺序无关），只改变同一批里的邻居。
RERANK_LENGTH_SORTED = os.getenv("RERANK_LENGTH_SORTED", "1").strip().lower() not in {
    "0", "false", "no", "off",
}
# 精排用的每路候选深度（0 = 沿用 top_k×3）。生产取 25：见 ADR-013 深池重测——
# 15→25 用 +0.16s 换 +2.7pt Recall@5，是性价比拐点；再往上耗时近似线性增长
# （精排成本对候选数线性、单候选 ≈14ms 恒定，GPU 也不例外），而质量只多 1~2pt。
# 设 0 退回默认 15，设 40/100 可复现深池曲线（报告 app/eval/results/2026-10-10__rerank-deep-pool.md）。
RERANK_CANDIDATE_K = int(os.getenv("RERANK_CANDIDATE_K", "25"))
# query+chunk 拼接后的 token 上限（XLM-R 硬上限 512）；chunk 已控制在 400 字符内，
# 正常不会触发截断，留作防御
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "512"))

# 扫描页 OCR（C4 的实现端）——见 app/rag/ocr.py
# 只对 detect_scanned_pages 检出 needs_ocr 的页做 OCR，把架构图 / 信息图 / 位图表格里
# **提取不到**的文字补回检索链路（实测样本：具身智能 p15 架构图，PyMuPDF 仅提出 38 字符）。
# 引擎用 rapidocr-onnxruntime（PP-OCR 系列，纯 onnx，模型随 wheel 内置、离线可用）。
# 三态：auto（默认，装了引擎才启用）/ 1 强制尝试 / 0 关闭。引擎缺失时静默降级，
# 扫描页仍带 needs_ocr 标记入库，不影响整体摄入。
OCR_ENABLED = os.getenv("OCR_ENABLED", "auto").strip().lower()
OCR_DISABLED = OCR_ENABLED in {"0", "false", "no", "off"}
# 渲染 dpi：识别率与耗时的折中。150 实测单页 ~3s、A4 → 1240×1754；
# 提到 200 耗时 +25% 而识别行数几乎不变（位图表格 79 → 78 行），故默认 150。
OCR_DPI = int(os.getenv("OCR_DPI", "150"))
# 行置信度下限：低于此值的行丢弃，滤掉水印 / 装饰噪声
OCR_MIN_SCORE = float(os.getenv("OCR_MIN_SCORE", "0.5"))


def rerank_onnx_subdir() -> str:
    """
    解析 ``RERANK_ONNX_SUBDIR``：``auto`` 时优先 fp16，没有该目录则回退 fp32

    解析结果同时用于「模型文件是否存在」的前置检查与建模，所以两条路天然一致——
    不会出现「检查 fp32、加载 fp16」这种错位。
    """
    if RERANK_ONNX_SUBDIR != "auto":
        return RERANK_ONNX_SUBDIR
    if (RERANK_MODEL_DIR / "onnx_fp16" / "model.onnx").exists():
        return "onnx_fp16"
    return "onnx"


def backend_kb(kb_name: str) -> str:
    """
    把「本地索引目录名」映射为「Qdrant / ES 里的 kb 值」

    生产环境后缀为空，映射是恒等变换；做解析器对照时设 ``RAG_KB_SUFFIX``（如
    ``__docling``），两臂在同一 collection / 索引里各占一份互不可见的 kb 分区。
    """
    return f"{kb_name}{KB_BACKEND_SUFFIX}"


def list_knowledge_base_names() -> list[str]:
    """
    列出所有可用知识库名称（KNOWLEDGE_BASE_ROOT 下的直接子目录）

    目录即知识库，不引入额外的注册表：新增知识库 = 新建目录 + 放入文档 + 重新摄入。
    """
    if not KNOWLEDGE_BASE_ROOT.exists():
        return []
    return sorted(p.name for p in KNOWLEDGE_BASE_ROOT.iterdir() if p.is_dir())
