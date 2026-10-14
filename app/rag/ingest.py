"""
知识库摄入模块

流水线：扫描知识库目录 → 提取 → 切分 → 落盘原文 → embedding 写 Qdrant → 写 ES 关键词索引。
索引按知识库组织，三处存储（见 ADR-007 / ADR-008）：

    app/rag/indexes/<kb_name>/chunks.jsonl   每个 chunk 一行 JSON
                                             （id / doc / heading / text / lang / cjk_ratio / zh_text
                                              + page_index / page_label / section_path / chunk_type /
                                              table / content_hash / parser_version / needs_ocr；
                                              chunk_type ∈ {body, table, ocr}，其中 ocr = 扫描页 OCR 产物）
    app/rag/indexes/<kb_name>/meta.json      文件哈希清单、索引模式、文档级 page_map，用于增量摄入与降级标记
    Qdrant collection (payload.kb = <kb_name>)   chunk 向量
    ES 索引 (字段 kb = <kb_name>)                chunk 原文与元数据（IK 分词倒排索引）

三处通过 chunk 在 chunks.jsonl 中的行号（seq）对齐，检索时在应用层做 RRF 融合。
chunks.jsonl 始终是原文权威副本：embedding 未配置则不写向量，ES 未启动则不写关键词索引，
两者都只是可选增强——进程内 rank_bm25 会兜底关键词路，知识库能力不消失。

增量策略：按文件 sha256 跳过未变化的文档，只重摄入新增或修改的文件；
索引 schema 版本变化（meta.schema 对不上）时强制全量重建，保证新增字段不会漏补。

语言字段：每份文档在提取后做一次**文档级**语言判定，把 lang / cjk_ratio 盖到该文档
的所有 chunk 上，并由 lang 派生 zh_text（中文文档=原文，英文文档=空）——这是检索侧
字段级语言路由的摄入期准备，见 ADR-010。

PDF 解析器选择（⑦ 切换）：`config.PDF_PARSER == "layout"`（默认）时 PDF 走自研版面解析
链路（`app/rag/pdf_parser.py`，见 docs/pdf-parsing-plan.md ③~⑥），产出 schema v3 全字段
chunk；设为 `"chunker"` 则回退旧的 pypdf 按页提取链路。

非 PDF 解析器选择：`config.STRUCT_PARSER == "struct"`（默认）时 docx / md / txt / pptx /
html 走统一结构化解析链路（`app/rag/struct_parser.py`）——各格式先转成统一的中间表示
（标题 / 正文 / 表格），再按**与 PDF 相同的契约**产出 schema v3 全字段 chunk：
`section_path` / `chunk_type` / `table` 都是真值，`page_index` / `page_label` 留空
（这些格式没有"页"的概念）。设为 `"chunker"` 则回退旧提取链路，并由 `_v3_defaults`
补上中性默认值——注意旧链路**不读 docx 的 `document.tables`，表格内容会整块丢失**。
"""

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from app.rag import config
from app.rag.chunker import SUPPORTED_SUFFIXES, TextBlock, chunk_text, extract_document
from app.rag.keyword_store import ElasticsearchUnavailable, keyword_store
from app.rag.language import classify_document
from app.rag.vector_store import QdrantUnavailable, vector_store

logger = logging.getLogger(__name__)


def _file_sha256(file_path: Path) -> str:
    """计算文件内容哈希，作为增量摄入的变更判断依据"""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _embed_texts(texts: list[str]) -> np.ndarray:
    """
    调用 OpenAI 兼容的 /embeddings 接口，把文本批量转向量

    分批请求避免单请求过大；失败时重试 2 次（embedding 是摄入期一次性成本，
    值得重试到成功，避免索引半途而废）。
    :param texts: 待向量化的文本列表
    :return: (N, dim) float32 矩阵
    """
    import requests

    # 防御性裁剪：TEI 对超过模型 token 上限的输入直接返回 413 而不是截断，
    # 这里按字符数上限先裁一刀，保证任何配置下都不会因单条过长中断整批摄入
    texts = [t[: config.EMBEDDING_MAX_CHARS] for t in texts]

    # 本地 embedding 服务一律直连、不走系统代理：requests 默认会读取环境变量里的
    # HTTP_PROXY（开发机/CI 常注入代理），把 localhost 请求转发给代理会带来额外延迟，
    # 并在大批量摄入时造成间歇性读超时（实测：报错里的连接端口是代理端口而非 8081）。
    session = requests.Session()
    session.trust_env = False

    headers = {"Authorization": f"Bearer {config.EMBEDDING_API_KEY}"}
    all_vectors: list[list[float]] = []

    for start in range(0, len(texts), config.EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + config.EMBEDDING_BATCH_SIZE]
        payload = {
            "model": config.EMBEDDING_MODEL,
            "input": batch,
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = session.post(
                    f"{config.EMBEDDING_BASE_URL}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=config.EMBEDDING_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                # 按 index 归位，防止服务端乱序返回
                batch_vectors = [None] * len(batch)
                for item in data:
                    batch_vectors[item["index"]] = item["embedding"]
                all_vectors.extend(batch_vectors)
                last_error = None
                break
            except Exception as e:  # noqa: BLE001 —— 摄入期重试，最后一次仍失败才上抛
                last_error = e
                time.sleep(1.5 * (attempt + 1))
        if last_error is not None:
            raise RuntimeError(f"embedding 请求失败（已重试 3 次）: {last_error}")

    return np.array(all_vectors, dtype=np.float32)


def _pdf_text_for_language(pdf_path: Path) -> str:
    """
    快速取 PDF 全文纯文本，**仅用于文档级语言判定**

    为什么不复用解析器产物：自研链路需要 ``doc_lang`` 作为输入（英文文档才启用
    「全大写」标题信号），而语言判定又需要全文——形成先有鸡还是先有蛋。这里用
    PyMuPDF 的纯文本提取（不做 span 聚类 / 线框解析）先跑一遍拿到全文，成本远低于
    完整解析，摄入期一次性开销可忽略。
    """
    import pymupdf

    document = pymupdf.open(str(pdf_path))
    try:
        return "\n".join(page.get_text() for page in document)
    finally:
        document.close()


# 统一结构化解析器产出的 chunk 用 config.STRUCT_PARSER_VERSION（struct-*）；
# 这里的版本号只给**回退路径**（PDF_PARSER != layout 或 STRUCT_PARSER=chunker）用
_CHUNKER_PARSER_VERSION = "chunker-1.0.0"


def _v3_defaults(chunk: dict) -> dict:
    """
    给**回退链路**的 chunk 补齐 schema v3 字段的**中性默认值**（就地修改并返回）

    默认链路的两个解析器（pdf_parser / struct_parser）产出的这些字段都是真值；
    补默认值只为保证 chunks.jsonl 内字段齐整，下游读 ``page_index`` /
    ``section_path`` 时不必到处 ``.get`` 兜底。
    """
    chunk.setdefault("page_index", None)
    chunk.setdefault("page_label", None)
    chunk.setdefault("section_path", [])
    chunk.setdefault("chunk_type", "body")
    chunk.setdefault("table", None)
    chunk.setdefault("parser_version", _CHUNKER_PARSER_VERSION)
    chunk.setdefault("needs_ocr", False)
    if not chunk.get("content_hash"):
        chunk["content_hash"] = hashlib.sha256(
            chunk.get("text", "").encode("utf-8")
        ).hexdigest()[:16]
    return chunk


def ingest_knowledge_base(kb_name: str, force: bool = False) -> dict:
    """
    摄入单个知识库：提取 → 切分 → 落盘原文 → 写向量库(Qdrant) + 写关键词索引(ES)

    :param kb_name: 知识库目录名（KNOWLEDGE_BASE_ROOT 下的直接子目录）
    :param force: True 时忽略文件哈希全量重建索引
    :return: 摄入摘要 dict（文件数、chunk 数、是否启用向量/关键词索引、耗时）
    """
    kb_dir = config.KNOWLEDGE_BASE_ROOT / kb_name
    if not kb_dir.is_dir():
        raise FileNotFoundError(f"知识库不存在: {kb_name}")

    index_dir = config.INDEX_ROOT / kb_name
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path = index_dir / "meta.json"

    # 读取上次的文件哈希清单；force 或首次摄入时视为空
    old_files: dict[str, str] = {}
    old_doc_langs: dict[str, dict] = {}
    old_page_maps: dict[str, dict] = {}
    old_doc_parsers: dict[str, str] = {}
    if meta_path.exists() and not force:
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        old_files = old_meta.get("files", {})
        old_doc_langs = old_meta.get("doc_langs", {})
        old_page_maps = old_meta.get("page_maps", {})
        old_doc_parsers = old_meta.get("doc_parsers", {})
        # 索引 schema 升级（如 v2 新增 lang / cjk_ratio / zh_text）时旧索引字段不全，
        # 而文件内容哈希没变——仅凭哈希会判成「无变化」，新字段永远补不上，
        # 检索侧的字段路由就会拿不到 zh_text 而静默失效。故直接转全量重建。
        if old_meta.get("schema") != config.INDEX_SCHEMA_VERSION:
            logger.info(
                "检测到索引 schema 变更（%s → %s），本次改为全量重建",
                old_meta.get("schema"),
                config.INDEX_SCHEMA_VERSION,
            )
            force = True

    # 增量跳过前先校验后端数据完整性：文档没变、但某个后端（Qdrant / ES）的文档数
    # 与本地原文索引对不上时，说明上次摄入半途失败或后端数据卷被清空——此时仅凭文件
    # 哈希会误判为「无变化」，导致后端永远补不齐。改为走全量重建路径。
    if not force and _backend_data_missing(kb_name, index_dir):
        logger.info("检测到 %s 的后端索引缺失或不完整，本次改为全量重建", kb_name)
        force = True

    # 扫描支持格式的文档，按哈希判断哪些需要（重）摄入
    doc_files = sorted(
        p for p in kb_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not doc_files:
        raise FileNotFoundError(f"知识库 {kb_name} 中没有可摄入的文档: {kb_dir}")

    # 先算一遍各文件哈希：既要判断增量，也要在发现文档被修改/删除时提前转全量
    current_hashes = {str(p.relative_to(kb_dir)): _file_sha256(p) for p in doc_files}

    # 文档被修改或删除时无法安全增量：修改会留下旧 chunk 对应的孤儿向量/ES 文档，
    # 删除更是无从按增量清理。这两种情况一律转全量重建（规模小，代价可接受）；
    # 只有「纯新增文档」才走增量追加路径。
    if not force:
        modified = {n for n, h in current_hashes.items() if n in old_files and old_files[n] != h}
        deleted = set(old_files) - set(current_hashes)
        if modified or deleted:
            logger.info(
                "检测到文档变更或删除（%s），本次改为全量重建以保证索引一致",
                "、".join(sorted(modified | deleted)),
            )
            force = True

    unchanged_files: dict[str, str] = {}
    doc_langs: dict[str, dict] = {}  # 每份文档的语言判定结果，落 meta 便于观测与调阈值
    doc_page_maps: dict[str, dict[str, str]] = {}  # 每份文档的 page_index → page_label（C5）
    doc_parsers: dict[str, str] = {}  # 每份文档实际使用的解析器版本（三条链路各不相同）
    chunks: list[dict] = []
    for doc_path in doc_files:
        rel_name = str(doc_path.relative_to(kb_dir))
        file_hash = current_hashes[rel_name]
        if not force and rel_name in old_files and old_files[rel_name] == file_hash:
            unchanged_files[rel_name] = file_hash
            continue

        # 单份文档解析失败**不应中断整库摄入**：解析器依赖第三方库（PyMuPDF / python-docx /
        # python-pptx / python-ocr），一份损坏的文件不该让其余文档都入不了库。失败时记录
        # 错误并**跳过该文档**——它不进 meta.files，下次摄入仍会被重试；解析中途已写入的
        # 文档级状态（语言 / 页码映射 / 解析器版本）一并回滚，避免落成幽灵条目。
        try:
            suffix = doc_path.suffix.lower()
            if suffix == ".pdf" and config.PDF_PARSER == "layout":
                # ---- 自研版面解析链路（⑥ 组装 + ⑦ 切换）：产出 schema v3 全字段 chunk ----
                from app.rag.pdf_parser import parse_document

                # 语言判定在**文档级**做（见 app/rag/language.py）：先用纯文本提取拿全文，
                # 再作为解析器输入（英文文档才启用「全大写」标题信号）
                doc_lang, doc_cjk_ratio = classify_document(_pdf_text_for_language(doc_path))
                doc_langs[rel_name] = {"lang": doc_lang, "cjk_ratio": doc_cjk_ratio}
                parsed = parse_document(
                    doc_path, rel_name, doc_lang=doc_lang, doc_cjk_ratio=doc_cjk_ratio
                )
                chunks.extend(parsed.chunks)
                # page_map 是 C5 的文档级物理页 → 印刷页码映射，落 meta 供展示/溯源
                doc_page_maps[rel_name] = {str(k): v for k, v in parsed.page_map.items()}
                doc_parsers[rel_name] = config.PARSER_VERSION
            elif suffix != ".pdf" and config.STRUCT_PARSER == "struct":
                # ---- 统一结构化链路（docx / md / txt / pptx / html，见 ADR-017）----
                # 各格式先转成统一中间表示，再按**与 PDF 完全相同的契约**产出 v3 chunk：
                # section_path / chunk_type / table 都是真值（page_index 留空，这些格式
                # 没有"页"的概念）。
                from app.rag.struct_parser import assemble, blocks_text, extract_blocks

                struct_blocks = extract_blocks(doc_path)
                # 语言判定同样在**文档级**做：中间表示拼成全文算一次
                doc_lang, doc_cjk_ratio = classify_document(blocks_text(struct_blocks))
                doc_langs[rel_name] = {"lang": doc_lang, "cjk_ratio": doc_cjk_ratio}
                chunks.extend(assemble(struct_blocks, rel_name, doc_lang, doc_cjk_ratio))
                doc_parsers[rel_name] = config.STRUCT_PARSER_VERSION
            else:
                # ---- 旧链路回退：PDF 非 layout 模式，或 STRUCT_PARSER=chunker ----
                # 非 PDF 走这里时结构字段只能补中性默认值（section_path=[] /
                # chunk_type 恒为 body / table=None），且 docx 的表格内容会整块丢失。
                blocks: list[TextBlock] = extract_document(doc_path)
                # 语言判定在**文档级**做：合并全文算一次，再传给切分器统一盖章。
                # 块级判定会被英文报告里的中文碎片块带偏（见 app/rag/language.py）
                doc_lang, doc_cjk_ratio = classify_document("\n".join(b.text for b in blocks))
                doc_langs[rel_name] = {"lang": doc_lang, "cjk_ratio": doc_cjk_ratio}
                # block_index 参与 chunk id 构造，保证同标题的多个块之间 id 也不冲突
                for block_index, block in enumerate(blocks, start=1):
                    for chunk in chunk_text(
                        block,
                        rel_name,
                        block_index=block_index,
                        doc_lang=doc_lang,
                        doc_cjk_ratio=doc_cjk_ratio,
                    ):
                        chunks.append(_v3_defaults(chunk))
                doc_parsers[rel_name] = _CHUNKER_PARSER_VERSION
        except Exception as exc:  # noqa: BLE001 —— 解析器可能抛任意第三方库异常
            logger.error(
                "文档解析失败，已跳过（该文档本次不会入库，其余文档照常处理）：%s | %s: %s",
                rel_name,
                type(exc).__name__,
                exc,
            )
            doc_langs.pop(rel_name, None)
            doc_page_maps.pop(rel_name, None)
            doc_parsers.pop(rel_name, None)
            continue
        unchanged_files[rel_name] = file_hash

    # 哈希全部一致且未强制重建：无需任何写入
    if not chunks and len(unchanged_files) == len(old_files):
        return {
            "kb": kb_name,
            "updated": False,
            "docs": len(doc_files),
            "chunks": _count_chunks(index_dir),
            "vector": config.EMBEDDING_ENABLED,
            "keyword": None,  # 未变更，本次没有写 ES
        }

    # 落盘原文。这份文件是 BM25 路的唯一数据源，也是 Qdrant / ES 的 seq 对齐基准——
    # seq 就是行号，所以**增量追加时既有行必须原样保留在原位置**（行号不能变），
    # 否则既有向量/关键词点的 seq 会指到别的 chunk 上，检索命中后取到错误原文。
    chunks_path = index_dir / "chunks.jsonl"

    # 本次要（重新）处理的文档：正常情况下其旧 chunk 不在文件里（新增文档本就没有）；
    # 这层过滤是为了兜住「chunks.jsonl 已写、meta.json 未及更新时进程中断」的窗口，
    # 避免同一文档下一轮被当成新增再追加一次。此类行只可能出现在文件尾部
    # （新文档总是追加在末尾），因此剔除它们不会移动其它文档的行号。
    processed_docs = {name for name in current_hashes if name not in unchanged_files}
    retained_lines: list[str] = []
    if not force and chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    if json.loads(line).get("doc") in processed_docs:
                        continue
                except json.JSONDecodeError:
                    pass  # 保留无法解析的行，宁可留着也不打乱行号
                retained_lines.append(line)
    seq_offset = len(retained_lines)

    with open(chunks_path, "w", encoding="utf-8") as f:
        for line in retained_lines:
            f.write(line if line.endswith("\n") else line + "\n")
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # 向量化（可选）：写入 Qdrant（见 ADR-007）
    # 先编码成功再清理旧点，避免编码中途失败把已有向量清空
    if config.EMBEDDING_ENABLED:
        vectors = _embed_texts([c["text"] for c in chunks])
        try:
            if force:
                vector_store.delete_kb(config.backend_kb(kb_name))
            vector_store.upsert(
                config.backend_kb(kb_name), chunks, vectors, seq_offset=seq_offset
            )
        except QdrantUnavailable as e:
            raise RuntimeError(
                f"向量写入 Qdrant 失败，请确认向量库已启动（{config.QDRANT_URL}）: {e}"
            ) from e

    # 关键词索引（可选）：写入 ES（见 ADR-008）
    # ES 未启动时不中断摄入——进程内 rank_bm25 会兜底关键词路，只记告警；
    # 这与向量路不同：Qdrant 失败会中断（否则 chunk 与向量不一致），
    # 而 ES 只是关键词路的加速实现，缺失时功能降级但不消失。
    es_docs = 0
    if config.ES_ENABLED:
        try:
            if force:
                keyword_store.delete_kb(config.backend_kb(kb_name))
            es_docs = keyword_store.index_kb(
                config.backend_kb(kb_name), chunks, seq_offset=seq_offset
            )
        except ElasticsearchUnavailable as e:
            logger.warning(
                "ES 关键词索引写入跳过（%s），关键词路将回退进程内 BM25: %s",
                config.ES_URL,
                e,
            )

    # 早期版本把向量存为本地 vectors.npy，改用 Qdrant 后清理历史残留
    stale_vectors = index_dir / "vectors.npy"
    if stale_vectors.exists():
        stale_vectors.unlink()

    meta = {
        "schema": config.INDEX_SCHEMA_VERSION,
        "files": unchanged_files,
        "vector_enabled": config.EMBEDDING_ENABLED,
        "vector_store": "qdrant" if config.EMBEDDING_ENABLED else "",
        "qdrant_collection": config.QDRANT_COLLECTION if config.EMBEDDING_ENABLED else "",
        "embedding_model": config.EMBEDDING_MODEL if config.EMBEDDING_ENABLED else "",
        # 关键词索引：写入成功记 elasticsearch，失败或未启用记本地 BM25 兜底
        "keyword_store": "elasticsearch" if es_docs else "local-bm25",
        "es_index": config.ES_INDEX if es_docs else "",
        "es_docs": es_docs,
        # 每份文档的语言判定结果：增量摄入时新文档覆盖同名旧条目，未变更文档沿用上次结果；
        # 已被删除的文档条目在此过滤掉，避免 meta 里留下幽灵文档
        "doc_langs": {
            name: lang
            for name, lang in {**old_doc_langs, **doc_langs}.items()
            if name in current_hashes
        },
        # PDF 解析器标识 + 版本：便于区分新旧索引、做 A/B 与回溯（⑦）
        "pdf_parser": config.PDF_PARSER,
        "parser_version": config.PARSER_VERSION,
        # 每份文档**实际**使用的解析器版本：PDF 走 layout-*、非 PDF 走 struct-*、
        # 回退路径走 chunker-*。库级 parser_version 只代表 PDF 解析器，混合库
        # （既有 PDF 又有 docx 等）必须看这张表才知道每份文档是谁解析的
        "doc_parsers": {
            name: version
            for name, version in {**old_doc_parsers, **doc_parsers}.items()
            if name in current_hashes
        },
        # 文档级 page_map（page_index → page_label，见 C5）：与 doc_langs 同样增量合并、
        # 过滤掉已删除文档；未走自研链路的格式（docx/md/txt）没有页码映射，留空
        "page_maps": {
            name: mapping
            for name, mapping in {**old_page_maps, **doc_page_maps}.items()
            if name in current_hashes
        },
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "kb": kb_name,
        "updated": True,
        "docs": len(doc_files),
        # 报告本次索引后的总 chunk 数（增量时含未变更文档的既有 chunk），与三处 count 对齐
        "chunks": seq_offset + len(chunks),
        "new_chunks": len(chunks),
        "vector": config.EMBEDDING_ENABLED,
        "keyword": bool(es_docs),
    }


def ingest_all(force: bool = False) -> list[dict]:
    """摄入全部知识库，供命令行一次性重建使用"""
    results = []
    for kb_name in config.list_knowledge_base_names():
        results.append(ingest_knowledge_base(kb_name, force=force))
    return results


def _count_chunks(index_dir: Path) -> int:
    """统计已有索引的 chunk 行数（用于「无需更新」时的摘要展示）"""
    chunks_path = index_dir / "chunks.jsonl"
    if not chunks_path.exists():
        return 0
    with open(chunks_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _backend_data_missing(kb_name: str, index_dir: Path) -> bool:
    """
    校验向量库 / ES 中该知识库的数据量是否与本地原文索引一致

    任一后端数量对不上即返回 True（调用方转全量重建）。
    后端不可达时不做判定——「服务没启动」不等于「数据丢了」，
    那种情况由检索层的降级链处理，不该触发一次昂贵的重建。
    """
    expected = _count_chunks(index_dir)
    if expected == 0:
        return False  # 尚未摄入过，走正常流程即可

    if config.EMBEDDING_ENABLED:
        try:
            if vector_store.count(config.backend_kb(kb_name)) != expected:
                return True
        except QdrantUnavailable:
            pass
        except Exception as exc:
            # 连接层异常（Docker 未启动时的 502 / 超时等）同样按「不可达」处理：
            # 本函数只判定「后端数据是否齐整」，服务没起来是另一回事，不该让摄入
            # 崩在完整性校验上（见上方 docstring 的约定）。
            logger.warning("向量库不可达，跳过完整性校验: %s", exc)

    if config.ES_ENABLED:
        try:
            if keyword_store.count(config.backend_kb(kb_name)) != expected:
                return True
        except ElasticsearchUnavailable:
            pass
        except Exception as exc:
            logger.warning("关键词索引不可达，跳过完整性校验: %s", exc)

    return False


if __name__ == "__main__":
    # 命令行入口：python -m app.rag.ingest [--force]
    import sys

    # 让摄入过程中的告警（如同步 ES 失败）能在命令行看到
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    results = ingest_all(force="--force" in sys.argv)
    for item in results:
        status = "已更新" if item["updated"] else "无变化"
        keyword = item.get("keyword")
        if keyword is None:
            keyword_status = "未变更"
        elif keyword:
            keyword_status = f"已索引({config.ES_INDEX})"
        else:
            keyword_status = "回退本地 rank-bm25"
        vector_status = "已索引" if item["vector"] else "未启用"
        new_chunks = item.get("new_chunks")
        if status == "已更新" and new_chunks is not None and new_chunks != item["chunks"]:
            # 增量追加：既有 chunk 未重算，单独标出本次真正新增的量
            chunk_info = f"chunk {item['chunks']} 条（本次新增 {new_chunks}）"
        else:
            chunk_info = f"chunk {item['chunks']} 条"
        print(
            f"[{item['kb']}] {status} | 文档 {item['docs']} 份 | {chunk_info}\n"
            f"    关键词路(ES): {keyword_status} | 向量路(Qdrant): {vector_status}"
        )
