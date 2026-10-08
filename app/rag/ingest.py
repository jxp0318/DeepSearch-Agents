"""
知识库摄入模块

流水线：扫描知识库目录 → 提取 → 切分 → embedding（可选）→ 写入向量库 + 落盘原文。
索引按知识库组织，原文与向量分两处存储（见 ADR-007）：

    app/rag/indexes/<kb_name>/chunks.jsonl   每个 chunk 一行 JSON（id / doc / heading / text）
    app/rag/indexes/<kb_name>/meta.json      文件哈希清单与索引模式，用于增量摄入与降级标记
    Qdrant collection (payload.kb = <kb_name>)   chunk 向量 + 原文副本

两条数据通过 chunk 在 chunks.jsonl 中的行号（seq）对齐：BM25 走本地原文，
向量走 Qdrant，检索时在应用层做 RRF 融合。
embedding 未配置时不写向量，检索退回纯 BM25。

增量策略：按文件 sha256 跳过未变化的文档，只重摄入新增或修改的文件。
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from app.rag import config
from app.rag.chunker import SUPPORTED_SUFFIXES, TextBlock, chunk_text, extract_document
from app.rag.vector_store import QdrantUnavailable, vector_store


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
                resp = requests.post(
                    f"{config.EMBEDDING_BASE_URL}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=30,
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


def ingest_knowledge_base(kb_name: str, force: bool = False) -> dict:
    """
    摄入单个知识库：提取 → 切分 → 向量化 → 写向量库 + 落盘原文

    :param kb_name: 知识库目录名（KNOWLEDGE_BASE_ROOT 下的直接子目录）
    :param force: True 时忽略文件哈希全量重建索引
    :return: 摄入摘要 dict（文件数、chunk 数、是否启用向量、耗时）
    """
    kb_dir = config.KNOWLEDGE_BASE_ROOT / kb_name
    if not kb_dir.is_dir():
        raise FileNotFoundError(f"知识库不存在: {kb_name}")

    index_dir = config.INDEX_ROOT / kb_name
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path = index_dir / "meta.json"

    # 读取上次的文件哈希清单；force 或首次摄入时视为空
    old_files: dict[str, str] = {}
    if meta_path.exists() and not force:
        old_files = json.loads(meta_path.read_text(encoding="utf-8")).get("files", {})

    # 扫描支持格式的文档，按哈希判断哪些需要（重）摄入
    doc_files = sorted(
        p for p in kb_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not doc_files:
        raise FileNotFoundError(f"知识库 {kb_name} 中没有可摄入的文档: {kb_dir}")

    unchanged_files: dict[str, str] = {}
    chunks: list[dict] = []
    for doc_path in doc_files:
        file_hash = _file_sha256(doc_path)
        rel_name = str(doc_path.relative_to(kb_dir))
        if rel_name in old_files and old_files[rel_name] == file_hash and not force:
            unchanged_files[rel_name] = file_hash
            continue
        blocks: list[TextBlock] = extract_document(doc_path)
        # block_index 参与 chunk id 构造，保证同标题的多个块之间 id 也不冲突
        for block_index, block in enumerate(blocks, start=1):
            chunks.extend(chunk_text(block, rel_name, block_index=block_index))
        unchanged_files[rel_name] = file_hash

    # 哈希全部一致且未强制重建：无需任何写入
    if not chunks and len(unchanged_files) == len(old_files):
        return {
            "kb": kb_name,
            "updated": False,
            "docs": len(doc_files),
            "chunks": _count_chunks(index_dir),
            "vector": config.EMBEDDING_ENABLED,
        }

    # 落盘原文：一次性重写（知识库规模小，全量重建比合并补丁简单可靠）
    # 这份文件是 BM25 路的唯一数据源，也是向量 payload 的 seq 对齐基准
    chunks_path = index_dir / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # 向量化（可选）：写入 Qdrant（见 ADR-007）
    # 先编码成功再清理旧点，避免编码中途失败把已有向量清空
    if config.EMBEDDING_ENABLED:
        vectors = _embed_texts([c["text"] for c in chunks])
        try:
            if force:
                vector_store.delete_kb(kb_name)
            vector_store.upsert(kb_name, chunks, vectors)
        except QdrantUnavailable as e:
            raise RuntimeError(
                f"向量写入 Qdrant 失败，请确认向量库已启动（{config.QDRANT_URL}）: {e}"
            ) from e

    # 早期版本把向量存为本地 vectors.npy，改用 Qdrant 后清理历史残留
    stale_vectors = index_dir / "vectors.npy"
    if stale_vectors.exists():
        stale_vectors.unlink()

    meta = {
        "files": unchanged_files,
        "vector_enabled": config.EMBEDDING_ENABLED,
        "vector_store": "qdrant" if config.EMBEDDING_ENABLED else "",
        "qdrant_collection": config.QDRANT_COLLECTION if config.EMBEDDING_ENABLED else "",
        "embedding_model": config.EMBEDDING_MODEL if config.EMBEDDING_ENABLED else "",
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "kb": kb_name,
        "updated": True,
        "docs": len(doc_files),
        "chunks": len(chunks),
        "vector": config.EMBEDDING_ENABLED,
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


if __name__ == "__main__":
    # 命令行入口：python -m app.rag.ingest [--force]
    import sys

    results = ingest_all(force="--force" in sys.argv)
    for item in results:
        status = "已更新" if item["updated"] else "无变化"
        vector_status = "向量(Qdrant)+BM25" if item["vector"] else "纯 BM25"
        print(
            f"[{item['kb']}] {status} | 文档 {item['docs']} 份 | "
            f"chunk {item['chunks']} 条 | 检索模式: {vector_status}"
        )
