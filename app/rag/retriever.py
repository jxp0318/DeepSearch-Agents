"""
知识库检索模块

双路检索架构：
1. BM25 路（始终可用）：jieba 分词 + Okapi BM25，覆盖关键词精确匹配
2. 向量路（Qdrant）：query embedding 与 Qdrant 中的 chunk 向量做相似度检索，覆盖语义改写

两路结果用 RRF（Reciprocal Rank Fusion）融合排序——两路得分量纲不可比
（词频统计 vs 余弦），RRF 只看排名不看分值，天然免调参。

可用性三级降级（见 ADR-006 / ADR-007）：
- embedding 未配置              → 纯 BM25
- embedding 已配置但 Qdrant 不可达 → 纯 BM25（不报错、不阻塞任务）
- 两者就绪                       → 双路 RRF 融合

存储分工：原文在本地 chunks.jsonl（BM25 依赖），向量在 Qdrant（向量路依赖），
两路结果通过 chunk 在 jsonl 中的行号（seq）对齐，融合与取原文都靠它。
索引按知识库懒加载并缓存：首次查询后常驻内存，进程内重复查询零 IO。
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.rag import config
from app.rag.chunker import SUPPORTED_SUFFIXES, _SENTENCE_BREAK_RE  # 句读正则做检索侧兜底分词

logger = logging.getLogger(__name__)


@dataclass
class KBIndex:
    """单个知识库的内存索引：chunk 原文 + BM25 结构（向量不驻留内存，按需查 Qdrant）"""

    chunks: list[dict]
    bm25: object = None
    tokenized_corpus: list[list[str]] = field(default_factory=list)


class Retriever:
    """知识库检索器：持有全部已加载索引，向工具层提供查询入口"""

    def __init__(self) -> None:
        self._indexes: dict[str, KBIndex] = {}
        self._vector_degraded = False  # 向量库降级只告警一次，避免刷屏

    # ---------- 索引加载 ----------

    def _load_index(self, kb_name: str) -> KBIndex:
        """加载并缓存单个知识库索引；未摄入的知识库抛异常由工具层转成提示"""
        if kb_name in self._indexes:
            return self._indexes[kb_name]

        chunks_path = config.INDEX_ROOT / kb_name / "chunks.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"知识库 {kb_name} 尚未摄入索引，请先执行: python -m app.rag.ingest"
            )

        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

        index = KBIndex(chunks=chunks)
        self._build_bm25(index)
        self._indexes[kb_name] = index
        return index

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """jieba 分词，过滤空白与单字符标点；jieba 不可用时按句读粗切保底"""
        try:
            import jieba

            return [t for t in jieba.lcut(text) if t.strip()]
        except ImportError:
            return [t for t in _SENTENCE_BREAK_RE.split(text) if t.strip()]

    def _build_bm25(self, index: KBIndex) -> None:
        """对全部 chunk 建一次 BM25，随索引缓存"""
        from rank_bm25 import BM25Okapi

        index.tokenized_corpus = [self._tokenize(c["text"]) for c in index.chunks]
        index.bm25 = BM25Okapi(index.tokenized_corpus) if index.chunks else None

    # ---------- 检索 ----------

    def search(
        self,
        kb_name: str,
        query: str,
        top_k: int | None = None,
        mode: str = "hybrid",
    ) -> list[dict]:
        """
        检索单个知识库，返回排序后的 top-k chunk（附来源与得分）

        :param kb_name: 知识库目录名
        :param query: 检索问题
        :param top_k: 返回条数，缺省用 config.DEFAULT_TOP_K
        :param mode: hybrid（默认，双路 RRF 融合）| bm25 | vector
                     后两种仅用于评测对比检索质量，生产路径走 hybrid
        :return: [{...chunk 字段, "score": 融合排名分}]，得分越高越相关
        """
        top_k = top_k or config.DEFAULT_TOP_K
        index = self._load_index(kb_name)
        if not index.chunks:
            return []

        # 两路候选各取 3 倍 top_k，给 RRF 融合留足交集空间
        candidate_k = top_k * 3
        rankings: list[list[tuple[int, float]]] = []
        if mode in ("hybrid", "bm25"):
            rankings.append(self._bm25_search(index, query, candidate_k))
        if mode in ("hybrid", "vector"):
            vector_ranking = self._vector_search_safe(kb_name, query, candidate_k)
            if vector_ranking:
                rankings.append(vector_ranking)

        if not rankings:
            # 指定 vector 模式但向量路不可用：降级 BM25，避免调用方拿到空结果
            rankings.append(self._bm25_search(index, query, candidate_k))

        fused = self._rrf_fuse(*rankings)
        results = []
        for seq, score in fused[:top_k]:
            chunk = dict(index.chunks[seq])
            chunk["score"] = round(score, 4)
            results.append(chunk)
        return results

    def _vector_search_safe(
        self, kb_name: str, query: str, k: int
    ) -> list[tuple[int, float]]:
        """
        向量路检索，任何失败都降级为「无向量结果」而不是中断整个检索

        向量库是增强项：embedding 服务未就绪、Qdrant 未启动或查询超时，
        都不应该让知识库检索整体失败——退回 BM25 仍然可用。
        """
        if not config.EMBEDDING_ENABLED:
            return []
        try:
            from app.rag.vector_store import QdrantUnavailable, vector_store

            query_vector = self._embed_query(query)
            return vector_store.search(kb_name, query_vector, k)
        except QdrantUnavailable as e:
            if not self._vector_degraded:
                logger.warning("向量检索不可用，已降级为纯 BM25: %s", e)
                self._vector_degraded = True
            return []
        except Exception as e:  # noqa: BLE001 —— embedding 调用失败同样降级
            if not self._vector_degraded:
                logger.warning("query embedding 失败，已降级为纯 BM25: %s", e)
                self._vector_degraded = True
            return []

    @staticmethod
    def _embed_query(query: str) -> list[float]:
        """把查询文本转向量（复用摄入侧同一端点，保证向量空间一致）"""
        from app.rag.ingest import _embed_texts

        return _embed_texts([query])[0].tolist()

    @staticmethod
    def _bm25_search(index: KBIndex, query: str, k: int) -> list[tuple[int, float]]:
        """BM25 关键词检索，返回 (chunk 序号, 原始得分) 按得分降序"""
        if index.bm25 is None:
            return []
        scores = index.bm25.get_scores(Retriever._tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    @staticmethod
    def _rrf_fuse(*rankings: list[tuple[int, float]], k: int = 60) -> list[tuple[int, float]]:
        """
        Reciprocal Rank Fusion：score = Σ 1/(k + rank)

        两路得分量纲完全不同（BM25 是词频统计、向量是余弦），直接加权拼接
        需要校准权重；RRF 只看排名不看分值，天然免调参。参数按路数可变，
        便于评测时单独跑 BM25 或向量单路。
        """
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, (seq, _) in enumerate(ranking, start=1):
                fused[seq] = fused.get(seq, 0.0) + 1.0 / (k + rank)
        return sorted(fused.items(), key=lambda item: item[1], reverse=True)

    # ---------- 工具层辅助 ----------

    def kb_summary(self) -> list[dict]:
        """列出全部知识库及其文档构成，供 list_knowledge_bases 工具展示"""
        summaries = []
        for kb_name in config.list_knowledge_base_names():
            kb_dir = config.KNOWLEDGE_BASE_ROOT / kb_name
            docs = sorted(
                p.name
                for p in kb_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
            )
            index_dir: Path = config.INDEX_ROOT / kb_name
            indexed = (index_dir / "chunks.jsonl").exists()
            summaries.append({"name": kb_name, "docs": docs, "indexed": indexed})
        return summaries


# 模块级单例：工具层复用同一份索引缓存，避免每次调用重建 BM25
retriever = Retriever()
