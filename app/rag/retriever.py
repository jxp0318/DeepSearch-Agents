"""
知识库检索模块

双路检索架构：
1. BM25 路（始终可用）：jieba 分词 + Okapi BM25，覆盖关键词精确匹配
2. 向量路（可选）：query embedding 与 chunk 向量做余弦相似度，覆盖语义改写

两路结果用 RRF（Reciprocal Rank Fusion）融合排序——不用调权拼分，
按「两路都认可的块排名靠前」的朴素逻辑融合，对 5 份 PDF 规模足够。

embedding 未配置或索引未向量化时自动降级为纯 BM25，不报错、不阻塞。
索引按知识库懒加载并缓存：首次查询后常驻内存，进程内重复查询零 IO。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.rag import config
from app.rag.chunker import SUPPORTED_SUFFIXES, _SENTENCE_BREAK_RE  # 句读正则做检索侧兜底分词


@dataclass
class KBIndex:
    """单个知识库的内存索引：chunks + 向量矩阵 + BM25 结构"""

    chunks: list[dict]
    vector_matrix: np.ndarray | None
    vector_enabled: bool
    bm25: object = None
    tokenized_corpus: list[list[str]] = field(default_factory=list)


class Retriever:
    """知识库检索器：持有全部已加载索引，向工具层提供查询入口"""

    def __init__(self) -> None:
        self._indexes: dict[str, KBIndex] = {}

    # ---------- 索引加载 ----------

    def _load_index(self, kb_name: str) -> KBIndex:
        """加载并缓存单个知识库索引；未摄入的知识库抛异常由工具层转成提示"""
        if kb_name in self._indexes:
            return self._indexes[kb_name]

        index_dir = config.INDEX_ROOT / kb_name
        chunks_path = index_dir / "chunks.jsonl"
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

        vectors_path = index_dir / "vectors.npy"
        vector_matrix = np.load(vectors_path) if vectors_path.exists() else None

        index = KBIndex(
            chunks=chunks,
            vector_matrix=vector_matrix,
            vector_enabled=vector_matrix is not None,
        )
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

    def search(self, kb_name: str, query: str, top_k: int | None = None) -> list[dict]:
        """
        检索单个知识库，返回融合排序后的 top-k chunk（附来源与得分）

        :param kb_name: 知识库目录名
        :param query: 检索问题
        :param top_k: 返回条数，缺省用 config.DEFAULT_TOP_K
        :return: [{...chunk 字段, "score": 融合排名分}]，得分越高越相关
        """
        top_k = top_k or config.DEFAULT_TOP_K
        index = self._load_index(kb_name)
        if not index.chunks:
            return []

        # 两路候选各取 3 倍 top_k，给 RRF 融合留足交集空间
        candidate_k = top_k * 3
        bm25_ranking = self._bm25_search(index, query, candidate_k)
        vector_ranking = (
            self._vector_search(index, query, candidate_k)
            if index.vector_enabled
            else []
        )

        fused = self._rrf_fuse(bm25_ranking, vector_ranking)
        results = []
        for chunk_id, score in fused[:top_k]:
            chunk = dict(index.chunks[chunk_id])
            chunk["score"] = round(score, 4)
            results.append(chunk)
        return results

    @staticmethod
    def _bm25_search(index: KBIndex, query: str, k: int) -> list[tuple[int, float]]:
        """BM25 关键词检索，返回 (chunk 序号, 原始得分) 按得分降序"""
        if index.bm25 is None:
            return []
        scores = index.bm25.get_scores(Retriever._tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    @staticmethod
    def _vector_search(index: KBIndex, query: str, k: int) -> list[tuple[int, float]]:
        """向量语义检索：query embedding 与全部 chunk 做余弦相似度"""
        from app.rag.ingest import _embed_texts

        query_vector = _embed_texts([query])[0]
        # 余弦相似度：矩阵已归一化时点积即余弦；这里显式归一化保证语义正确
        norms = np.linalg.norm(index.vector_matrix, axis=1) * np.linalg.norm(query_vector)
        norms[norms == 0] = 1e-10
        similarities = (index.vector_matrix @ query_vector) / norms
        order = np.argsort(similarities)[::-1][:k]
        return [(int(i), float(similarities[i])) for i in order]

    @staticmethod
    def _rrf_fuse(
        ranking_a: list[tuple[int, float]], ranking_b: list[tuple[int, float]], k: int = 60
    ) -> list[tuple[int, float]]:
        """
        Reciprocal Rank Fusion：score = Σ 1/(k + rank)

        两路得分量纲完全不同（BM25 是词频统计、向量是余弦），直接加权拼接
        需要校准权重；RRF 只看排名不看分值，天然免调参。
        """
        fused: dict[int, float] = {}
        for ranking in (ranking_a, ranking_b):
            for rank, (chunk_id, _) in enumerate(ranking, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
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
            summaries.append(
                {"name": kb_name, "docs": docs, "indexed": indexed}
            )
        return summaries


# 模块级单例：工具层复用同一份索引缓存，避免每次调用重建 BM25
retriever = Retriever()
