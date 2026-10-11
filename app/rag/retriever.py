"""
知识库检索模块

双路检索架构，两路都是「可选增强 + 可降级」：
1. 关键词路：Elasticsearch（IK 中文分词 + 倒排索引 + BM25 打分）—— 见 ADR-008；
   ES 不可达时回退进程内 rank_bm25（jieba 分词 + Okapi BM25），功能不消失
2. 向量路：Qdrant（query embedding 与 chunk 向量做余弦相似度）—— 见 ADR-007；
   embedding 未配置或 Qdrant 不可达时该路缺席

两路结果用 RRF（Reciprocal Rank Fusion）融合排序——得分量纲不可比
（BM25 词频统计 vs 余弦），RRF 只看排名不看分值，天然免调参。

**字段级语言路由**（ADR-010）：关键词路的正文有两个字段——`text`（原文）与
`zh_text`（中文可检索视图，英文文档为空）。查询先判语言，中文 query 只打
`zh_text`、英文 query 只打 `text`、混排 query 两个都打。这样「中文 query ×
英文库」时关键词路天生拿不到候选，融合自动退化为向量单路——RRF 的双路命中
奖励在失效路存在时反而会拖低排名（实测中文组 Recall@5 46.2% → 69.2%）。
库内语言与查询语言完全不匹配时，整路直接跳过，既省一次 ES 查询也更干净。

**精排层**（ADR-012）：mode="hybrid_rerank" 在双路 RRF 融合之后再跑一次
cross-encoder 精排（进程内 ONNX，见 app/rag/reranker.py）。RRF 只看名次不看分值，
双路都命中的块必赢单路 rank-1，跨语言时还会被「假公共 token」污染——精排正是吃
这块排序缺口：离线实测 Recall@5 59.3% → 84.3%。精排**不改变候选集**，只在候选池
内部重排，所以「候选池召回」仍是 Recall@5 的理论上限。

可用性降级链：
- ES 可达 + 向量路就绪            → 关键词(ES) + 向量 双路 RRF（最佳）
- ES 不可达                       → 关键词路回退进程内 rank_bm25
- embedding 未配置 / Qdrant 不可达 → 向量路缺席，仅关键词路（单路同样返回结果）
- 两路都不可用                     → 进程内 rank_bm25 兜底，知识库能力始终可用
- 精排不可用/失败                  → 保持 RRF 原序（hybrid_rerank 自动退化为 hybrid）

存储分工：原文权威副本在本地 chunks.jsonl（本地 BM25 与 seq 对齐基准），
向量在 Qdrant、关键词索引在 ES；三处通过 chunk 在 jsonl 中的行号（seq）对齐，
融合与取原文都靠它。
索引按知识库懒加载并缓存：首次查询后常驻内存，进程内重复查询零 IO。
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.rag import config
from app.rag.chunker import SUPPORTED_SUFFIXES, _SENTENCE_BREAK_RE  # 句读正则做检索侧兜底分词
from app.rag.language import classify_query

logger = logging.getLogger(__name__)


@dataclass
class KBIndex:
    """单个知识库的内存索引：chunk 原文 + BM25 结构（向量不驻留内存，按需查 Qdrant）"""

    chunks: list[dict]
    bm25: object = None
    tokenized_corpus: list[list[str]] = field(default_factory=list)
    # 库内出现的语言集合（来自 chunk 的 lang 字段）。用于「中文 query × 纯英文库」
    # 这类组合下整路跳过关键词检索（见 ADR-010）。旧 schema 索引没有 lang 字段，
    # 此时为空集，语言路由判定自动失效、按原样跑，不会误伤老索引。
    langs: set[str] = field(default_factory=set)


# 生产检索模式：双路 RRF 融合 + cross-encoder 精排。精排不可用时（模型缺失 / 未装
# onnxruntime / 无 CUDA）自动退化为 hybrid 并告警一次，所以不需要额外的开关判断。
PRODUCTION_MODE = "hybrid_rerank"


class Retriever:
    """知识库检索器：持有全部已加载索引，向工具层提供查询入口"""

    def __init__(self) -> None:
        self._indexes: dict[str, KBIndex] = {}
        self._vector_degraded = False   # 向量路降级只告警一次，避免刷屏
        self._keyword_degraded = False  # 关键词路回退本地 BM25 同样只告警一次
        self._rerank_degraded = False   # 精排不可用回退 RRF 原序，同样只告警一次

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
        index.langs = {c.get("lang", "") for c in chunks} - {""}
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
        mode: str = PRODUCTION_MODE,
    ) -> list[dict]:
        """
        检索单个知识库，返回排序后的 top-k chunk（附来源与得分）

        :param kb_name: 知识库目录名
        :param query: 检索问题
        :param top_k: 返回条数，缺省用 config.DEFAULT_TOP_K
        :param mode: hybrid_rerank（**生产默认**，双路 RRF 融合后再 cross-encoder 精排）
                     | hybrid（双路 RRF 融合，精排的对照基线）| keyword（仅关键词路）
                     | bm25（仅进程内 BM25）| vector（仅向量路）
                     后三种仅用于评测对比检索质量。hybrid_rerank 在精排不可用时
                     （模型缺失 / 未装依赖 / 无 CUDA）自动退化为 hybrid 并告警一次。
        :return: [{...chunk 字段, "score": 排序分}]，得分越高越相关。
                 hybrid_rerank 下 score 是精排的归一化分数（0~1），其余模式是 RRF 分
        """
        top_k = top_k or config.DEFAULT_TOP_K
        fused = self._fused_ranking(kb_name, query, top_k, mode)
        return self._materialize(kb_name, fused[:top_k])

    def search_with_pool(
        self,
        kb_name: str,
        query: str,
        top_k: int | None = None,
        mode: str = PRODUCTION_MODE,
    ) -> tuple[list[dict], list[int]]:
        """
        与 search 完全相同的检索，但额外返回**截断前**的候选池序号列表

        候选池 = 两路各取 candidate_k（缺省 top_k×3）后 RRF 融合去重的集合。rerank
        只能在这个集合内重排，故「候选池召回」是 Recall@5 的理论上限——评测用它量化
        精排还有多少空间可吃。

        hybrid_rerank 模式下，候选池仍是**融合**（精排前）的集合：精排只重排不增删，
        所以池成员与是否精排无关，池召回上限也保持不变。
        """
        top_k = top_k or config.DEFAULT_TOP_K
        fused = self._fused_ranking(kb_name, query, top_k, mode)
        return self._materialize(kb_name, fused[:top_k]), [seq for seq, _ in fused]

    def candidate_pool(
        self,
        kb_name: str,
        query: str,
        top_k: int | None = None,
        mode: str = PRODUCTION_MODE,
        candidate_k: int | None = None,
    ) -> list[int]:
        """
        只返回融合去重后的候选池序号（截断成 top_k 之前）

        candidate_k 缺省沿用生产配置（top_k×3，每路）；传入更大的值可测「加深候选池」
        能达到的上限（评测用，不改变 search 的生产行为）。

        刻意**不做精排**（apply_rerank=False）：本方法只回答「候选集里有哪些」,
        精排不改变集合成员；诊断时容易把 candidate_k 放大到 100，白跑一次精排没意义。
        """
        fused = self._fused_ranking(kb_name, query, top_k, mode, candidate_k, apply_rerank=False)
        return [seq for seq, _ in fused]

    def _fused_ranking(
        self,
        kb_name: str,
        query: str,
        top_k: int | None,
        mode: str,
        candidate_k: int | None = None,
        apply_rerank: bool = True,
    ) -> list[tuple[int, float]]:
        """构建排序后的完整候选池（**不截断**），search / search_with_pool / candidate_pool 共用"""
        top_k = top_k or config.DEFAULT_TOP_K
        # hybrid_rerank 的融合部分与 hybrid 完全一致，只是融合后再精排一次——
        # 这样「精排 vs 不精排」的差异纯粹来自精排，不受融合行为变化干扰
        fusion_mode = "hybrid" if mode == "hybrid_rerank" else mode
        index = self._load_index(kb_name)
        if not index.chunks:
            return []

        # 查询语言决定关键词路打哪个字段（字段级路由，见 ADR-010）：
        # zh → 只打 zh_text（英文文档的块天生不在候选里）
        # en → 只打 text
        # mixed → 两个正文字段都打
        query_lang = classify_query(query)
        keyword_applicable = self._keyword_route_applicable(index, query_lang)

        # 两路候选各取 cand_k（缺省 3 倍 top_k），给 RRF 融合留足交集空间。
        # 精排模式下可由 RERANK_CANDIDATE_K 单独加深池（GPU 上加候选近乎免费），
        # 不影响 hybrid 的生产行为。
        if candidate_k:
            cand_k = candidate_k
        elif mode == "hybrid_rerank" and config.RERANK_CANDIDATE_K > 0:
            cand_k = config.RERANK_CANDIDATE_K
        else:
            cand_k = top_k * 3
        rankings: list[list[tuple[int, float]]] = []
        if fusion_mode in ("hybrid", "keyword") and keyword_applicable:
            rankings.append(
                self._keyword_search_safe(kb_name, index, query, cand_k, query_lang)
            )
        if fusion_mode == "bm25":
            # 仅进程内 BM25：评测对照用，用于量化「ES 关键词路」相对手写实现的增益
            rankings.append(self._bm25_search(index, query, cand_k))
        if fusion_mode in ("hybrid", "vector"):
            vector_ranking = self._vector_search_safe(kb_name, query, cand_k)
            if vector_ranking:
                rankings.append(vector_ranking)

        if not rankings and fusion_mode != "keyword" and keyword_applicable:
            # 向量路不可用（未配置 embedding / Qdrant 不可达）时降级 BM25，避免拿到空结果。
            # 两个例外：
            # - keyword 模式：该模式的语义就是「只看关键词路」，路由判不适用时空集才对
            # - 关键词路已被语言路由判定为不适用（中文 query × 纯英文库）：此时退回 BM25
            #   等于把噪声重新灌回来，返回空结果才是正确答案
            rankings.append(self._bm25_search(index, query, cand_k))

        fused = self._rrf_fuse(*rankings)
        if mode == "hybrid_rerank" and apply_rerank:
            fused = self._rerank_order(kb_name, query, fused)
        return fused

    def _rerank_order(
        self, kb_name: str, query: str, fused: list[tuple[int, float]]
    ) -> list[tuple[int, float]]:
        """
        用 cross-encoder 对候选池重排（ADR-012）

        **只重排、不增删**：成员集合不变，故候选池召回仍是 Recall@5 的上限。
        精排不可用或推理失败时返回原序（RRF 融合顺序）并告警一次——检索的可用性
        不依赖精排，缺模型、没装 onnxruntime、无 CUDA 都只是拿不到那 25 个点的提升。
        """
        if not fused:
            return fused
        from app.rag.reranker import reranker

        index = self._load_index(kb_name)
        texts = [index.chunks[seq]["text"] for seq, _ in fused]
        ranked = reranker.rerank(query, texts)
        if ranked is None:
            if not self._rerank_degraded:
                logger.warning("精排不可用，已回落到 RRF 融合排序：%s", reranker.status())
                self._rerank_degraded = True
            return fused
        return [(fused[pool_idx][0], score) for pool_idx, score in ranked]

    def _materialize(self, kb_name: str, fused: list[tuple[int, float]]) -> list[dict]:
        """把融合结果 (seq, score) 还原成 chunk 字典列表（附 score）"""
        index = self._load_index(kb_name)
        results = []
        for seq, score in fused:
            chunk = dict(index.chunks[seq])
            chunk["score"] = round(score, 4)
            results.append(chunk)
        return results

    def _keyword_search_safe(
        self,
        kb_name: str,
        index: KBIndex,
        query: str,
        k: int,
        query_lang: str = "mixed",
    ) -> list[tuple[int, float]]:
        """
        关键词路检索：Elasticsearch 优先，任何失败都回退进程内 BM25

        关键词路是知识库检索的基本盘——ES 没启动、索引未就绪或查询超时，
        都不应该让检索失败；退回进程内 rank_bm25 仍能拿到结果（功能不消失，
        只是少了倒排索引与 IK 分词带来的精度）。

        **已知取舍**：回退到进程内 BM25 时不做语言字段路由（本地索引是按 text
        原文建的 jieba 语料），只有在 ES 不可达时才会走到这条路径；做门控需要
        同时维护 text / zh_text 两套语料，收益（罕见降级场景）不抵复杂度。
        """
        if not config.ES_ENABLED:
            return self._bm25_search(index, query, k)
        try:
            from app.rag.keyword_store import ElasticsearchUnavailable, keyword_store

            return keyword_store.search(kb_name, query, k, query_lang=query_lang)
        except ElasticsearchUnavailable as e:
            if not self._keyword_degraded:
                logger.warning("ES 关键词检索不可用，已回退进程内 BM25: %s", e)
                self._keyword_degraded = True
            return self._bm25_search(index, query, k)
        except Exception as e:  # noqa: BLE001 —— 兜底：任何异常都不应中断检索
            if not self._keyword_degraded:
                logger.warning("ES 关键词检索异常，已回退进程内 BM25: %s", e)
                self._keyword_degraded = True
            return self._bm25_search(index, query, k)

    @staticmethod
    def _keyword_route_applicable(index: KBIndex, query_lang: str) -> bool:
        """
        关键词路这次是否值得跑（字段级语言路由，见 ADR-010）

        「中文 query × 纯英文库」这类组合下，该库所有 chunk 的 zh_text 都是空，
        关键词路必然返回空候选。跳过它有双重好处：
        1. **省掉一次 ES 查询**——实测跨语言查询的 keyword 候选 7/7、6/6 为空；
        2. **保证融合退化为有效单路**——RRF 的「双路命中奖励」在失效路存在时会
           把噪声块推上去（中文 query 下 Recall@5 从 69.2% 掉到 46.2%），
           整路不参与比事后过滤候干净。

        混排 query 永远要跑（两个字段都打）；索引没有 lang 字段（旧 schema）时
        不做判定、按原样跑，避免误伤。
        """
        if query_lang == "mixed" or not index.langs:
            return True
        return query_lang in index.langs

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
