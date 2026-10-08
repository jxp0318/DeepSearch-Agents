"""
关键词检索模块（Elasticsearch）

职责边界：只管 chunk 原文的倒排索引与关键词检索，不碰向量、不碰本地 BM25 统计量。
- 原文的权威副本在本地 chunks.jsonl；ES 保存一份可检索的副本（含元数据）；
- 向量在 Qdrant，关键词索引在 ES；两路都靠 chunk 在 chunks.jsonl 中的行号（seq）对齐。

这是双路检索里的「关键词路」的生产实现：IK 中文分词 + 倒排索引 + BM25 打分
（BM25 是 Elasticsearch 自 7.0 起的默认相似度算法，ES 是引擎、BM25 是算法）。

设计取舍（详见 ADR-008）：
1. **单索引 + kb 字段过滤**，与 Qdrant 的建模方式对称；新增知识库不必建新索引。
2. **_id 用 uuid5(kb::chunk_id) 派生**，确定性 ID 让重复摄入天然幂等。
3. **分词器可回退**：优先 IK（`ik_max_word` 索引 / `ik_smart` 查询），探测不到时
   退回内置 `standard`——ES 装了 IK 插件才有最佳中文效果，但缺失也不能让系统崩。
4. **不可用时降级而非报错**：ES 是关键词路的增强实现，连不上时检索层回退到进程内
   rank_bm25，知识库能力不消失（抛 ElasticsearchUnavailable 交由上层处理）。
"""

from __future__ import annotations

import logging
import uuid

from app.rag import config

logger = logging.getLogger(__name__)

# 固定命名空间：保证同一 (kb, chunk_id) 在不同进程、不同时间映射到同一 _id
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

# 单次 bulk 的文档数：控制单请求体量
_BULK_BATCH = 256

# 中文分词器候选：首选 IK，回退内置 standard
_IK_INDEX_ANALYZER = "ik_max_word"   # 索引期：细粒度切分，提高召回
_IK_SEARCH_ANALYZER = "ik_smart"     # 查询期：粗粒度切分，提高精度
_FALLBACK_ANALYZER = "standard"      # ES 内置：中文按字切分，效果一般但保证可用


class ElasticsearchUnavailable(RuntimeError):
    """ES 不可达或查询失败，调用方应回退到进程内 BM25"""


class ElasticsearchKeywordStore:
    """ES 封装：懒连接、按知识库写入与关键词检索"""

    def __init__(self) -> None:
        self._client = None
        self._analyzer: str | None = None  # 探测到的可用分词器，缓存避免重复探测

    # ---------- 连接 ----------

    def _get_client(self):
        """懒创建客户端；连接失败统一转成 ElasticsearchUnavailable 供上层降级"""
        if self._client is None:
            try:
                from elasticsearch import Elasticsearch
            except ImportError as e:  # pragma: no cover —— 依赖缺失属于部署问题
                raise ElasticsearchUnavailable(f"未安装 elasticsearch 客户端: {e}") from e

            auth = (
                (config.ES_USERNAME, config.ES_PASSWORD)
                if config.ES_USERNAME
                else None
            )
            try:
                self._client = Elasticsearch(
                    hosts=[config.ES_URL],
                    basic_auth=auth,
                    request_timeout=config.ES_TIMEOUT,
                    # 本地部署用的是 http://（xpack 安全已关），不做 TLS 校验
                    verify_certs=False,
                )
            except Exception as e:  # noqa: BLE001 —— 连接失败即降级
                raise ElasticsearchUnavailable(
                    f"连接 Elasticsearch 失败（{config.ES_URL}）: {e}"
                ) from e
        return self._client

    @property
    def available(self) -> bool:
        """ES 已启用且可达时才走 ES 关键词路，否则回退进程内 BM25"""
        if not config.ES_ENABLED:
            return False
        try:
            return bool(self._get_client().ping())
        except Exception:  # noqa: BLE001 —— 探测失败按不可用处理
            return False

    def _resolve_analyzer(self, client) -> str:
        """
        探测 IK 分词器是否可用，结果缓存

        ES 装了 analysis-ik 插件才有 ik_max_word；缺失时回退内置 standard
        （中文按单字切分，召回够但精度差）。返回索引期使用的分词器名。
        """
        if self._analyzer is not None:
            return self._analyzer
        try:
            client.indices.analyze(analyzer=_IK_INDEX_ANALYZER, text="中文分词器探测")
            self._analyzer = _IK_INDEX_ANALYZER
        except Exception:  # noqa: BLE001 —— 插件缺失属预期情况，不算错误
            logger.warning(
                "Elasticsearch 未检测到 %s 分词器（缺少 analysis-ik 插件？），"
                "已回退内置 %s，中文检索精度会下降",
                _IK_INDEX_ANALYZER,
                _FALLBACK_ANALYZER,
            )
            self._analyzer = _FALLBACK_ANALYZER
        return self._analyzer

    def _search_analyzer(self) -> str:
        """查询期分词器：IK 可用时用细粒度 lower 版本，否则与索引期一致"""
        if self._analyzer == _IK_INDEX_ANALYZER:
            return _IK_SEARCH_ANALYZER
        return _FALLBACK_ANALYZER

    # ---------- 写入 ----------

    def index_kb(self, kb_name: str, chunks: list[dict]) -> int:
        """
        把一个知识库的全部 chunk 写入 ES

        :param kb_name: 知识库名（写入文档作为过滤字段）
        :param chunks: chunk 列表，seq 取其列表下标（与 chunks.jsonl 行号一致）
        :return: 成功写入的文档数
        :raises ElasticsearchUnavailable: ES 不可达
        """
        from elasticsearch.helpers import bulk

        client = self._get_client()
        self._ensure_index(client)

        actions = [
            {
                "_index": config.ES_INDEX,
                "_id": self._doc_id(kb_name, chunk["id"]),
                "_source": {
                    "kb": kb_name,
                    # seq = chunk 在 chunks.jsonl 中的行号，两路检索靠它对齐
                    "seq": seq,
                    "chunk_id": chunk["id"],
                    "doc": chunk.get("doc", ""),
                    "heading": chunk.get("heading", ""),
                    "text": chunk.get("text", ""),
                },
            }
            for seq, chunk in enumerate(chunks)
        ]

        written = 0
        for start in range(0, len(actions), _BULK_BATCH):
            try:
                success, errors = bulk(
                    client,
                    actions[start : start + _BULK_BATCH],
                    refresh=True,  # 写入立即可检索，避免摄入后立刻查询查不到
                    raise_on_error=False,
                )
            except Exception as e:  # noqa: BLE001 —— 批量写入失败统一降级
                raise ElasticsearchUnavailable(f"ES 批量写入失败: {e}") from e
            written += int(success)
            if errors:
                logger.warning("ES 写入存在失败文档: %d 条，首条: %s", len(errors), errors[0])

        logger.info("ES 写入完成: kb=%s docs=%d index=%s", kb_name, written, config.ES_INDEX)
        return written

    def delete_kb(self, kb_name: str) -> None:
        """删除某知识库的全部文档：全量重建前调用，避免旧文档残留成孤儿数据"""
        client = self._get_client()
        if not client.indices.exists(index=config.ES_INDEX):
            return
        try:
            client.delete_by_query(
                index=config.ES_INDEX,
                query={"term": {"kb": kb_name}},
                refresh=True,
                conflicts="proceed",
            )
        except Exception as e:  # noqa: BLE001 —— 删除失败不致命，后续 upsert 会覆盖
            logger.warning("ES 删除知识库旧文档失败（不影响继续写入）: %s", e)

    def _ensure_index(self, client) -> None:
        """索引不存在时按 IK 分词器建 mapping"""
        if client.indices.exists(index=config.ES_INDEX):
            # 索引已存在时仍解析一次分词器，供检索期使用
            self._resolve_analyzer(client)
            return

        analyzer = self._resolve_analyzer(client)
        search_analyzer = self._search_analyzer()
        client.indices.create(
            index=config.ES_INDEX,
            mappings={
                "properties": {
                    "kb": {"type": "keyword"},
                    "seq": {"type": "integer"},
                    "chunk_id": {"type": "keyword"},
                    "doc": {"type": "keyword"},
                    # 章节标题命中通常意味着整段相关，故检索期给更高权重
                    "heading": {
                        "type": "text",
                        "analyzer": analyzer,
                        "search_analyzer": search_analyzer,
                    },
                    "text": {
                        "type": "text",
                        "analyzer": analyzer,
                        "search_analyzer": search_analyzer,
                    },
                }
            },
        )
        logger.info(
            "ES 创建索引: %s analyzer=%s search_analyzer=%s",
            config.ES_INDEX,
            analyzer,
            search_analyzer,
        )

    # ---------- 检索 ----------

    def search(self, kb_name: str, query: str, limit: int) -> list[tuple[int, float]]:
        """
        在指定知识库内做关键词检索（kb 字段过滤）

        用 multi_match 覆盖正文与章节标题（标题权重 2 倍），打分由 ES 内置 BM25 给出。

        :param kb_name: 知识库名
        :param query: 检索问题
        :param limit: 返回条数
        :return: [(chunk 在 chunks.jsonl 中的行号, 相关度得分)]，按得分降序
        :raises ElasticsearchUnavailable: ES 不可达，调用方应回退本地 BM25
        """
        try:
            client = self._get_client()
            if not client.indices.exists(index=config.ES_INDEX):
                # 索引未就绪（从未摄入，或摄入时 ES 未启动）——按「ES 路当前不可服务」
                # 处理，交由检索层回退进程内 BM25，而不是返回空结果骗上层说没命中
                raise ElasticsearchUnavailable(
                    f"ES 索引 {config.ES_INDEX} 不存在，请先执行 python -m app.rag.ingest"
                )
            self._resolve_analyzer(client)
            resp = client.search(
                index=config.ES_INDEX,
                query={
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["text^1.0", "heading^2.0"],
                                }
                            }
                        ],
                        # 过滤不参与打分，等价于 Qdrant 的 payload filter
                        "filter": [{"term": {"kb": kb_name}}],
                    }
                },
                size=limit,
                _source=["seq"],
            )
        except ElasticsearchUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 —— 查询异常同样降级处理
            raise ElasticsearchUnavailable(f"ES 检索失败: {e}") from e

        return [
            (int(hit["_source"]["seq"]), float(hit["_score"]))
            for hit in resp["hits"]["hits"]
            if "seq" in hit.get("_source", {})
        ]

    def count(self, kb_name: str) -> int:
        """统计某知识库已写入的文档数，用于摄入摘要与自检"""
        client = self._get_client()
        if not client.indices.exists(index=config.ES_INDEX):
            return 0
        result = client.count(index=config.ES_INDEX, query={"term": {"kb": kb_name}})
        return int(result["count"])

    @staticmethod
    def _doc_id(kb_name: str, chunk_id: str) -> str:
        """把 (知识库, chunk 业务 ID) 映射为确定性文档 ID"""
        return str(uuid.uuid5(_NAMESPACE, f"{kb_name}::{chunk_id}"))


# 模块级单例：ingest 与 retriever 共用同一客户端
keyword_store = ElasticsearchKeywordStore()
