"""
向量库模块（Qdrant）

职责边界：只管 chunk 向量的写入与检索，不碰原文、不碰 BM25。
- 原文与元数据保存在本地 chunks.jsonl，BM25 路依赖它；
- 向量保存在 Qdrant，向量路依赖它；两路通过 chunk 在 jsonl 中的行号（seq）对齐。

设计取舍（详见 ADR-007）：
1. **单 collection + payload 过滤**，而不是「一个知识库一个 collection」——
   新增知识库无需建表，并为「跨库检索 + 按来源过滤」留出演进空间。
2. **point id 用 uuid5(kb::chunk_id) 派生**——确定性 ID 让重复摄入天然幂等，
   upsert 命中同一点而非产生重复。
3. **不可用时降级而非报错**——向量库是检索质量的增强项，不是系统可用性的前提；
   连接失败抛 QdrantUnavailable，由检索层捕获后退回纯 BM25。
"""

from __future__ import annotations

import logging
import uuid

import numpy as np

from app.rag import config

logger = logging.getLogger(__name__)

# 固定命名空间：保证同一 (kb, chunk_id) 在不同进程、不同时间映射到同一 point id
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

# 单次 upsert 的点数：Qdrant 单请求体量有限，分批写入更稳
_UPSERT_BATCH = 128


class QdrantUnavailable(RuntimeError):
    """向量库不可达或查询失败，调用方应降级为纯 BM25"""


class QdrantVectorStore:
    """Qdrant 封装：懒连接、按知识库写入与检索"""

    def __init__(self) -> None:
        self._client = None

    # ---------- 连接 ----------

    def _get_client(self):
        """懒创建客户端；连接失败统一转成 QdrantUnavailable 供上层降级"""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as e:  # pragma: no cover —— 依赖缺失属于部署问题
                raise QdrantUnavailable(f"未安装 qdrant-client: {e}") from e
            try:
                self._client = QdrantClient(
                    url=config.QDRANT_URL,
                    api_key=config.QDRANT_API_KEY or None,
                    timeout=config.QDRANT_TIMEOUT,
                )
            except Exception as e:  # noqa: BLE001 —— 连接失败即降级
                raise QdrantUnavailable(f"连接 Qdrant 失败（{config.QDRANT_URL}）: {e}") from e
        return self._client

    @property
    def available(self) -> bool:
        """embedding 已配置且 Qdrant 可达时才走向量路；任一不满足则纯 BM25"""
        if not config.EMBEDDING_ENABLED:
            return False
        try:
            self._get_client().get_collections()
            return True
        except Exception:  # noqa: BLE001 —— 探测失败按不可用处理
            return False

    # ---------- 写入 ----------

    def upsert(self, kb_name: str, chunks: list[dict], vectors: np.ndarray) -> int:
        """
        把一个知识库的全部 chunk 向量写入 Qdrant

        :param kb_name: 知识库名（写入 payload 作为过滤字段）
        :param chunks: chunk 列表，顺序必须与 vectors 行序一致
        :param vectors: (N, dim) 向量矩阵
        :return: 写入的点数
        """
        client = self._get_client()
        dim = int(vectors.shape[1])
        self._ensure_collection(client, dim)

        points = [
            {
                "id": self._point_id(kb_name, chunk["id"]),
                "vector": vector.tolist(),
                "payload": {
                    "kb": kb_name,
                    # seq = chunk 在 chunks.jsonl 中的行号，两路检索靠它对齐
                    "seq": seq,
                    "chunk_id": chunk["id"],
                    "doc": chunk.get("doc", ""),
                    "heading": chunk.get("heading", ""),
                    # 原文一并存入 payload：Qdrant dashboard 里能直接看到命中内容，
                    # 也让跨库检索时可以脱离本地文件独立返回结果
                    "text": chunk.get("text", ""),
                },
            }
            for seq, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]

        for start in range(0, len(points), _UPSERT_BATCH):
            client.upsert(
                collection_name=config.QDRANT_COLLECTION,
                points=points[start : start + _UPSERT_BATCH],
                wait=True,
            )
        logger.info("Qdrant 写入完成: kb=%s points=%d dim=%d", kb_name, len(points), dim)
        return len(points)

    def delete_kb(self, kb_name: str) -> None:
        """删除某知识库的全部向量：全量重建前调用，避免旧点残留成孤儿数据"""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client = self._get_client()
        if not client.collection_exists(config.QDRANT_COLLECTION):
            return
        client.delete(
            collection_name=config.QDRANT_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="kb", match=MatchValue(value=kb_name))]
            ),
            wait=True,
        )

    def _ensure_collection(self, client, dim: int) -> None:
        """collection 不存在时按向量维度创建，距离用余弦（与归一化语义一致）"""
        from qdrant_client.models import Distance, VectorParams

        if client.collection_exists(config.QDRANT_COLLECTION):
            return
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Qdrant 创建 collection: %s dim=%d", config.QDRANT_COLLECTION, dim)

    # ---------- 检索 ----------

    def search(
        self, kb_name: str, query_vector: list[float], limit: int
    ) -> list[tuple[int, float]]:
        """
        在指定知识库内做向量检索（payload 过滤 kb 字段）

        :param kb_name: 知识库名
        :param query_vector: 查询向量
        :param limit: 返回条数
        :return: [(chunk 在 chunks.jsonl 中的行号, 相似度)]，按相似度降序
        :raises QdrantUnavailable: 向量库不可达，调用方应降级为 BM25
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        try:
            client = self._get_client()
            if not client.collection_exists(config.QDRANT_COLLECTION):
                return []
            hits = client.query_points(
                collection_name=config.QDRANT_COLLECTION,
                query=query_vector,
                query_filter=Filter(
                    must=[FieldCondition(key="kb", match=MatchValue(value=kb_name))]
                ),
                limit=limit,
                with_payload=True,
            ).points
        except QdrantUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 —— 查询异常同样降级处理
            raise QdrantUnavailable(f"Qdrant 检索失败: {e}") from e

        return [
            (int(hit.payload["seq"]), float(hit.score))
            for hit in hits
            if hit.payload and "seq" in hit.payload
        ]

    def count(self, kb_name: str) -> int:
        """统计某知识库已写入的向量数，用于摄入摘要与自检"""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client = self._get_client()
        if not client.collection_exists(config.QDRANT_COLLECTION):
            return 0
        result = client.count(
            collection_name=config.QDRANT_COLLECTION,
            count_filter=Filter(
                must=[FieldCondition(key="kb", match=MatchValue(value=kb_name))]
            ),
            exact=True,
        )
        return int(result.count)

    @staticmethod
    def _point_id(kb_name: str, chunk_id: str) -> str:
        """把 (知识库, chunk 业务 ID) 映射为确定性 UUID"""
        return str(uuid.uuid5(_NAMESPACE, f"{kb_name}::{chunk_id}"))


# 模块级单例：ingest 与 retriever 共用同一客户端
vector_store = QdrantVectorStore()
