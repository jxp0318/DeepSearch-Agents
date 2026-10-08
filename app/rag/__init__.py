"""
自建 RAG 模块

替代原 RAGFlow 外部服务（ADR-006）：文档摄入 → 结构感知切分 → 原文落盘 →
向量写入 Qdrant（ADR-007）、关键词索引写入 Elasticsearch（ADR-008）→
关键词路与向量路双路检索、RRF 融合排序。
检索与生成分离，由知识库子智能体拿到原始片段后自行综合，
而不是把问题转发给外部 Chat 服务。

模块构成：
- config        知识库根目录、切分参数、embedding / Qdrant / ES 配置
- chunker       结构感知切分（MD 按标题 / DOCX 按 heading / PDF 按页）
- ingest        摄入流水线：提取 → 切分 → 落盘原文 → 写 Qdrant / ES
- retriever     双路检索 + RRF 融合 + 分级降级
- vector_store  Qdrant 封装（向量路）
- keyword_store Elasticsearch 封装（关键词路，IK 分词）
"""
