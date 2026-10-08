"""
检索质量对比脚本（评测辅助工具）

对同一批问题分别用四种模式检索，并排打印 Top-K 结果，用于回答：
- ES（IK 分词 + 倒排索引 + BM25 打分）相比进程内 rank_bm25 手写实现的增益
- 向量路相比关键词路修好了哪些 case（关键词撞车、同义改写）
- RRF 融合是否把两路的优点都保留下来

四种模式（对应 retriever.search 的 mode 参数）：
  bm25     仅进程内 rank_bm25（jieba 分词 + Okapi BM25），ES 缺席时的兜底实现
  keyword  仅关键词路：ES 优先；ES 不可达时脚本会提示并自动退回进程内 BM25
  vector   仅向量路：Qdrant 余弦相似度（依赖 embedding 服务）
  hybrid   关键词路 + 向量路，RRF 融合（生产路径）

用法（在项目根目录）：
    python -m scripts.compare_retrieval                    # 用内置样例问题
    python -m scripts.compare_retrieval "自定义问题" 电商行业
"""

import sys

from app.rag import config
from app.rag.retriever import retriever

# 内置样例：前两条是纯关键词检索冒烟测试中暴露的典型问题
DEFAULT_CASES = [
    ("金融行业", "贝莱德对全球股票市场的核心观点和配置建议"),
    ("金融行业", "央行对下一阶段货币政策的基调"),
    ("电商行业", "数字人直播主要应用于哪些电商场景"),
]

TOP_K = 3
MODES = ("bm25", "keyword", "vector", "hybrid")


def _fmt(results: list[dict]) -> list[str]:
    """把检索结果压缩成「文档·章节 | 文本开头」的短行"""
    if not results:
        return ["  (无结果)"]
    return [
        f"  {i}. [{r['score']}] {r['doc'][:24]} · {r['heading'][:12]} | {r['text'][:38]}"
        for i, r in enumerate(results, start=1)
    ]


def _service_status() -> str:
    """打印各检索后端的可用性，避免把「降级结果」误读成「该模式的效果」"""
    from app.rag.keyword_store import keyword_store
    from app.rag.vector_store import vector_store

    es_state = "可用" if keyword_store.available else "不可用（keyword 模式将回退进程内 BM25）"
    qdrant_state = (
        "可用" if vector_store.available else "不可用（vector 模式将回退关键词路）"
    )
    return (
        f"后端状态 | ES({config.ES_URL}): {es_state} | "
        f"embedding: {'已配置' if config.EMBEDDING_ENABLED else '未配置'} | "
        f"Qdrant({config.QDRANT_URL}): {qdrant_state}"
    )


def main() -> None:
    if len(sys.argv) > 1:
        query = sys.argv[1]
        kb = sys.argv[2] if len(sys.argv) > 2 else "金融行业"
        cases = [(kb, query)]
    else:
        cases = DEFAULT_CASES

    print(_service_status())

    for kb, query in cases:
        print(f"\n{'=' * 78}\n知识库「{kb}」 | 问题: {query}\n{'=' * 78}")
        for mode in MODES:
            print(f"\n--- 模式: {mode} ---")
            try:
                results = retriever.search(kb, query, top_k=TOP_K, mode=mode)
                print("\n".join(_fmt(results)))
            except Exception as e:  # noqa: BLE001 —— 对比脚本，单模式失败不影响其他模式
                print(f"  (该模式不可用: {e})")


if __name__ == "__main__":
    main()
