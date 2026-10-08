"""
检索质量对比脚本（评测辅助工具）

对同一批问题分别用三种模式检索，并排打印 Top-K 结果，用于回答：
- 向量路相比纯 BM25 到底修好了哪些 case（关键词撞车、同义改写）
- RRF 融合是否把两路的优点都保留下来

用法（在项目根目录）：
    python -m scripts.compare_retrieval                    # 用内置样例问题
    python -m scripts.compare_retrieval "自定义问题" 电商行业
"""

import sys

from app.rag.retriever import retriever

# 内置样例：前两条是 BM25 冒烟测试中暴露的典型问题
DEFAULT_CASES = [
    ("金融行业", "贝莱德对全球股票市场的核心观点和配置建议"),
    ("金融行业", "央行对下一阶段货币政策的基调"),
    ("电商行业", "数字人直播主要应用于哪些电商场景"),
]

TOP_K = 3


def _fmt(results: list[dict]) -> list[str]:
    """把检索结果压缩成「文档·章节 | 文本开头」的短行"""
    if not results:
        return ["  (无结果)"]
    return [
        f"  {i}. [{r['score']}] {r['doc'][:24]} · {r['heading'][:12]} | {r['text'][:38]}"
        for i, r in enumerate(results, start=1)
    ]


def main() -> None:
    if len(sys.argv) > 1:
        query = sys.argv[1]
        kb = sys.argv[2] if len(sys.argv) > 2 else "金融行业"
        cases = [(kb, query)]
    else:
        cases = DEFAULT_CASES

    for kb, query in cases:
        print(f"\n{'=' * 78}\n知识库「{kb}」 | 问题: {query}\n{'=' * 78}")
        for mode in ("bm25", "vector", "hybrid"):
            print(f"\n--- 模式: {mode} ---")
            try:
                results = retriever.search(kb, query, top_k=TOP_K, mode=mode)
                print("\n".join(_fmt(results)))
            except Exception as e:  # noqa: BLE001 —— 对比脚本，单模式失败不影响其他模式
                print(f"  (该模式不可用: {e})")


if __name__ == "__main__":
    main()
