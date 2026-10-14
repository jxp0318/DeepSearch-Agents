"""
语言判定模块（文档级 / 查询级）

字段级语言路由的地基：摄入期给每个 chunk 盖 ``lang`` / ``cjk_ratio`` / ``zh_text``
字段，查询期判断 query 属于中文还是中英混排，据此决定关键词路打哪个字段。

判据只有一条：``cjk_ratio`` = 汉字字符数 ÷ 非空白字符数。不引入语言检测模型、
不调用 LLM——摄入链路上任何外部依赖都会把它变成不确定的长尾操作，而这条判据
纯本地、零成本、可复现。

两个层级用两套阈值（见 ADR-010）：

- **文档级** ``classify_document``：整份文档合并后算一次，≥ 0.20 视为中文。
  必须是文档级而不是块级——英文报告里夹带的中文碎片块（实测 cjk≈0.15~0.17）
  若按块判定会被误判成中文，而它们恰恰是主要的噪声来源。
- **查询级** ``classify_query``：query 很短、比例分布与长文档不同，故用 0.30 判中文，
  其余一律判中英混排。混排 query 实测需要关键词路（带英文 token 能命中英文块），
  不能一刀切砍掉。**2026-10-12 起纯英文查询已下线**（英→中 / 英→英 不再支持），
  故原「≤0.02 判英文」的分支已删除，只保留 zh / mixed 两类。

**这批阈值是先验值，未做校准实验**（当前索引规模不足以扫描）。它们的作用是
「不误杀」而非「最优化」：真正的隔离由 ``zh_text`` 字段本身完成——英文文档的
``zh_text`` 为空，噪声块天生不在候选里，不依赖任何阈值过滤。阈值只负责路由决策。
"""

import re

from app.rag.config import (
    LANG_DOC_ZH_MIN_RATIO,
    LANG_QUERY_ZH_MIN_RATIO,
)

# 汉字区间：基本区 + 扩展 A + 兼容表意文字。不含中文标点/全角符号——
# 本比率要衡量的是「汉字占比」，标点混进来会稀释分母、让判定偏松。
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

_WHITESPACE_RE = re.compile(r"\s+")


def cjk_ratio(text: str) -> float:
    """
    计算文本的汉字占比

    分母是**非空白字符数**（去掉所有空白后统计），英文文本因此不会因为词间空格
    而低估中文比例；分母为 0 时返回 0.0。

    :param text: 任意文本
    :return: 0.0 ~ 1.0 的比值（未取整，调用方按需 round）
    """
    if not text:
        return 0.0
    compact = _WHITESPACE_RE.sub("", text)
    if not compact:
        return 0.0
    return len(_CJK_RE.findall(compact)) / len(compact)


def classify_document(text: str) -> tuple[str, float]:
    """
    文档级语言判定：返回 (lang, cjk_ratio)

    调用方应传入**整份文档合并后的全文**，而非单个块或单页——块级判定会把
    英文报告里的中文碎片块误判成中文（见模块注释）。

    :return: ("zh" | "en", 四舍五入到 4 位的 cjk_ratio)
    """
    ratio = cjk_ratio(text)
    return ("zh" if ratio >= LANG_DOC_ZH_MIN_RATIO else "en"), round(ratio, 4)


def classify_query(text: str) -> str:
    """
    查询级语言判定：返回 "zh" | "mixed"

    汉字占比 ≥ ``LANG_QUERY_ZH_MIN_RATIO`` 判中文，其余判中英混排——混排 query 需要
    同时打 ``zh_text`` 与 ``text`` 两个字段，这是实测结论（混合组 Recall@5：需要
    关键词路 69.9% vs 纯向量 55.1%）。

    **2026-10-12 起不再支持纯英文查询**（英→中 / 英→英 已下线），原先「占比 ≤ 0.02
    判 en」的分支已删除；纯英文输入现在落到 "mixed"，不影响中文与混排两条路径。
    """
    ratio = cjk_ratio(text)
    if ratio >= LANG_QUERY_ZH_MIN_RATIO:
        return "zh"
    return "mixed"
