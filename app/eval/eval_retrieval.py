"""
检索层量化评测

回答的问题：四种检索模式（进程内 BM25 / ES 关键词 / 向量 / RRF 融合）在各知识库上的
检索质量如何，以及质量缺口是「没召回到」还是「排得太靠后」。

题库在 app/eval/retrieval-tasks.yaml（不在本文件里），本脚本只负责跑与算。

--------------------------------------------------------------------------
指标口径（主指标 2026-10-11 换为 chunk 级 gold，见 docs/eval-testset-v3.md）
--------------------------------------------------------------------------
**相关块的判定**（2026-10-11 起，本文件最重要的口径变更）：
题库里每道可跑题都标了 **chunk 级 gold**（`gold:` 字段，主键 = content_hash）。Top-k 里的
chunk「是否相关」直接查它是否在 gold 集里 —— **真值**，不再用「含任一锚点即相关」的近似。
无 gold 的题才回落到锚点近似，报告里会明确标注。

为什么换：fact 级 micro Recall@5 的分母是「该题的 fact 数」，命中又靠子串包含，**切块粒度一变
数字整体漂移** —— v2（400 字滑窗 + 80 字重叠）→ v3（按节切块 + 表格独立成块）同题同口径
从 87.0% 掉到 79.6%，其中相当一部分是粒度差异而非检索质量差异。gold 口径以「标准答案的出处」
为分母，语义稳定得多。

检索质量（本脚本负责，★ 项的相关块均由 gold 判定）：
  ★ Context Recall@5       【核心】chunk 级：Top-5 覆盖的 gold chunk 数 / 该题 gold 总数。
                           汇总用 micro（Σ命中 / Σgold）。回答「标准答案的出处被召回了多少」。
  ★ Context Precision@5    【核心】chunk 级 AP@5（位置加权）：相关块 = Top-5 里的 gold 块。
                           AP@k = Σ_{i: 相关} (前 i 个中相关数 / i) / R，R = Top-5 内相关块数，
                           R=0 记 0。同时反映「占比」与「位置」——同为 2 个相关块，
                           排第 1、3 位（0.83）远高于排第 4、5 位（0.35）。
  ★ MRR@5                  【核心】首个 gold 块的排名倒数，Top-5 外记 0，跨题平均。
                           唯一直接回答「第一个相关结果排得够不够靠前」的指标。
  NDCG@5                   标准归一化折损累计增益（二值相关性）：DCG = Σ rel_i / log2(i+1)，
                           IDCG = 相关块全排最前的理想 DCG，NDCG = DCG / IDCG。

参考列（兼容历史的 fact 级口径，**不当主指标**）：
  fact Recall@5            fact 级 micro：Top-5 覆盖的锚点事实数 / 该题锚点总数。
                           ⚠ 跨切块版本不可比（见上），只留作历史对照。
  Context Precision@5（占比）  相关块数 / 5。位置无关的裸占比。

检索上限（默认即算，与 search 同一次调用、无额外成本）：
  候选池召回（gold）            融合去重后、截断成 top_k 之前的候选集合覆盖的 gold 比例。
                              rerank 只能在这个集合内重排，故它就是 chunk Rec@5 的理论上限；
                              「池召回(gold) − ★Rec@5」= 精排可吃掉的排序空间。
  候选池召回（fact，参考）        同上但用锚点事实口径（历史列，仅存 payload）。

诊断列（--diag 才计算，需额外检索）：
  Recall@20（gold / fact 两口径） 用来区分「没召回」与「排得靠后」——@5 低而 @20 高即排序问题。
                              注意 @20 这一路用 top_k=20 → 每路候选 60，池比生产深得多，
                              它不是「生产池的召回」，别拿它反推「池里没有」。
  候选池召回@100（gold）        把每路候选深度从 15（=5×3）加深到 100 时的候选池覆盖。

阶段诊断（--diag / --diag-table 提供，**不额外检索**）：
  逐 gold 块排名                每个 gold 块在 向量腿 / 关键词腿 / RRF(精排前) / 精排后 四个
                              阶段的**独立**排名（不取最小排名，避免多 gold 题掩盖部分漏召回），
                              以及是否进入候选池。全部取自本次检索的真实中间结果。
  keyword_status                found / searched_not_found / skipped_by_router——区分「语言路由
                              跳过了关键词路」与「跑了但没召回」，避免把 null 误读成 0 命中。
  top5_distractors              最终 Top-5 里排在首个正确块之前的非相关块（关联 hash/seq/doc），
                              用于看「谁把正确块挤下去了」。
  阶段归因                      A=未进候选池(召回层) / B=池内但融合排名>5 / C=RRF 在 5 内被精排挤出。

性能：
  查询延迟 P50 / P95 / max     逐条查询的 wall-clock（含向量路的 embedding 调用）。
                              P95 在样本量 < 100 时 ≈ 第 2 慢的那条，统计上不稳健，
                              故同时报 P50 / max 与样本数，不单看 P95。

不在本脚本内（属 app/eval 第二阶段，见 eval.md）：
  Faithfulness / Answer Correctness——需要真实生成的答案、参考答案与 LLM judge。

--------------------------------------------------------------------------
命中判定口径（重要）
--------------------------------------------------------------------------
相关块：有 gold 的题查 content_hash 是否在 gold 集里（真值）；无 gold 的题回落到
「Top-k 拼接文本里出现任一锚点」。fact 级参考列仍用「锚点出现即算该 fact 被召回」。

gold 按**答案单元**组织：每个单元是一组「等价块」（同一答案的不同出处，如「执行摘要」
与「正文」都完整叙述同一事实），命中任一即该单元命中；单元之间是 AND。Rec 的分母 =
单元数（不是块数），这样等价块不会稀释分母（2026-10-12 修，见 docs/eval-testset-v3.md §7）。
**只标注可从原文客观判定的锚点（数值、专名、术语）**，描述型事实不配锚点、不入库——
这样每个数字都可复现、可争论，不带主观判定。锚点校准铁律见题库文件头部注释；
gold 的生成与校验见 `python -m app.eval.build_gold [--check]`。

--------------------------------------------------------------------------
用法（在项目根目录）
--------------------------------------------------------------------------
    python -m app.eval.eval_retrieval                  # 全部题目 × 五种模式
    python -m app.eval.eval_retrieval icct-q1 blk-q2   # 只跑指定题
    python -m app.eval.eval_retrieval --mode hybrid    # 只跑指定模式
    python -m app.eval.eval_retrieval --mode hybrid_rerank   # 只跑精排
    python -m app.eval.eval_retrieval --diag           # 附带 Recall@20 诊断列 + 逐 gold 块阶段排名
    python -m app.eval.eval_retrieval --diag-table     # 阶段诊断表（一行 = 题×查询语言，含归因）
    python -m app.eval.eval_retrieval --list           # 只列题库，不检索

**查询语言口径 = 中文（zh）+ 中英混排（mixed）**：2026-10-12 起纯英文查询（英→中 /
英→英）已从题库与检索链路整体下线，只保留中文与中英混排两类问法。跨语言召回
「中→英」（中文问句检索英文文档）照常测——它靠的是中文问句本身，不依赖英文查询。
见 docs/eval-testset-v3.md §7.9。

题目的知识库或对应文档（case 的 doc 字段）尚未入库时会被自动跳过并在报告里列明，
不计入任何指标（不会当成 0 分）。跳过判定在**文档级**——一个库含多份文档时，只看库
是否存在会把「锚在未入库文档上」的题目当成有效题。

结果落盘：每次运行自动写入 app/eval/results/，文件名为
    {时间戳}__retrieval__{模式}.md 与 .json（同一次运行两种格式同前缀）。
文件名带秒级时间戳，历史文件不会被覆盖；同一秒内重复运行会追加 -2、-3 后缀。
"""

import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

import yaml

from app.eval.reporting import save_run, timestamp
from app.rag import config
from app.rag.retriever import retriever

TASKS_PATH = Path(__file__).resolve().parent / "retrieval-tasks.yaml"

MODES = ("bm25", "keyword", "vector", "hybrid", "hybrid_rerank")
MODE_W = 13  # 模式列宽（容纳 hybrid_rerank）
TOP_K = 5
DIAG_K = 20
POOL_DEEP_K = 100  # 诊断用：把每路候选深度从 top_k×3(=15) 加深到 100
PCTS = (50, 95)

LANG_LABEL = {"zh": "中文Query", "mixed": "混合Query"}
DOC_LANG_LABEL = {"zh": "中文文档", "en": "英文文档"}

# 题库支持的查询语言（纯英文查询已于 2026-10-12 下线，故无 "en"）
LANGS_ALL = ("zh", "mixed")


# ---------------------------------------------------------------------------
# 题库加载
# ---------------------------------------------------------------------------
def load_tasks() -> list[dict]:
    """读取题库并做结构校验：字段缺失或锚点为空的题目直接报错，避免静默跑出假指标"""
    data = yaml.safe_load(TASKS_PATH.read_text(encoding="utf-8"))
    tasks = data.get("cases") or []
    if not tasks:
        raise RuntimeError(f"题库为空: {TASKS_PATH}")

    for case in tasks:
        cid = case.get("id", "<无 id>")
        for field in ("kb", "doc", "doc_lang", "queries", "facts"):
            if not case.get(field):
                raise ValueError(f"题目 {cid} 缺少字段 {field}")
        if "zh" not in case["queries"]:
            raise ValueError(f"题目 {cid} 的问句缺少 zh（中文问句是基准，必须有）")
        for fact in case["facts"]:
            if not fact.get("patterns"):
                raise ValueError(f"题目 {cid} 的事实「{fact.get('name')}」没有锚点")
    return tasks


def _indexed_docs(kb: str) -> set[str]:
    """该知识库已入库的文档名集合（读 meta.json.files；无 meta / 解析失败记空）"""
    meta_path = config.INDEX_ROOT / kb / "meta.json"
    if not meta_path.exists():
        return set()
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return set((meta.get("files") or {}).keys())


def _skip_reason(case: dict) -> str | None:
    """题目不可跑时返回原因；可跑返回 None。

    判定必须下沉到**文档级**：一个知识库常含多份文档，只看库是否存在会把「锚在未入库
    文档上」的题目当成有效题——锚点必然全部落空，于是 0 分混进均值（2026-10-10 踩到：
    新能源汽车出海只入了 ICCT 时，锚在德勤 / IEA 上的 kb-08~10 被误判为可跑）。
    故除「库已入库」外，还要求 case["doc"] 出现在库内 meta.files 清单里。
    """
    if not (config.INDEX_ROOT / case["kb"] / "chunks.jsonl").exists():
        return f"{case['kb']} 未入库"
    doc = case.get("doc")
    if doc and doc not in _indexed_docs(case["kb"]):
        return f"文档未入库：{doc}"
    return None


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def _fact_ranks(case: dict, top_texts: list[str]) -> dict[str, int | None]:
    """每个 fact 首次出现的排名（1-based）；未出现记 None"""
    ranks: dict[str, int | None] = {}
    for fact in case["facts"]:
        rank = None
        for i, text in enumerate(top_texts, start=1):
            if any(p in text for p in fact["patterns"]):
                rank = i
                break
        ranks[fact["name"]] = rank
    return ranks


def average_precision_at_k(flags: list[bool]) -> float:
    """
    AP@k（位置加权平均精度，RAGAs 的 Context Precision 口径）

    flags[i] 表示第 i+1 位的结果是否相关。R = k 内相关数；R=0 时返回 0
    （没有任何相关结果时，「排得准不准」无从谈起，记 0 而非 1）。
    """
    total_rel = sum(1 for f in flags if f)
    if total_rel == 0:
        return 0.0
    cum = 0
    acc = 0.0
    for i, flag in enumerate(flags, start=1):
        if flag:
            cum += 1
            acc += cum / i
    return acc / total_rel


def ndcg_at_k(flags: list[bool]) -> float:
    """
    NDCG@k（二值相关性版，业界标准口径）

    DCG  = Σ_{i=1..k} rel_i / log2(i+1)（名次越靠后折扣越重）
    IDCG = 理想排序（相关块全排最前）的 DCG = Σ_{i=1..R} 1/log2(i+1)
    NDCG = DCG / IDCG；R=0（无相关块）时记 0。

    与 AP@5 的分工：AP 用「逐点精度的平均」，对「排太后」惩罚更陡；NDCG 用对数折扣，
    更平缓。两者并排可区分「相关块太少」与「相关块排太后」两种不同的坏法。
    """
    rel = [1.0 if f else 0.0 for f in flags]
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rel, start=1))
    n_rel = int(sum(rel))
    if n_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg else 0.0


def percentile(values: list[float], pct: float) -> float:
    """最近秩法分位数；样本量小时（如 n=39 的 P95）本身不稳健，报告中一并给出样本数"""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[idx]


def _hits_in_texts(case: dict, texts: list[str]) -> int:
    """这批文本覆盖了几个事实（锚点出现即算命中）——候选池用"""
    ranks = _fact_ranks(case, texts)
    return sum(1 for v in ranks.values() if v is not None)


# gold 块未进 Top-5 的原因分层（阶段诊断表用）
STAGE_LABEL = {
    "hit": "命中@5",
    "A_recall": "A 未进候选池（召回层）",
    "B_fusion": "B 池内但融合排名>5",
    "C_rerank": "C RRF在5内但被精排挤出",
}
STAGE_ORDER = ("hit", "A_recall", "B_fusion", "C_rerank")


def _fmt_rank(rank: int | None) -> str:
    """排名格式化：None → '-'（未出现在该阶段）"""
    return "-" if rank is None else str(rank)


def _rank_in(seq_order: list[int], seq: int | None) -> int | None:
    """seq 在某个阶段排名序列里的 1-based 位置；不在其中返回 None"""
    if seq is None:
        return None
    try:
        return seq_order.index(seq) + 1
    except ValueError:
        return None


def _stage_of_block(seq: int | None, in_pool: bool, final_rank: int | None,
                    upstream_rank: int | None, has_late_stage: bool) -> str:
    """把一个 gold 块「没进 Top-5」的原因归到某一层（用于阶段诊断表）

    - hit        已进 Top-5
    - A_recall   既不在向量腿也不在关键词腿 → 真实召回失败
    - C_rerank   在池内、且上一阶段（RRF）排名 ≤5，却被后续精排挤出 → 排序层（精排）
    - B_fusion   在池内，但融合排名本就 >5 → 融合/单路排序层
    """
    if seq is None:
        return "A_recall"
    if final_rank is not None and final_rank <= TOP_K:
        return "hit"
    if not in_pool:
        return "A_recall"
    if has_late_stage and upstream_rank is not None and upstream_rank <= TOP_K:
        return "C_rerank"
    return "B_fusion"


# 跨解析器对照开关（--no-gold，见 main）：gold 主键是 content_hash，换解析器后块边界
# 全变、hash 必然对不上（不是召回差，是「标准答案换了一套切法」）。此时强制作废 gold，
# 所有题按「含任一锚点」近似判定相关块，两臂在同一口径下才可比。
FORCE_ANCHOR_RELEVANCE = False


def _gold_units(case: dict) -> list[list[str]]:
    """该题的 chunk 级 gold，按「答案单元」组织。

    每个单元是一组 **等价块**（同一答案的不同出处，如「执行摘要」与「正文」都完整
    叙述同一事实），命中任一即该单元命中；单元之间是 AND（都要被覆盖）。

    元素可以是：
      - ``"hash"``        → 单块单元；
      - ``["h1","h2"]``   → 多块 OR 单元（等价块）。

    兼容旧格式（元素全是字符串）＝ 全部单块单元。
    """
    if FORCE_ANCHOR_RELEVANCE:
        return []
    raw = case.get("gold") or []
    return [([x] if isinstance(x, str) else list(x)) for x in raw]


def measure_query(case: dict, query: str, lang: str, mode: str, diag: bool,
                  trace: bool = False) -> dict:
    """跑一条查询并算出该条的全部指标

    相关块判定：有 gold 用 gold（真值），无 gold 回落到「含任一锚点」的近似。

    ``trace=True`` 时从**本次检索**的真实中间结果里补记：每个 gold 块在 向量 / 关键词 /
    RRF / 精排 四阶段的独立排名、是否进候选池、阶段归因，以及 Top-5 干扰块。**不额外
    发起检索**，故不改变任何指标数值。
    """
    import time

    t0 = time.perf_counter()
    tr: dict | None = None
    if trace:
        results, pool_seqs, tr = retriever.search_with_pool(
            case["kb"], query, top_k=TOP_K, mode=mode, trace=True)
    else:
        results, pool_seqs = retriever.search_with_pool(
            case["kb"], query, top_k=TOP_K, mode=mode)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    chunks = retriever._load_index(case["kb"]).chunks
    units = _gold_units(case)
    gold = {h for u in units for h in u}  # 相关块全集（位置类指标用）
    has_gold = bool(units)

    top_texts = [r["text"] for r in results]
    # 相关块：gold 真值优先；无 gold 时用锚点近似（报告里会标注）
    if has_gold:
        flag5 = [r.get("content_hash") in gold for r in results]
    else:
        flag5 = [any(p in t for f in case["facts"] for p in f["patterns"]) for t in top_texts]

    # fact 级（参考列，口径不变）
    ranks = _fact_ranks(case, top_texts)
    hits5 = sum(1 for r in ranks.values() if r is not None)

    # chunk 级（主指标，gold 真值）：Top-5 覆盖了几个答案单元（单元内命中任一即算）
    top_hashes = [r.get("content_hash") for r in results]
    top_set = set(top_hashes)
    gold_hits5 = sum(1 for u in units if top_set & set(u)) if has_gold else None

    first_rel = next((i for i, f in enumerate(flag5, start=1) if f), None)

    # 候选池（生产深度）：融合去重后、截断成 top_k 之前的集合。已由 search_with_pool 取回，
    # 无额外检索成本。同时算 gold 口径（主）与 fact 口径（参考）。
    pool_hashes = [chunks[s].get("content_hash") for s in pool_seqs]
    pool_hits = _hits_in_texts(case, [chunks[s]["text"] for s in pool_seqs])
    pool_set = set(pool_hashes)
    gold_pool_hits = sum(1 for u in units if pool_set & set(u)) if has_gold else None

    rec = {
        "id": case["id"],
        "kb": case["kb"],
        "query_lang": lang,
        "doc_lang": case["doc_lang"],
        "mode": mode,
        # fact 级（参考）
        "n_facts": len(case["facts"]),
        "hits5": hits5,
        "recall5": hits5 / len(case["facts"]),
        # chunk 级（主指标）
        "n_gold": len(units),
        "has_gold": has_gold,
        "rel_source": "gold" if has_gold else "anchor",
        "gold_hits5": gold_hits5,
        "gold_recall5": (gold_hits5 / len(units)) if has_gold else None,
        # 位置类（相关块由 flag5 定义，故 gold 优先）
        "precision5_ap": average_precision_at_k(flag5),
        "precision5_ratio": (sum(flag5) / TOP_K) if top_texts else 0.0,
        "ndcg5": ndcg_at_k(flag5),
        "mrr5": (1.0 / first_rel) if first_rel else 0.0,
        # 候选池
        "pool_size": len(pool_seqs),
        "pool_hits": pool_hits,
        "pool_recall": pool_hits / len(case["facts"]),
        "gold_pool_hits": gold_pool_hits,
        "gold_pool_recall": (gold_pool_hits / len(units)) if has_gold else None,
        "latency_ms": elapsed_ms,
        "ranks": ranks,
        "n_results": len(results),
        "docs": sorted({r["doc"] for r in results}),
    }

    if tr is not None:
        # content_hash → seq（同文本可能多块共享 hash，取首个 seq 做定位锚点；
        # 命中判定走集合，不受影响）
        seq_of_hash: dict[str, int] = {}
        for i, c in enumerate(chunks):
            seq_of_hash.setdefault(c.get("content_hash"), i)

        pool_seq_set = set(pool_seqs)
        v_order = tr["vector_seqs"]
        k_order = tr["keyword_seqs"]
        r_order = tr["rrf_seqs"]
        rr_order = tr["rerank_seqs"]  # None ⇒ 本模式无精排阶段
        has_late_stage = rr_order is not None
        final_order = rr_order if has_late_stage else r_order

        gold_blocks: list[dict] = []
        for u in units:  # 单元内逐块独立记录（不取最小排名）
            for h in u:
                seq = seq_of_hash.get(h)
                c = chunks[seq] if seq is not None else {}
                v_rank = _rank_in(v_order, seq)
                k_rank = _rank_in(k_order, seq)
                r_rank = _rank_in(r_order, seq)
                rr_rank = _rank_in(rr_order, seq) if has_late_stage else None
                gold_blocks.append({
                    "hash": h,
                    "seq": seq,
                    "doc": c.get("doc"),
                    "type": c.get("chunk_type"),
                    "page": c.get("page_index"),
                    "in_pool": (seq in pool_seq_set) if seq is not None else False,
                    "vector_rank": v_rank,
                    "keyword_rank": k_rank,
                    "rrf_rank": r_rank,
                    "rerank_rank": rr_rank,
                    "stage": _stage_of_block(
                        seq,
                        (seq in pool_seq_set) if seq is not None else False,
                        _rank_in(final_order, seq),
                        r_rank,
                        has_late_stage,
                    ),
                })
        rec["gold_blocks"] = gold_blocks
        rec["keyword_status"] = tr["keyword_status"]
        rec["candidate_k"] = tr["candidate_k"]
        rec["pool_size_pre_rerank"] = len(r_order)
        # 三段耗时（毫秒）：① 召回+RRF 融合 ② 精排单独 ③ 检索层合计。
        # 与 latency_ms 的分工：latency_ms 是**端到端**（含 _materialize 取原文与对象构造），
        # 这三项是**检索层内部**拆解，用来回答「那 900ms 到底花在哪」。
        rec["t_fusion_ms"] = tr.get("t_fusion_ms")
        rec["t_rerank_ms"] = tr.get("t_rerank_ms")
        rec["t_total_ms"] = tr.get("t_total_ms")

        # Top-5 干扰块：排在首个正确块**之前**的非相关块（Top-5 无正确块时全部计入）
        top_hashes = [r.get("content_hash") for r in results]
        first_gold = next((i for i, h in enumerate(top_hashes) if h in gold), None)
        distractors: list[dict] = []
        for i, r in enumerate(results):
            h = r.get("content_hash")
            if h in gold:
                continue
            if first_gold is not None and i > first_gold:
                continue
            distractors.append({
                "rank": i + 1,
                "hash": h,
                "seq": seq_of_hash.get(h),
                "doc": r.get("doc"),
                "type": r.get("chunk_type"),
                "page": r.get("page_index"),
            })
        rec["top5_distractors"] = distractors

    if diag:
        # 诊断 1：放大到 @20，判断「没召回」还是「排得靠后」（gold 与 fact 两口径）
        deep = retriever.search(case["kb"], query, top_k=DIAG_K, mode=mode)
        deep_ranks = _fact_ranks(case, [r["text"] for r in deep])
        rec["hits20"] = sum(1 for r in deep_ranks.values() if r is not None)
        rec["recall20"] = rec["hits20"] / len(case["facts"])
        if has_gold:
            dset = {r.get("content_hash") for r in deep}
            g20 = sum(1 for u in units if dset & set(u))
            rec["gold_hits20"] = g20
            rec["gold_recall20"] = g20 / len(units)
        # 诊断 2：把每路候选深度加深到 100，看候选池上限
        deep_pool = retriever.candidate_pool(
            case["kb"], query, top_k=TOP_K, mode=mode, candidate_k=POOL_DEEP_K
        )
        deep_pool_hits = _hits_in_texts(case, [chunks[s]["text"] for s in deep_pool])
        rec["pool_hits100"] = deep_pool_hits
        rec["pool_recall100"] = deep_pool_hits / len(case["facts"])
        if has_gold:
            dpset = {chunks[s].get("content_hash") for s in deep_pool}
            gp = sum(1 for u in units if dpset & set(u))
            rec["gold_pool_hits100"] = gp
            rec["gold_pool_recall100"] = gp / len(units)

    return rec


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def aggregate(records: list[dict], diag: bool) -> dict:
    """把一批逐条结果聚合成一组指标。

    主指标 chunk 级用 micro（Σ gold 命中 / Σ gold 总数）——「整个测试集里标准答案的出处
    被覆盖了多少」。fact 级 Recall@5 同理但分母是锚点数，作参考列。
    位置类（AP / MRR / NDCG / 占比）天然是题级量，一律题均（macro）。
    """
    if not records:
        return {}
    total_facts = sum(r["n_facts"] for r in records)
    gold_recs = [r for r in records if r.get("has_gold")]
    total_gold = sum(r["n_gold"] for r in gold_recs)
    lat = [r["latency_ms"] for r in records]
    out = {
        "n_queries": len(records),
        "n_gold_queries": len(gold_recs),
        "n_gold_units": total_gold,
        # ---- chunk 级（主指标，gold 真值）----
        "gold_recall5_micro": (sum(r["gold_hits5"] for r in gold_recs) / total_gold) if total_gold else None,
        "gold_pool_recall_micro": (
            sum(r["gold_pool_hits"] for r in gold_recs) / total_gold) if total_gold else None,
        # ---- 位置类（相关块 = gold；无 gold 的题回落锚点近似）----
        "precision5_ap": statistics.fmean(r["precision5_ap"] for r in records),
        "precision5_ratio": statistics.fmean(r["precision5_ratio"] for r in records),
        "ndcg5": statistics.fmean(r["ndcg5"] for r in records),
        "mrr5": statistics.fmean(r["mrr5"] for r in records),
        # ---- fact 级（参考列）----
        "recall5_micro": sum(r["hits5"] for r in records) / total_facts,
        "recall5_macro": statistics.fmean(r["recall5"] for r in records),
        "pool_recall_micro": sum(r["pool_hits"] for r in records) / total_facts,
        "latency_p50": percentile(lat, 50),
        "latency_p95": percentile(lat, 95),
        "latency_max": max(lat),
    }
    # 三段耗时拆解（仅 trace 运行时才有值；缺值给 None 而不是 0，避免误读成「不耗时」）
    for name, key in (("fusion", "t_fusion_ms"), ("rerank", "t_rerank_ms"),
                      ("search_layer", "t_total_ms")):
        vals = [r[key] for r in records if r.get(key) is not None]
        if not vals:
            out[f"{name}_p50_ms"] = out[f"{name}_p95_ms"] = out[f"{name}_max_ms"] = None
            continue
        out[f"{name}_p50_ms"] = percentile(vals, 50)
        out[f"{name}_p95_ms"] = percentile(vals, 95)
        out[f"{name}_max_ms"] = max(vals)
        out[f"{name}_n"] = len(vals)
    if diag:
        out["recall20_micro"] = sum(r["hits20"] for r in records) / total_facts
        out["pool_recall100_micro"] = sum(r["pool_hits100"] for r in records) / total_facts
        if total_gold:
            out["gold_recall20_micro"] = sum(r["gold_hits20"] for r in gold_recs) / total_gold
            out["gold_pool_recall100_micro"] = (
                sum(r["gold_pool_hits100"] for r in gold_recs) / total_gold)
    return out


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}ms"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _service_status() -> str:
    """打印后端可用性，避免把「降级结果」误读成「该模式的效果」"""
    from app.rag.keyword_store import keyword_store
    from app.rag.reranker import reranker
    from app.rag.vector_store import vector_store

    return (
        f"后端状态 | ES({config.ES_URL}): {'可用' if keyword_store.available else '不可用'}"
        f" | embedding: {'已配置' if config.EMBEDDING_ENABLED else '未配置'}"
        f" | Qdrant({config.QDRANT_URL}): {'可用' if vector_store.available else '不可用'}"
        f" | 精排: {reranker.status()}"
    )


def main() -> None:
    # 解析参数：--mode 后面跟的是**取值**（如 hybrid_rerank），不以 -- 开头，
    # 必须先把它连同取值一起从位置参数里摘掉，否则会被当成题目 id 过滤（曾导致题库全空）。
    global FORCE_ANCHOR_RELEVANCE
    FORCE_ANCHOR_RELEVANCE = "--no-gold" in sys.argv

    argv = sys.argv[1:]
    modes = MODES
    if "--mode" in argv:
        i = argv.index("--mode")
        modes = (argv[i + 1],)
        argv = argv[:i] + argv[i + 2:]
    diag = "--diag" in argv
    diag_table = "--diag-table" in argv
    timing = "--timing" in argv
    # --rerank-k 后面跟的是**取值**（如 15），不以 -- 开头，必须先把它连同取值一起
    # 从位置参数里摘掉，否则会被当成题目 id 过滤（与 --mode 同一个坑）。
    rerank_k: int | None = None
    if "--rerank-k" in argv:
        i = argv.index("--rerank-k")
        if i + 1 >= len(argv):
            raise SystemExit("--rerank-k 需要一个整数取值，例如 --rerank-k 15")
        try:
            rerank_k = int(argv[i + 1])
        except ValueError:
            raise SystemExit(f"--rerank-k 取值必须是整数，收到 {argv[i + 1]!r}")
        argv = argv[:i] + argv[i + 2:]
    # --rerank-bs：单次前向的候选批大小（同样的坑，取值不以 -- 开头）
    rerank_bs: int | None = None
    if "--rerank-bs" in argv:
        i = argv.index("--rerank-bs")
        if i + 1 >= len(argv):
            raise SystemExit("--rerank-bs 需要一个整数取值，例如 --rerank-bs 8")
        try:
            rerank_bs = int(argv[i + 1])
        except ValueError:
            raise SystemExit(f"--rerank-bs 取值必须是整数，收到 {argv[i + 1]!r}")
        argv = argv[:i] + argv[i + 2:]
    # trace 只读地采集中间排名与三段耗时：--diag / --diag-table / --timing 都需要
    # （全部取自本次检索的真实中间结果，不额外发起检索）
    trace = diag or diag_table or timing
    args = [a for a in argv if not a.startswith("--")]

    # 精排候选深度可被本次运行覆盖（候选池深度扫描用；不改磁盘上的配置，也不影响
    # 其它进程）。0 表示「沿用 top_k×3」。
    if rerank_k is not None:
        config.RERANK_CANDIDATE_K = rerank_k
    if rerank_bs is not None:
        config.RERANK_BATCH_SIZE = rerank_bs

    tasks = load_tasks()
    if args:
        tasks = [c for c in tasks if c["id"] in args]

    if "--list" in sys.argv:
        total_gold = sum(len(c.get("gold") or []) for c in tasks)
        print(f"题库：{TASKS_PATH}")
        print(f"共 {len(tasks)} 道题 / {sum(len(c['facts']) for c in tasks)} 个锚点 / "
              f"{total_gold} 个 gold 单元\n")
        print(f"{'id':<10} {'kb':<12} {'文档':<5} {'问句':<14} {'锚点':>4} {'gold':>5}  知识库状态")
        print("-" * 80)
        for case in tasks:
            reason = _skip_reason(case)
            langs = "+".join(case["queries"].keys())
            print(
                f"{case['id']:<10} {case['kb']:<12} {case['doc_lang']:<5} {langs:<14} "
                f"{len(case['facts']):>4} {len(case.get('gold') or []):>5}  "
                f"{'可跑' if reason is None else f'跳过（{reason}）'}"
            )
        return

    lines: list[str] = []

    def emit(text: str = "") -> None:
        """同时打印到终端与收集到报告，供最终落盘"""
        print(text)
        lines.append(text)

    # 1) 过滤掉未入库的库；这类题不计入指标（不是 0 分）
    runnable: list[dict] = []
    skipped: list[dict] = []
    for case in tasks:
        if _skip_reason(case) is None:
            runnable.append(case)
        else:
            skipped.append(case)
    if not runnable:
        print("没有任何可跑的题目：题目的知识库或对应文档都尚未入库。")
        print("请先执行：python -m app.rag.ingest")
        for case in skipped:
            print(f"  跳过 {case['id']}（{_skip_reason(case)}）")
        return

    # 2) 预取每个库的 chunk 原文，用于判定「相关块」
    kb_texts: dict[str, list[str]] = {}
    for kb in sorted({c["kb"] for c in runnable}):
        kb_texts[kb] = [c["text"] for c in retriever._load_index(kb).chunks]

    # 2.5) 预热精排：模型加载（2.2GB）与 CUDA 上下文初始化只做一次，把它从首条
    # 查询的计时里摘出去，否则 hybrid_rerank 的延迟会被这一笔一次性开销污染。
    if "hybrid_rerank" in modes:
        from app.rag.reranker import reranker

        reranker.warmup()

    run_at = datetime.now()
    generated_at = run_at.isoformat(timespec="seconds")
    emit(f"# 检索层评测报告（{generated_at}）")
    emit()
    emit(f"- 测试方式：{'+'.join(modes)}")
    if FORCE_ANCHOR_RELEVANCE:
        emit("- ⚠ 已启用 --no-gold：相关块一律按「含任一锚点」近似判定（跨解析器对照专用），"
             "chunk 级 gold 列不适用")
    n_gold_units = sum(len(c.get("gold") or []) for c in runnable)
    emit(f"- 题库：{TASKS_PATH.name}（跑 {len(runnable)} 题 / "
         f"{sum(len(c['facts']) for c in runnable)} 个锚点 / {n_gold_units} 个 gold 单元）")
    if skipped:
        detail = "；".join(f"{c['id']}（{_skip_reason(c)}）" for c in skipped)
        emit(f"- 跳过（库或文档未入库，不计入指标）：{detail}")
    no_gold = [c["id"] for c in runnable if not c.get("gold")]
    if no_gold:
        emit(f"- ⚠ 无 chunk 级 gold 的题（相关块回落锚点近似，不计入 chunk 级 micro）：{'、'.join(no_gold)}")
    emit("- 指标口径：★Context Recall@5（chunk 级·gold 真值 micro）、★Context Precision@5（chunk 级 AP@5）、"
         "★MRR@5（chunk 级）、NDCG@5；参考列 fact Rec@5（fact 级 micro，跨切块版本不可比）；"
         "候选池召回(gold)、查询延迟 P50/P95/max；Top-k = 5")
    emit(f"- {_service_status()}")
    if "hybrid_rerank" in modes:
        from app.rag.reranker import reranker as _rr

        override = []
        if rerank_k is not None:
            override.append(f"--rerank-k {rerank_k}")
        if rerank_bs is not None:
            override.append(f"--rerank-bs {rerank_bs}")
        emit(f"- 精排配置：候选深度 RERANK_CANDIDATE_K={config.RERANK_CANDIDATE_K}（每路）、"
             f"batch_size={config.RERANK_BATCH_SIZE}、RERANK_MAX_LENGTH={config.RERANK_MAX_LENGTH}"
             + (f"  ← 本次运行覆盖：{'、'.join(override)}" if override else ""))
        # 权重标识写进报告：fp32 / fp16 两臂的差异必须在报告里可追溯（同名 model.onnx，
        # 只有子目录不同），否则事后无法判断某次数字出自哪份权重。
        _sub = config.rerank_onnx_subdir()
        _wdir = config.RERANK_MODEL_DIR / _sub
        _bytes = sum(p.stat().st_size for p in _wdir.glob("model.onnx*")) if _wdir.is_dir() else 0
        emit(f"- 精排执行：{_rr.status()}；权重 {_sub}/model.onnx（{_bytes / 2**30:.2f} GiB，"
             f"RERANK_ONNX_SUBDIR={config.RERANK_ONNX_SUBDIR}）")
    emit()

    # 3) 逐模式跑全部查询
    per_mode: dict[str, list[dict]] = {}
    for mode in modes:
        records: list[dict] = []
        for case in runnable:
            for lang, query in case["queries"].items():
                records.append(measure_query(case, query, lang, mode, diag, trace))
        per_mode[mode] = records

    summary = {mode: aggregate(recs, diag) for mode, recs in per_mode.items()}

    # ---- 汇总表（主指标 = chunk 级 gold）----
    emit("=" * 110)
    emit("汇总（行 = 检索模式；★ = 主指标，相关块由 chunk 级 gold 判定）")
    emit("=" * 110)
    emit(f"{'模式':<13} | {'★Rec@5':>8} | {'★Prec@5':>8} | {'★MRR@5':>7} | {'NDCG@5':>7} | "
         f"{'占比@5':>7} | {'factRec@5':>9} | {'延迟':^26}")
    emit(f"{'':<9} | {'(chunk)':>8} | {'(AP)':>8} | {'':>7} | {'':>7} | "
         f"{'':>7} | {'(参考)':>9} | {'P50':>8} {'P95':>8} {'max':>8}")
    emit("-" * 110)
    for mode in modes:
        s = summary[mode]
        emit(
            f"{mode:<13} | {_fmt_pct(s['gold_recall5_micro']):>8} | {s['precision5_ap']:>8.3f} | "
            f"{s['mrr5']:>7.3f} | {s['ndcg5']:>7.3f} | "
            f"{_fmt_pct(s['precision5_ratio']):>7} | {_fmt_pct(s['recall5_micro']):>9} | "
            f"{_fmt_ms(s['latency_p50']):>8} {_fmt_ms(s['latency_p95']):>8} {_fmt_ms(s['latency_max']):>8}"
        )
    emit()
    emit("★Rec@5(chunk) = Top-5 覆盖的 gold chunk 数 / 该题 gold 总数（micro）；")
    emit("★Prec@5/★MRR@5/NDCG@5/占比@5 为题均，相关块 = Top-5 里的 gold 块。")
    emit("factRec@5 是历史的 fact 级 micro 口径，**跨切块版本不可比**，仅留作对照（见文件头口径说明）。")
    emit(f"延迟样本数 = {summary[modes[0]]['n_queries']} 条/模式"
         + ("（样本量小时 P95 ≈ 第 2 慢那条，请连同 P50 / max 一起看）" if summary[modes[0]]["n_queries"] < 100 else ""))

    # ---- 检索上限：候选池召回(gold) vs chunk Rec@5（精排能补多少）----
    emit()
    emit("=" * 110)
    emit("检索上限（候选池召回 = 融合去重后、截断成 top-5 之前的候选集合覆盖的 gold 比例）")
    emit("=" * 110)
    if diag:
        emit(f"{'模式':<13} | {'池召回(gold)':>12} | {'★Rec@5':>8} | {'精排可吃':>8} | "
             f"{'池召回@100(gold)':>17} | {'加深收益':>8}")
        emit("-" * 100)
    else:
        emit(f"{'模式':<13} | {'池召回(gold)':>12} | {'★Rec@5':>8} | {'精排可吃':>8}")
        emit("-" * 60)
    for mode in modes:
        s = summary[mode]
        pr, cr = s.get("gold_pool_recall_micro"), s.get("gold_recall5_micro")
        gap = f"{pr - cr:+.1%}" if (pr is not None and cr is not None) else "n/a"
        row = f"{mode:<13} | {_fmt_pct(pr):>12} | {_fmt_pct(cr):>8} | {gap:>8}"
        if diag:
            p100 = s.get("gold_pool_recall100_micro")
            gain = f"{p100 - pr:+.1%}" if (p100 is not None and pr is not None) else "n/a"
            row += f" | {_fmt_pct(p100):>17} | {gain:>8}"
        emit(row)
    emit("精排可吃 = 池召回(gold) − chunk Rec@5：rerank 在当前候选池内最多能补回的 gold 比例（排序缺口）。")
    emit("加深收益 = 池召回@100 − 池召回@生产（gold 口径）：把每路候选从 15 加深到 100 的召回缺口。")
    emit("参考：fact 口径的池召回 " + "；".join(
        f"{m} {_fmt_pct(summary[m]['pool_recall_micro'])}" for m in modes))

    if diag:
        emit()
        emit("诊断列（放大到 @20：@5 低而 @20 高 = 排序问题，而不是没召回）")
        emit(f"{'模式':<13} | {'Rec@5(gold)':>12} | {'Rec@20(gold)':>13} | {'差额':>8} | "
             f"{'Rec@5(fact)':>12} | {'Rec@20(fact)':>13}")
        emit("-" * 94)
        for mode in modes:
            s = summary[mode]
            g5, g20 = s.get("gold_recall5_micro"), s.get("gold_recall20_micro")
            d = f"{g20 - g5:+.1%}" if (g5 is not None and g20 is not None) else "n/a"
            emit(f"{mode:<13} | {_fmt_pct(g5):>12} | {_fmt_pct(g20):>13} | {d:>8} | "
                 f"{_fmt_pct(s['recall5_micro']):>12} | {_fmt_pct(s['recall20_micro']):>13}")

    # ---- 按查询语言分组 ----
    langs = [lg for lg in LANGS_ALL if any(lg in c["queries"] for c in runnable)]
    if len(langs) > 1:
        emit()
        emit("=" * 104)
        emit("按查询语言分组（避免一种语言的改善掩盖另一种的退化）")
        emit("=" * 104)
        emit(f"{'模式':<13} | " + " | ".join(
            f"{LANG_LABEL[lg]:^34}" for lg in langs))
        emit(f"{'':<9} | " + " | ".join(
            f"{'★Rec@5':>8} {'★Prec@5':>8} {'MRR@5':>7} {'NDCG@5':>8}" for _ in langs))
        emit("-" * 124)
        for mode in modes:
            cells = []
            for lg in langs:
                sub = [r for r in per_mode[mode] if r["query_lang"] == lg]
                a = aggregate(sub, diag)
                cells.append(
                    f"{_fmt_pct(a.get('gold_recall5_micro')):>8} {a['precision5_ap']:>8.3f} "
                    f"{a['mrr5']:>7.3f} {a['ndcg5']:>8.3f}"
                    if a else f"{'n/a':>8} {'n/a':>8} {'n/a':>7} {'n/a':>8}"
                )
            emit(f"{mode:<13} | " + " | ".join(cells))

    # ---- 按「查询语言 × 文档语言」分组 ----
    if len(langs) > 1:
        emit()
        emit("=" * 104)
        emit("按「查询语言 × 文档语言」分组（跨语言失配的直接证据）")
        emit("=" * 104)
        pairs = [(lg, dl) for lg in langs for dl in ("zh", "en")
                 if any(r["query_lang"] == lg and r["doc_lang"] == dl for r in per_mode[modes[0]])]
        emit(f"{'模式':<13} | " + " | ".join(
            f"{LANG_LABEL[lg]}×{DOC_LANG_LABEL[dl]:^10}" for lg, dl in pairs))
        emit("-" * 108)
        for mode in modes:
            cells = []
            for lg, dl in pairs:
                sub = [r for r in per_mode[mode]
                       if r["query_lang"] == lg and r["doc_lang"] == dl]
                a = aggregate(sub, diag)
                cells.append(f"{_fmt_pct(a.get('gold_recall5_micro')):>16}" if a else f"{'n/a':>16}")
            emit(f"{mode:<13} | " + " | ".join(cells))

    # ---- 明细：逐条列出 gold 命中与未召回的 fact ----
    emit()
    emit("=" * 110)
    emit("明细（逐条查询；gold 为 chunk 级命中，未召回 fact 用于定位锚点缺口）")
    emit("=" * 110)
    for mode in modes:
        emit(f"\n--- {mode} ---")
        for r in per_mode[mode]:
            missed = [n for n, rk in r["ranks"].items() if rk is None]
            hit = [f"{n}@{rk}" for n, rk in r["ranks"].items() if rk]
            g = f"{r['gold_hits5']}/{r['n_gold']}" if r["has_gold"] else "n/a"
            emit(f"  {r['id']}/{r['query_lang']}  gold {g} | fact {len(hit)}/{r['n_facts']}"
                 f" | Prec@5(AP) {r['precision5_ap']:.2f} | {r['latency_ms']:.0f}ms")
            if hit:
                emit(f"        命中 fact: {'、'.join(hit)}")
            if missed:
                emit(f"        未召回 fact: {'、'.join(missed)}")
            if trace and r.get("gold_blocks"):
                emit(f"        [阶段] keyword_status={r.get('keyword_status')}"
                     f"  候选池(精排前)={r.get('pool_size_pre_rerank')}")
                for b in r["gold_blocks"]:
                    emit(f"          {b['hash']} seq={b['seq']} {b['type']} p{b['page']}"
                         f" | vec={_fmt_rank(b['vector_rank'])} kw={_fmt_rank(b['keyword_rank'])}"
                         f" rrf={_fmt_rank(b['rrf_rank'])} rerank={_fmt_rank(b['rerank_rank'])}"
                         f" in_pool={'Y' if b['in_pool'] else 'N'} → {STAGE_LABEL[b['stage']]}")
                for d in r.get("top5_distractors", []):
                    emit(f"          ⤷ 干扰#{d['rank']} {d['hash']} seq={d['seq']}"
                         f" {d['type']} p{d['page']} doc={d['doc']}")

    # ---- RRF 融合 vs 向量单路（简历可用的对比数字）----
    if "hybrid" in summary and "vector" in summary:
        h, v = summary["hybrid"], summary["vector"]
        emit()
        emit("=" * 104)
        emit("RRF 融合（hybrid）vs 向量单路（vector）")
        emit("=" * 104)
        emit(f"  ★ Context Recall@5    : {_fmt_pct(v['gold_recall5_micro'])} → {_fmt_pct(h['gold_recall5_micro'])}"
             f"  ({h['gold_recall5_micro'] - v['gold_recall5_micro']:+.1%})   [chunk 级 gold]")
        emit(f"  ★ Context Precision@5 : {v['precision5_ap']:.3f} → {h['precision5_ap']:.3f}"
             f"  ({h['precision5_ap'] - v['precision5_ap']:+.3f})")
        emit(f"  ★ MRR@5               : {v['mrr5']:.3f} → {h['mrr5']:.3f}"
             f"  ({h['mrr5'] - v['mrr5']:+.3f})")
        emit(f"  NDCG@5                : {v['ndcg5']:.3f} → {h['ndcg5']:.3f}"
             f"  ({h['ndcg5'] - v['ndcg5']:+.3f})")
        emit(f"  候选池召回(gold)      : {_fmt_pct(v['gold_pool_recall_micro'])} → {_fmt_pct(h['gold_pool_recall_micro'])}"
             f"  ({h['gold_pool_recall_micro'] - v['gold_pool_recall_micro']:+.1%})")

    # ---- 精排（hybrid_rerank）vs 纯 RRF（hybrid）：检索侧最有效的杠杆 ----
    if "hybrid" in summary and "hybrid_rerank" in summary:
        h, r = summary["hybrid"], summary["hybrid_rerank"]
        emit()
        emit("=" * 104)
        emit("精排（hybrid_rerank）vs 纯 RRF 融合（hybrid）")
        emit("=" * 104)
        emit(f"  ★ Context Recall@5    : {_fmt_pct(h['gold_recall5_micro'])} → {_fmt_pct(r['gold_recall5_micro'])}"
             f"  ({r['gold_recall5_micro'] - h['gold_recall5_micro']:+.1%})   [chunk 级 gold]")
        emit(f"  ★ Context Precision@5 : {h['precision5_ap']:.3f} → {r['precision5_ap']:.3f}"
             f"  ({r['precision5_ap'] - h['precision5_ap']:+.3f})")
        emit(f"  ★ MRR@5               : {h['mrr5']:.3f} → {r['mrr5']:.3f}"
             f"  ({r['mrr5'] - h['mrr5']:+.3f})")
        emit(f"  NDCG@5                : {h['ndcg5']:.3f} → {r['ndcg5']:.3f}"
             f"  ({r['ndcg5'] - h['ndcg5']:+.3f})")
        emit(f"  命中率@5（占比）      : {_fmt_pct(h['precision5_ratio'])} → {_fmt_pct(r['precision5_ratio'])}"
             f"  ({r['precision5_ratio'] - h['precision5_ratio']:+.1%})")
        emit(f"  延迟 P50 / P95        : {_fmt_ms(h['latency_p50'])} / {_fmt_ms(h['latency_p95'])}"
             f" → {_fmt_ms(r['latency_p50'])} / {_fmt_ms(r['latency_p95'])}")
        emit(f"  候选池召回(gold)      : {_fmt_pct(h['gold_pool_recall_micro'])} → {_fmt_pct(r['gold_pool_recall_micro'])}"
             f"  ({r['gold_pool_recall_micro'] - h['gold_pool_recall_micro']:+.1%})")
        if abs(r["gold_pool_recall_micro"] - h["gold_pool_recall_micro"]) < 1e-9:
            emit("  两模式候选池召回相同（本模式下池深一致）⇒「池召回 − chunk Rec@5」即精排吃掉的排序缺口。")
        else:
            emit("  ⚠ 两模式候选池深度不同（hybrid 每路 top_k×3 vs 精排 RERANK_CANDIDATE_K）——")
            emit("    此处 hybrid→hybrid_rerank 的涨幅同时含「池加深」与「重排」两部分，不可只归因于精排。")
        if abs(r["gold_recall5_micro"] - h["gold_recall5_micro"]) < 1e-9:
            emit("  ⚠ 两模式结果完全一致 —— 精排未生效（检查上方「后端状态 | 精排」一行）")

    # ---- 耗时拆解（--timing）：三个独立耗时，各出 P50 / P95 / max ----
    if timing:
        emit()
        emit("=" * 104)
        emit("耗时拆解（三段独立计时；P95 用最近秩法——样本量小时约等于第 2 慢那条）")
        emit("=" * 104)
        emit("①召回+RRF 与 ②精排 是**检索层内部**拆解；③检索层合计 = ①+②+结果整理；"
             "④端到端 = ③ + _materialize（取原文、构造字典），即对外可观测的延迟。")
        emit("")
        for mode in modes:
            s = summary[mode]
            n = s.get("fusion_n")
            if not n:
                emit(f"  {mode}：无耗时拆解（需 --timing / --diag / --diag-table 运行）")
                emit()
                continue
            f50, f95, fmx = s["fusion_p50_ms"], s["fusion_p95_ms"], s["fusion_max_ms"]
            r50, r95, rmx = s["rerank_p50_ms"], s["rerank_p95_ms"], s["rerank_max_ms"]
            t50, t95, tmx = s["search_layer_p50_ms"], s["search_layer_p95_ms"], s["search_layer_max_ms"]
            e50, e95, emx = s["latency_p50"], s["latency_p95"], s["latency_max"]
            emit(f"  【{mode}】样本 {n} 条")
            emit(f"    ① 召回 + RRF 融合   : P50 {_fmt_ms(f50):>7} | P95 {_fmt_ms(f95):>7} | max {_fmt_ms(fmx):>7}")
            emit(f"    ② 精排单独推理      : P50 {_fmt_ms(r50):>7} | P95 {_fmt_ms(r95):>7} | max {_fmt_ms(rmx):>7}")
            emit(f"    ③ 检索层合计        : P50 {_fmt_ms(t50):>7} | P95 {_fmt_ms(t95):>7} | max {_fmt_ms(tmx):>7}")
            emit(f"    ④ 端到端(③+取原文)  : P50 {_fmt_ms(e50):>7} | P95 {_fmt_ms(e95):>7} | max {_fmt_ms(emx):>7}")
            if t50:
                # ⚠️ 分位数不可加：① 的 P50 + ② 的 P50 与 ③ 的 P50 会有几个百分点的出入
                # （两者不是同一批查询的同一分位），故这里只作**粗略占比**看。
                emit(f"    占比（按各自 P50：① {f50 / t50:.0%} / ② {r50 / t50:.0%}"
                     f" / 取原文 {((e50 - t50) / e50 if e50 else 0):.0%}；分位数不可加，仅供粗看）")
            emit("")

    # ---- 阶段诊断表（--diag-table）：gold 块在四阶段的独立排名 + 归因 ----
    if diag_table:
        from collections import Counter

        src_mode = "hybrid_rerank" if "hybrid_rerank" in per_mode else (modes[-1] if modes else None)
        recs = per_mode.get(src_mode, []) if src_mode else []
        emit()
        emit("=" * 120)
        emit("阶段诊断表（一行 = 题 × 查询语言）")
        emit("=" * 120)
        emit(f"取自 {src_mode} 模式**本次实际检索**的中间排名（不额外检索）；"
             "vec/kw/rrf/rerank = 该 gold 块在各阶段的独立排名，'-' = 未出现在该阶段")
        emit("阶段归因：" + " / ".join(STAGE_LABEL[k] for k in STAGE_ORDER))
        emit("")
        all_blocks = [b for r in recs for b in r.get("gold_blocks", [])]
        cnt = Counter(b["stage"] for b in all_blocks)
        emit(f"gold 块总数 = {len(all_blocks)}；阶段分布：" +
             "、".join(f"{STAGE_LABEL[k]}={cnt.get(k, 0)}" for k in STAGE_ORDER))
        kstat = Counter(r.get("keyword_status") for r in recs if r.get("has_gold"))
        emit("keyword_status 分布（gold 题）：" +
             "、".join(f"{k}={v}" for k, v in sorted(kstat.items(), key=lambda x: str(x[0]))))
        emit("")
        for r in recs:
            if not r.get("has_gold"):
                continue
            emit(f"— {r['id']} [{r['query_lang']}] {r['kb']} {DOC_LANG_LABEL[r['doc_lang']]}"
                 f" | gold单元 {r['gold_hits5']}/{r['n_gold']}"
                 f" | kw_status={r.get('keyword_status')} 池={r.get('pool_size_pre_rerank')}")
            for b in r.get("gold_blocks", []):
                emit(f"    {b['hash']} seq={str(b['seq']):>4} {str(b['type']):<6} p{str(b['page']):<4}"
                     f" | vec={_fmt_rank(b['vector_rank']):>4} kw={_fmt_rank(b['keyword_rank']):>4}"
                     f" rrf={_fmt_rank(b['rrf_rank']):>4} rerank={_fmt_rank(b['rerank_rank']):>4}"
                     f" | in_pool={'Y' if b['in_pool'] else 'N'} → {STAGE_LABEL[b['stage']]}")
            for d in r.get("top5_distractors", []):
                emit(f"    ⤷ 干扰#{d['rank']} {d['hash']} seq={d['seq']} {d['type']} p{d['page']}")
        emit("")
        emit("按「查询语言 × 文档语言」汇总阶段分布（复用现有 2×2 口径）：")
        for lg in langs:
            for dl in ("zh", "en"):
                sub_q = [r for r in recs if r["query_lang"] == lg and r["doc_lang"] == dl]
                sub_b = [b for r in sub_q for b in r.get("gold_blocks", [])]
                if not sub_b:
                    continue
                c = Counter(b["stage"] for b in sub_b)
                dist = "、".join(f"{STAGE_LABEL[k]}={c[k]}" for k in STAGE_ORDER if c.get(k))
                ks = Counter(r.get("keyword_status") for r in sub_q if r.get("has_gold"))
                emit(f"  {LANG_LABEL[lg]}×{DOC_LANG_LABEL[dl]:<8} n块={len(sub_b):>3}  {dist}"
                     f"  [kw_status: {'、'.join(f'{k}={v}' for k, v in sorted(ks.items(), key=lambda x: str(x[0])))}]")
        emit("")

    # ---- 落盘：每次运行独立留存，历史文件不会被覆盖 ----
    payload = {
        "generated_at": generated_at,
        "kind": "retrieval",
        "modes": list(modes),
        "top_k": TOP_K,
        "diag": diag,
        "diag_table": diag_table,
        "timing": timing,
        "trace": trace,
        "rerank_candidate_k": config.RERANK_CANDIDATE_K,
        "rerank_batch_size": config.RERANK_BATCH_SIZE,
        "rerank_onnx": f"{config.rerank_onnx_subdir()}/model.onnx",
        "rerank_length_sorted": config.RERANK_LENGTH_SORTED,
        "pool_deep_k": POOL_DEEP_K if diag else None,
        "tasks_file": TASKS_PATH.name,
        "query_langs": list(langs),
        # 口径声明：主指标是 chunk 级 gold；summary 里 recall5_micro 是 fact 级参考列
        "primary_metric": "gold_recall5_micro",
        "relevance_source": "gold(content_hash) if present else anchor-substring",
        "gold_units_total": n_gold_units,
        "cases_run": [c["id"] for c in runnable],
        "cases_no_gold": no_gold,
        "cases_skipped": [c["id"] for c in skipped],
        "service_status": _service_status(),
        "summary": summary,
        "by_query_lang": {
            mode: {lg: aggregate([r for r in per_mode[mode] if r["query_lang"] == lg], diag)
                   for lg in langs}
            for mode in modes
        },
        "per_query": per_mode,
    }
    paths = save_run("retrieval", list(modes), "\n".join(lines) + "\n", payload, ts=timestamp(run_at))
    emit()
    emit("结果已落盘（不覆盖历史）：")
    for p in paths:
        emit(f"  {p}")


if __name__ == "__main__":
    main()
