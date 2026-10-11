"""
检索层量化评测

回答的问题：四种检索模式（进程内 BM25 / ES 关键词 / 向量 / RRF 融合）在各知识库上的
检索质量如何，以及质量缺口是「没召回到」还是「排得太靠后」。

题库在 app/eval/retrieval-tasks.yaml（不在本文件里），本脚本只负责跑与算。

--------------------------------------------------------------------------
指标口径（2026-10-10 按用户要求重定，向业界 RAG 评测体系对齐）
--------------------------------------------------------------------------
检索质量（本脚本负责）：
  ★ Context Recall@5       【核心】fact 级：Top-5 覆盖的锚点事实数 / 该题锚点总数。
                           决定下游 agent 的能力天花板——检索不到，模型再强也答不出。
                           汇总用 micro 口径（Σ命中 / Σ事实）；另给 macro（题均）作参考，
                           两者差距大说明总分被「锚点多的大题」主导。
  ★ Context Precision@5    【核心】chunk 级 AP@5（位置加权）：相关块 = 含任一锚点的块。
                           AP@k = Σ_{i: 相关} (前 i 个中相关数 / i) / R，R = Top-5 内相关块数，
                           R=0 记 0。它同时反映「占比」与「位置」——同样是 2 个相关块，
                           排在第 1、3 位（0.83）比排在第 4、5 位（0.35）高得多。
  ★ MRR@5                  【核心】首个相关块的排名倒数，Top-5 外记 0，跨题平均。
                           唯一直接回答「第一个相关结果排得够不够靠前」的指标。
  NDCG@5                   标准归一化折损累计增益（二值相关性）：DCG = Σ rel_i / log2(i+1)，
                           IDCG = 相关块全排最前的理想 DCG，NDCG = DCG / IDCG。
                           与 AP@5 同为「位置敏感 + 归一化」，但对「排太后」惩罚更平缓
                           （对数折扣 vs 逐点精度平均），两者并排能分出排太后的严重程度。

参考列：
  Context Precision@5（占比）  相关块数 / 5。位置无关的裸占比，对应「检索结果里有多少
                              条用得上」；与 AP@5 一起看能区分「占比低」与「排得靠后」。
  题均 Context Recall@5        macro 口径（每道查询等权），与 micro 对照用。

检索上限（默认即算，与 search 同一次调用、无额外成本）：
  候选池召回（生产深度）        融合去重后、截断成 top_k 之前的候选集合覆盖的锚点比例。
                              rerank 只能在这个集合内重排，故它就是本配置下 Recall@5
                              的理论上限；「候选池召回 − ★Recall@5」= 精排可吃掉的排序空间。

诊断列（--diag 才计算，需额外检索）：
  Context Recall@20           用来区分「没召回」与「排得靠后」——@5 低而 @20 高即排序问题。
                              ⚠ @20 这一路用 top_k=20 → 每路候选 60，候选池比生产深得多，
                              它不是「生产池的召回」，别拿它反推「池里没有」。
  候选池召回@100               把每路候选深度从 15（=5×3）加深到 100 时的候选池覆盖，
                              用来固化「加深池即可覆盖几乎全部锚点」这一结论（缺口在排序不在召回）。

性能：
  查询延迟 P50 / P95 / max     逐条查询的 wall-clock（含向量路的 embedding 调用）。
                              P95 在样本量 < 100 时 ≈ 第 2 慢的那条，统计上不稳健，
                              故同时报 P50 / max 与样本数，不单看 P95。
                              注意这是「检索延迟」；含 LLM 生成的端到端延迟属第二阶段。

不在本脚本内（属 app/eval 第二阶段，见 eval.md）：
  Faithfulness / Answer Correctness——需要真实生成的答案、参考答案与 LLM judge。

--------------------------------------------------------------------------
命中判定口径（重要）
--------------------------------------------------------------------------
每个 fact 配一组「事实锚点」字符串，Top-k 拼接文本中任一锚点出现即视为该 fact 被召回。
**只标注可从原文客观判定的锚点（数值、专名、术语）**，描述型事实不配锚点、不入库——
这样每个数字都可复现、可争论，不带主观判定。锚点校准铁律见题库文件头部注释。

--------------------------------------------------------------------------
用法（在项目根目录）
--------------------------------------------------------------------------
    python -m app.eval.eval_retrieval                  # 全部题目 × 五种模式
    python -m app.eval.eval_retrieval icct-q1 blk-q2   # 只跑指定题
    python -m app.eval.eval_retrieval --mode hybrid    # 只跑指定模式
    python -m app.eval.eval_retrieval --mode hybrid_rerank   # 只跑精排
    python -m app.eval.eval_retrieval --diag           # 附带 Recall@20 诊断列
    python -m app.eval.eval_retrieval --list           # 只列题库，不检索

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

LANG_LABEL = {"zh": "中文Query", "en": "英文Query", "mixed": "混合Query"}
DOC_LANG_LABEL = {"zh": "中文文档", "en": "英文文档"}


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


def measure_query(case: dict, query: str, lang: str, mode: str, diag: bool) -> dict:
    """跑一条查询并算出该条的全部指标"""
    import time

    t0 = time.perf_counter()
    results, pool_seqs = retriever.search_with_pool(case["kb"], query, top_k=TOP_K, mode=mode)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    top_texts = [r["text"] for r in results]
    flag5 = [any(p in t for f in case["facts"] for p in f["patterns"]) for t in top_texts]

    ranks = _fact_ranks(case, top_texts)
    hits5 = sum(1 for r in ranks.values() if r is not None)

    first_rel = next((i for i, f in enumerate(flag5, start=1) if f), None)

    # 候选池召回（生产深度）：融合去重后、截断成 top_k 之前的候选集合覆盖了多少锚点。
    # 这批 chunk 已由 search_with_pool 取回，无额外检索成本。
    chunks = retriever._load_index(case["kb"]).chunks
    pool_hits = _hits_in_texts(case, [chunks[s]["text"] for s in pool_seqs])

    rec = {
        "id": case["id"],
        "kb": case["kb"],
        "query_lang": lang,
        "doc_lang": case["doc_lang"],
        "mode": mode,
        "n_facts": len(case["facts"]),
        "hits5": hits5,
        "recall5": hits5 / len(case["facts"]),
        "precision5_ap": average_precision_at_k(flag5),
        "precision5_ratio": (sum(flag5) / TOP_K) if top_texts else 0.0,
        "ndcg5": ndcg_at_k(flag5),
        "mrr5": (1.0 / first_rel) if first_rel else 0.0,
        "pool_size": len(pool_seqs),
        "pool_hits": pool_hits,
        "pool_recall": pool_hits / len(case["facts"]),
        "latency_ms": elapsed_ms,
        "ranks": ranks,
        "n_results": len(results),
        "docs": sorted({r["doc"] for r in results}),
    }

    if diag:
        # 诊断 1：放大到 @20，判断「没召回」还是「排得靠后」
        deep = retriever.search(case["kb"], query, top_k=DIAG_K, mode=mode)
        deep_ranks = _fact_ranks(case, [r["text"] for r in deep])
        rec["hits20"] = sum(1 for r in deep_ranks.values() if r is not None)
        rec["recall20"] = rec["hits20"] / len(case["facts"])
        # 诊断 2：把每路候选深度加深到 100，看候选池上限（固化「缺口在排序不在召回」）
        deep_pool = retriever.candidate_pool(
            case["kb"], query, top_k=TOP_K, mode=mode, candidate_k=POOL_DEEP_K
        )
        deep_pool_hits = _hits_in_texts(case, [chunks[s]["text"] for s in deep_pool])
        rec["pool_hits100"] = deep_pool_hits
        rec["pool_recall100"] = deep_pool_hits / len(case["facts"])

    return rec


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def aggregate(records: list[dict], diag: bool) -> dict:
    """把一批逐条结果聚合成一组指标。

    Recall@5 给 micro（Σ命中/Σ事实）与 macro（题均）两个口径：
    micro 代表「整个测试集的证据覆盖率」，macro 代表「一道题平均能凑齐多少」。
    Precision / MRR 天然是题级量，一律用题均（macro）。
    """
    if not records:
        return {}
    total_facts = sum(r["n_facts"] for r in records)
    lat = [r["latency_ms"] for r in records]
    out = {
        "n_queries": len(records),
        "recall5_micro": sum(r["hits5"] for r in records) / total_facts,
        "recall5_macro": statistics.fmean(r["recall5"] for r in records),
        "precision5_ap": statistics.fmean(r["precision5_ap"] for r in records),
        "precision5_ratio": statistics.fmean(r["precision5_ratio"] for r in records),
        "ndcg5": statistics.fmean(r["ndcg5"] for r in records),
        "mrr5": statistics.fmean(r["mrr5"] for r in records),
        "pool_recall_micro": sum(r["pool_hits"] for r in records) / total_facts,
        "latency_p50": percentile(lat, 50),
        "latency_p95": percentile(lat, 95),
        "latency_max": max(lat),
    }
    if diag:
        out["recall20_micro"] = sum(r["hits20"] for r in records) / total_facts
        out["pool_recall100_micro"] = sum(r["pool_hits100"] for r in records) / total_facts
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
    argv = sys.argv[1:]
    modes = MODES
    if "--mode" in argv:
        i = argv.index("--mode")
        modes = (argv[i + 1],)
        argv = argv[:i] + argv[i + 2:]
    diag = "--diag" in argv
    args = [a for a in argv if not a.startswith("--")]

    tasks = load_tasks()
    if args:
        tasks = [c for c in tasks if c["id"] in args]

    if "--list" in sys.argv:
        print(f"题库：{TASKS_PATH}")
        print(f"共 {len(tasks)} 道题 / {sum(len(c['facts']) for c in tasks)} 个锚点\n")
        print(f"{'id':<10} {'kb':<10} {'文档':<5} {'问句':<14} {'锚点':>4}  知识库状态")
        print("-" * 72)
        for case in tasks:
            reason = _skip_reason(case)
            langs = "+".join(case["queries"].keys())
            print(
                f"{case['id']:<10} {case['kb']:<10} {case['doc_lang']:<5} {langs:<14} "
                f"{len(case['facts']):>4}  {'可跑' if reason is None else f'跳过（{reason}）'}"
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
    emit(f"- 题库：{TASKS_PATH.name}（跑 {len(runnable)} 题 / "
         f"{sum(len(c['facts']) for c in runnable)} 个锚点）")
    if skipped:
        detail = "；".join(f"{c['id']}（{_skip_reason(c)}）" for c in skipped)
        emit(f"- 跳过（库或文档未入库，不计入指标）：{detail}")
    emit("- 指标口径：★Context Recall@5（fact 级 micro）、★Context Precision@5（chunk 级 AP@5）、"
         "★MRR@5（chunk 级）、NDCG@5、候选池召回（检索上限）、查询延迟 P50/P95/max；Top-k = 5")
    emit(f"- {_service_status()}")
    emit()

    # 3) 逐模式跑全部查询
    per_mode: dict[str, list[dict]] = {}
    for mode in modes:
        records: list[dict] = []
        for case in runnable:
            for lang, query in case["queries"].items():
                records.append(measure_query(case, query, lang, mode, diag))
        per_mode[mode] = records

    summary = {mode: aggregate(recs, diag) for mode, recs in per_mode.items()}

    # ---- 汇总表 ----
    emit("=" * 104)
    emit("汇总（行 = 检索模式；★ 为核心指标）")
    emit("=" * 104)
    emit(f"{'模式':<13} | {'★Recall@5':>9} | {'★Prec@5(AP)':>11} | {'★MRR@5':>7} | {'NDCG@5':>7} | "
         f"{'占比@5':>7} | {'题均@5':>7} | {'延迟':^26}")
    emit(f"{'':<9} | {'(micro)':>9} | {'':>11} | {'':>7} | {'':>7} | "
         f"{'':>7} | {'(macro)':>7} | {'P50':>8} {'P95':>8} {'max':>8}")
    emit("-" * 108)
    for mode in modes:
        s = summary[mode]
        emit(
            f"{mode:<13} | {_fmt_pct(s['recall5_micro']):>9} | {s['precision5_ap']:>11.3f} | "
            f"{s['mrr5']:>7.3f} | {s['ndcg5']:>7.3f} | "
            f"{_fmt_pct(s['precision5_ratio']):>7} | {_fmt_pct(s['recall5_macro']):>7} | "
            f"{_fmt_ms(s['latency_p50']):>8} {_fmt_ms(s['latency_p95']):>8} {_fmt_ms(s['latency_max']):>8}"
        )
    emit()
    emit(f"延迟样本数 = {summary[modes[0]]['n_queries']} 条/模式"
         + ("（样本量小时 P95 ≈ 第 2 慢那条，请连同 P50 / max 一起看）" if summary[modes[0]]["n_queries"] < 100 else ""))

    # ---- 检索上限：候选池召回 vs Recall@5（精排能补多少）----
    emit()
    emit("=" * 104)
    emit("检索上限（候选池召回 = 融合去重后、截断成 top-5 之前的候选集合覆盖的锚点比例）")
    emit("=" * 104)
    if diag:
        emit(f"{'模式':<13} | {'池召回(生产)':>12} | {'★Recall@5':>9} | {'精排可吃':>8} | "
             f"{'池召回@100/路':>13} | {'加深收益':>8}")
        emit("-" * 88)
    else:
        emit(f"{'模式':<13} | {'池召回(生产)':>12} | {'★Recall@5':>9} | {'精排可吃':>8}")
        emit("-" * 56)
    for mode in modes:
        s = summary[mode]
        row = (f"{mode:<13} | {_fmt_pct(s['pool_recall_micro']):>12} | "
               f"{_fmt_pct(s['recall5_micro']):>9} | "
               f"{s['pool_recall_micro'] - s['recall5_micro']:>+8.1%}")
        if diag:
            row += (f" | {_fmt_pct(s['pool_recall100_micro']):>13} | "
                    f"{s['pool_recall100_micro'] - s['pool_recall_micro']:>+8.1%}")
        emit(row)
    emit("精排可吃 = 池召回 − Recall@5：rerank 在当前候选池内最多能补回的锚点比例（排序缺口）。")
    emit("加深收益 = 池召回@100 − 池召回@生产：每路候选从 15 加深到 100 多覆盖的锚点（召回缺口）。")

    if diag:
        emit()
        emit("诊断列（放大到 @20：@5 低而 @20 高 = 排序问题，而不是没召回）")
        emit(f"{'模式':<13} | {'Recall@5':>9} | {'Recall@20':>10} | {'差额':>7}")
        emit("-" * 50)
        for mode in modes:
            s = summary[mode]
            emit(f"{mode:<13} | {_fmt_pct(s['recall5_micro']):>9} | "
                 f"{_fmt_pct(s['recall20_micro']):>10} | "
                 f"{s['recall20_micro'] - s['recall5_micro']:>+7.1%}")

    # ---- 按查询语言分组 ----
    langs = [lg for lg in ("zh", "en", "mixed") if any(lg in c["queries"] for c in runnable)]
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
                    f"{_fmt_pct(a['recall5_micro']):>8} {a['precision5_ap']:>8.3f} "
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
                cells.append(f"{_fmt_pct(a['recall5_micro']):>16}" if a else f"{'n/a':>16}")
            emit(f"{mode:<13} | " + " | ".join(cells))

    # ---- 明细：定位是哪个事实没被召回 ----
    emit()
    emit("=" * 104)
    emit("明细（逐条查询；只列出未命中的事实，便于定位召回缺口）")
    emit("=" * 104)
    for mode in modes:
        emit(f"\n--- {mode} ---")
        for r in per_mode[mode]:
            missed = [n for n, rk in r["ranks"].items() if rk is None]
            hit = [f"{n}@{rk}" for n, rk in r["ranks"].items() if rk]
            line = (f"  {r['id']}/{r['query_lang']}  命中 {len(hit)}/{r['n_facts']}"
                    f" | Prec@5(AP) {r['precision5_ap']:.2f} | {r['latency_ms']:.0f}ms")
            if hit:
                line += f"\n        命中: {'、'.join(hit)}"
            emit(line)
            if missed:
                emit(f"        未召回: {'、'.join(missed)}")

    # ---- RRF 融合 vs 向量单路（简历可用的对比数字）----
    if "hybrid" in summary and "vector" in summary:
        h, v = summary["hybrid"], summary["vector"]
        emit()
        emit("=" * 104)
        emit("RRF 融合（hybrid）vs 向量单路（vector）")
        emit("=" * 104)
        emit(f"  ★ Context Recall@5    : {_fmt_pct(v['recall5_micro'])} → {_fmt_pct(h['recall5_micro'])}"
             f"  ({h['recall5_micro'] - v['recall5_micro']:+.1%})")
        emit(f"  ★ Context Precision@5 : {v['precision5_ap']:.3f} → {h['precision5_ap']:.3f}"
             f"  ({h['precision5_ap'] - v['precision5_ap']:+.3f})")
        emit(f"  ★ MRR@5               : {v['mrr5']:.3f} → {h['mrr5']:.3f}"
             f"  ({h['mrr5'] - v['mrr5']:+.3f})")
        emit(f"  NDCG@5                : {v['ndcg5']:.3f} → {h['ndcg5']:.3f}"
             f"  ({h['ndcg5'] - v['ndcg5']:+.3f})")
        emit(f"  候选池召回(生产)      : {_fmt_pct(v['pool_recall_micro'])} → {_fmt_pct(h['pool_recall_micro'])}"
             f"  ({h['pool_recall_micro'] - v['pool_recall_micro']:+.1%})")

    # ---- 精排（hybrid_rerank）vs 纯 RRF（hybrid）：检索侧最有效的杠杆 ----
    if "hybrid" in summary and "hybrid_rerank" in summary:
        h, r = summary["hybrid"], summary["hybrid_rerank"]
        emit()
        emit("=" * 104)
        emit("精排（hybrid_rerank）vs 纯 RRF 融合（hybrid）")
        emit("=" * 104)
        emit(f"  ★ Context Recall@5    : {_fmt_pct(h['recall5_micro'])} → {_fmt_pct(r['recall5_micro'])}"
             f"  ({r['recall5_micro'] - h['recall5_micro']:+.1%})")
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
        emit("  候选池召回两模式相同（精排只重排不增删），故「池召回 − Recall@5」的差值")
        emit("  就是精排实际吃掉的排序缺口。")
        if abs(r["recall5_micro"] - h["recall5_micro"]) < 1e-9:
            emit("  ⚠ 两模式结果完全一致 —— 精排未生效（检查上方「后端状态 | 精排」一行）")

    # ---- 落盘：每次运行独立留存，历史文件不会被覆盖 ----
    payload = {
        "generated_at": generated_at,
        "kind": "retrieval",
        "modes": list(modes),
        "top_k": TOP_K,
        "diag": diag,
        "pool_deep_k": POOL_DEEP_K if diag else None,
        "tasks_file": TASKS_PATH.name,
        "cases_run": [c["id"] for c in runnable],
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
