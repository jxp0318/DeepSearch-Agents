"""
精排（cross-encoder）微基准：量化 batch size 与候选数对单次推理耗时的影响

回答两个具体问题：
1. **精排是不是批量推理？** —— 是。reranker.rerank() 用 `config.RERANK_BATCH_SIZE`
   把候选切批，`_encode()` 把 (query, chunk) 对编码成规整矩阵做**一次前向**，
   不是在 Python 里循环逐篇调用。本脚本用 batch_size=1（等价于逐篇循环）与批量
   对比，把「批量到底省了多少」量化出来。
2. **候选数增加时耗时是否线性？** —— 给候选池深度扫描提供耗时侧的实测依据。

用法::

    python -m app.eval.bench_rerank                 # 默认 batch × K 矩阵
    python -m app.eval.bench_rerank --reps 20
    python -m app.eval.bench_rerank --json out.json

只读：不写任何索引、不改生产配置。
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from app.rag import config  # noqa: E402


def _load_candidates(kb: str, n: int) -> list[str]:
    """从某个知识库的 chunks.jsonl 里取前 n 条真实 chunk 原文（不建 BM25，不走 ES/Qdrant）"""
    path = config.INDEX_ROOT / kb / "chunks.jsonl"
    if not path.exists():
        raise SystemExit(f"索引不存在：{path}（请先 python -m app.rag.ingest）")
    texts: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(json.loads(line)["text"])
            if len(texts) >= n:
                break
    return texts


def _time_once(reranker, query: str, texts: list[str], bs: int) -> tuple[float, float]:
    """跑一次精排，返回 (毫秒, 平均补齐长度)

    平均补齐长度 = Σ(批内条数 × 该批最长序列) / 候选数，即「每条序列实际参与计算的
    token 槽位」。它和真实平均长度之比就是 padding 浪费——本脚本用它验证「批越小越快」
    的机制确实是补齐而不是 kernel 选型。
    """
    encs = reranker._tokenizer.encode_batch([(query, t) for t in texts])
    plan = reranker._batch_plan(encs, bs)
    slots = sum(len(idxs) * max(len(encs[i].ids) for i in idxs) for idxs in plan)
    raw_avg = sum(len(e.ids) for e in encs) / len(encs)
    t0 = time.perf_counter()
    out = reranker.rerank(query, texts, batch_size=bs)
    ms = (time.perf_counter() - t0) * 1000
    if out is None:
        raise SystemExit("精排不可用（返回 None）——请检查 RERANK_ENABLED / CUDA / ONNX 模型")
    return ms, slots / len(texts) - raw_avg  # 返回的是「补齐多出来的槽位」


def main() -> None:
    ap = argparse.ArgumentParser(description="精排 batch / 候选数 微基准")
    ap.add_argument("--kb", default="人工智能产业", help="取样 chunk 的知识库目录名")
    ap.add_argument("--query", default="具身智能的核心技术路线有哪些？", help="固定查询")
    ap.add_argument("--reps", type=int, default=15, help="每个配置的重复次数")
    ap.add_argument("--warmup", type=int, default=3, help="每配置先跑几次不计入")
    ap.add_argument("--ks", default="5,10,15,25", help="候选数列表")
    ap.add_argument("--batches", default="1,8,16,25", help="batch size 列表（1 = 逐篇循环）")
    ap.add_argument("--no-sort", action="store_true",
                    help="关闭按长度排序切批（复现「原序切批」的对照臂）")
    ap.add_argument("--json", default="", help="把原始结果写到该 json 路径")
    args = ap.parse_args()

    if args.no_sort:
        config.RERANK_LENGTH_SORTED = False

    from app.rag.reranker import reranker

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    texts = _load_candidates(args.kb, max(ks))

    ok = reranker.warmup()
    print(f"精排状态：{reranker.status()}  (warmup={'成功' if ok else '失败'})")
    print(f"知识库：{args.kb}  候选文本：{len(texts)} 条  查询：{args.query!r}")
    print(f"按长度排序切批 RERANK_LENGTH_SORTED = {config.RERANK_LENGTH_SORTED}")
    print(f"重复：每个配置 warmup {args.warmup} 次 + 计时 {args.reps} 次\n")

    header = (f"{'候选数':>6} | {'batch':>5} | {'中位延迟':>9} | {'均值':>9} | {'P95':>9} | "
              f"{'单候选':>7} | {'补齐开销':>8} | {'批次数':>5}")
    print(header)
    print("-" * len(header) * 1)

    # 轮转取样：所有配置按 rep 交替执行，让每个配置经历同样的 GPU 温度 / 功耗 / 时钟
    # 状态。嵌套循环（一个配置连跑 18 次再换下一个）会把「先跑的配置顺便给 GPU 热了身」
    # 混进结果——实测同一 K 下 batch=8 比 batch=16 还快，就是顺序效应而不是真实差异。
    configs = [(k, bs) for k in ks for bs in batches]
    samples_of: dict[tuple[int, int], list[float]] = {c: [] for c in configs}
    pad_of: dict[tuple[int, int], float] = {}

    for k, bs in configs:  # 先各暖一遍，把 kernel 编译 / cudnn 选型等一次性开销摊掉
        for _ in range(args.warmup):
            _time_once(reranker, args.query, texts[:k], bs)
        pad_of[(k, bs)] = 0.0

    for _ in range(args.reps):
        for k, bs in configs:
            ms, pad_extra = _time_once(reranker, args.query, texts[:k], bs)
            samples_of[(k, bs)].append(ms)
            pad_of[(k, bs)] = pad_extra

    raw: list[dict] = []
    for k in ks:
        for bs in batches:
            samples = samples_of[(k, bs)]
            pad_extra = pad_of[(k, bs)]
            med = statistics.median(samples)
            n_batch = (k + bs - 1) // bs
            print(f"{k:>6} | {bs:>5} | {med:>8.1f}ms | {statistics.fmean(samples):>8.1f}ms | "
                  f"{sorted(samples)[max(0, int(0.95 * len(samples)) - 1)]:>8.1f}ms | "
                  f"{med / k:>6.2f}ms | {pad_extra:>7.1f}t | {n_batch:>5}")
            raw.append({
                "k": k, "batch_size": bs, "n_batches": n_batch,
                "padding_extra_tokens_per_seq": round(pad_extra, 1),
                "median_ms": round(med, 2),
                "mean_ms": round(statistics.fmean(samples), 2),
                "p95_ms": round(sorted(samples)[max(0, int(0.95 * len(samples)) - 1)], 2),
                "min_ms": round(min(samples), 2), "max_ms": round(max(samples), 2),
                "per_candidate_ms": round(med / k, 3),
                "samples_ms": [round(s, 2) for s in samples],
            })
        print()

    if args.json:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "provider": reranker.provider,
            "kb": args.kb,
            "query": args.query,
            "reps": args.reps,
            "length_sorted": config.RERANK_LENGTH_SORTED,
            "results": raw,
        }
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"原始结果已写入 {args.json}")


if __name__ == "__main__":
    main()
