"""
并发压测：问答式检索在「同时来 N 个请求」时的吞吐与延迟分布

为什么需要它：单线程跑的 P50 只说明「一个人用起来多快」。检索链路里三段的并发特性
完全不同——
- 召回（向量路）走 **HTTP 到 TEI**，服务端有 ``MAX_CONCURRENT_REQUESTS`` 上限，超了要排队；
- 召回（关键词路）走 **HTTP 到 ES**；
- 精排是 **本进程内的 ONNX + CUDA**，同进程内多个线程共享一张 GPU，``session.run``
  本身线程安全，但 GPU 上会互相排队（kernel 串行）；
所以「并发上去以后到底还能不能打」必须实测，不能从单线程数字外推。

用法::

    python -m app.eval.bench_concurrency                       # 1/2/4/8 并发，生产模式
    python -m app.eval.bench_concurrency --concurrency 1,4,16
    python -m app.eval.bench_concurrency --mode hybrid --json out.json

只读：不写索引、不改配置。
"""

import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from app.eval.eval_retrieval import (  # noqa: E402
    TOP_K,
    _skip_reason,
    load_tasks,
    percentile,
)
from app.rag.retriever import retriever  # noqa: E402


def _collect_queries(max_queries: int | None) -> list[tuple[str, str]]:
    """(kb, query) 列表：取题库里所有可跑题目的全部查询（zh + mixed）"""
    out: list[tuple[str, str]] = []
    for case in load_tasks():
        if _skip_reason(case) is not None:
            continue
        for _lang, q in case["queries"].items():
            out.append((case["kb"], q))
    if max_queries:
        out = out[:max_queries]
    return out


def _run_round(work: list[tuple[str, str]], concurrency: int, mode: str) -> list[float]:
    """用 concurrency 个线程把 work 全部跑一遍，返回每条请求的端到端毫秒数（按完成顺序）"""
    lat: list[float] = []
    lock = threading.Lock()
    idx = 0
    n = len(work)

    def worker() -> None:
        nonlocal idx
        while True:
            with lock:
                if idx >= n:
                    return
                kb, query = work[idx]
                idx += 1
            t0 = time.perf_counter()
            retriever.search(kb, query, top_k=TOP_K, mode=mode)
            ms = (time.perf_counter() - t0) * 1000
            with lock:
                lat.append(ms)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(lambda _: worker(), range(concurrency)))
    return lat


def main() -> None:
    ap = argparse.ArgumentParser(description="检索并发压测")
    ap.add_argument("--mode", default="hybrid_rerank", help="检索模式（默认生产：hybrid_rerank）")
    ap.add_argument("--concurrency", default="1,2,4,8", help="并发度列表")
    ap.add_argument("--repeats", type=int, default=1, help="每个并发度重复几轮（取并集算分位）")
    ap.add_argument("--max-queries", type=int, default=0, help="只用前 N 条查询（0 = 全部）")
    ap.add_argument("--json", default="", help="把结果写到该 json 路径")
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    work = _collect_queries(args.max_queries or None)
    if not work:
        raise SystemExit("没有可跑的查询（知识库未入库？先 python -m app.rag.ingest）")

    print(f"模式：{args.mode} | 查询数/轮：{len(work)} | 并发度：{levels} | 每档 {args.repeats} 轮")
    # 暖机：加载索引 + BM25、预热精排（2.2GB 模型与 CUDA 上下文只该付一次）
    if args.mode == "hybrid_rerank":
        from app.rag.reranker import reranker

        reranker.warmup()
        print(f"精排：{reranker.status()}")
    for kb in sorted({kb for kb, _ in work}):
        retriever._load_index(kb)
    retriever.search(work[0][0], work[0][1], top_k=TOP_K, mode=args.mode)
    print()

    header = (f"{'并发':>4} | {'请求数':>6} | {'墙钟':>7} | {'吞吐(QPS)':>9} | {'P50':>8} | {'P95':>8} | "
              f"{'max':>9} | {'吞吐加速':>8} | {'延迟劣化':>8}")
    print(header)
    print("-" * len(header))

    raw: list[dict] = []
    baseline_qps: float | None = None
    baseline_p50: float | None = None
    for c in levels:
        all_lat: list[float] = []
        qps_list: list[float] = []
        walls: list[float] = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            all_lat.extend(_run_round(work, c, args.mode))
            wall = time.perf_counter() - t0
            # 每轮固定提交 len(work) 个请求（C 个 worker 抢同一个队列），
            # 所以吞吐 = 请求数 / 墙钟——**不能**再乘并发度（那样会把 C=1 之外的
            # 数字凭空放大 C 倍，看起来像线性扩展，实际是延迟堆高）。
            qps_list.append(len(work) / wall)
            walls.append(wall)
        qps = statistics.fmean(qps_list)
        p50 = statistics.median(all_lat)
        p95 = percentile(all_lat, 95)
        if baseline_qps is None:
            baseline_qps, baseline_p50 = qps, p50
            speedup = degrade = 1.0
        else:
            speedup = qps / baseline_qps            # 相对单并发的吞吐倍数
            degrade = p50 / baseline_p50            # 单请求延迟劣化倍数
        print(f"{c:>4} | {len(all_lat):>6} | {statistics.fmean(walls):>6.1f}s | {qps:>9.1f} | "
              f"{p50:>7.0f}ms | {p95:>7.0f}ms | {max(all_lat):>8.0f}ms | "
              f"{speedup:>7.2f}x | {degrade:>7.2f}x")
        raw.append({
            "concurrency": c, "n_requests": len(all_lat), "qps": round(qps, 2),
            "wall_s": round(statistics.fmean(walls), 2),
            "p50_ms": round(p50, 1), "p95_ms": round(p95, 1), "max_ms": round(max(all_lat), 1),
            "min_ms": round(min(all_lat), 1),
            "speedup_vs_1": round(speedup, 3), "latency_degrade_vs_1": round(degrade, 3),
            "wall_repeats": args.repeats,
        })

    print()
    print("说明：吞吐加速 = 当前 QPS / 单并发 QPS；延迟劣化 = 当前 P50 / 单并发 P50。")
    print("      理想线性扩展下吞吐加速 ≈ 并发度、延迟劣化 ≈ 1.0；")
    print("      若吞吐加速远低于并发度而延迟按并发度线性上涨，说明瓶颈是**串行资源**")
    print("      （这里就是唯一那张 GPU：query embedding 与精排都在上面排队）。")

    if args.json:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "mode": args.mode,
            "n_queries_per_round": len(work),
            "results": raw,
        }
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"原始结果已写入 {args.json}")


if __name__ == "__main__":
    main()
