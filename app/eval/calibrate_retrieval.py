"""检索题库的锚点校准诊断（只读）

对 ``app/eval/retrieval-tasks.yaml`` 的每个锚点，在**当前索引**里统计：

    n_hits   命中 chunk 数（库内，含全部文档）
    ratio    n_hits / 库内 chunk 总数
    where    命中块的定位（content_hash / page_index / chunk_type）

并标出四类问题（题库文件头部「锚点校准铁律」的自动化版本）：

    FAIL-0      0 命中 —— 锚点在库里找不到。切块/解析变更后失配，或写错。
    WARN-BROAD  ratio > 3% —— 锚点过泛，会把大量无关块判成相关（铁律 #2）。
    WARN-CROSS  命中了非答案文档 —— 同库跨文档假阳性（铁律 #2 的变体）。
    WARN-SPACE  原文匹配 0 命中、但「压掉空白后」能命中 —— 撞上 PDF 双空格（铁律 #4）。

产出：终端表格 + 可选 md 报告（--write）。

用法（项目根目录）：
    python -m app.eval.calibrate_retrieval                 # 全部题
    python -m app.eval.calibrate_retrieval icct-q8 blk-q1  # 只校准指定题
    python -m app.eval.calibrate_retrieval --write         # 顺带落盘 md 报告
"""

import json
import re
import sys
from pathlib import Path

from app.eval.eval_retrieval import _skip_reason, load_tasks
from app.rag import config

BROAD_RATIO = 0.03  # 铁律 #5：占比 > 3% 一律回炉
OUT_DIR = Path(".workbuddy/scripts/out")
NORM_RE = re.compile(r"[\s\u3000]+")


def _norm(text: str) -> str:
    """压掉所有空白（PDF 提取常留双空格，见铁律 #4）"""
    return NORM_RE.sub("", text or "")


def _load_chunks(kb: str) -> list[dict]:
    path = config.INDEX_ROOT / kb / "chunks.jsonl"
    chunks = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            c = json.loads(line)
            c["_seq"] = i
            chunks.append(c)
    return chunks


def _loc(c: dict) -> str:
    return f"{c['content_hash']}@p{c.get('page_index')}/{c.get('chunk_type')}"


def analyze_case(case: dict, chunks: list[dict]) -> list[dict]:
    total = len(chunks)
    texts = [c.get("text") or "" for c in chunks]
    ntexts = [_norm(t) for t in texts]
    rows: list[dict] = []
    for fact in case["facts"]:
        for pat in fact["patterns"]:
            exact = [c for c, t in zip(chunks, texts) if pat in t]
            np = _norm(pat)
            norm_hits = [c for c, t in zip(chunks, ntexts) if np and np in t]
            hits = exact or norm_hits
            ratio = len(hits) / total if total else 0.0
            cross = sorted({c["doc"] for c in hits if c["doc"] != case["doc"]})
            flag = "OK"
            if not hits:
                flag = "FAIL-0"
            elif not exact and norm_hits:
                flag = "WARN-SPACE"
            elif ratio > BROAD_RATIO:
                flag = "WARN-BROAD"
            elif cross:
                flag = "WARN-CROSS"
            rows.append({
                "fact": fact["name"],
                "pattern": pat,
                "flag": flag,
                "n_hits": len(hits),
                "ratio": ratio,
                "cross_docs": cross,
                "locs": [_loc(c) for c in hits[:4]],
            })
    return rows


def main() -> None:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--write" in sys.argv

    tasks = load_tasks()
    if argv:
        tasks = [c for c in tasks if c["id"] in argv]

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    n_anchor = n_fail = n_broad = n_cross = n_space = n_skip = 0
    emit("# 锚点校准诊断（对当前索引）")
    emit()
    emit(f"- 索引 schema：{config.INDEX_SCHEMA_VERSION}  parser：{config.PARSER_VERSION}")
    emit(f"- 阈值：占比 > {BROAD_RATIO:.0%} 记 WARN-BROAD")
    emit()

    for case in tasks:
        reason = _skip_reason(case)
        if reason is not None:
            n_skip += len(case["facts"])
            emit(f"## {case['id']}  ⏭ 跳过（{reason}）")
            emit()
            continue
        chunks = _load_chunks(case["kb"])
        rows = analyze_case(case, chunks)
        n_anchor += len(rows)
        bad = [r for r in rows if r["flag"] != "OK"]
        n_fail += sum(1 for r in rows if r["flag"] == "FAIL-0")
        n_broad += sum(1 for r in rows if r["flag"] == "WARN-BROAD")
        n_cross += sum(1 for r in rows if r["flag"] == "WARN-CROSS")
        n_space += sum(1 for r in rows if r["flag"] == "WARN-SPACE")
        mark = "✅" if not bad else "⚠️"
        emit(f"## {mark} {case['id']}   [{case['kb']} / {len(chunks)} chunk]")
        emit()
        emit("| fact | pattern | flag | hits | ratio | where |")
        emit("|---|---|---|---:|---:|---|")
        for r in rows:
            locs = " ".join(r["locs"]) if r["locs"] else "—"
            cross = f"  ⚠跨文档:{','.join(r['cross_docs'])}" if r["cross_docs"] else ""
            emit(f"| {r['fact']} | `{r['pattern']}` | {r['flag']} | {r['n_hits']} | "
                 f"{r['ratio']:.2%} | {locs}{cross} |")
        emit()

    emit("## 汇总")
    emit()
    emit(f"- 锚点总数（可跑题）：{n_anchor}；跳过题锚点：{n_skip}")
    emit(f"- FAIL-0 {n_fail} / WARN-BROAD {n_broad} / WARN-CROSS {n_cross} / WARN-SPACE {n_space}")
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        p = OUT_DIR / "anchor_calibration_v3.md"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n报告已落盘：{p}")


if __name__ == "__main__":
    main()
