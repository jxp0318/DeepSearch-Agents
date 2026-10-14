"""检索题库的 chunk 级 gold 生成/校验工具

对 ``retrieval-tasks.yaml`` 的每道**可跑题**，按每个 fact 的锚点在当前索引里定位
应命中的 chunk，产出可粘贴回题库的 ``gold:`` 块。

gold 的定义（**最小覆盖集**，2026-10-11 修订）
--------------------------------------------------------------------------
gold = **能覆盖该题全部锚点的最少块集合** = 该题标准答案的出处。

为什么不直接把「所有含锚点的块」都收进 gold：切块本身有重叠（节内超长仍保留
80 字 ``CHUNK_OVERLAP``、超大表格分块要重复表头行），同一句话会同时落进相邻两块。
若两块都算 gold，分母就虚大——模型召回了前半块、却被判「后半块没召回」。实测
v3 索引上 39 个 gold 里有 18 个（46.2%）属于这种冗余，最小覆盖集下只有 27 个。

主键是 ``content_hash``（schema v3 字段，随 chunk 文本稳定），这样：
  - 评测时「相关块」是**真值**（不是「含任一锚点」的近似）；
  - 重切块 / 换解析器后 hash 对不上，``--check`` 会立刻报「gold 失效」，提示重标。

用法（项目根目录）：
    python -m app.eval.build_gold              # 打印每题的 gold 块（预览）
    python -m app.eval.build_gold --check      # 校验题库已有 gold：失效 / 冗余 / 新增
    python -m app.eval.build_gold --apply      # 把生成结果直接回写 retrieval-tasks.yaml
    python -m app.eval.build_gold --apply icct-q1 blk-q4   # 只处理指定题
"""

import json
import re
import sys
from pathlib import Path

from app.eval.eval_retrieval import _skip_reason, load_tasks
from app.rag import config

NORM_RE = re.compile(r"[\s\u3000]+")
TASKS_FILE = Path(__file__).with_name("retrieval-tasks.yaml")


def _norm(text: str) -> str:
    return NORM_RE.sub("", text or "")


def _load_chunks(kb: str) -> list[dict]:
    path = config.INDEX_ROOT / kb / "chunks.jsonl"
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            c = json.loads(line)
            c["_seq"] = i
            out.append(c)
    return out


def _fact_hits(case: dict, chunks: list[dict]) -> list[tuple[str, int, frozenset]]:
    """返回 [(content_hash, seq, 该块命中的 fact 下标集合)]，只保留命中 ≥1 个 fact 的块"""
    out = []
    for c in chunks:
        t = c.get("text") or ""
        nt = _norm(t)
        covered = frozenset(
            i
            for i, f in enumerate(case["facts"])
            if any(p in t or (_norm(p) and _norm(p) in nt) for p in f["patterns"])
        )
        if covered:
            out.append((c["content_hash"], c["_seq"], covered))
    return out


def _min_cover(
    items: list[tuple[str, int, frozenset]], n_facts: int
) -> tuple[list[str], set[int]]:
    """贪心集合覆盖：返回 (最少块 hash 按 seq 排序, 未被任何块覆盖的 fact 下标)"""
    universe: set[int] = set()
    for _, _, s in items:
        universe |= set(s)
    uncovered = set(range(n_facts)) - universe  # 库内根本没有块含该锚点
    remaining = set(universe)
    pool = list(items)
    chosen: list[tuple[str, int, frozenset]] = []
    while remaining:
        best = None
        best_gain = 0
        for h, seq, s in pool:
            gain = len(s & remaining)
            # 覆盖数优先；同分取 seq 更小（更靠前）的块，保证结果稳定可复现
            if gain > best_gain or (gain == best_gain and gain > 0 and best and seq < best[1]):
                best, best_gain = (h, seq, s), gain
        if best is None or best_gain == 0:
            break
        chosen.append(best)
        remaining -= best[2]
        pool.remove(best)
    uncovered |= remaining  # 理论上不会残留（universe 内必然可覆盖），留作防御
    return [h for h, _, _ in sorted(chosen, key=lambda x: x[1])], uncovered


def _redundant(hashes: list[str], items: list[tuple[str, int, frozenset]]) -> list[str]:
    """在给定块集合里找出「命中集合被其他块完全覆盖」的块（对本集合没有独立贡献）"""
    cov = {h: s for h, _, s in items}
    out = []
    for h in hashes:
        if h not in cov:
            continue
        others = [cov[o] for o in hashes if o in cov and o != h]
        if others and cov[h] <= frozenset().union(*others):
            out.append(h)
    return out


def _gold_units(case: dict, items: list[tuple[str, int, frozenset]], gold: list[str]) -> list[list[str]]:
    """把最小覆盖集 ``gold`` 组织成「答案单元」（``list[list[hash]]``）。

    当最小集是**单块**（即存在能独立覆盖全锚点的块）且这种块不止一个时，把它们合并成
    一个 OR 单元——它们互为**等价出处**（同一答案被多个块完整叙述，如「执行摘要」与
    「正文」），命中任一即算该单元命中；否则每块各成一个单元。
    """
    n_facts = len(case["facts"])
    seq_of = {h: seq for h, seq, _ in items}
    full = sorted({h for h, _, s in items if len(s) == n_facts}, key=lambda h: seq_of.get(h, 10**9))
    if len(gold) == 1 and len(full) > 1:
        return [full]
    return [[h] for h in gold]


def _gold_for_case(
    case: dict, chunks: list[dict]
) -> tuple[list[list[str]], dict[str, dict], list[str], list[str]]:
    """返回 (gold 单元, 每块说明[按 hash], 警告, 冗余项)"""
    by_hash = {c["content_hash"]: c for c in chunks}
    items = _fact_hits(case, chunks)
    gold, uncovered = _min_cover(items, len(case["facts"]))
    units = _gold_units(case, items, gold)

    cov = {h: s for h, _, s in items}
    detail: dict[str, dict] = {}
    for h in sorted({h for u in units for h in u}, key=lambda x: by_hash[x].get("_seq", 0)):
        c = by_hash[h]
        names = [f["name"] for i, f in enumerate(case["facts"]) if i in cov.get(h, frozenset())]
        detail[h] = {"hash": h, "page": c.get("page_index"), "type": c.get("chunk_type"),
                     "facts": names}

    notes = []
    for i in sorted(uncovered):
        notes.append(f"⚠ fact「{case['facts'][i]['name']}」锚点当前 0 命中，gold 缺该事实")

    # 等价组内部的块互相完全覆盖，属「预期冗余」，不做 WARN
    has_or_group = any(len(u) > 1 for u in units)
    redundant = [] if has_or_group else _redundant([h for u in units for h in u], items)
    return units, detail, notes, redundant


def _unit_sig(units: list[list[str]]) -> frozenset:
    """单元的“形状”签名，用来判断两组单元是否等价（忽略块的先后顺序）"""
    return frozenset(frozenset(u) for u in units)


def _yaml_block(case_id: str, units: list[list[str]], detail: dict[str, dict]) -> list[str]:
    n_units = len(units)
    n_blocks = sum(len(u) for u in units)
    if any(len(u) > 1 for u in units):
        head = (f"gold: {n_units} 个答案单元（{n_blocks} 块）—— 单元内命中任一即算命中，"
                f"主键 = content_hash")
    else:
        head = f"gold: 最小覆盖集（{n_units} 块）—— 覆盖全部锚点的最少块，主键 = content_hash"
    lines = [f"    # {head}"]
    for u in units:
        if len(u) == 1:
            d = detail[u[0]]
            lines.append(f"    #   p{d['page']}/{d['type']} {d['hash']} ← {'、'.join(d['facts'])}")
        else:
            lines.append("    #   [等价组] 以下块互为同一答案的等价出处，命中任一即算该单元命中：")
            for h in u:
                d = detail[h]
                lines.append(f"    #     p{d['page']}/{d['type']} {d['hash']} ← {'、'.join(d['facts'])}")
    lines.append(_gold_line(units))
    return lines


def _gold_line(units: list[list[str]]) -> str:
    """题库里 ``gold:`` 那一行的文本（单块直接列 hash；等价组用嵌套 flow 列表）"""
    parts = ["[" + ", ".join(u) + "]" if len(u) > 1 else u[0] for u in units]
    return f"    gold: [{', '.join(parts)}]"


def _apply(gold_by_case: dict[str, tuple[list[list[str]], dict[str, dict]]]) -> int:
    """把生成的 gold 块回写题库（只动每题 gold 及其上方注释，其余原样保留）

    两种情形都要覆盖：
      - 该题**已有** ``gold: [...]`` 行 → 原地替换（连同上方注释）；
      - 该题**还没有** gold 行（新增题）→ 在 facts 块之后、下一个 ``- id:`` 之前插入。
        早期版本只做前者，导致新题 `--apply` 静默回写 0 题（2026-10-11 踩到）。
    """
    lines = TASKS_FILE.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    cur_id: str | None = None
    handled: set[str] = set()
    changed = 0

    def flush(cid: str | None) -> None:
        """给尚未写过 gold 的题补插一段 gold 块（插在已收集内容末尾）"""
        nonlocal changed
        if cid is None or cid not in gold_by_case or cid in handled:
            return
        trail: list[str] = []  # 暂时摘下尾部空行，让 gold 紧贴 facts
        while out and out[-1].strip() == "":
            trail.append(out.pop())
        new_gold, detail = gold_by_case[cid]
        out.extend(_yaml_block(cid, new_gold, detail))
        out.extend(reversed(trail))
        handled.add(cid)
        changed += 1

    for line in lines:
        m = re.match(r"\s*-\s*id:\s*(\S+)", line)
        if m:
            flush(cur_id)  # 上一个题块到此结束，缺 gold 就补
            cur_id = m.group(1)
        if re.match(r"\s*gold:\s*\[", line) and cur_id in gold_by_case:
            # 回退：把紧邻上方、缩进 >0 的注释行一并吃掉（那是本 gold 的说明）
            start = len(out)
            while start > 0 and out[start - 1].strip().startswith("#") and out[start - 1].startswith(" "):
                start -= 1
            new_gold, detail = gold_by_case[cur_id]
            out[start:] = _yaml_block(cur_id, new_gold, detail)
            handled.add(cur_id)
            changed += 1
        else:
            out.append(line)
    flush(cur_id)  # 文件末尾的最后一个题块
    TASKS_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    apply_ = "--apply" in sys.argv

    tasks = load_tasks()
    if argv:
        tasks = [c for c in tasks if c["id"] in argv]

    generated: dict[str, tuple[list[list[str]], dict[str, dict]]] = {}
    for case in tasks:
        reason = _skip_reason(case)
        if reason is not None:
            print(f"# {case['id']}: 跳过（{reason}）")
            continue
        chunks = _load_chunks(case["kb"])
        by_hash = {c["content_hash"]: c for c in chunks}
        gold, detail, notes, redundant = _gold_for_case(case, chunks)
        generated[case["id"]] = (gold, detail)

        if check:
            have_units = [([x] if isinstance(x, str) else list(x)) for x in (case.get("gold") or [])]
            have = [h for u in have_units for h in u]
            gen = [h for u in gold for h in u]
            missing = [h for h in have if h not in by_hash]
            extra = [h for h in gen if h not in have]
            stale = [h for h in have if h in by_hash and h not in gen]
            struct_diff = _unit_sig(have_units) != _unit_sig(gold)
            if missing:
                status = "❌ gold 失效"
            elif extra or stale:
                status = "⚠ 与最小覆盖集不一致"
            elif struct_diff:
                status = "⚠ 等价组划分与生成结果不同"
            else:
                status = "✅"
            print(f"{case['id']:<9} {status}  题内单元={len(have_units)} 生成单元={len(gold)}"
                  f"  失效={len(missing)}  应删={len(stale)}  应补={len(extra)}")
            for h in missing:
                print(f"    ✗ {h} 不在当前索引")
            for u in gold:
                if len(u) > 1:
                    flag = "" if not struct_diff else "（与题内划分不同）"
                    print(f"    [等价组] 命中任一即算命中{flag}：{' ⟺ '.join(u)}")
            for h in redundant:
                print(f"    WARN-REDUNDANT {h}：命中集合被其他块完全覆盖（冗余，不提供独立证据）")
            for n in notes:
                print(f"    {n}")
            continue

        n_blocks = sum(len(u) for u in gold)
        print(f"  # ----- {case['id']}  gold={len(gold)} 单元 / {n_blocks} 块 -----")
        for u in gold:
            if len(u) == 1:
                d = detail[u[0]]
                print(f"  #   p{d['page']}/{d['type']} {d['hash']} ← {'、'.join(d['facts'])}")
            else:
                print("  #   [等价组] 命中任一即算该单元命中：")
                for h in u:
                    d = detail[h]
                    print(f"  #     p{d['page']}/{d['type']} {d['hash']} ← {'、'.join(d['facts'])}")
        print(_gold_line(gold))
        for n in notes:
            print(f"    # {n}")
        print()

    if apply_ and generated:
        n = _apply(generated)
        print(f"\n已回写 {n} 题的 gold 到 {TASKS_FILE.name}")
    elif not check:
        print("# 用 --apply 直接把上面结果回写题库；或手工粘贴。")


if __name__ == "__main__":
    main()
