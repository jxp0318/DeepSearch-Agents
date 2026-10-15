"""
生成层评测 runner（阶段四 / eval.md 第 10 节）

与 `eval_retrieval.py` 的分工：那个只考「检索员」（给定查询能否捞到正确 chunk），
这个考「整个研究员」——完整跑一次任务，产出 7 个指标：

    1. 任务成功率      verdict=pass 的任务比例（同一任务跑 N 次，≥半数 pass 记 pass）
    2. 答案正确率      gold_facts 命中率（judge）+ gold_answer 数值校验（规则层）
    3. Faithfulness    claim 级配对：该结论自己标注的编号，其 chunk 原文是否含该 claim 的锚点
    4. 引用准确率      required_citations 里 must_cite 的 claim 是否真的带上了合规引用
    5. 平均工具调用次数 monitor tool_start 事件计数
    6. P50 / P95 延迟   session_created → task_result 的墙钟时间
    7. 单任务 API 成本  usage_metadata 按 DeepSeek 峰谷价折算

前置条件（重要）：
  * 服务必须带评测采集开关启动，否则引用编号无法核对「是否指向真实片段」：
        EVAL_CAPTURE=1 EVAL_TRACE_DIR=app/eval/results/<run_id>/traces \
        uvicorn app.api.server:app --port 8001
    关闭时（默认）采集代码完全不执行，线上零开销。
  * 依赖服务：ds-mysql（DB 题）、ds-embedding / ds-qdrant / ds-elasticsearch（KB 题）、
    ds-redis（checkpointer）。缺失时检索会降级，报告会标注降级提示。

用法：
    python -m app.eval.eval_agent --list
    python -m app.eval.eval_agent --ids db-03,kb-07 --repeat 1        # 小规模冒烟
    python -m app.eval.eval_agent --category knowledge_base --repeat 3  # 正式跑批
    python -m app.eval.eval_agent --repeat 3 --no-judge                # 只测过程指标（省钱）
"""

import argparse
import asyncio
import datetime
import json
import os
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import httpx
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from app.agent.llm import model

# DeepSeek 峰谷分时计价（2026-08-17 起生效，单位：元 / 百万 tokens）
# 高峰时段 = 北京时间 09:00-12:00 与 14:00-18:00；其余（夜间/周末/节假日）为空闲时段
PRICE_PEAK = {"cached_in": 0.04, "input": 2.0, "output": 8.0}      # deepseek-v4-flash
PRICE_OFFPEAK = {"cached_in": 0.02, "input": 1.0, "output": 4.0}
PEAK_HOURS = {(9, 12), (14, 18)}

TASKS_PATH = Path(__file__).parent / "eval-tasks.yaml"
RESULTS_DIR = Path(__file__).parent / "results"

# 报告里的引用编号：【引用B3-2】
CITATION_RE = re.compile(r"【引用B(\d+)-(\d+)】")
# judge 抄回来的编号常带各种包装（【引用B3-2】、"B3-2"、"3-2"），只认 B3-2 这一种写法
CID_RE = re.compile(r"B\d+-\d+")
URL_RE = re.compile(r"https?://[^\s，。；、）)\]]+")


# ---------------------------------------------------------------------------
# judge 的结构化输出（沿用 ADR-002：端点只支持 json_mode，不支持 json_schema）
# ---------------------------------------------------------------------------
class FaithfulClaim(BaseModel):
    """单条「必引 claim」的溯源核对结果"""

    claim: str = Field(
        default="", description="该条结论的原文描述，须照抄清单原文，不得改写或合并"
    )
    citation_ids: List[str] = Field(
        default_factory=list,
        description="从报告承载该结论的句子末尾原样抄出的引用编号，形如 B12-3；"
        "同一句有多个编号就都抄；该结论没有标注任何编号时给空数组",
    )
    supported: bool = Field(default=False, description="这些编号指向的来源是否真的支撑该论断")
    reason: str = Field(default="", description="判断理由（一句话）")


class JudgeResult(BaseModel):
    """单题判分结果"""

    fact_hits: List[dict] = Field(
        default_factory=list,
        description="逐条 gold_fact 的核对结果：{fact, covered(bool), evidence}",
    )
    verdict: str = Field(default="fail", description="pass | partial | fail")
    answer_correct: float = Field(
        default=0.0, description="答案正确率 0~1：gold_facts 覆盖率（数值题以 gold_answer 校验为准）",
    )
    faithful_claims: List[FaithfulClaim] = Field(
        default_factory=list,
        description="逐条必引 claim 的溯源核对，元素数应等于必引 claim 数",
    )
    unsupported_claims: List[str] = Field(
        default_factory=list, description="报告中有明显问题但缺乏信源支撑的论断",
    )
    reason: str = Field(default="", description="一句话判分理由")

    @field_validator("unsupported_claims", mode="before")
    @classmethod
    def _tolerate_object_items(cls, value: Any) -> list[str]:
        """
        把 unsupported_claims 里的对象元素压成字符串。

        模型经常把这栏返回成 `[{"claim": ..., "reason": ...}]`，而声明是 `List[str]`——
        原样会让**整个 JudgeResult 解析失败**，于是「判分器格式没兜住」被记成「被测任务失败」
        （2026-10-15 批次一里 db-02 / db-03 就是这么被误判 ❌ 的，judge 其实判的 pass）。
        """
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(str(item.get("claim") or item.get("fact") or item))
            else:
                out.append(str(item))
        return out


_judge_json_mode = model.with_structured_output(JudgeResult, method="json_mode")


def cited_digest(trace: "RunTrace", max_chars: int = 160) -> str:
    """
    把「报告里出现的引用编号 → 实际交给模型的片段原文」压成清单，喂给 judge。

    judge 要判「这条来源是否真的支撑该论断」，就必须看得见来源内容；只给它编号和
    文档名，它只能靠自己的先验猜，那个 supported 字段就是空的。采集关闭时返回空串。
    """
    known = trace.captured.get("citations", {})
    lines = []
    for cid in dict.fromkeys(trace.citation_ids_in_report):
        entry = known.get(cid)
        if not entry:
            continue
        text = re.sub(r"\s+", " ", entry.get("text", ""))[:max_chars]
        lines.append(f"- {cid}｜《{entry.get('doc', '')}》{entry.get('heading', '')}：{text}")
    return "\n".join(lines)


def judge_prompt(task: dict, trace: "RunTrace") -> str:
    """构造 judge 输入：原问题 + 最终报告 + gold_facts + 引用片段原文（不输入执行轨迹）"""
    facts = "\n".join(f"- {f}" for f in task.get("gold_facts", []))
    claims = "\n".join(
        f"- {c.get('claim', '')}（{c.get('cite_kind', 'chunk')}）"
        for c in (task.get("required_citations") or [])
        if c.get("must_cite")
    )
    digest = cited_digest(trace)
    sources = (
        f"""
[报告引用编号对应的片段原文]（检索实际交付给模型的内容，按编号索引）
{digest}
"""
        if digest
        else ""
    )
    return f"""你是研究质量评审专家。请判断下面这份研究报告是否足以回答用户的原始问题。

[用户原始问题]
{task["query"]}

[研究报告]
{trace.report}
{sources}
[必须覆盖的关键事实清单]
{facts}

[必须标注来源的结论清单]
{claims}

判分要求：
1. 逐条核对事实清单里的事实是否被报告覆盖（语义匹配，不要求逐字一致）；
2. 判定 verdict：完全覆盖关键事实且正面回答了问题记 pass；覆盖部分记 partial；未覆盖或答非所问记 fail；
3. answer_correct 取已覆盖事实数 / 清单总数（0~1）；
4. 对每条「必须标注来源的结论」，按清单里括号标注的**来源类型**分别判，不要用同一把尺：
   · `chunk`（知识库片段）——**citation_ids**：在报告里找到承载这条结论的句子，把句末的
     `【引用B12-3】` 原样抄成 `"B12-3"`（去掉「引用」二字，保留 B 和横线）；一句话里有多个
     编号就都抄；这条结论在报告里没有任何编号标注时给空数组。**supported**：对照上面
     「引用编号对应的片段原文」，判这些编号指向的原文**是否真的支撑该论断**；原文里找不到
     对应内容、编号在原文清单里不存在、或 citation_ids 为空，都记 false；
   · `query_cond`（数据库）——这类结论**本来就没有【引用】编号：citation_ids 一律给空数组，
     不得因为它是空就判 false**。改判报告有没有写明**表名与筛选/聚合口径**（例如 `inventory`
     表、按 `drug_id` 汇总 `SUM(quantity_on_hand)`）：写了且口径与结论对得上记 true，
     只给结论没给查询口径记 false；
   · `url`（网络）——同样 **citation_ids 给空数组**。改判报告有没有为这条结论给出**可核查的
     来源链接**：有链接且与论断相称记 true，没有链接记 false；
5. `faithful_claims` 必须逐条覆盖「必须标注来源的结论」清单，条数与清单相同，
   `claim` 字段照抄清单原文——漏一条按不支持计，多出清单的条目无效；
6. 只评**结果质量**，不考虑报告的排版与篇幅；不要因为引用格式问题判 fail（格式由规则层单独判）。

只输出一个 JSON 对象，不要输出解释文字或 Markdown 代码块：
{{
  "fact_hits": [{{"fact": "...", "covered": true, "evidence": "报告原文片段"}}],
  "verdict": "pass",
  "answer_correct": 0.8,
  "faithful_claims": [
    {{"claim": "...", "citation_ids": ["B12-3"], "supported": true, "reason": "..."}}
  ],
  "unsupported_claims": [],
  "reason": "一句话理由"
}}"""


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class RunTrace:
    """单次任务运行的完整轨迹"""

    task_id: str
    run_index: int
    thread_id: str
    ok: bool = False
    error: str = ""
    report: str = ""
    tool_calls: int = 0
    tool_names: list[str] = field(default_factory=list)
    assistants: list[str] = field(default_factory=list)
    rounds: int = 0
    reflected: bool = False          # 是否发生过补搜（reflection_supplement）
    rounds_detail: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    latency_s: float = 0.0
    tokens: dict = field(default_factory=dict)
    cost_cny: float = 0.0
    files: list[str] = field(default_factory=list)
    citation_ids_in_report: list[str] = field(default_factory=list)
    captured: dict = field(default_factory=dict)   # 采集到的引用编号 → chunk 映射
    # judge 从报告里抄出的「claim → 该句自己标注的引用编号」配对，规则层据此做逐条核验
    claim_ids_from_judge: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class TaskResult:
    """单题的多次运行汇总"""

    task: dict
    traces: list[RunTrace] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.task["id"]


# ---------------------------------------------------------------------------
# 运行：走真实 HTTP + WebSocket 链路，与前端使用方式一致
# ---------------------------------------------------------------------------
async def _heartbeat(ws) -> None:
    while True:
        await asyncio.sleep(15)
        try:
            await ws.send("ping")
        except Exception:
            return


def load_captured(thread_id: str, trace_dir: Optional[Path]) -> dict:
    """
    读取服务端落盘的采集文件。

    返回 {"citations": {引用编号: {doc, heading, score, text}}, "web_urls": [...],
          "sql_calls": [...]}
    """
    out: dict[str, Any] = {"citations": {}, "web_urls": [], "db_calls": []}
    if not trace_dir:
        return out
    folder = trace_dir / thread_id
    if not folder.exists():
        return out

    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = path.name
        if name.startswith("kb_search"):
            for c in data.get("citations", []):
                out["citations"][c["id"]] = {
                    "doc": c.get("doc", ""),
                    "heading": c.get("heading", ""),
                    "score": c.get("score", 0),
                    "text": c.get("text", ""),
                }
        elif name.startswith("web_search"):
            out["web_urls"].extend(r.get("url", "") for r in data.get("results", []))
        elif name.startswith("db"):
            out["db_calls"].append(data)
    return out


async def run_once(
    task: dict,
    run_index: int,
    api: str,
    ws_base: str,
    timeout: int,
    trace_dir: Optional[Path],
) -> RunTrace:
    """跑一次任务并收集全过程数据"""
    import websockets

    thread_id = f"eval-{task['id']}-r{run_index}-{int(time.time()) % 100000}"
    trace = RunTrace(task_id=task["id"], run_index=run_index, thread_id=thread_id)

    ws_url = f"{ws_base}/{thread_id}"
    try:
        async with websockets.connect(ws_url) as ws:
            hb = asyncio.create_task(_heartbeat(ws))
            try:
                async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
                    resp = await client.post(
                        f"{api}/api/task",
                        json={"query": task["query"], "thread_id": thread_id},
                    )
                    if resp.status_code != 200:
                        trace.error = f"POST /api/task -> {resp.status_code} {resp.text[:200]}"
                        return trace
                post_at = datetime.datetime.now()
                started = time.time()

                while True:
                    if time.time() - started > timeout:
                        trace.error = f"超时 {timeout}s"
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    except asyncio.TimeoutError:
                        trace.error = "WebSocket 等待超时"
                        break

                    payload = json.loads(raw)
                    if payload.get("type") != "monitor_event":
                        continue
                    # 丢弃 POST 之前的历史事件回放
                    try:
                        if datetime.datetime.fromisoformat(payload["timestamp"]) < post_at:
                            continue
                    except (KeyError, ValueError):
                        pass

                    event = payload.get("event", "")
                    data = payload.get("data") or {}
                    if event == "tool_start":
                        trace.tool_calls += 1
                        trace.tool_names.append(data.get("tool_name", "?"))
                    elif event == "assistant_call":
                        trace.assistants.append(data.get("assistant_name", "?"))
                    elif event == "token_usage":
                        # 成本指标的唯一数据源：逐次模型调用的 usage 明细
                        for k in ("input_tokens", "output_tokens", "cached_tokens"):
                            trace.tokens[k] = trace.tokens.get(k, 0) + int(data.get(k) or 0)
                    elif event == "reflection_supplement":
                        trace.reflected = True
                    elif event == "round_result":
                        trace.rounds = max(trace.rounds, int(data.get("round") or 0))
                        trace.stop_reason = data.get("stop_reason", "")
                        trace.rounds_detail.append(
                            {"round": data.get("round"), "stop": data.get("stop_reason")}
                        )
                    elif event == "task_result":
                        trace.report = data.get("result", "") or ""
                        trace.partial = bool(data.get("partial"))
                        trace.ok = bool(trace.report)
                        trace.latency_s = time.time() - started
                        break
                    elif event == "error":
                        trace.error = payload.get("message", "")[:300]
                        break
            finally:
                hb.cancel()
    except Exception as e:
        trace.error = f"{type(e).__name__}: {e}"[:300]

    # 从会话目录取产物清单
    out_dir = Path("app/output") / f"session_{thread_id}"
    if out_dir.exists():
        trace.files = [p.name for p in out_dir.rglob("*") if p.is_file()]

    trace.captured = load_captured(thread_id, trace_dir)
    trace.citation_ids_in_report = [
        f"B{b}-{i}" for b, i in CITATION_RE.findall(trace.report or "")
    ]
    return trace


# ---------------------------------------------------------------------------
# 规则层判分：不需要 LLM 的部分
# ---------------------------------------------------------------------------
def check_gold_answer(task: dict, report: str) -> Optional[float]:
    """
    DB 数值类精确校验：gold_answer 里的每个数值是否出现在报告中（容差 ±1%）。

    :return: 命中率 0~1；该题没有 gold_answer（或其中无数值字段）时返回 None
    """
    gold = task.get("gold_answer")
    if not gold:
        return None

    # 展平出所有应出现的数值与名称
    expected: list[tuple[str, float]] = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in {"quantity", "total_amount", "quantity_sold", "amount"} and isinstance(v, (int, float)):
                    expected.append((k, float(v)))
                else:
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(gold)
    if not expected:
        return None

    # 报告里的数字（去掉千分位与逗号）
    numbers = set()
    for m in re.finditer(r"[\d,]+(?:\.\d+)?", report.replace(" ", "")):
        try:
            numbers.add(float(m.group(0).replace(",", "")))
        except ValueError:
            pass

    hit = 0
    for _, value in expected:
        if any(abs(n - value) <= max(abs(value) * 0.01, 1e-6) for n in numbers):
            hit += 1
    return hit / len(expected)


def check_citations(task: dict, trace: RunTrace) -> dict:
    """
    引用准确率（规则层，可复现、不依赖 LLM）。

    对每条 must_cite 的 claim：
      · cite_kind=chunk      → 报告里必须出现至少一个【引用BN-I】，且该编号在本次
                              采集到的映射里真实存在（不存在 = 凭空编的编号）
      · cite_kind=query_cond → 报告里必须写明表名（drugs/inventory/sales_records）
      · cite_kind=url        → 报告里必须出现 http(s) 链接
    返回 {"required", "cited", "resolvable", "detail": [...]}。
    """
    report = trace.report or ""
    required = task.get("required_citations") or []
    detail: list[dict] = []
    cited = 0
    resolvable = 0

    known_ids = set(trace.captured.get("citations", {}))
    report_ids = set(trace.citation_ids_in_report)

    for item in required:
        if not item.get("must_cite"):
            continue
        kind = item.get("cite_kind", "chunk")
        claim = item.get("claim", "")
        if kind == "chunk":
            hit_ids = report_ids & known_ids if known_ids else report_ids
            ok = bool(hit_ids)
        elif kind == "query_cond":
            tables = ("drugs", "inventory", "sales_records")
            ok = any(t in report for t in tables)
        elif kind == "url":
            ok = bool(URL_RE.search(report))
        else:
            ok = False
        cited += int(ok)
        detail.append({"claim": claim, "kind": kind, "cited": ok})

    # 「编号可解析」独立于 must_cite 清单：分母是报告里出现的**去重引用编号总数**，
    # 分子是其中能在本次采集映射里找到的编号数 —— 这才是「有没有凭空编编号」。
    # 采集关闭（known_ids 为空）时不产出该指标。
    resolvable_total = len(report_ids)
    resolvable_hit = len(report_ids & known_ids)

    return {
        "required": len(required),
        "cited": cited,
        "resolvable_total": resolvable_total,
        "resolvable_hit": resolvable_hit,
        "detail": detail,
    }


def chunk_claims(task: dict) -> list[dict]:
    """全部「必引 + anchor_match + 引用类型是知识库 chunk」的 claim（是否已配锚点都算）"""
    return [
        c
        for c in (task.get("required_citations") or [])
        if c.get("must_cite")
        and c.get("anchor_match")
        and c.get("cite_kind", "chunk") == "chunk"
    ]


def anchored_claims(task: dict) -> list[dict]:
    """
    参与 Faithfulness 核验的 claim 清单：`chunk_claims` 里已经配好 `anchors` 的那些。

    锚点**跟着 claim 走**（`required_citations[].anchors`），不再是整题共用一组——
    否则「数字人」「白皮书」这种满篇都有的宽词会让每条 claim 都自动满足，指标失去区分度。
    DB（query_cond）与网络（url）类 claim 不参与：前者由 gold_answer 数值校验覆盖，
    后者的原文不在采集映射里。
    """
    return [c for c in chunk_claims(task) if c.get("anchors")]


def must_cite_claims(task: dict) -> list[dict]:
    """judge 层的固定分母：全部 must_cite 的 claim（judge 漏答按不支持计，不得缩小分母）"""
    return [c for c in (task.get("required_citations") or []) if c.get("must_cite")]


def normalize_ids(raw: Any) -> list[str]:
    """把 judge 抄回的编号规整成 B{批次}-{序号}；认不出格式的丢弃，顺序去重"""
    out: list[str] = []
    for item in raw if isinstance(raw, list) else [raw]:
        m = CID_RE.search(str(item or ""))
        if m and m.group(0) not in out:
            out.append(m.group(0))
    return out


def match_faithful_claims(task: dict, jr: JudgeResult) -> dict[str, FaithfulClaim]:
    """
    把 judge 返回的 faithful_claims 对回题库 claim。

    先按原文精确匹配，退到互为子串（judge 常把描述截短）。对不上的条目直接丢弃——
    它是自己发明的 claim 文本，不能拿来给别的 claim 送分。
    """
    out: dict[str, FaithfulClaim] = {}
    entries = list(jr.faithful_claims or [])
    for item in must_cite_claims(task):
        text = (item.get("claim") or "").strip()
        if not text:
            continue
        hit = next((e for e in entries if e.claim.strip() == text), None)
        if hit is None:
            hit = next(
                (e for e in entries if e.claim.strip() and (text in e.claim or e.claim in text)),
                None,
            )
        if hit is not None:
            out[text] = hit
    return out


def judge_faithfulness_score(task: dict, jr: JudgeResult) -> Optional[float]:
    """
    Faithfulness（judge 层）：supported 数 / **题库必引 claim 数**。

    旧实现是 `[1.0 for c in claims if c.supported]` 再求均值——列表里只可能有 1.0，
    均值恒等于 1；一条都不被支撑时列表为空，这次运行整个被跳过而不是记 0，
    所以这个指标只能输出 1.0 或空。改为固定分母，漏答与不支持同样扣分。
    """
    claims = must_cite_claims(task)
    if not claims:
        return None
    matched = match_faithful_claims(task, jr)
    supported = sum(
        1
        for c in claims
        if (entry := matched.get((c.get("claim") or "").strip())) is not None and entry.supported
    )
    return supported / len(claims)


def check_faithfulness(task: dict, trace: RunTrace) -> dict:
    """
    Faithfulness 规则层：**claim 配对后的锚点核验**（2026-10-15 改口径）。

    只核验每条 claim **自己标注的那些编号**：编号在本次采集映射里真实存在、
    且对应 chunk 原文含该 claim 的锚点，才算「有据」。任一编号成立记该 claim 通过。

    与上一版的差别（这是本指标能不能 defend 的关键）：
      · 上一版拿报告里**全部**编号去匹配（实测 157 个编号 vs 5 条 claim），只要有任何
        一个编号的 chunk 含锚点就放行——等于在问「这篇报告里存不存在含这些词的证据」，
        模型把不相干段落挂上去也照样得分。
      · 本版要求「这句话挂的编号，原文里真的有这个数」。编号挂错、挂的是编造的编号，
        都记不过，并在 detail 里区分 bogus（小抄里没有）与 off_topic（存在但不含锚点）。
      · judge 抄回的编号还必须在**报告里真的出现过**（`not_in_report`）：否则它可能是
        照着采集映射猜的——采集里真实存在又含锚点的编号，恰好是最容易被猜中的那一类。

    锚点也换成 claim 级（`required_citations[].anchors`），不再用整题共用的
    `faithfulness_anchors`——宽词会让任何一条引用都自动满足。

    没有 judge 配对（`--no-judge` 或该次判分失败）时本指标**不产出**，而不是退回
    上一版那种必然虚高的宽松口径。
    """
    claims = anchored_claims(task)
    known = trace.captured.get("citations", {})
    if not claims or not known or not trace.claim_ids_from_judge:
        return {"applicable": False, "score": None, "checked": len(claims)}

    traced = paired = 0
    detail = []
    report_ids = set(trace.citation_ids_in_report)
    for item in claims:
        text = (item.get("claim") or "").strip()
        anchors = [str(a) for a in (item.get("anchors") or [])]
        ids = trace.claim_ids_from_judge.get(text, [])
        paired += int(bool(ids))
        # 编号必须**真的出现在报告里**：只在小抄存在不算——judge 可能照着小抄猜一个
        in_report = [cid for cid in ids if cid in report_ids]
        hits = [
            cid
            for cid in in_report
            if cid in known and any(a in known[cid].get("text", "") for a in anchors)
        ]
        absent = [cid for cid in ids if cid not in report_ids]
        bogus = [cid for cid in in_report if cid not in known]
        off_topic = [cid for cid in in_report if cid in known and cid not in hits]
        traced += int(bool(hits))
        detail.append(
            {
                "claim": text,
                "anchors": anchors,
                "cited_ids": ids,
                "traced": bool(hits),
                "hits": hits,
                "not_in_report": absent,
                "bogus_ids": bogus,
                "off_topic_ids": off_topic,
            }
        )
    return {
        "applicable": True,
        "score": traced / len(claims),
        "checked": len(claims),
        "paired": paired,
        "detail": detail,
    }


def check_anchors(tasks: list[dict]) -> int:
    """
    题库自检：每个 claim 级锚点必须能在语料里**逐字命中**。

    锚点是字符串匹配的真值来源，写错一个字符、或者切块改版（`content_hash` 那套）之后
    失效，都会让 Faithfulness 静默变成 0——与其跑完一遍 agent 才发现，不如在这里当场暴露。
    判据与检索层 `calibrate_retrieval` 同源：MISS 判失败；全库命中占比 > 3% 记 BROAD
    （满篇都含的宽词等于送分，正是上一版「数字人 / 白皮书」失去区分度的原因）。

    :return: 有 MISS 时返回 1，供命令行退出码使用
    """
    from app.rag import config  # 延迟导入：--list 之类不需要 RAG 依赖

    indexes: list[tuple[str, list[str]]] = []
    for kb in config.list_knowledge_base_names():
        path = config.INDEX_ROOT / kb / "chunks.jsonl"
        if not path.exists():
            continue
        texts = [
            json.loads(line).get("text", "")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        indexes.append((kb, texts))
    if not indexes:
        print("没有任何已摄入的知识库索引 —— 先跑 `python -m app.rag.ingest`")
        return 1
    total = sum(len(texts) for _, texts in indexes)
    print(f"语料：{len(indexes)} 个知识库 / {total} 块")

    n_ok = n_broad = n_miss = n_claims = 0
    for task in tasks:
        for item in anchored_claims(task):
            n_claims += 1
            for anchor in item.get("anchors") or []:
                hits = sum(1 for _, texts in indexes for t in texts if anchor in t)
                ratio = hits / total if total else 0.0
                if hits == 0:
                    n_miss += 1
                    flag = "MISS "
                elif ratio > 0.03:
                    n_broad += 1
                    flag = "BROAD"
                else:
                    n_ok += 1
                    continue
                print(
                    f"  [{flag}] {task['id']:<8} {(item.get('claim') or '')[:24]:<26} "
                    f"{anchor!r} 命中 {hits} ({ratio:.2%})"
                )
    print(
        f"\n参与核验的 claim {n_claims} 条；锚点 ok {n_ok} / BROAD {n_broad} / MISS {n_miss}"
        f"；未配锚点的必引 chunk claim {sum(len(chunk_claims(t)) - len(anchored_claims(t)) for t in tasks)} 条"
    )
    return 1 if n_miss else 0


def current_prices() -> dict:
    """按当前北京时间判断处于峰谷哪一档"""
    hour = datetime.datetime.now().hour + datetime.datetime.now().minute / 60
    is_peak = any(start <= hour < end for start, end in PEAK_HOURS)
    return {"tier": "peak" if is_peak else "offpeak", **(PRICE_PEAK if is_peak else PRICE_OFFPEAK)}


def compute_cost(usage_accum: dict) -> float:
    """
    按 DeepSeek 峰谷价折算成本（元）。

    usage_accum 需含 input_tokens / output_tokens / cached_tokens。
    缓存命中部分按 cached_in 单价、未命中部分按 input 单价。
    """
    p = current_prices()
    cached = min(usage_accum.get("cached_tokens", 0), usage_accum.get("input_tokens", 0))
    fresh = usage_accum.get("input_tokens", 0) - cached
    cost = (
        cached * p["cached_in"] + fresh * p["input"] + usage_accum.get("output_tokens", 0) * p["output"]
    ) / 1_000_000
    return round(cost, 6)


def percentile(values: list[float], q: float) -> Optional[float]:
    """线性插值分位数（q 取 0~1）"""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


# ---------------------------------------------------------------------------
# 判分：规则层 + LLM judge
# ---------------------------------------------------------------------------
async def judge_task(task: dict, result: TaskResult, use_judge: bool) -> dict:
    """单题判分：返回该题的 7 指标明细"""
    traces = result.traces
    n = len(traces)

    verdicts: list[Optional[str]] = []
    judge_scores = []
    faith_judge: list[float] = []
    judge_details: list[dict] = []
    judge_errors = 0
    for t in traces:
        if not t.report:
            # 连报告都没有 = 任务本身失败，照旧计入分母按不通过算
            verdicts.append("no_report")
            continue
        if not use_judge:
            verdicts.append("unjudged")
            continue
        jr = None
        for attempt in (1, 2):
            try:
                jr = _judge_json_mode.invoke(judge_prompt(task, t))
                break
            except Exception as e:
                print(
                    f"[Judge] {task['id']} 第{attempt}次判分失败："
                    f"{type(e).__name__}: {str(e)[:160]}"
                )
        if jr is None:
            # 判分器失败不能冒充被测任务失败：留 None，成功率分母里剔除
            verdicts.append(None)
            judge_errors += 1
            continue
        verdicts.append(jr.verdict)
        judge_scores.append(float(jr.answer_correct))
        # 语义级 Faithfulness：固定分母 = 题库必引 claim 数（漏答按不支持计）
        fs = judge_faithfulness_score(task, jr)
        if fs is not None:
            faith_judge.append(fs)
        # 顺带把 judge 抄回的「claim → 编号」交回 trace，供规则层逐条核验
        t.claim_ids_from_judge = {
            key: normalize_ids(entry.citation_ids)
            for key, entry in match_faithful_claims(task, jr).items()
        }
        judge_details.append(
            {
                "verdict": jr.verdict,
                "reason": jr.reason,
                "unsupported_claims": jr.unsupported_claims,
                "faithful_claims": [
                    {
                        "claim": c.claim,
                        "citation_ids": normalize_ids(c.citation_ids),
                        "supported": c.supported,
                        "reason": c.reason,
                    }
                    for c in (jr.faithful_claims or [])
                ],
            }
        )

    # 任务成功率：≥半数 pass 记 pass（run1 次时以该次为准）
    # 分母只算「真正判过分的运行」——判分器崩掉的运行不参与，避免 infra 故障被读成能力不足
    judged = [v for v in verdicts if v not in (None, "unjudged")]
    if not use_judge:
        success = all(t.ok for t in traces)
    elif judged:
        success = sum(1 for v in judged if v == "pass") * 2 >= len(judged)
    else:
        success = None

    # 答案正确率：judge 覆盖率与 gold_answer 数值校验取均值（后者优先）
    rule_scores = [check_gold_answer(task, t.report) for t in traces if t.report]
    rule_scores = [s for s in rule_scores if s is not None]
    if rule_scores:
        correctness = statistics.fmean(rule_scores)
    elif judge_scores:
        correctness = statistics.fmean(judge_scores)
    else:
        correctness = None

    citations = [check_citations(task, t) for t in traces if t.report]
    faith = [check_faithfulness(task, t) for t in traces if t.report]
    faith_ok = [f for f in faith if f.get("applicable")]
    faith_rule = [f["score"] for f in faith_ok if f["score"] is not None]

    latencies = [t.latency_s for t in traces if t.ok]
    return {
        "task_id": task["id"],
        "category": task["category"],
        "difficulty": task["difficulty"],
        "runs": n,
        "success": success,
        "judge_errors": judge_errors,
        "verdicts": verdicts,
        "judge_details": judge_details,
        "correctness": correctness,
        "citation_required": sum(c["required"] for c in citations),
        "citation_cited": sum(c["cited"] for c in citations),
        "citation_resolvable_total": sum(c["resolvable_total"] for c in citations),
        "citation_resolvable_hit": sum(c["resolvable_hit"] for c in citations),
        "citation_detail": citations,
        "faithfulness_rule": statistics.fmean(faith_rule) if faith_rule else None,
        "faithfulness_judge": statistics.fmean(faith_judge) if faith_judge else None,
        "faithfulness_checked": sum(f["checked"] for f in faith_ok),
        "faithfulness_paired": sum(f.get("paired", 0) for f in faith_ok),
        # 题库漏标 anchors 的必引 chunk claim：不参与配对核验，但必须看得见有几个
        "claims_unanchored": len(chunk_claims(task)) - len(anchored_claims(task)),
        "faithfulness_detail": [f.get("detail") for f in faith_ok],
        "tool_calls": [t.tool_calls for t in traces],
        "rounds": [t.rounds for t in traces],
        "reflected": any(t.reflected for t in traces),
        "files_ok": all(any(f.endswith((".md", ".pdf")) for f in t.files) for t in traces)
        if task.get("require_file")
        else None,
        "latencies": latencies,
        "cost": sum(t.cost_cny for t in traces),
        "tokens": merge_tokens([t.tokens for t in traces]),
        "overshoot": [t.tool_calls for t in traces if t.tool_calls > task.get("max_tool_calls", 10**9)],
    }


def merge_tokens(dicts: list[dict]) -> dict:
    out = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    for d in dicts:
        for k in out:
            out[k] += d.get(k, 0)
    return out


# ---------------------------------------------------------------------------
# 汇总与报告
# ---------------------------------------------------------------------------
def summarize(results: list[dict], meta: dict) -> dict:
    """把逐题结果汇总成 7 个指标"""
    succ = [r["success"] for r in results if r["success"] is not None]
    unrated = sum(1 for r in results if r["success"] is None)
    judge_err = sum(r.get("judge_errors", 0) for r in results)
    corr = [r["correctness"] for r in results if r["correctness"] is not None]
    cited = sum(r["citation_cited"] for r in results)
    required = sum(r["citation_required"] for r in results)
    res_total = sum(r["citation_resolvable_total"] for r in results)
    res_hit = sum(r["citation_resolvable_hit"] for r in results)
    lat = [x for r in results for x in r["latencies"]]
    tool_calls = [x for r in results for x in r["tool_calls"]]
    costs = [r["cost"] for r in results if r["cost"] > 0]
    rounds = [x for r in results for x in r["rounds"]]
    faith_rule = [r["faithfulness_rule"] for r in results if r["faithfulness_rule"] is not None]
    faith_judge = [r["faithfulness_judge"] for r in results if r["faithfulness_judge"] is not None]
    checked = sum(r["faithfulness_checked"] for r in results)
    paired = sum(r["faithfulness_paired"] for r in results)
    unanchored = sum(r["claims_unanchored"] for r in results)

    return {
        "meta": meta,
        "任务成功率": round(sum(succ) / len(succ), 4) if succ else None,
        # 成功率的分母：判分器失败而未参评的题不进分子也不进分母
        "任务成功率_分母": len(succ),
        "判分失败运行数": judge_err,
        "答案正确率": round(statistics.fmean(corr), 4) if corr else None,
        "Faithfulness_claim锚点核验": round(statistics.fmean(faith_rule), 4) if faith_rule else None,
        "Faithfulness_claim语义支撑": round(statistics.fmean(faith_judge), 4) if faith_judge else None,
        # 规则层的成立前提：judge 得把「这句话挂的是哪个编号」抄回来。抄不回来就没得核验
        "claim编号配对率": round(paired / checked, 4) if checked else None,
        "claim漏标锚点数": unanchored,
        "引用准确率_带引用": round(cited / required, 4) if required else None,
        "引用准确率_编号可解析": round(res_hit / res_total, 4) if res_total else None,
        "平均工具调用次数": round(statistics.fmean(tool_calls), 1) if tool_calls else None,
        "平均迭代轮数": round(statistics.fmean(rounds), 2) if rounds else None,
        "反思触发题数": sum(1 for r in results if r["reflected"]),
        "P50_延迟s": round(percentile(lat, 0.5), 1) if lat else None,
        "P95_延迟s": round(percentile(lat, 0.95), 1) if lat else None,
        "单任务平均成本_元": round(statistics.fmean(costs), 4) if costs else None,
        "总成本_元": round(sum(costs), 3),
        "总token_输入": sum(r["tokens"]["input_tokens"] for r in results),
        "总token_输出": sum(r["tokens"]["output_tokens"] for r in results),
        "总token_缓存命中": sum(r["tokens"]["cached_tokens"] for r in results),
        "跑偏题数": sum(1 for r in results if r["overshoot"]),
        "require_file_达标率": (
            round(
                sum(1 for r in results if r["files_ok"] is True)
                / sum(1 for r in results if r["files_ok"] is not None),
                4,
            )
            if any(r["files_ok"] is not None for r in results)
            else None
        ),
    }


def render_report(summary: dict, results: list[dict], run_id: str) -> str:
    """把汇总渲染成人看的 markdown 表"""
    m = summary["meta"]
    lines = [
        f"# 生成层评测报告 {run_id}",
        "",
        f"- 口径：**单臂（最新版）** = 反思 {m['rounds']} 轮 + 引用规范 + 检索 {m['retrieval']}",
        f"- 规模：{m['tasks']} 题 × {m['repeat']} 次 = {m['total_runs']} 次运行",
        f"- 判分：{'LLM judge + 规则层' if m['judge'] else '仅规则层（--no-judge）'}",
        f"- 价格档：**{m['price_tier']}**（DeepSeek 峰谷分时计价）",
        "",
        "## 7 指标",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
    ]
    for key in [
        "任务成功率",
        "任务成功率_分母",
        "判分失败运行数",
        "答案正确率",
        "Faithfulness_claim锚点核验",
        "Faithfulness_claim语义支撑",
        "claim编号配对率",
        "claim漏标锚点数",
        "引用准确率_带引用",
        "引用准确率_编号可解析",
        "平均工具调用次数",
        "平均迭代轮数",
        "反思触发题数",
        "P50_延迟s",
        "P95_延迟s",
        "单任务平均成本_元",
        "总成本_元",
        "总token_输入",
        "总token_输出",
        "总token_缓存命中",
        "跑偏题数",
        "require_file_达标率",
    ]:
        v = summary.get(key)
        lines.append(f"| {key} | {v if v is not None else '—'} |")

    lines += [
        "",
        "## 分类细分",
        "",
        "| category | 题数 | 成功率 | 答案正确率 | 引用准确率 | 平均工具调用 |",
        "|---|---|---|---|---|---|",
    ]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    for cat, rs in by_cat.items():
        succ = sum(1 for r in rs if r["success"]) / len(rs)
        corr = [r["correctness"] for r in rs if r["correctness"] is not None]
        cited = sum(r["citation_cited"] for r in rs)
        req = sum(r["citation_required"] for r in rs)
        tc = statistics.fmean([x for r in rs for x in r["tool_calls"]]) if rs else 0
        lines.append(
            f"| {cat} | {len(rs)} | {succ:.0%} | "
            f"{statistics.fmean(corr):.2f} | {cited}/{req} | {tc:.1f} |"
            if corr
            else f"| {cat} | {len(rs)} | {succ:.0%} | — | {cited}/{req} | {tc:.1f} |"
        )

    lines += [
        "",
        "## 逐题明细",
        "",
        "| 题号 | 类别 | 成功 | 正确率 | Faith(锚点核验) | Faith(语义支撑) | 编号配对 | 引用 | 工具调用 | 轮数 | 延迟s | 成本¥ |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        latency = f"{statistics.fmean(r['latencies']):.0f}" if r["latencies"] else "—"
        lines.append(
            f"| {r['task_id']} | {r['category']} | {'✅' if r['success'] else '❌'} | "
            f"{r['correctness'] if r['correctness'] is not None else '—'} | "
            f"{(r['faithfulness_rule'] if r['faithfulness_rule'] is not None else '—')} | "
            f"{(r['faithfulness_judge'] if r['faithfulness_judge'] is not None else '—')} | "
            f"{r['faithfulness_paired']}/{r['faithfulness_checked']} | "
            f"{r['citation_cited']}/{r['citation_required']} | "
            f"{statistics.fmean(r['tool_calls']):.0f} | "
            f"{statistics.fmean(r['rounds']):.1f} | {latency} | {r['cost']:.4f} |"
        )

    lines += [
        "",
        "## 口径与已知边界",
        "",
        "1. **单臂**：无「原始版」对照，表中为绝对值，不含提升幅度。",
        "2. **P95 样本量**：题量 "
        f"{m['tasks']} × {m['repeat']} 次 = {m['total_runs']} 个样本，P95 统计上不稳，"
        "同时给出 P50 作参考（沿用 eval.md §3.4 的口径边界）。",
        "3. **Faithfulness 是 claim 级配对核验**：分母 = 题库 `anchor_match` 的必引 claim 数；"
        "分子 = judge 抄回「这句话挂的是哪个编号」后，该编号在本次采集映射里真实存在、"
        "且对应 chunk 原文含**该 claim 自己的**锚点。编号挂错记 `off_topic`、编造的编号记 "
        "`bogus`，两者都判不过。不是逐句自然语言推理 entailment。",
        "4. **本指标依赖 judge 抄编号**：「claim编号配对率」不足 1 说明 judge 没照做，"
        "读数必须与配对率一起看；`--no-judge` 时本指标不产出（而不是虚高的宽松口径）。",
        "5. **claim漏标锚点数** = 声明了 `anchor_match` 却没配 `anchors` 的 claim 条数，"
        "这些题不参与核验，属于标注欠账，应补齐。",
        "6. **引用准确率拆两列**：「带引用」= 该标注的都标了；「编号可解析」= 标出来的"
        "编号在本次采集映射里真实存在（防凭空编编号）。采集关闭时后者为空。",
        "7. 判分模型与被测模型同源，存在自评偏向；如需更硬的结论可换 judge 模型交叉抽检。"
        "本版已把「引用片段原文」一并喂给 judge，否则它无从判断来源是否真能支撑论断。",
        "8. **判分器失败不折算成任务失败**：judge 两次调用/解析都失败时该运行记「未评」，"
        "既不进成功率分子也不进分母（见「判分失败运行数」与分母列）；只有「连报告都没产出」"
        "才按不通过计。此前 db-02/db-03 是因 `unsupported_claims` 返回对象数组导致解析失败、"
        "被误记 ❌，现已在校验层兜住。",
        "9. **Faithfulness 语义支撑按来源类型分判**：`chunk` 类核引用编号与原文；"
        "`query_cond`（DB）类核「表名 + 筛选/聚合口径」；`url`（网络）类核「是否给出可核查链接」——"
        "后两类本就没有【引用】编号，早先「编号为空即判 false」的写法会让 DB/网络题的语义支撑恒为 0。",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
async def main_async(args) -> int:
    load_dotenv()
    tasks = yaml.safe_load(TASKS_PATH.read_text(encoding="utf-8"))

    if args.list:
        print(f"共 {len(tasks)} 题：")
        for t in tasks:
            flag = "R" if t.get("must_reflect") else " "
            print(f"  [{flag}] {t['id']:<8} {t['difficulty']:<6} {t['query'][:46]}")
        return 0

    if args.check_anchors:
        return check_anchors(tasks)

    selected = tasks
    if args.ids:
        wanted = {s.strip() for s in args.ids.split(",")}
        selected = [t for t in tasks if t["id"] in wanted]
    if args.category:
        selected = [t for t in selected if t["category"] == args.category]
    if not selected:
        print("没有匹配到题目")
        return 1

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    trace_dir = Path(args.trace_dir) if args.trace_dir else RESULTS_DIR / run_id / "traces"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    capture = Path(args.trace_dir).exists() if args.trace_dir else False
    if not capture:
        print(
            "⚠️  未发现采集目录 —— Faithfulness 与「引用编号可解析」将为空。\n"
            "    需要它们请用 EVAL_CAPTURE=1 EVAL_TRACE_DIR=<dir> 启动服务后重跑。"
        )

    print(f"\n===== 生成层评测 {run_id} =====")
    print(f"题目 {len(selected)} 道 × {args.repeat} 次 = {len(selected) * args.repeat} 次运行")
    print(f"价格档：{current_prices()['tier']}（北京时 {datetime.datetime.now():%H:%M}）")
    print(f"采集目录：{trace_dir if capture else '（无）'}\n")

    results: list[TaskResult] = []
    for i, task in enumerate(selected, start=1):
        print(f"[{i}/{len(selected)}] {task['id']} {task['query'][:40]}")
        tr = TaskResult(task=task)
        for r in range(1, args.repeat + 1):
            trace = await run_once(task, r, args.api, args.ws, args.timeout, trace_dir)
            tr.traces.append(trace)
            cost = compute_cost(trace.tokens) if trace.tokens else 0.0
            trace.cost_cny = cost
            status = "ok" if trace.ok else f"失败({trace.error[:40]})"
            print(
                f"    第{r}次: {status} | 工具{trace.tool_calls}次 轮数{trace.rounds} "
                f"延迟{trace.latency_s:.0f}s token≈{trace.tokens.get('input_tokens', 0)}+"
                f"{trace.tokens.get('output_tokens', 0)} 引用{len(trace.citation_ids_in_report)}处"
                f" 成本¥{cost:.4f}"
            )
        results.append(tr)

    print("\n===== 判分 =====")
    judged = []
    for tr in results:
        j = await judge_task(tr.task, tr, use_judge=not args.no_judge)
        judged.append(j)
        faith = f"{j['faithfulness_rule']:.2f}" if j["faithfulness_rule"] is not None else "—"
        print(
            f"  {j['task_id']:<8} {'PASS' if j['success'] else 'FAIL'} "
            f"正确率={j['correctness'] if j['correctness'] is not None else '—'} "
            f"Faith(锚点)={faith} 编号配对={j['faithfulness_paired']}/{j['faithfulness_checked']} "
            f"引用={j['citation_cited']}/{j['citation_required']} "
            f"轮数={j['rounds']} 延迟={['%.0f' % x for x in j['latencies']]}"
        )

    meta = {
        "run_id": run_id,
        "tasks": len(selected),
        "repeat": args.repeat,
        "total_runs": len(selected) * args.repeat,
        "judge": not args.no_judge,
        "price_tier": current_prices()["tier"],
        "rounds": os.getenv("REFLECTION_MAX_ROUNDS", "3"),
        "retrieval": os.getenv("PRODUCTION_MODE", "hybrid_rerank"),
        "capture": capture,
    }
    summary = summarize(judged, meta)

    (run_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "results": judged}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report = render_report(summary, judged, run_id)
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    for key in [
        "任务成功率",
        "任务成功率_分母",
        "判分失败运行数",
        "答案正确率",
        "Faithfulness_claim锚点核验",
        "Faithfulness_claim语义支撑",
        "claim编号配对率",
        "claim漏标锚点数",
        "引用准确率_带引用",
        "引用准确率_编号可解析",
        "平均工具调用次数",
        "P50_延迟s",
        "P95_延迟s",
        "单任务平均成本_元",
        "总成本_元",
    ]:
        v = summary.get(key)
        print(f"  {key:<22} {v if v is not None else '—'}")
    if not args.no_judge and summary.get("claim编号配对率") in (None, 0):
        print(
            "⚠️  judge 没有把「claim → 引用编号」抄回来，claim 级锚点核验无从进行。\n"
            "    检查采集目录是否有 kb_search.*.json，以及报告里是否真的带【引用BN-I】。"
        )
    print(f"\n报告：{run_dir / 'report.md'}")
    print(f"明细：{run_dir / 'summary.json'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="生成层评测 runner（7 指标）")
    p.add_argument("--ids", default="", help="只跑指定题号，逗号分隔")
    p.add_argument("--category", default="", help="只跑某一类：database/knowledge_base/web/mixed")
    p.add_argument("--repeat", type=int, default=3, help="每题重复次数（默认 3）")
    p.add_argument(
        "--timeout",
        type=int,
        default=420,
        help="单次任务超时秒数（默认 420；执行层已有工具预算熔断，卡死题会提前收尾）",
    )
    p.add_argument("--api", default="http://localhost:8001")
    p.add_argument("--ws", default="ws://localhost:8001/ws")
    p.add_argument("--trace-dir", default="", help="采集目录（须与服务端 EVAL_TRACE_DIR 一致）")
    p.add_argument("--no-judge", action="store_true", help="跳过 LLM judge，只算规则层与过程指标")
    p.add_argument(
        "--check-anchors",
        action="store_true",
        help="只自检题库 claim 锚点能否在语料里逐字命中（不跑 agent、不花 API 钱）",
    )
    p.add_argument("--list", action="store_true", help="列出题目后退出")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
