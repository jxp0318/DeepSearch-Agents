"""
自建知识库工具模块

给知识库子智能体提供的两个 LangChain 工具：
list_knowledge_bases 用于发现可用知识库及其文档构成，
search_knowledge_base 对指定知识库执行双路检索，返回原始片段。

与原 RAGFlow 工具的本质区别：这里返回的是**检索到的原文片段**，由子智能体
自行阅读、筛选并综合成回答；而不是把问题转发给外部 Chat 服务拿现成答案。
检索（R）与生成（G）分离，子智能体就是那个 G。

检索走 retriever 的**生产模式 hybrid_rerank**（双路 RRF 融合 + cross-encoder 精排，
见 ADR-012）；精排不可用时 retriever 内部自动退化为纯 RRF，工具层无需感知。
"""

from itertools import count
import os
import time
from typing import Annotated

from langchain_core.tools import tool

from app.api.context import get_thread_context
from app.api.monitor import (
    check_call_budget,
    eval_capture_enabled,
    monitor,
    register_tool_result,
)
from app.rag.retriever import retriever

# ---------------------------------------------------------------------------
# 引用批次号分配（ADR-019）
#
# 每条检索片段带一个【引用B{批次号}-{批内序号}】编号，批次号保证「跨批次不重号」：
# 子智能体在一次任务里会连续检索十几批（提示词要求多角度检索），若每批都从
# 【引用1】重新编号，最终文档里会出现大量同名编号，模型只能自己发明前缀消歧。
#
# 计数器放在 Redis（key = refbatch:{thread_id}，INCR 原子自增）而不是进程内存：
# 进程内存计数在服务重启后归零，同一会话跨重启继续追问时编号会与历史撞号。
#
# TTL 用滑动窗口（每次自增顺手 EXPIRE 刷新）：活跃会话永不过期，冷会话自动回收，
# 避免只增不减的 key 无限堆积。Redis 不可用时退回进程内计数器——引用编号只是
# 溯源标记，降级后依然可用（编号连续性由进程内单调递增保证）。
# ---------------------------------------------------------------------------
_REDIS_URL = os.getenv("REDIS_URL", "").strip()
_BATCH_KEY_PREFIX = "refbatch:"
_BATCH_TTL_SECONDS = int(float(os.getenv("REF_BATCH_TTL_DAYS", "7")) * 86400)
# Redis 故障退避：30s 起，每次失败翻倍，上限 5 分钟。
# 为什么必须退避：计数器分配在**检索关键路径**上，Redis 不可达时若每次检索都重连，
# 实测单次连接失败要耗十几秒，会把整条检索链路拖垮。退避后最坏情况是
# 「5 分钟才重试一次」，代价可以忽略。
_REDIS_BACKOFF_MIN_SECONDS = 30
_REDIS_BACKOFF_MAX_SECONDS = 300
_REDIS_CONNECT_TIMEOUT = 1

_local_batch = count(1)
_redis_client = None
_redis_backoff_until = 0.0
_redis_backoff_seconds = _REDIS_BACKOFF_MIN_SECONDS


def _get_redis():
    """惰性获取 Redis 客户端；不可用时返回 None 并进入指数退避"""
    global _redis_client, _redis_backoff_until, _redis_backoff_seconds

    if not _REDIS_URL:
        return None
    if time.time() < _redis_backoff_until:
        return None
    if _redis_client is None:
        try:
            import redis as redis_lib

            client = redis_lib.Redis.from_url(
                _REDIS_URL,
                socket_connect_timeout=_REDIS_CONNECT_TIMEOUT,
                socket_timeout=_REDIS_CONNECT_TIMEOUT,
                decode_responses=True,
            )
            client.ping()
            _redis_client = client
            _redis_backoff_seconds = _REDIS_BACKOFF_MIN_SECONDS  # 连通后重置退避
        except Exception as e:
            wait = min(_redis_backoff_seconds, _REDIS_BACKOFF_MAX_SECONDS)
            _redis_backoff_until = time.time() + wait
            print(
                f"[KB] 引用批次计数器连不上 Redis（{type(e).__name__}），"
                f"{wait:.0f}s 内不再重试，期间用进程内计数器"
            )
            _redis_backoff_seconds = wait * 2
    return _redis_client


def warmup_ref_batch_counter() -> bool:
    """
    预热引用批次计数器（会话开始时调用一次）。

    放在 run_deep_agent 开头并放到线程池里跑有两个好处：Redis 可用时，
    后续每次检索直接复用连接；不可用时，退避状态在第一次检索之前就确定，
    检索路径上不会付出任何连接代价。

    :return: 计数器是否可用（False 表示本次会话将使用进程内计数器）
    """
    return _get_redis() is not None


def _next_batch_no() -> int:
    """
    分配本次检索的批次号。

    优先用 Redis 按会话自增（跨重启不归零）；Redis 不可用或无 thread_id 时
    退回进程内计数器。两条路径都是单调递增，区别只在重启后是否连续。
    """
    thread_id = get_thread_context()
    client = _get_redis() if thread_id else None
    if client is not None:
        key = f"{_BATCH_KEY_PREFIX}{thread_id}"
        try:
            pipe = client.pipeline()
            pipe.incr(key)
            # 滑动窗口：每次自增都把过期时间推后，活跃会话不会被回收
            pipe.expire(key, _BATCH_TTL_SECONDS)
            return int(pipe.execute()[0])
        except Exception as e:
            # 连接已建立但命令失败：丢弃客户端并进入退避，下一次走进程内计数器
            global _redis_client, _redis_backoff_until, _redis_backoff_seconds
            _redis_client = None
            wait = min(_redis_backoff_seconds, _REDIS_BACKOFF_MAX_SECONDS)
            _redis_backoff_until = time.time() + wait
            print(f"[KB] 引用批次号分配失败（{e}），{wait:.0f}s 内退回进程内计数器")
            _redis_backoff_seconds = wait * 2
    return next(_local_batch)


@tool
def list_knowledge_bases() -> str:
    """
    查询当前系统中有哪些可用知识库，以及每个知识库包含的文档

    作用：让模型先了解「哪个知识库装了哪类文档」，再决定向哪个知识库检索。
    调用 search_knowledge_base 之前，应先调用本工具确认知识库名称。
    :return: 知识库名称、文档清单与索引状态；无知识库时返回中文提示
    """
    monitor.report_tool(tool_name="知识库列表查询工具：list_knowledge_bases")

    summaries = retriever.kb_summary()
    if not summaries:
        return "当前没有任何可用知识库"

    lines = []
    for item in summaries:
        status = "已就绪" if item["indexed"] else "未摄入索引"
        docs = "、".join(item["docs"]) if item["docs"] else "无文档"
        lines.append(
            f"知识库名称: {item['name']}（{status}）| 包含文档: {docs}"
        )
    return "\n".join(lines)


@tool
def search_knowledge_base(
    kb_name: Annotated[str, "知识库名称，必须来自 list_knowledge_bases 的返回结果"],
    query: Annotated[str, "检索问题，围绕用户原始需求提炼出的具体信息需求"],
    top_k: Annotated[int, "返回的片段数量，默认 5；问题复杂时可增大到 10"] = 5,
) -> str:
    """
    对指定知识库执行检索，返回与问题最相关的原文片段

    返回内容为知识库文档中的原始片段（含来源文档与章节定位），不是现成答案。
    请基于片段内容自行综合回答，并在回答中注明信息来自哪份文档。
    每条片段都带唯一引用编号【引用B{批次}-{批内序号}】，回答时必须原样沿用该编号标注来源。
    检索走 retriever 的生产模式 hybrid_rerank（双路 RRF 融合 + cross-encoder 精排）。
    :param kb_name: 知识库名称
    :param query: 检索问题
    :param top_k: 返回片段数
    :return: 片段列表文本；检索失败或无结果时返回中文提示
    """
    monitor.report_tool(
        tool_name="知识库检索工具：search_knowledge_base",
        args={"kb_name": kb_name, "query": query, "top_k": top_k},
        retrieval=True,
    )

    # 预算熔断：超上限后本次检索**不执行**，直接把收尾指令还给模型。
    # 放在领取批次号之前——没交付片段就不该占用编号，否则「编号连续不重号」会出现空洞
    blocked = check_call_budget("search_knowledge_base")
    if blocked:
        return blocked

    # 批次号在真正检索之前领取：即使本次检索失败，这次调用也占用一个批次号，
    # 保证「编号连续且不重号」这一语义对模型稳定
    batch = _next_batch_no()

    try:
        results = retriever.search(kb_name, query, top_k=top_k)
    except FileNotFoundError as e:
        return f"知识库不可用: {e}"
    except Exception as e:  # noqa: BLE001 —— 工具层兜底，不让检索异常冒到图顶层
        return f"知识库检索失败，错误原因: {e}"

    if not results:
        return (
            f"在知识库「{kb_name}」中没有检索到与问题相关的内容，"
            "建议换个提问角度重试，或说明该知识库暂无相关资料"
        )

    # 片段按相关度降序拼接，标注来源文档与章节，供子智能体引用与溯源。
    # 引用编号 = 批次号 + 批内序号（如【引用B3-2】）：批次号全局单调递增，
    # 因此子智能体连续检索十几批也不会重号，最终文档可逐条溯源。
    parts = [
        f"在知识库「{kb_name}」中检索到 {len(results)} 条相关片段"
        f"（第 {batch} 批检索，每条带有唯一引用编号）："
    ]
    for seq, chunk in enumerate(results, start=1):
        source = chunk["doc"]
        if chunk.get("heading"):
            source += f" · {chunk['heading']}"
        parts.append(
            f"\n【引用B{batch}-{seq}】来源: {source}（相关度 {chunk['score']:.2f}）"
            f"\n{chunk['text']}"
        )
    parts.append(
        "\n\n引用规范（必须遵守，用于最终交付物的溯源）："
        "\n1. 回答中凡使用上述片段的信息，必须在对应结论后原样标注引用编号，如【引用B3-2】；"
        "\n2. 回答末尾必须附「引用来源」清单：编号 → 文档名 · 章节；"
        "\n3. 编号格式为【引用B{批次号}-{批内序号}】，由系统按会话统一分配、跨批次不重号，"
        "禁止改写、合并或自行添加前缀。"
    )

    # 评测采集（EVAL_CAPTURE=1 时）：把「引用编号 → 片段原文」的映射落盘，
    # 评测 runner 据此判定引用准确率（编号是否真实、是否指向支持该结论的块）
    # 与 Faithfulness（论断能否被引用块支撑）。关闭时本段完全不执行。
    if eval_capture_enabled():
        register_tool_result(
            get_thread_context(),
            "kb_search",
            {
                "kb": kb_name,
                "query": query,
                "batch": batch,
                "citations": [
                    {
                        "id": f"B{batch}-{i}",
                        "doc": chunk["doc"],
                        "heading": chunk.get("heading", ""),
                        "score": chunk["score"],
                        "text": chunk["text"],
                    }
                    for i, chunk in enumerate(results, start=1)
                ],
            },
        )

    return "\n".join(parts)
