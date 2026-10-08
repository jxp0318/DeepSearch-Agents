"""
Tavily 网络搜索工具模块

封装 internet_search 工具，供网络搜索子智能体检索互联网公开信息
工具内部会先通过 monitor 上报调用参数，再请求 Tavily API 返回结构化搜索结果

容错设计（对应线上出现过的 ConnectionResetError 10054）：
1. 传输层：给 requests.Session 挂带退避的 Retry，处理连接池里已被对端关闭的
   keep-alive 连接——这类「复用坏连接」是间歇性连接重置最常见的原因
2. 应用层：整体失败后按指数退避再试若干次，覆盖代理抖动等短暂不可用
3. 兜底：仍然失败时不再抛异常，而是返回一段可读的失败说明。
   工具抛异常会顺着 LangGraph 冒到图顶层，把整轮任务连同已搜集的资料一起作废；
   返回文本则让模型有机会换关键词、改用其他助手，或如实说明该部分信息未获取到
"""

import json
import os
import time
from typing import Literal

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from requests.adapters import HTTPAdapter
from tavily import TavilyClient
from urllib3.util.retry import Retry

from app.api.monitor import monitor

load_dotenv()

# 单次请求超时：搜索接口正常 1~3 秒返回，30 秒足够容忍慢链路
# 偏大是因为 include_raw_content=True 时服务端需要抓取正文，耗时会更长
SEARCH_TIMEOUT = float(os.getenv("TAVILY_TIMEOUT", "30"))
# 传输层重试次数（连接失败 / 读超时 / 5xx / 429 都会重试）
TRANSPORT_RETRIES = int(os.getenv("TAVILY_TRANSPORT_RETRIES", "3"))
# 应用层尝试次数：传输层重试全部耗尽后，整体再试的次数
MAX_ATTEMPTS = int(os.getenv("TAVILY_MAX_ATTEMPTS", "2"))
# 退避基数，第 n 次重试等待 BACKOFF_BASE * 2^(n-1) 秒
BACKOFF_BASE = float(os.getenv("TAVILY_BACKOFF_BASE", "0.8"))

# 这几种错误重试没有意义，只会浪费时间和配额：密钥、权限、参数、配额问题
NON_RETRYABLE_ERRORS = (
    "InvalidAPIKeyError",
    "MissingAPIKeyError",
    "ForbiddenError",
    "BadRequestError",
    "UsageLimitExceededError",
)


def _build_session() -> requests.Session:
    """
    构建带重试的连接池会话

    requests 默认 max_retries=0：连接池里一条已经被代理或服务端关闭的 keep-alive
    连接被复用时，会立刻抛出 ConnectionError，不做任何重试。搜索是 POST 请求，
    还需要显式把 POST 加进 allowed_methods，否则 urllib3 默认只重试幂等方法。
    """
    retry = Retry(
        total=TRANSPORT_RETRIES,
        connect=TRANSPORT_RETRIES,
        read=TRANSPORT_RETRIES,
        status=TRANSPORT_RETRIES,
        backoff_factor=BACKOFF_BASE,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        respect_retry_after_header=True,
        # 重试耗尽后把响应交回上层，由业务代码统一判定，便于记录具体状态码
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# TavilyClient 是实际访问搜索服务的客户端；模块级复用可避免每次工具调用重复初始化
# session 走带重试的连接池；如所在网络需要代理，可设置 TAVILY_HTTP_PROXY / TAVILY_HTTPS_PROXY
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY"),
    session=_build_session(),
)


def _is_retryable(exc: BaseException) -> bool:
    """判断异常是否值得重试：网络类可重试，鉴权 / 配额 / 参数类不可重试"""
    return type(exc).__name__ not in NON_RETRYABLE_ERRORS


def _describe_error(exc: BaseException) -> tuple[str, str]:
    """
    把底层异常翻译成「原因 + 说明」，同时给前端展示和给模型做决策依据

    :return: (错误类型标签, 可读说明)
    """
    name = type(exc).__name__

    if isinstance(exc, (requests.exceptions.ConnectionError, ConnectionResetError)):
        return (
            "连接被中断",
            "到搜索服务的连接被重置或不可达，常见原因是网络链路不稳定、代理中断或本地网络切换",
        )
    if isinstance(exc, requests.exceptions.Timeout) or name == "TimeoutError":
        return "请求超时", f"搜索服务在 {SEARCH_TIMEOUT:.0f} 秒内没有返回结果"
    if name == "InvalidAPIKeyError":
        return "密钥无效", "TAVILY_API_KEY 无效或已失效，请检查 .env 配置"
    if name == "UsageLimitExceededError":
        return "配额耗尽", "Tavily 账户的调用配额已用完"
    if name == "ForbiddenError":
        return "权限不足", "当前 API Key 无权访问该接口"
    if isinstance(exc, requests.exceptions.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", "未知")
        return "服务端错误", f"搜索服务返回 HTTP {status}"
    return name, str(exc) or "未提供更多信息"


def _search_with_retry(**kwargs) -> dict:
    """带指数退避的整体重试，传输层重试耗尽后仍失败才会走到这里"""
    last_error: BaseException | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return tavily_client.search(**kwargs)
        except Exception as exc:
            last_error = exc

            # 鉴权 / 配额类错误重试无意义，直接抛出交给上层兜底
            if not _is_retryable(exc):
                raise

            if attempt < MAX_ATTEMPTS:
                delay = BACKOFF_BASE * (2 ** (attempt - 1))
                print(
                    f"[Tavily] 第 {attempt}/{MAX_ATTEMPTS} 次尝试失败"
                    f"（{type(exc).__name__}），{delay:.1f}s 后重试"
                )
                time.sleep(delay)

    assert last_error is not None
    raise last_error


# @tool 会把函数签名和 docstring 暴露给 DeepAgents，模型据此决定是否调用以及如何填参
@tool
def internet_search(
    query: str,
    topic: Literal["news", "finance", "general"] = "general",
    max_results: int = 5,
    include_raw_content: bool = False,
):
    """
    根据用户问题检索互联网公开信息

    注意：本工具只用于外部公开网页、新闻、政策等信息，不用于查询业务数据库或 RAGFlow 私有知识库
    :param query: 搜索关键词或自然语言问题
    :param topic: 搜索主题，可选 news、finance、general
    :param max_results: 返回的最大结果数
    :param include_raw_content: 是否返回网页原文内容；False 返回摘要，True 尝试返回更完整正文
    :return: Tavily 返回的结构化搜索结果；失败时返回 JSON 格式的错误说明（不抛异常）
    """
    # 工具内部埋点比外层 stream 解析更直接：只要工具被调用，前端就能看到本次搜索参数
    # 这里只上报查询参数，不上报搜索结果正文，避免监控事件体过大
    monitor.report_tool(
        tool_name="网络搜索工具",
        args={
            "query": query,
            "topic": topic,
            "max_results": max_results,
            "include_raw_content": include_raw_content,
        },
    )

    started_at = time.time()
    try:
        result = _search_with_retry(
            query=query,
            topic=topic,
            max_results=max_results,
            include_raw_content=include_raw_content,
            timeout=SEARCH_TIMEOUT,
        )
        print(f"[Tavily] 搜索成功，耗时 {time.time() - started_at:.1f}s")
        return result
    except Exception as exc:
        # 兜底：搜索失败不能把整轮任务连带作废，转成结构化错误交给模型决策
        reason, detail = _describe_error(exc)
        cost = time.time() - started_at
        print(f"[Tavily] 搜索最终失败（{reason}），累计耗时 {cost:.1f}s：{exc}")

        # 失败也上报前端：用户能区分「模型没去搜」和「搜了但网络断了」
        monitor.report_tool_error(
            tool_name="网络搜索工具",
            reason=reason,
            detail=detail,
            context={"query": query, "attempts": MAX_ATTEMPTS},
        )

        return json.dumps(
            {
                "error": True,
                "reason": reason,
                "detail": detail,
                "query": query,
                "guidance": (
                    "本次网络搜索未取得结果。请勿用完全相同的查询重复调用本工具；"
                    "可以尝试更换更简短的关键词，或改用知识库检索助手 / 数据库查询助手获取信息；"
                    "若仍无法取得，请在最终回答中明确说明该部分信息因检索失败而缺失，不要编造内容。"
                ),
            },
            ensure_ascii=False,
        )


if __name__ == "__main__":
    from pprint import pprint

    # 本地调试入口：直接运行本文件可验证 TAVILY_API_KEY 和 Tavily API 是否可用
    pprint(
        internet_search.invoke(
            {"query": "2026中国法定节假日放假安排表，我天天都想要放假"}
        )
    )
