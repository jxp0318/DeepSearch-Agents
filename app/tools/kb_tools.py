"""
自建知识库工具模块

给知识库子智能体提供的两个 LangChain 工具：
list_knowledge_bases 用于发现可用知识库及其文档构成，
search_knowledge_base 对指定知识库执行双路检索，返回原始片段。

与原 RAGFlow 工具的本质区别：这里返回的是**检索到的原文片段**，由子智能体
自行阅读、筛选并综合成回答；而不是把问题转发给外部 Chat 服务拿现成答案。
检索（R）与生成（G）分离，子智能体就是那个 G。
"""

from typing import Annotated

from langchain_core.tools import tool

from app.api.monitor import monitor
from app.rag.retriever import retriever


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
    :param kb_name: 知识库名称
    :param query: 检索问题
    :param top_k: 返回片段数
    :return: 片段列表文本；检索失败或无结果时返回中文提示
    """
    monitor.report_tool(
        tool_name="知识库检索工具：search_knowledge_base",
        args={"kb_name": kb_name, "query": query, "top_k": top_k},
    )

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

    # 片段按相关度降序拼接，标注来源文档与章节，供子智能体引用与溯源
    parts = [f"在知识库「{kb_name}」中检索到 {len(results)} 条相关片段："]
    for seq, chunk in enumerate(results, start=1):
        source = chunk["doc"]
        if chunk.get("heading"):
            source += f" · {chunk['heading']}"
        parts.append(f"\n【片段{seq}】来源: {source}（相关度 {chunk['score']}）\n{chunk['text']}")
    return "\n".join(parts)
