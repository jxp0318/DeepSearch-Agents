"""
DeepAgents 快速入门：搜索工具 + 非流式调用

演示如何把 Tavily 搜索封装成 LangChain 工具，并交给 DeepAgent 自动决策调用
本示例使用 invoke 一次性获取最终结果，适合理解 DeepAgent 的最小运行链路
用户问题 -> 大模型决策 -> 工具调用 -> 大模型整理最终回答
"""

import os
from typing import Literal

from deepagents import create_deep_agent
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from tavily import TavilyClient

# 读取项目根目录中的 .env，示例依赖 LLM_QWEN_MAX 和 TAVILY_API_KEY
load_dotenv(find_dotenv())

llm_name = os.getenv("LLM_QWEN_MAX")
tavily_key = os.getenv("TAVILY_API_KEY")


# Tavily 客户端负责真正的联网搜索，工具函数中会复用这个客户端
tavily_client = TavilyClient(api_key=tavily_key)


@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["news", "finance", "general"] = "general",
    include_raw_content: bool = False,
):
    """
    互联网搜索工具

    DeepAgent 会根据工具描述和参数签名，自动决定是否调用该工具
    :param query: 搜索关键词
    :param max_results: 返回结果数量
    :param topic: 查询主题，可选 news、finance、general
    :param include_raw_content: 是否返回更详细的原文内容，include_raw_content=False 时返回摘要内容；True 时会尝试返回更完整的网页原文
    :return: Tavily 搜索结果
    """
    print(
        f"开始调用网络搜索工具，核心参数为：{query},{max_results},{topic},{include_raw_content}"
    )
    return tavily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )


# 使用 OpenAI 兼容接口初始化千问模型
# 模型名从 .env 的 LLM_QWEN_MAX 读取，例如 qwen-max
llm = init_chat_model(model=llm_name, model_provider="openai")

# 创建 DeepAgent，模型负责推理和规划，tools 提供可被调用的外部能力
# 当前示例不配置子智能体，重点观察“主智能体 + 搜索工具”的基本流程
deep_agent = create_deep_agent(
    model=llm,
    tools=[internet_search],
    subagents=[],
    system_prompt="""
    你是一名严谨的研究员，可以使用 internet_search 工具检索网络信息。
    请根据检索结果进行归纳、分析和交叉验证，生成一份结构清晰、信息可靠的中文报告。
    """,
)

# 非流式执行，invoke 会等整条 agent 链路完成后，一次性返回最终状态
result = deep_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "请查询人工智能和机器人领域的热门新闻信息，并整理为一份简要报告。",
            }
        ]
    }
)

# 运行时会先看到工具函数内部的日志，例如：
# 开始调用网络搜索工具，核心参数为：人工智能 机器人 热门新闻,5,news,False
#
#
# result 中会保留完整消息轨迹，便于观察模型决策、工具返回和最终回答。
# 结合本示例的实际输出，messages 通常包含下面四类关键消息：
#
# 1. HumanMessage
#    用户原始问题：
#    请查询人工智能和机器人领域的热门新闻信息，并整理为一份简要报告。
#
# 2. AIMessage(content="", tool_calls=[...])
#    这不是最终回答，而是模型决定调用工具。
#    重点看 tool_calls 中的工具名和参数，例如：
#    name: internet_search
#    args: {
#        "query": "人工智能 机器人 热门新闻",
#        "topic": "news",
#        "max_results": 5,
#        "include_raw_content": False,
#    }
#
# 3. ToolMessage(name="internet_search")
#    Agent 根据 tool_calls 真正执行 internet_search 后得到的工具返回。
#    Tavily 返回的数据通常包含 query、results、url、title、content、published_date 等字段。
#
# 4. AIMessage(content="...")
#    模型基于工具返回结果整理出的最终报告。
#    到这一步 tool_calls 为空，content 中才是用户最终需要阅读的自然语言答案。
print(result)

# messages 的最后一条通常就是 DeepAgent 整理后的最终回答。
# 如果只关心最终报告，可以直接读取 result["messages"][-1].content，
# 不必把完整的 result 执行轨迹展示给最终用户。
print(result["messages"][-1].content)
