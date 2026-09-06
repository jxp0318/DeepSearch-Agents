"""
DeepAgents 快速入门：搜索工具 + 流式解析

演示如何使用 stream 逐步读取 DeepAgent 的执行过程
相比 invoke 只拿最终结果，stream 更适合观察 agent 的中间状态
模型决定调用工具、工具返回结果、模型生成最终回答
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
    include_raw_content=False 时返回摘要内容；True 时会尝试返回更完整的网页原文
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

# 流式执行，stream 会在每个图节点完成后产出一个 chunk
# 常见节点包括 model（模型决策或最终回答）和 tools（工具执行结果）
# 流式处理结果
stream = deep_agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "请查询人工智能和机器人领域的热门新闻信息，并整理为一份简要报告。",
            }
        ]
    }
)

# 循环获取块
for chunk in stream:
    # chunk 是一个按节点名组织的字典，例如
    # {"model": {"messages": [...]}} 或 {"tools": {"messages": [...]}}
    for node_name, state in chunk.items():
        # DeepAgents 内部中间件也可能产出空状态或非消息状态，这里只解析消息类状态
        if not state or "messages" not in state:
            continue

        messages = state["messages"]
        if not messages or not isinstance(messages, list):
            continue

        # 每个 chunk 的最后一条消息，通常就是这个节点本次产出的核心信息
        last_msg = messages[-1]

        if node_name == "model":
            # 情况一：模型决定调用工具或子智能体
            # model 节点有两类重点事件
            # 1. tool_calls 非空，模型决定下一步调用工具或子智能体
            # 2. content 非空，模型已经生成最终回答
            if last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    if tool_call["name"] == "task":
                        print(
                            f"【大模型】决定调用子智能体：{tool_call['args']['subagent_type']}"
                        )
                    else:
                        print(
                            f"【大模型】决定调用工具：{tool_call['name']} 传入的参数：{tool_call['args']}"
                        )

            # 情况二：模型生成最终结果
            elif last_msg.content:
                print(f"【大模型】最终执行的结果：{last_msg.content}")

        # 情况三：工具执行完成，返回结果
        elif node_name == "tools":
            # tools 节点返回的是具体工具的执行结果，通常可以推送给前端展示执行进度
            tool_return_result = last_msg.content[:100] + "..."
            tool_name = last_msg.name
            print(f"【agent】调用了{tool_name}工具，返回的结果为：{tool_return_result}")
