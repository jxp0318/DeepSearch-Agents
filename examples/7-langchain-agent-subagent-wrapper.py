"""
DeepAgents 子智能体：接入 LangChain 资料检索 Agent

演示如何把 LangChain create_agent 创建的工具型 Agent 封装成 DeepAgents 子智能体。
这个例子模拟“深度研搜”中的资料检索子任务：
用户问题 -> 主智能体决策 -> task 调用 research_retriever_agent
-> LangChain Agent 自主选择公开资料工具或内部知识库工具 -> 主智能体整理最终回答

对应本仓库已有 LangChain 知识点：
- @tool 工具封装：参考 案例与源码-2-LangChain框架/08-tools/QueryWeatherTool.py
- create_agent 工具型 Agent：参考 案例与源码-2-LangChain框架/12-agent/AgentSmartSelectV1.0.py
- 结构化结果意识：参考 案例与源码-2-LangChain框架/05_parser/StructuredOutput_TypedDict.py
"""

import os

from deepagents import CompiledSubAgent, create_deep_agent
from dotenv import find_dotenv, load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv(find_dotenv())

llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai",
)


@tool
def search_public_web(query: str) -> str:
    """
    检索公开网络资料。

    参数:
        query: 用户关注的研究主题或关键词。

    返回:
        模拟的公开资料摘要。真实项目中可以替换为 Tavily、搜索引擎 API 或爬虫服务。
    """
    print(f"【LangChain Tool】检索公开资料：{query}")
    return (
        "公开资料检索结果：\n"
        "1. 多家机构认为具身智能正在推动机器人从单点自动化走向通用任务执行。\n"
        "2. 机器人产业热点集中在大模型控制、灵巧手、低成本传感器和仿真训练。\n"
        "3. 公开报道显示，头部公司正在加速布局工业、仓储、家庭服务等场景。"
    )


@tool
def search_internal_knowledge_base(query: str) -> str:
    """
    检索企业内部知识库。

    参数:
        query: 用户关注的研究主题或关键词。

    返回:
        模拟的内部知识库摘要。真实项目中可以替换为 Qdrant、Elasticsearch 或企业知识库。
    """
    print(f"【LangChain Tool】检索内部知识库：{query}")
    return (
        "内部知识库检索结果：\n"
        "1. 历史项目复盘显示，客户最关注机器人方案的稳定性、部署周期和维护成本。\n"
        "2. 销售材料中高频卖点包括：多模态感知、自动任务分解、远程运维和持续学习。\n"
        "3. 交付团队建议在研究报告中单独列出落地风险和数据闭环建设方案。"
    )


research_retriever_agent = create_agent(
    model=llm,
    tools=[search_public_web, search_internal_knowledge_base],
    system_prompt="""
    你是资料检索助手，负责为深度研究任务收集资料。
    当用户需要行业公开信息时，调用 search_public_web。
    当用户需要企业内部经验、项目复盘或知识库内容时，调用 search_internal_knowledge_base。
    如果问题同时涉及公开趋势和内部经验，可以两个工具都调用。
    最后请用中文输出：资料来源、关键发现、后续建议。
    """,
)


research_retriever_subagent = CompiledSubAgent(
    name="research_retriever_agent",
    description="用于检索公开资料和企业内部知识库，适合为深度研究报告收集证据和背景信息。",
    runnable=research_retriever_agent,
)


deep_agent = create_deep_agent(
    model=llm,
    tools=[],
    system_prompt="""
    你是深度研搜系统的主智能体。
    当用户需要查找资料、收集证据、检索公开信息或内部知识库时，必须调用 research_retriever_agent。
    你不直接检索资料，只负责分派任务并整理子智能体返回的结果。
    """,
    subagents=[research_retriever_subagent],
)


for chunk in deep_agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "请同时检索人工智能机器人行业的公开趋势，以及我们内部知识库里的项目经验。",
            }
        ]
    }
):
    print(chunk)
