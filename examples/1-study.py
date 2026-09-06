"""
DeepAgents 快速入门示例：使用 Tavily 搜索工具

本示例演示如何：
1. 从 .env 文件读取配置（大模型密钥、Tavily API 密钥）
2. 创建 Tavily 搜索工具
3. 初始化 DeepAgent 智能体
4. 执行搜索任务并输出最终结果

适合新手学习 DeepAgent 的基本用法
"""

import os
import time
from typing import Literal

# 导入必要的库
from deepagents import create_deep_agent  # DeepAgent 核心创建函数
from dotenv import find_dotenv, load_dotenv  # 环境变量加载工具
from langchain.chat_models import init_chat_model  # LangChain 模型初始化
from langchain.tools import tool  # 工具装饰器
from tavily import TavilyClient  # Tavily 搜索客户端
import requests  # 用于捕获网络异常

# ==================== 第一步：加载环境变量 ====================
# find_dotenv() 自动查找项目根目录的 .env 文件
# load_dotenv() 将 .env 中的键值对加载到系统环境变量中
load_dotenv(find_dotenv())

# 从环境变量中读取配置
# LLM_QWEN_MAX: 指定使用的大模型名称（如 qwen-max、deepseek-v4-pro 等）
llm_name = os.getenv("LLM_QWEN_MAX")

# TAVILY_API_KEY: Tavily 搜索引擎的 API 密钥（需要在 https://app.tavily.com/ 注册获取）
tavily_key = os.getenv("TAVILY_API_KEY")

# 检查是否成功读取配置
if not llm_name or not tavily_key:
    raise ValueError("请在 .env 文件中配置 LLM_QWEN_MAX 和 TAVILY_API_KEY")

print(f"✅ 已加载配置：")
print(f"   - 大模型: {llm_name}")
print(f"   - Tavily API: {'*' * 8}{tavily_key[-4:] if tavily_key else '未配置'}")

# ==================== 第二步：初始化 Tavily 客户端 ====================
# TavilyClient 是 Tavily 官方提供的 Python SDK 客户端
# 传入 API 密钥后，可以调用 search() 方法进行搜索
tavily_client = TavilyClient(api_key=tavily_key)


# ==================== 第三步：定义搜索工具 ====================
# @tool 装饰器将普通函数转换为 LangChain 工具
# DeepAgent 会根据函数的 docstring（文档字符串）和参数签名，自动理解何时调用此工具
@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["news", "finance", "general"] = "general",
    include_raw_content: bool = False,
):
    """
    互联网搜索工具 - 用于检索网络上的最新信息

    当用户需要查询实时信息、新闻、行业动态时，DeepAgent 会自动调用此工具。

    参数说明：
    :param query: 搜索关键词（必填），例如："人工智能 最新进展"
    :param max_results: 返回结果数量（可选，默认5条），范围建议 3-10
    :param topic: 搜索主题分类（可选，默认 general）
                  - "news": 新闻类内容
                  - "finance": 金融财经类内容
                  - "general": 通用搜索
    :param include_raw_content: 是否返回完整网页内容（可选，默认 False）
                                - False: 只返回摘要（速度快）
                                - True: 返回完整原文（信息更全但速度慢）

    :return: Tavily 搜索结果字典，包含 title、url、content、published_date 等字段
    """
    # 打印日志，方便调试时查看工具调用情况
    print(f"\n🔍 正在调用搜索工具...")
    print(f"   关键词: {query}")
    print(f"   结果数: {max_results}")
    print(f"   主题: {topic}")

    # 添加重试机制，应对网络不稳定问题
    max_retries = 3
    retry_delay = 2  # 秒

    for attempt in range(1, max_retries + 1):
        try:
            # 调用 Tavily API 执行实际搜索
            search_result = tavily_client.search(
                query=query,
                max_results=max_results,
                topic=topic,
                include_raw_content=include_raw_content,
            )

            print(f"   ✅ 搜索完成，返回 {len(search_result.get('results', []))} 条结果\n")
            return search_result

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                ConnectionResetError) as e:
            if attempt < max_retries:
                print(f"   ⚠️  网络连接失败 (尝试 {attempt}/{max_retries}): {str(e)[:50]}")
                print(f"   🔄 {retry_delay} 秒后重试...\n")
                time.sleep(retry_delay)
            else:
                print(f"   ❌ 搜索失败，已重试 {max_retries} 次")
                raise Exception(f"Tavily API 连接失败: {str(e)}")


# ==================== 第四步：初始化大语言模型 ====================
# init_chat_model 是 LangChain 提供的统一模型初始化接口
# model_provider="openai" 表示使用 OpenAI 兼容格式的 API
#
# 重要：虽然这里写的是 "openai"，但实际上可以连接任何兼容 OpenAI 格式的 API
# 例如：DeepSeek、通义千问、Moonshot 等
# 具体连接哪个服务，由 .env 中的 OPENAI_BASE_URL 决定
llm = init_chat_model(
    model=llm_name,           # 模型名称，从 .env 读取
    model_provider="openai"   # 使用 OpenAI 兼容格式
)

print(f"✅ 大模型初始化完成: {llm_name}\n")


# ==================== 第五步：创建 DeepAgent ====================
# create_deep_agent 是 DeepAgents 框架的核心函数
# 它创建一个能够自主规划、调用工具、生成报告的智能体

deep_agent = create_deep_agent(
    model=llm,                    # 传入初始化好的大语言模型
    tools=[internet_search],      # 传入工具列表，可以有多个工具
    subagents=[],                 # 子智能体列表（本示例不使用）

    # system_prompt: 系统提示词，定义智能体的角色和行为准则
    # 这个提示词会告诉 AI：
    # 1. 它的身份是什么
    # 2. 可以使用哪些工具
    # 3. 应该如何处理信息和生成回答
    system_prompt="""
    你是一名专业的研究分析师，具备以下能力：

    1. **信息检索**: 你可以使用 internet_search 工具搜索互联网上的最新信息
    2. **信息分析**: 对搜索结果进行归纳、整理、交叉验证
    3. **报告生成**: 基于可靠信息生成结构清晰、逻辑严谨的中文报告

    工作流程：
    - 首先分析用户需求，确定是否需要搜索
    - 如果需要，调用 internet_search 工具获取信息
    - 对搜索结果进行分析整合
    - 最后生成一份简洁、准确、有洞察力的中文报告

    注意事项：
    - 优先使用最新的信息来源
    - 如果信息存在矛盾，需要明确指出
    - 报告要结构清晰，重点突出
    - 用词专业但不晦涩，让普通读者也能理解
    """
)

print("✅ DeepAgent 创建完成\n")
print("=" * 60)


# ==================== 第六步：执行任务 ====================
# 定义用户问题
user_question = "请查询2025年人工智能领域的重大突破和热门趋势，并整理为一份简要报告。"

print(f"📝 用户问题：{user_question}\n")
print("=" * 60)
print("⏳ DeepAgent 正在思考和处理...\n")

# invoke() 方法：非流式执行，等待整个流程完成后一次性返回结果
# 执行过程：
# 1. 用户问题 -> 2. 模型分析 -> 3. 决定是否调用工具 ->
# 4. 执行工具 -> 5. 模型整理结果 -> 6. 生成最终回答
result = deep_agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": user_question,
        }
    ]
})

# ==================== 第七步：输出最终结果 ====================
print("\n" + "=" * 60)
print("📊 最终报告\n")
print("=" * 60)

# result["messages"] 包含了完整的对话历史
# 最后一条消息就是 DeepAgent 生成的最终回答
final_answer = result["messages"][-1].content

print(final_answer)

print("\n" + "=" * 60)
print("✅ 任务完成！")
