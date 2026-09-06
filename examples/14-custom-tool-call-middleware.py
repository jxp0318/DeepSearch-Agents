"""
DeepAgents Middleware：自定义工具调用中间件

演示如何使用 @wrap_tool_call 包住一次工具调用
本示例在工具执行前后打印日志，帮助理解 request、handler 和 handler(request)
用户请求 -> Agent 决定调用工具 -> log_tool_call 中间件 -> add_numbers 工具 -> 返回结果
"""

import os
import time

from deepagents import create_deep_agent
from dotenv import find_dotenv, load_dotenv
from langchain.agents.middleware import wrap_tool_call
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(find_dotenv())


@tool
def add_numbers(a: int, b: int):
    """
    计算两个数字的和

    :param a: 第一个数字
    :param b: 第二个数字
    :return: 两个数字相加后的结果
    """
    time.sleep(0.5)  # 模拟工具本身的耗时，便于观察中间件前后日志
    result = a + b
    print(f"[工具执行] {a} + {b} = {result}")
    return result


@wrap_tool_call
def log_tool_call(request, handler):
    """
    工具调用中间件：在目标工具执行前后打印调用信息

    :param request: 本次工具调用请求，包含工具名和工具参数等信息
    :param handler: 真正执行目标工具的调用器
    :return: 目标工具的最终返回结果
    """
    print("--------进入了工具中间件----------")
    print(f"request : {request}")
    print(f"handler : {handler}")

    # 前置增强：这里可以记录日志、做权限校验，或者改写 request 中的工具参数
    # handler(request) 是真正执行目标工具的关键步骤；不调用它，工具就会被中间件拦住
    result = handler(request)

    # 后置增强：这里可以包装返回结果、做敏感信息过滤，或者记录工具耗时
    print("--------退出工具中间件----------")
    print(f"result:{result}")

    return result


llm = init_chat_model(model=os.getenv("LLM_QWEN_MAX"), model_provider="openai")


# 自定义中间件和框架内置中间件一样，都需要放到 middleware 列表里才会生效
deep_agent = create_deep_agent(
    model=llm,
    tools=[add_numbers],
    checkpointer=InMemorySaver(),
    middleware=[log_tool_call],
    system_prompt="你是一个计算器助手，使用add_numbers工具完成加法计算，回答仅返回计算结果。",
)


if __name__ == "__main__":
    # 使用固定 thread_id，便于把一次测试调用归到同一个会话线程
    thread_config = {"configurable": {"thread_id": "middleware_test_1"}}

    result = deep_agent.invoke(
        {"messages": [{"role": "user", "content": "帮我计算 100 + 200 的结果"}]},
        config=thread_config,
    )

    print("\n=== 最终回复 ===")
    print(result["messages"][-1].content)
