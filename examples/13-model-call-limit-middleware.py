"""
DeepAgents Middleware：模型调用限制

演示如何把 LangChain 提供的 ModelCallLimitMiddleware 配置到 DeepAgent
本示例通过很小的 run_limit 和 thread_limit，观察模型调用次数超限后的行为
用户请求 -> Agent 调用模型规划动作 -> 中间件检查调用次数 -> 超限后按 exit_behavior 处理
"""

import os

from deepagents import create_deep_agent
from dotenv import find_dotenv, load_dotenv
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(find_dotenv())


llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai",
)


@tool
def delete_database(table_name: str):
    """
    删除指定数据库表

    :param table_name: 要删除的表名
    :return: 操作的返回结果
    """
    print(f"调用 delete_database 工具，删除数据表：{table_name}")
    return f"已删除数据表：{table_name}"


@tool
def delete_file(file_name: str):
    """
    删除指定文件

    :param file_name: 要删除的文件名
    :return: 操作的返回结果
    """
    print(f"调用 delete_file 工具，删除文件：{file_name}")
    return f"已删除文件：{file_name}"


@tool
def select_database(table_name: str):
    """
    查询指定数据库表

    :param table_name: 要查询的表名
    :return: 查询结果
    """
    print(f"调用 select_database 工具，查询数据表：{table_name}")
    return f"已查询数据表：{table_name}"


# checkpointer 用来记录同一条线程的执行状态
# thread_id 也是 thread_limit 判断“同一个会话线程”的依据
checkpointer = InMemorySaver()
thread_config = {"configurable": {"thread_id": "erdaye"}}


# middleware 可以配置到主智能体，也可以配置到子智能体
# 这里重点演示主智能体的模型调用次数限制
main_agent = create_deep_agent(
    model=llm,
    tools=[delete_database, delete_file, select_database],
    checkpointer=checkpointer,
    system_prompt="回答使用中文，调用对应的工具实现对应的功能",
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=1,  # 同一个 thread_id 下累计最多调用 1 次模型
            run_limit=1,  # 当前这次 invoke 内最多调用 1 次模型
            exit_behavior="error",  # 超限后抛出异常，便于后端统一捕获处理
        )
    ],
    # 本章重点是 middleware 调用限制，因此这里关闭人工审批拦截
    # 如果要演示危险动作审批，可以把高风险工具配置为 True 或 allowed_decisions
    interrupt_on={
        "delete_database": False,
        "delete_file": False,
        "select_database": False,
    },
)

# 这个请求通常需要多次“模型规划 -> 工具调用 -> 模型整理结果”
# run_limit=1 会让示例更容易触发模型调用上限
result_1 = main_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "先查询 product 表的数据，再删除 user 表，最后删除 zhaoweifeng.txt 文件",
            }
        ]
    },
    config=thread_config,
)

print(f"最终结果：{result_1['messages'][-1].content}")
