"""
DeepAgents 人机协作：编辑工具参数 + 恢复执行

演示人工在审批阶段修改模型准备调用的工具参数，然后继续恢复执行
本示例和审批示例的主流程相同，区别是 delete_database 使用 edit 修改参数后放行
用户问题 -> 命中高危工具后中断 -> 人工编辑工具参数 -> Command(resume=...) 恢复执行
"""

import os

from deepagents import create_deep_agent
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv(find_dotenv())

llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai",
)


@tool
def delete_database(table_name: str):
    """
    删除数据库表

    这是高风险动作，所以会在 interrupt_on 中配置为执行前中断
    示例只返回模拟结果，不会真的删除数据库表
    """
    print(f"调用 delete_database 工具，准备删除 {table_name} 表")
    return f"已删除表：{table_name}"


@tool
def delete_file(file_name: str):
    """
    删除文件

    这是高风险动作，所以会在 interrupt_on 中配置为执行前中断
    示例只返回模拟结果，不会真的删除本地文件
    """
    print(f"调用 delete_file 工具，准备删除 {file_name} 文件")
    return f"已删除文件：{file_name}"


@tool
def select_database(table_name: str):
    """
    查询数据库表数据

    查询动作属于低风险读操作，本示例不会对它做人工审批
    """
    print(f"调用 select_database 工具，查询 {table_name} 表数据")
    return f"已查询表 {table_name} 的数据"


# 人机协作必须配置 checkpointer
# 第一次执行命中中断点时，Agent 会把暂停位置保存到检查点中
checkpointer = InMemorySaver()

# 恢复执行时必须使用相同 thread_id
# 可以把 thread_id 理解成一次任务的唯一执行线程 ID
thread_config = {
    "configurable": {
        "thread_id": "hitl-edit-demo",
    }
}


main_agent = create_deep_agent(
    model=llm,
    tools=[delete_database, delete_file, select_database],
    checkpointer=checkpointer,
    system_prompt="""
    你是一个负责执行数据库和文件操作的智能助手
    请根据用户需求调用合适的工具，并使用中文回复执行结果
    """,
    # interrupt_on 用工具名配置哪些动作需要人工审批
    # True 表示使用默认审批选项：approve、edit、reject
    # False 表示该工具不需要中断，可以直接执行
    interrupt_on={
        "delete_database": True,
        "delete_file": True,
        "select_database": False,
    },
)


# 第一次 invoke 会正常规划任务
# delete_database 和 delete_file 命中 interrupt_on，本轮会暂停并返回待审核动作
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


# 命中人机协作中断时，结果中会出现 __interrupt__
# action_requests 是模型准备执行、但还没真正执行的高风险工具调用列表
interrupts = result_1.get("__interrupt__", [])

if interrupts:
    action_requests = interrupts[0].value["action_requests"]
    print(
        "本次需要审核的工具数量："
        f"{len(action_requests)}，具体拦截的工具："
        f"{[action['name'] for action in action_requests]}"
    )

    decisions = []
    for action in action_requests:
        action_name = action["name"]

        # edit 表示人工不拒绝这个工具调用，但要先修正工具名或参数
        # edited_action 中的 name 是最终要执行的工具名，args 是修正后的工具参数
        if action_name == "delete_database":
            decisions.append(
                {
                    "type": "edit",
                    "edited_action": {
                        "name": action_name,
                        "args": {
                            "table_name": "archived_user",
                        },
                    },
                }
            )
        elif action_name == "delete_file":
            decisions.append({"type": "approve"})

    # 第二次 invoke 不再传用户原始问题，而是传 Command(resume=...)
    # config 必须继续使用第一次相同的 thread_id，Agent 才能找到之前暂停的位置
    result_2 = main_agent.invoke(
        Command(
            resume={
                "decisions": decisions,
            }
        ),
        config=thread_config,
    )

    print(f"最终结果：{result_2['messages'][-1].content}")
