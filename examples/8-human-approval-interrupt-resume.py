"""
DeepAgents 人机协作：中断审批 + 恢复执行

演示如何在高风险工具真正执行前暂停，让人工决定 approve 或 reject
本示例使用 interrupt_on 拦截删除数据库表和删除文件这两个高危动作
用户问题 -> 模型规划工具调用 -> 命中高危工具后中断 -> 人工审批 -> Command(resume=...) 恢复执行
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
        "thread_id": "hitl-approval-demo",
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
# select_database 是低风险工具，可以直接执行
# delete_database 和 delete_file 命中 interrupt_on，本轮会暂停，不会真正执行这两个工具
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
# __interrupt__ 是一个列表，里面保存 Interrupt 对象
# 每个 Interrupt 的 value 是一个字典，核心结构可以理解为：
# {
#     "action_requests": [
#         {"name": "delete_database", "args": {"table_name": "user"}},
#         {"name": "delete_file", "args": {"file_name": "zhaoweifeng.txt"}},
#     ],
#     "review_configs": [
#         {"action_name": "delete_database", "allowed_decisions": ["approve", "edit", "reject"]},
#         {"action_name": "delete_file", "allowed_decisions": ["approve", "edit", "reject"]},
#     ],
# }
# action_requests 表示模型准备执行、但还没真正执行的高风险工具调用
# review_configs 表示每个被拦截工具允许的人工决策类型
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

        # decisions 的顺序要和 action_requests 的顺序保持一致
        # reject 表示拒绝执行该工具；approve 表示允许继续执行该工具
        if action_name == "delete_database":
            decisions.append({"type": "reject"})
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
