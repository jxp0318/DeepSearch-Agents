"""
DeepAgents Backend：FilesystemBackend 本地文件存储

演示如何把 Agent 的文件系统操作映射到本地目录
本示例使用 FilesystemBackend，让 Agent 生成的文件落到 agent_workspace
用户问题 -> Agent 判断是否需要写文件 -> 文件工具写入虚拟文件系统 -> FilesystemBackend 保存到本地目录
"""

import os
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(find_dotenv())


# 准备本地工作目录
# Agent 写入的文件最终会落到这个目录中，便于本地调试和查看结果
workspace_dir = Path("./agent_workspace").resolve()
workspace_dir.mkdir(parents=True, exist_ok=True)


# FilesystemBackend 把 DeepAgents 的文件系统操作映射到本地磁盘
# root_dir 是真实存储目录
# virtual_mode=True 表示开启虚拟沙箱，限制 Agent 只能在 root_dir 范围内读写
file_backend = FilesystemBackend(
    root_dir=workspace_dir,
    virtual_mode=True,
)


llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai",
)


# 这里把 backend 指定为 file_backend
# Agent 内置文件工具产生的文件读写，会交给 FilesystemBackend 处理
main_agent = create_deep_agent(
    model=llm,
    tools=[],
    backend=file_backend,
    system_prompt="""
    你是一个智能助手，可以使用文件工具进行文件读写
    只有在用户明确要求创建或写入文件时，才可以创建文件
    """,
)


# 第一次请求只是查询介绍，没有明确要求写文件
# 用来观察 Agent 是否会遵守提示词，不主动创建文件
print("第 1 次调用：用户没有明确要求创建文件")
result_1 = main_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "帮我查询下 Python 语言的介绍",
            }
        ]
    }
)
print(f"最终结果：{result_1['messages'][-1].content}")


# 第二次请求明确要求把内容写入 java.txt
# 此时 Agent 可以调用文件工具，FilesystemBackend 会把文件保存到 agent_workspace
print("第 2 次调用：用户明确要求创建文件")
result_2 = main_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "帮我查询下 Java 语言的介绍，并写到 java.txt 文件中",
            }
        ]
    }
)
print(f"最终结果：{result_2['messages'][-1].content}")
