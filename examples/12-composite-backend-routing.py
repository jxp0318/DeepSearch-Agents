"""
DeepAgents Backend：CompositeBackend 混合存储路由

演示如何根据文件路径前缀把不同文件写入不同 Backend
本示例使用 FilesystemBackend 保存普通文件，使用 StoreBackend 保存 /store/ 路径下的重要记忆
普通路径 -> 本地 agent_workspace 目录
/store/ 路径 -> InMemoryStore
"""

import os
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore

load_dotenv(find_dotenv())


# Store 用来保存 /store/ 路径下的重要记忆
# 这里使用 InMemoryStore 便于教学观察，生产环境可以替换成持久化 Store
store = InMemoryStore()


llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai",
)


def create_composite_backend(runtime):
    """
    创建混合 Backend

    create_deep_agent 会把 runtime 传入这个工厂函数
    StoreBackend 需要 runtime 才能连接到 Agent 配置中的 store
    """
    workspace_dir = Path("./agent_workspace").resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # 默认后端：普通文件写入本地工作目录
    fs_backend = FilesystemBackend(
        root_dir=workspace_dir,
        virtual_mode=True,
    )

    # Store 后端：重要记忆写入 Key-Value Store
    store_backend = StoreBackend(runtime)

    # 路由规则：
    # 普通路径，例如 local.txt，走 default 的 FilesystemBackend
    # /store/ 开头的路径，例如 /store/memory.txt，走 StoreBackend
    return CompositeBackend(
        default=fs_backend,
        routes={
            "/store/": store_backend,
        },
    )


agent = create_deep_agent(
    model=llm,
    store=store,
    backend=create_composite_backend,
    tools=[],
    system_prompt="""
    你是一个智能助手
    普通文件请直接写入文件名，例如 local.txt
    重要记忆请写入 /store/ 目录，例如 /store/memory.txt
    """,
)


print("\n=== 测试混合存储 ===")
config = {"configurable": {"thread_id": "thread-composite"}}


# 这个用户指令会同时触发两种存储路径
# local.txt 会写到本地 agent_workspace
# /store/memory.txt 会通过 CompositeBackend 路由到 StoreBackend
user_input = (
    "1 创建本地文件 local.txt，内容为 本地文件\n"
    "2 创建记忆文件 /store/memory.txt，内容为 重要记忆"
)
print(f"用户指令：{user_input}")

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": user_input,
            }
        ]
    },
    config=config,
)

print(f"Agent 回复：{result['messages'][-1].content}")


print("\n=== 验证 Store 存储 ===")
# CompositeBackend 命中 /store/ 路由后，会把内容交给 StoreBackend
# 进入 Store 时，路由前缀可能会被剥离，因此可以直接打印 item 观察实际 key 和 value
items = store.search(("filesystem",))
for item in items:
    print(item)
