"""
DeepAgents 子智能体：异步流式执行 + 并发任务

演示如何把同步 stream 调用改成异步 astream 调用
本示例沿用天气、数学、翻译三个子智能体，并使用 asyncio.gather 并发执行多个任务
用户问题 -> 主智能体决策 -> task 分派子智能体 -> 异步读取执行过程

在 Web 服务或批量任务中，模型调用、工具调用、子智能体执行都可能等待 I/O。
异步写法可以让多个请求一起推进，而不是一个任务结束后再启动下一个任务。
"""

import asyncio
import os

from deepagents import create_deep_agent
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(find_dotenv())

# 使用 OpenAI 兼容接口初始化模型；温度调低，让路由和回答更稳定
llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    temperature=0.1,
    model_provider="openai",
)

# 子智能体配置和同步路由示例保持一致。
# description 主要服务于主智能体路由；system_prompt 则约束子智能体自己的行为。
weather_agent = {
    "name": "weather_helper",
    "description": "用于查询天气信息。当用户询问天气时，请调用此助手。",
    "system_prompt": """
    你是一个天气查询助手。
    无论用户查询哪个城市，请统一回复：今天天气晴朗，温度25度。
    """,
    "tools": [],
}

math_agent = {
    "name": "math_helper",
    "description": "用于处理数学计算问题。当用户询问加减乘除、数字计算、算数题时，请调用此助手。",
    "system_prompt": """
    你是一个严谨的数学助手。
    请帮助用户完成数学计算，并给出清晰、准确的答案。
    """,
    "tools": [],
}

translate_agent = {
    "name": "translate_helper",
    "description": "用于处理中英互译任务。当用户需要中文和英文之间的翻译时，请调用此助手。",
    "system_prompt": """
    你是一个中英翻译助手。
    如果输入是中文，请翻译成英文；如果输入是英文，请翻译成中文。
    """,
    "tools": [],
}


# 主智能体自己不挂普通工具，重点负责把任务路由到子智能体
main_agent = create_deep_agent(
    model=llm,
    tools=[],
    subagents=[weather_agent, math_agent, translate_agent],
    system_prompt="""
    你是一个负责统筹任务的主智能体。
    请根据用户需求选择合适的子智能体完成任务。
    你不直接执行天气查询、数学计算或翻译任务，而是通过子智能体完成。
    """,
)


async def test_stream(query):
    """
    异步流式执行一次用户问题，并打印主智能体的调度过程。

    和同步版相比，核心变化只有两处：
    - main_agent.stream(...) 变为 main_agent.astream(...)；
    - for chunk in stream 变为 async for chunk in stream。
    """
    # astream() 返回异步流对象，需要在 async 函数中用 async for 消费
    stream = main_agent.astream({"messages": [{"role": "user", "content": query}]})

    async for chunk in stream:
        # chunk 是按节点名组织的字典，例如 {"model": {"messages": [...]}}
        for node_name, state in chunk.items():
            # DeepAgents 内部可能产出空状态或非消息状态，这里只解析消息类状态
            if state is None or "messages" not in state:
                continue

            messages = state["messages"]
            if messages and isinstance(messages, list):
                # 每个节点本次产出的最后一条消息，通常就是最值得观察的信息
                last_msg = messages[-1]

                if node_name == "model":
                    # tool_calls 表示模型决定下一步调用工具或通过 task 分派子智能体
                    if last_msg.tool_calls:
                        for tool_call in last_msg.tool_calls:
                            if tool_call["name"] == "task":
                                # task 是 DeepAgents 内置的子智能体分派入口
                                print(
                                    f"【model】决定调用子智能体{tool_call['args']['subagent_type']}"
                                )
                            else:
                                print(
                                    f"【model】决定调用子工具{tool_call['name']},传入的参数为：{tool_call['args']}"
                                )
                    elif last_msg.content:
                        # 没有 tool_calls 且 content 非空，通常就是主智能体整理后的最终回复
                        print(f"【model】返回最终结果：{last_msg.content}")
                elif node_name == "tools":
                    # tools 节点返回普通工具结果；task 子智能体执行完后也会以工具消息形式返回
                    name = last_msg.name
                    content = last_msg.content
                    print(
                        f"【agent】调用了具体的工具{name},返回结果为：{content[:100] + '...'}"
                    )


# test_stream("北京今天的天气怎么样？")
# # test_stream("998+889 运算后等于多少？")
# #test_stream("请将'你是最棒的'翻译成英文，并且查询今天北京的天气信息。")
# test_stream("请将'你是最棒的'翻译成英文。")


if __name__ == "__main__":
    # asyncio.run(test_stream("北京今天的天气怎么样？"))
    async def batch_run():
        # 这里得到的是协程对象；传给 gather 后，会由事件循环并发调度
        task1 = test_stream("北京今天的天气怎么样？")
        task2 = test_stream("请将'你是最棒的'翻译成英文。")

        # 打印类型只是为了让初学者看到：调用 async 函数不会立刻执行，而是先返回 coroutine
        print(type(task1))
        print(type(task2))

        # gather 会等待两个协程都完成；哪个请求先拿到结果，就会先输出自己的流式片段
        await asyncio.gather(task1, task2)

    asyncio.run(batch_run())

"""
同步版参考 examples/3-dict-subagents-routing.py。
本文件的重点是把单任务 stream 观察，扩展为多任务异步并发观察。
"""
