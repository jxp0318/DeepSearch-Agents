"""
DeepAgents 子智能体：字典配置 + 同步流式路由

演示如何用普通 dict 定义轻量级子智能体，并注册到 DeepAgent 主智能体中
本示例使用 stream 逐步读取主智能体的路由过程，观察 task 工具如何分派子任务
用户问题 -> 主智能体决策 -> task 分派子智能体 -> 主智能体整理最终回答

注意：天气、计算、翻译在真实项目里也可以做成普通工具。
这里故意写成子智能体，是为了讲清楚 DeepAgents 的子智能体调度机制。
"""

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

# 字典式子智能体的核心字段：
# name 是子智能体唯一标识，流式输出里的 subagent_type 会对应它；
# description 主要给主智能体看，用来判断什么时候应该调用该助手；
# system_prompt 是子智能体自己的角色和行为约束；
# tools 是该子智能体可用的工具列表，不填 model 时通常继承主智能体模型。
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


# 主智能体自己不挂普通工具，重点负责根据用户问题分派给合适的子智能体
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


def test_stream(query):
    """
    同步流式执行一次用户问题，并打印主智能体的调度过程。

    stream() 会持续产出图节点状态，常见节点包括：
    - model：模型做出下一步决策，或生成最终回复；
    - tools：工具执行完成，子智能体的 task 结果也会从这里返回。
    """
    stream = main_agent.stream({"messages": [{"role": "user", "content": query}]})

    for chunk in stream:
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
                    # model 节点的 tool_calls 表示模型决定下一步调用工具或子智能体
                    if last_msg.tool_calls:
                        for tool_call in last_msg.tool_calls:
                            if tool_call["name"] == "task":
                                # DeepAgents 用内置 task 工具表示“分派给某个子智能体”
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


test_stream("北京今天的天气怎么样？")
# test_stream("998+889 运算后等于多少？")
# test_stream("请将'你是最棒的'翻译成英文，并且查询今天北京的天气信息。")
test_stream("请将'你是最棒的'翻译成英文。")
