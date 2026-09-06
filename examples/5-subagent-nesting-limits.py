"""
DeepAgents 子智能体：字典配置的嵌套边界

演示普通 dict 子智能体在多层级委派场景下的配置边界
本示例故意写出 CEO -> CTO -> Coder 的层级配置，用来观察 subagents 字段是否生效
主智能体 -> CTO 子智能体 -> Coder 子智能体

也就是说，直接在 cto_config 里硬塞 "subagents": [coder_config]，并不是推荐写法，
底层也不一定会把 Coder 识别成 CTO 的下级子智能体。

如果后续确实需要复杂多层级 Agent，建议使用 CompiledSubAgent，
或者把一段完整的 LangGraph / LangChain 流程封装后再挂到 DeepAgents。
"""

import os

from deepagents import create_deep_agent
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(find_dotenv())

llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    model_provider="openai",
)

# ---------------------------------------------------------
# 故意构造“多级委派”的教学配置
# ---------------------------------------------------------

# 1. 底层 Coder：理想情况下，它应该只负责具体代码实现
coder_config = {
    "name": "Coder",
    "description": "高级 Python 工程师，负责接收具体的编码任务并实现代码。",
    "system_prompt": """
    你是一名高级 Python 工程师。
    你的职责是接收具体的编码任务，并给出对应的代码实现。
    """,
    "tools": [],  # Coder 拥有默认的文件操作工具
}

# 2. 中间层 CTO：理想情况下，它负责拆解任务，再委派给 Coder
cto_config = {
    "name": "CTO",
    "description": "技术总监，负责将战略需求转化为技术任务，并分配给工程师。",
    # system_prompt 可以约束 CTO 的角色，但不能让 dict 自动支持嵌套子智能体
    "system_prompt": """
    你是技术总监。
    你不直接编写代码。
    你的职责包括：
    1. 分析 CEO 的需求。
    2. 设计技术方案。
    3. 调用 Coder 子代理完成具体的代码编写工作。
    """,
    "tools": [],
    # 关键边界：普通 dict 子智能体的标准字段里没有 subagents。
    # 这一行是“反例式演示”，用于观察底层是否忽略该字段，而不是推荐用法。
    "subagents": [coder_config],
}

# 3. 顶层 CEO：真正注册到 DeepAgents 的主智能体
ceo_agent = create_deep_agent(
    model=llm,
    name="CEO",
    # 顶层只注册 CTO。即便 CTO 配置里写了 subagents，也不代表 Coder 会被自动注册成功。
    system_prompt="""
    你是 CEO，负责公司战略决策。
    你不直接编写代码或操作文件。
    请将所有技术相关的开发任务委派给 CTO 处理。
    你的工作是验收 CTO 提交的结果。
    """,
    subagents=[cto_config],
)

# 运行 CEO 代理，观察最终流式输出里是否真的出现 CTO -> Coder 的二级委派
print(">>> 开始执行任务链...")
stream = ceo_agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "帮我开发一个贪吃蛇游戏，要求用 Python 实现，直接提供代码字符串即可。",
            }
        ]
    }
)

# 为了教学观察，这里直接打印原始 chunk，而不是解析 model/tools 节点。
#
# 结合实际输出，你重点看下面几个位置：
#
# 1. {'PatchToolCallsMiddleware.before_agent': None}
#    这是 DeepAgents / LangGraph 中间件产出的中间状态，不是智能体回复，可以先忽略。
#
# 2. 第一次 {'model': {'messages': [AIMessage(..., name='CEO', tool_calls=[...])]}}
#    关键看 tool_calls：
#    tool_call["name"] == "task"
#    tool_call["args"]["subagent_type"] == "CTO"
#    这说明顶层 CEO 成功把任务分派给了 CTO。
#
# 3. {'tools': {'messages': [ToolMessage(..., name='task', content='...')]}}
#    这里的 ToolMessage 是 task 工具的返回结果。
#    因为 task 对应子智能体调用，所以 content 中就是 CTO 子智能体返回给 CEO 的结果。
#
# 4. 第二次 {'model': {'messages': [AIMessage(..., name='CEO', tool_calls=[])]}}
#    tool_calls 为空，并且 content 有内容，说明 CEO 已经基于 CTO 的返回整理出了最终答案。
#
# 5. 最关键的观察结论：
#    输出里只出现了 subagent_type == "CTO"，没有出现 subagent_type == "Coder"。
#    这说明虽然 cto_config 里写了 "subagents": [coder_config]，
#    但普通 dict 子智能体并没有继续识别并触发 CTO -> Coder 的二级委派。
#
print("\n>>> 最终结果：")
for chunk in stream:
    print(chunk)
