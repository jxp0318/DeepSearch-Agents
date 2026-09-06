"""
DeepAgents Skill：通过 FilesystemBackend 加载外置技能

演示如何把本地 skills 目录注册给 DeepAgent
本示例使用 emoji-translator 技能，把用户文本转换成表情符号
用户请求 -> Agent 读取 Skill 元数据 -> 按需加载 SKILL.md -> 根据技能规则生成回复
"""

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(find_dotenv())


llm = init_chat_model(model="deepseek-v4-flash", model_provider="openai")


# Skill 文件需要通过 Backend 暴露给 Agent
# 这里把当前示例文件所在目录作为根目录，后续 skills 路径都相对于它查找
current_dir = Path(__file__).parent.resolve()
file_backend = FilesystemBackend(
    root_dir=current_dir,
    virtual_mode=True,
)


# skills=["skills"] 表示加载 current_dir/skills 目录下的技能包
# 每个技能包至少需要一个 SKILL.md，模型会先读取 name 和 description 判断是否触发
main_agent = create_deep_agent(
    model=llm,
    backend=file_backend,
    skills=[
        "skills",
    ],
    system_prompt="你是一个智能助手，可以使用 SKILL 技能",
)


# 这句话明确要求使用“表情翻译技能”，便于触发 emoji-translator/SKILL.md
query = "我早上起床晚了，赶公交车差点摔倒，还好最后到了公司。请你只用表情翻译技能。"
result = main_agent.invoke({"messages": [{"role": "user", "content": query}]})

print(f"最终输出结果：{result['messages'][-1].content}")

"""
注意事项：
1. 先配置 backend，再通过 skills 指定相对于 backend.root_dir 的技能目录。
2. 技能文件夹名建议和 SKILL.md 中的 name 保持一致，例如 emoji-translator。
3. Skill 采用渐进式加载：先看 YAML 元数据，再按需读取完整 SKILL.md。
4. description 要写清楚触发场景，否则模型可能不知道什么时候该加载该技能。
5. Skill 不是越多越好，功能重复或描述接近时会降低触发稳定性。
"""
