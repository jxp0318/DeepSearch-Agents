"""
知识库子智能体配置模块

将 app/prompt/prompts.yml 中的 knowledge_base 配置与自建 RAG 检索工具
组装成 DeepAgents 可识别的字典式子智能体。主智能体根据 description
决定是否把内部非结构化文档查询任务分派给它。
"""

from app.agent.prompts import sub_agents_content
from app.tools.kb_tools import list_knowledge_bases, search_knowledge_base

# 知识库子智能体处理内部非结构化文档，与网络搜索助手、数据库查询助手形成互补
# 它遵循「先查知识库列表 -> 再按需检索片段 -> 自行综合回答」的工作顺序：
# 检索（R）由自建 RAG 模块负责，生成（G）由子智能体本身负责（ADR-006）
knowledge_base_agent = {
    "name": sub_agents_content["knowledge_base"]["name"],
    "description": sub_agents_content["knowledge_base"]["description"],
    "system_prompt": sub_agents_content["knowledge_base"]["system_prompt"],
    "tools": [list_knowledge_bases, search_knowledge_base],
}
