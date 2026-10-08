"""
主智能体组装与反思循环执行模块

负责把模型、主提示词、文件类工具和三个专家子智能体组装成 DeepAgent，
并在其外层实现反思循环：每轮执行后由评估器判断信息是否充分，不充分则
识别缺口并驱动下一轮补充检索，直到充分、轮数耗尽或 token 预算用尽。

设计决策见 docs/design.md ADR-001（外层循环不改框架）与 ADR-003（事件语义分离）。
"""

import asyncio
import shutil
from pathlib import Path

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.llm import model
from app.agent.prompts import main_agent_content
from app.agent.reflection import (
    build_supplement_message,
    evaluate_sufficiency,
    get_reflection_config,
)
from app.agent.subagents.database_query_agent import database_query_agent
from app.agent.subagents.knowledge_base_agent import knowledge_base_agent
from app.agent.subagents.network_search_agent import network_search_agent
from app.api.context import (
    reset_session_context,
    set_session_context,
    set_thread_context,
)
from app.api.monitor import monitor

# 文件类工具由主智能体直接掌握，负责读取上传附件和生成最终交付文档
from app.tools.markdown_tools import generate_markdown
from app.tools.pdf_tools import convert_md_to_pdf
from app.tools.upload_file_read_tool import read_file_content

# 主智能体是调度中心：
# 1. tools 只放最终交付相关的文件工具
# 2. subagents 放网络、数据库、自建知识库（RAG）三类信息获取助手
# 3. checkpointer 通过 thread_id 保存同一会话中的执行上下文
main_agent = create_deep_agent(
    model=model,
    system_prompt=main_agent_content["system_prompt"],
    tools=[generate_markdown, convert_md_to_pdf, read_file_content],
    checkpointer=InMemorySaver(),
    subagents=[database_query_agent, network_search_agent, knowledge_base_agent],
)

# 当前文件位于 app/agent/main_agent.py，parents[1] 即 app 目录
project_root_path = Path(__file__).parents[1].resolve()


async def _run_agent_round(message: str, config: dict) -> tuple[str, int]:
    """
    执行一轮主智能体（单次图执行）

    复用同一 thread_id 的 checkpointer，因此后续轮次可以直接看到前面所有轮次的
    消息上下文，无需手动拼接历史。

    :param message: 本轮注入主智能体的消息（首轮为任务+工作环境指令，后续轮为补搜指令）
    :param config: LangGraph 运行配置，含 thread_id
    :return: (本轮最终回答文本, 本轮累计 token 数)
    """
    round_result = ""
    round_tokens = 0
    has_usage = False

    # astream 会持续产出模型节点、工具节点和子智能体节点的状态片段
    async for chunk in main_agent.astream(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    ):
        # chunk 形如 {"model": {"messages": [...]}}，这里主要关心模型最新消息
        for node_name, state in chunk.items():
            if not state or "messages" not in state:
                continue
            messages = state["messages"]
            if not (messages and isinstance(messages, list)):
                continue

            last_msg = messages[-1]

            # token 消耗取自模型消息的 usage_metadata，作为反思循环预算的计量依据
            usage = getattr(last_msg, "usage_metadata", None)
            if usage and isinstance(usage, dict):
                has_usage = True
                round_tokens += int(usage.get("total_tokens") or 0)

            if node_name != "model":
                continue

            if last_msg.tool_calls:
                # DeepAgents 调用子智能体时，本质上会产生名为 task 的工具调用
                for tool_call in last_msg.tool_calls:
                    if tool_call["name"] == "task":
                        # 子智能体调用单独上报，前端可以展示“正在调用哪个专家助手”
                        monitor.report_assistant(
                            tool_call["args"]["subagent_type"],
                            {"description": tool_call["args"]["description"]},
                        )
            elif last_msg.content:
                # 模型本轮不再调用工具时，这段文本就是本轮的最终回答。
                # 中间过程性文本会被后续更完整的回答覆盖，只保留最后一次。
                round_result = (
                    last_msg.content
                    if isinstance(last_msg.content, str)
                    else str(last_msg.content)
                )

    # 模型接口未返回 usage_metadata 时，按字符数粗估 token，保证预算约束仍然生效
    if not has_usage and round_result:
        round_tokens = max(1, len(round_result) // 2)

    return round_result, round_tokens


async def run_deep_agent(task_query, session_id):
    """
    执行主智能体反思循环（API 层统一入口）

    流程：准备会话目录 → 首轮执行 → 评估充分性 → 不充分则补搜迭代 → 输出最终结果。
    每轮的关键事件都通过 monitor 推送到前端，形成「搜 → 反思 → 补搜」的可视化链路。

    :param task_query: 前端提交的原始任务问题
    :param session_id: 当前任务 ID，同时用于 thread_id、输出目录和 WebSocket 定向推送
    """
    print(f"[MainAgent] 开始执行会话，session_id={session_id}")

    # 每个会话独立使用 output/session_{session_id}，避免不同用户的产物互相覆盖
    session_dir = project_root_path / "output" / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # 前端和工具使用绝对路径；提示词里只给模型相对路径，降低模型误用系统绝对路径的概率
    session_dir_str = str(session_dir).replace("\\", "/")
    relative_session_dir_str = str(session_dir.relative_to(project_root_path)).replace(
        "\\", "/"
    )

    # 上传文件先落在 updated/session_{session_id}，执行前复制到本次 output 工作目录
    # 这样读文件工具和生成文件工具都只需要围绕同一个 session_dir 工作
    updated_dir_path = project_root_path / "updated" / f"session_{session_id}"
    updated_info_prompt = ""
    if updated_dir_path.exists():
        files = [f.name for f in updated_dir_path.iterdir() if f.is_file()]
        if files:
            for filename in files:
                # copy2 会保留上传文件的修改时间、权限等元数据，便于后续排查文件来源
                shutil.copy2(updated_dir_path / filename, session_dir / filename)

            # 把上传文件列表注入用户消息，提醒模型先调用 read_file_content 获取附件内容
            updated_info_prompt = (
                "\n    [已上传文件] 已加载到工作目录:\n"
                + "\n".join([f"    - {f}" for f in files])
                + "\n    请优先使用工具（read_file_content）读取并参考这些文件。"
            )

    # ContextVar 让深层工具无需显式传参，也能拿到当前会话目录和 WebSocket thread_id
    session_dir_token = set_session_context(session_dir_str)
    session_id_token = set_thread_context(session_id)

    # 新任务开始：清空该 thread 的历史事件缓冲，避免回放上一轮任务的轨迹
    monitor.begin_task()

    # 前端拿到工作目录后，可以展示本次任务生成的 Markdown/PDF 等产物
    # 同时带上原始问题：页面刷新或跨标签页收到事件时，前端能还原用户提问
    monitor.report_session_dir(session_dir_str, task_query)

    # checkpointer 依赖 thread_id 区分会话记忆；同一 session_id 的多轮反思共用一条上下文
    config = {"configurable": {"thread_id": session_id}}

    # 工作环境指令是运行时动态补充的，约束模型只在当前会话目录读写文件
    path_instruction = f"""
    【工作环境指令】
    工作目录: {relative_session_dir_str}
    {updated_info_prompt}

    规则：
    1. 新生成文件必须保存到工作目录：'{relative_session_dir_str}/filename'
    2. 读取已上传的文件时，请直接将文件名（例如：'开篇.txt'）作为 filename 参数传入（read_file_content）读取工具，不要带上任何目录前缀。
    3. 使用相对路径，禁止使用绝对路径
    4. 若存在上传文件，请先分析内容
    """

    reflection_config = get_reflection_config()
    max_rounds = reflection_config["max_rounds"]
    token_budget = reflection_config["token_budget"]

    current_message = task_query + path_instruction
    final_result = ""
    total_tokens = 0

    try:
        for round_num in range(1, max_rounds + 1):
            print(f"[MainAgent] 第 {round_num}/{max_rounds} 轮执行开始")
            round_result, round_tokens = await _run_agent_round(current_message, config)
            total_tokens += round_tokens

            if round_result:
                final_result = round_result
            else:
                # 本轮没有产出文本：保留上一轮结果，避免反思循环把已有结论清空
                monitor.report_reflection_stopped(
                    "本轮未产出回答",
                    f"第 {round_num} 轮为空，沿用上一轮结果",
                )

            # 最后一轮不再触发评估，直接终止循环
            if round_num >= max_rounds:
                monitor.report_round_result(
                    round_num, final_result, stop_reason="达到最大轮数"
                )
                break

            # 反思评估：判断当前信息是否足以回答用户问题
            evaluation = evaluate_sufficiency(task_query, final_result)
            monitor.report_reflection_evaluation(round_num, evaluation)

            if evaluation.sufficient:
                monitor.report_round_result(round_num, final_result, stop_reason="信息充分")
                break

            # 预算兜底：评估判定不充分但已超 token 预算时，停止迭代，输出当前最好结果
            if total_tokens >= token_budget:
                monitor.report_reflection_stopped(
                    "token 预算耗尽",
                    f"已消耗 {total_tokens} tokens，预算 {token_budget}",
                )
                monitor.report_round_result(
                    round_num, final_result, stop_reason="预算耗尽"
                )
                break

            # 缺口为空说明无法生成可执行的补充查询，继续迭代只会空转
            if not evaluation.missing_dimensions and not evaluation.follow_up_queries:
                monitor.report_reflection_stopped(
                    "无明确信息缺口", "评估未给出可执行的补充方向"
                )
                monitor.report_round_result(
                    round_num, final_result, stop_reason="无补充方向"
                )
                break

            # 生成下一轮补搜指令（复用同一 thread_id，模型自带全部历史上下文）
            monitor.report_reflection_supplement(
                round_num + 1, evaluation.follow_up_queries
            )
            current_message = build_supplement_message(evaluation, round_num + 1)
            print(
                f"[MainAgent] 第 {round_num} 轮反思发现缺口，进入第 {round_num + 1} 轮补搜"
            )

        print(f"[MainAgent] 反思循环结束，累计消耗 token 约 {total_tokens}")
        monitor.report_task_result(final_result)

    except asyncio.CancelledError:
        monitor.report_task_cancelled()
        raise
    except Exception as e:
        # 异常兜底：把错误类型和原因一起告诉前端，避免只看到一串栈信息
        error_type = type(e).__name__
        detail = str(e) or "未提供更多信息"
        print(f"[MainAgent] 任务执行中断（{error_type}）：{detail}")

        monitor.report_error(
            reason=error_type,
            detail=detail,
            has_partial_result=bool(final_result),
        )

        # 已经产出过内容时按「部分结果」交付：网络抖动不该让整轮研搜成果作废
        if final_result:
            monitor.report_task_result(final_result, partial=True)
    finally:
        # 任务结束后恢复 ContextVar，避免后续请求复用到本次会话目录或 thread_id
        reset_session_context(session_dir_token, session_id_token)


if __name__ == "__main__":
    asyncio.run(
        run_deep_agent("从网络查询机器人信息，并生成Markdown文件", "test_session_001")
    )
