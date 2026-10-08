"""
反思循环评估模块

deepsearch 的核心机制：每轮执行结束后，由独立评估器判断当前收集的信息
是否足以回答用户原始问题；不充分时识别缺口维度并生成针对性补充查询，
驱动外层循环进入下一轮补搜。

设计决策见 docs/design.md ADR-001 / ADR-002。
"""

import json
import os
import re
from typing import List

from pydantic import BaseModel, Field

from app.agent.llm import model
from app.agent.prompts import reflection_content


class SufficiencyEvaluation(BaseModel):
    """信息充分性评估结果（评估器结构化输出的目标 schema）"""

    sufficient: bool = Field(
        description="当前信息是否已足以完整回答用户的问题"
    )
    missing_dimensions: List[str] = Field(
        default_factory=list,
        description="仍缺失的信息维度，例如：时间范围、具体数据、反方观点、来源可靠性等",
    )
    follow_up_queries: List[str] = Field(
        default_factory=list,
        description="针对缺口的补充查询建议，每条应是可直接执行的检索问题",
    )
    reasoning: str = Field(
        default="", description="简要说明判断依据（一到两句话）"
    )


# json_mode 用 response_format=json_object，兼容性最好；
# 部分 OpenAI 兼容端点不支持 json_schema，直接调用会返回 400
_evaluator_json_mode = model.with_structured_output(
    SufficiencyEvaluation, method="json_mode"
)

# 记录 json_mode 是否可用：一旦失败就不再重试，后续直接走文本解析，避免每轮白跑一次请求
_json_mode_supported: bool | None = None


def _extract_json_object(text: str) -> dict:
    """
    从模型输出里提取 JSON 对象

    模型可能返回 Markdown 代码块或前后带说明文字，这里做宽容解析。
    """
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("评估器输出中未找到 JSON 对象")

    return json.loads(cleaned[start : end + 1])


def _coerce_bool(value: object) -> bool:
    """兼容模型把布尔值写成字符串的情况"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "是"}
    return bool(value)


def _coerce_str_list(value: object) -> List[str]:
    """兼容模型返回字符串、列表或 null 的情况"""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _parse_evaluation_text(text: str) -> SufficiencyEvaluation:
    """把模型返回的 JSON 文本解析成评估结果对象"""
    data = _extract_json_object(text)
    return SufficiencyEvaluation(
        sufficient=_coerce_bool(data.get("sufficient", False)),
        missing_dimensions=_coerce_str_list(data.get("missing_dimensions")),
        follow_up_queries=_coerce_str_list(data.get("follow_up_queries")),
        reasoning=str(data.get("reasoning") or ""),
    )


def evaluate_sufficiency(task_query: str, round_result: str) -> SufficiencyEvaluation:
    """
    评估当前轮结果是否充分回答了用户问题

    调用链做了两层兜底：json_mode 结构化输出 → 纯文本 + 宽容 JSON 解析 →
    标记为“已充分”并终止迭代。评估器本身出问题不应让整个任务失败。

    :param task_query: 用户原始任务问题
    :param round_result: 本轮主智能体产出的最终回答
    :return: 结构化评估结果
    """
    global _json_mode_supported

    # 评估提示词里含 JSON 花括号，不能用 str.format（会把花括号当占位符），改用显式替换
    evaluation_prompt = (
        reflection_content["evaluator_prompt"]
        .replace("{task_query}", task_query)
        .replace("{round_result}", round_result)
    )

    if _json_mode_supported is not False:
        try:
            evaluation = _evaluator_json_mode.invoke(evaluation_prompt)
            _json_mode_supported = True
            return evaluation
        except Exception as e:
            _json_mode_supported = False
            print(f"[Reflection] json_mode 结构化输出不可用，回退文本解析: {e}")

    try:
        message = model.invoke(evaluation_prompt)
        return _parse_evaluation_text(str(message.content))
    except Exception as e:
        # 评估器失败不应拖垮整个任务：按“已充分”处理，返回上一轮结果
        # 这个降级路径记入事件流，评测阶段可统计评估器的故障率
        print(f"[Reflection] 评估器调用失败，降级为充分: {e}")
        return SufficiencyEvaluation(
            sufficient=True,
            missing_dimensions=[],
            follow_up_queries=[],
            reasoning=f"评估器调用异常，降级终止: {e}",
        )


def build_supplement_message(evaluation: SufficiencyEvaluation, round_num: int) -> str:
    """
    把评估缺口拼装成下一轮补搜指令

    :param evaluation: 不充分的评估结论
    :param round_num: 即将进入的轮次编号
    :return: 注入主智能体的补充搜索消息
    """
    missing = "\n".join(f"{i}. {m}" for i, m in enumerate(evaluation.missing_dimensions, 1))
    queries = "\n".join(f"{i}. {q}" for i, q in enumerate(evaluation.follow_up_queries, 1))

    return f"""
[第 {round_num} 轮反思补搜指令]
对上一轮结果的反思评估发现以下信息缺口：
{missing}

请针对上述缺口进行补充检索（建议的查询方向）：
{queries}

要求：
1. 优先补充缺失维度的信息，不要重复已完成的检索
2. 结合上一轮已有信息和本轮新信息，重新生成更完整的最终回答
3. 如果上一轮已生成输出文件，需要基于补充信息更新该文件
"""


def get_reflection_config() -> dict:
    """
    读取反思循环运行参数（环境变量可覆盖默认值）

    :return: {"max_rounds": int, "token_budget": int}
    """
    return {
        "max_rounds": int(os.getenv("REFLECTION_MAX_ROUNDS", "3")),
        "token_budget": int(os.getenv("REFLECTION_TOKEN_BUDGET", "150000")),
    }
