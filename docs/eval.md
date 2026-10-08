# 评测方案设计（阶段四）

> 目标：为 deepsearch-agents 建立一套可复现的评测集与判分协议，产出「反思循环前后对比」的量化数字。
> 核心认知：**deep research 系统的输出是研究报告，不是短答案**——不能用「标准答案精确匹配」的思路建评测集，必须用「关键事实命中 + 过程指标」的组合判分。
>
> 备注：知识库类任务运行在自建 RAG 之上（ADR-006），当前 embedding / Qdrant / Elasticsearch 均已接入。可用同一评测集跑出「进程内 BM25 vs ES 关键词 vs 向量 vs RRF 融合」四模式的检索质量对比（`python -m scripts.compare_retrieval`）——这是免费的第二组对比数据。

---

## 1. 任务结构（schema）

每条任务是一个 YAML 对象，评测 runner 直接消费：

```yaml
id: db-02                          # 唯一编号：{db|kb|web|mix}-{难度序号}
category: database                 # database | knowledge_base | web | mixed
difficulty: medium                 # easy | medium | hard
query: "..."                       # 用户问题原文（与真实使用一致，不注入提示）
expected_sources: [mysql]          # 预期主信源 → 用于检查工具路由是否正确
must_reflect: false                # 设计上「单次搜索信息不足」，应触发至少一轮补搜
gold_facts:                        # 判分点：报告中必须覆盖的关键事实
  - "..."                         # judge 逐条核对是否被报告覆盖（语义匹配，非逐字）
gold_answer: null                  # 可选：可程序化精确校验的答案（仅 DB 数值类）
```

## 2. 出题原则

四类信源各 5 条，每类内部覆盖 easy / medium / hard，并刻意包含**必须触发反思**的任务——没有这类任务，反思前后对比就做不出来。

| 原则 | 说明 |
|------|------|
| 信源真实 | 每条题必须贴着真实数据出：DB 题基于 `drugs/inventory/sales_records` 实际数据；KB 题基于知识库 5 份 PDF；禁止出「库里根本没有答案」的题 |
| 路由可判 | 每条声明 `expected_sources`，跑完后对照实际工具调用序列，得「路由准确率」 |
| 反思设计 | `must_reflect: true` 的任务天然信息不完整（跨文档、跨信源、需要多角度），单轮交付必然有缺口 |
| 答案稳定 | web 类题目答案会随时间漂移，gold_facts 只锚定「可验证的事实锚点」，判分看「论断是否有来源支撑」而非精确文本 |
| 难度递进 | easy=单点查询；medium=多表/跨文档/聚合；hard=多维聚合、跨信源综合、多轮搜索 |

## 3. 判分：三层结构

### 3.1 规则层（程序化，零成本、零争议）

- DB 数值类：报告中的关键数字与 SQL 实测值比对（容差 ±1% 处理四舍五入）
- 工程层：任务是否正常终止（非超预算中断）、工具路由是否命中 `expected_sources`

### 3.2 LLM-as-judge 层（质量判分核心）

judge 输入 = 原问题 + 最终报告 + gold_facts 清单（**不输入执行轨迹**，避免被过程带偏，只评结果质量）。

judge 用 `with_structured_output(..., method="json_mode")`——沿用 ADR-002 的结论，`json_schema` 在当前端点不可用。

输出结构：

```json
{
  "fact_hits":      [{"fact": "...", "covered": true, "evidence": "报告原文片段"}],
  "unsupported_claims": ["报告中无信源支撑的关键论断"],
  "completeness":   1-5,
  "verdict":        "pass | partial | fail",
  "reason":         "一句话理由"
}
```

派生指标：
- **事实命中率** = 命中 gold_facts 数 / 总数
- **幻觉率** = unsupported_claims 数 / 报告关键论断总数

### 3.3 过程层（来自 monitor 事件流 + usage_metadata）

| 指标 | 定义 | 来源 |
|------|------|------|
| 平均工具调用次数 | 每任务工具调用数均值 | tool_call 事件 |
| 平均迭代轮数 | 反思循环实际执行轮数 | round_result 事件 |
| 反思触发率 | `must_reflect` 任务中实际触发补搜的比例 | reflection_supplement 事件 |
| token 成本 | 每任务总消耗（输入+输出） | usage_metadata |
| 端到端延迟 | 提交到 task_result 的墙钟时间 | monitor 时间戳 |
| 路由准确率 | 实际信源 ⊇ expected_sources 的任务比例 | 工具调用序列 |

**反思触发率是本项目特有的指标**——它直接回答「反思循环是否真的在起作用」这个问题。

## 4. 对比实验协议

```text
配置 A（等效关闭反思）：REFLECTION_MAX_ROUNDS = 1，其余不变
配置 B（默认）：        REFLECTION_MAX_ROUNDS = 3，token 预算默认

每条任务 × 每配置 × 3 次重复（LLM 非确定性，单次结果不可信）
报告口径：3 次中 ≥2 次 pass 记为该任务 pass（多数投票）

总计运行数：20 任务 × 2 配置 × 3 次 = 120 次
```

产出表格（写入本文件第 6 节）：

| 指标 | 配置 A（无反思） | 配置 B（反思） | 变化 |
|------|------|------|------|
| 任务成功率 | 待测 | 待测 | 目标：量化「A% → B%」的提升 |

分 category 的细分表 + 归因分析（哪类任务提升最大、失败案例的失败原因分类）。

## 5. 校准流程（评测集本身也要测一遍）

1. **首跑校准**：每条任务先人工跑 1~2 遍，核对 gold_facts 与实际可得数据一致
   - DB 类：对照 SQL 实测值修正 gold_facts（初稿基于表结构推断，**必须**以库内真实数据为准）
   - KB 类：对照 PDF 原文核对表述
   - web 类：核对事实锚点当前仍可搜索到
2. **删题原则**：跑不通的题（信源缺失/歧义/依赖偶然搜索结果）直接删或改，不留「时灵时不灵」的题
3. **冻结**：校准后评测集冻结，后续所有对比跑同一份，保证可比性
4. **防泄漏**：评测集只在 judge 阶段使用，任何任务 query 不携带 gold_facts 进 agent

## 6. 成本估算与运行注意

- 按单任务平均 40k token 估算：120 次 ≈ 480 万 token；若成本敏感，可将 easy 题重复次数降为 2 次
- Tavily 间歇性故障已由分层重试兜底（ADR-005），但评测跑批仍建议避开网络高峰、失败任务单独补跑并记录
- 评测期间 `docker start deepsearch-mysql`（DB 类任务硬依赖）；知识库类任务需 embedding / Qdrant / Elasticsearch 服务在线
- 判分模型与被测模型同源，存在自评偏向——如实写入 eval 报告的方法论一节，有条件时换 judge 模型交叉抽检

## 7. 结果记录（评测完成后回填）

| 日期 | 配置 | 任务成功率 | 事实命中率 | 幻觉率 | 平均轮数 | 平均 token | 备注 |
|------|------|------|------|------|------|------|------|
| | A | | | | | | |
| | B | | | | | | |
