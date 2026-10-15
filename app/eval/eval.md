# 评测方案设计（阶段四）

> 目标：为 deepsearch-agents 建立一套可复现的评测集与判分协议，产出「反思循环前后对比」的量化数字。
> 核心认知：**deep research 系统的输出是研究报告，不是短答案**——不能用「标准答案精确匹配」的思路建评测集，必须用「关键事实命中 + 过程指标」的组合判分。
>
> 备注：知识库类任务运行在自建 RAG 之上（ADR-006），当前 embedding / Qdrant / Elasticsearch 均已接入，检索链路的生产默认模式为 `hybrid_rerank`（RRF 融合 + cross-encoder 精排，ADR-012；每路候选 25 条，ADR-013）。检索层可独立评测：`python -m app.eval.eval_retrieval` 用同一份题库跑出「进程内 BM25 vs ES 关键词 vs 向量 vs RRF 融合 vs RRF+精排」五模式的检索质量对比（指标口径见 3.4 节）——这是免费的第二组对比数据。精排将 Recall@5 从 59.3% 提到 87.0%（候选池上限 91.7%）。

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
  "answer_relevancy": 1-5,
  "relevancy_reason": "为什么给这个切题分（是否正面回答了用户所问）",
  "verdict":        "pass | partial | fail",
  "reason":         "一句话理由"
}
```

派生指标：
- **事实命中率** = 命中 gold_facts 数 / 总数
- **幻觉率** = unsupported_claims 数 / 报告关键论断总数
- **答案相关性** = `answer_relevancy` 跨任务均值（1–5 归一化到 0–1）

**completeness 与 answer_relevancy 是正交的两维，必须分开评**（对齐 RAGAs 的 Faithfulness / Answer Relevancy 之分）：

| 维度 | 回答的问题 | 反例 |
|------|-----------|------|
| completeness（+ 事实命中率、幻觉率） | 答得**全不全 / 对不对** | 漏了关键事实、或编造了没信源支撑的论断 |
| **answer_relevancy（新增）** | 答得**切不切题** | 「北京天气怎么样」→ 回一篇全对但无关的北京历史 —— Faithfulness 高、Answer Relevancy 低 |

现有 completeness 只测「信息覆盖」，测不出「跑题」；补上 answer_relevancy 后，「有据但答非所问」这类失败才可被量化。评测协议不引入 RAGAs 框架本体（见 ADR-009）。

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

### 3.4 检索层指标口径（2026-10-10 重定）

检索层与生成层**分开评测**：检索层不依赖 LLM，可完全程序化、零成本、可复现；生成层需要真实生成的答案与 LLM judge。两层各自独立成表，避免「检索差」与「生成差」互相掩盖。

| 类型 | 指标 | 衡量什么 | 口径 | 实现 |
|------|------|---------|------|------|
| 检索质量 | **★Context Recall@5（chunk 级）** | 标准答案的**出处**是否被检索出来 | gold 真值 micro：Top-5 覆盖的 gold chunk 数 / 该题 gold 总数（**2026-10-11 起为主指标**） | `eval_retrieval.py` |
| 检索质量 | Context Recall@5（fact 级，参考） | 关键**事实**是否被检索出来 | fact 级 micro：Top-5 覆盖的锚点事实数 / 该题锚点总数（**跨切块版本不可比**，降为参考列） | 同上 |
| 检索质量 | **Context Precision@5** | 检索结果的排序与占比是否合理 | chunk 级 AP@5（位置加权）：相关块 = Top-5 里的 gold 块 | 同上 |
| 检索质量 | **MRR@5** | 第一个相关结果排得是否靠前 | chunk 级：首个 gold 块的排名倒数，Top-5 外记 0，跨题平均 | 同上 |
| 检索质量 | **NDCG@5** | 排序整体质量（对数折扣 + 归一化） | chunk 级二值：DCG = Σ relᵢ/log₂(i+1)，NDCG = DCG / 理想 DCG | 同上 |
| 检索上限 | **候选池召回（gold）** | 当前配置的检索天花板 | 融合去重后、**截断成 top-5 之前**的候选集合覆盖的 gold 比例 | 同上 |
| 答案质量 | **Faithfulness** | 回答中的事实能否被检索到的上下文支持 | LLM judge：逐论断判「是否有 chunk 支撑」 | **第二阶段** |
| 答案质量 | **Answer Correctness** | 最终答案与参考答案相比是否正确 | LLM judge：与 gold answer 比对（事实重合 + 语义一致） | **第二阶段** |
| 性能 | **P95 延迟** | 95% 的请求在多长时间内完成 | 逐条 wall-clock 的 95 分位 | `eval_retrieval.py`（检索延迟） |
| 诊断（--diag） | **Recall@20 / 候选池召回@100** | 区分「没召回」与「排得靠后」 | @20 放大输出；候选池把每路深度从 15 加深到 100 | 同上 |

**Context Recall@5 的判定用事实锚点法，而非 RAGAs 的 reference-based 口径。** RAGAs 的 Context Recall 需要参考答案、再逐句判「该句能否归因到某 chunk」，属 LLM 判分，噪声大且不可复现；本项目用「每个 gold_fact 配一组可从原文客观判定的锚点字符串（数值、专名、术语）」实现同一语义，且每个数字都可争论、可复现。**只标注能从原文客观判定的锚点，描述型事实不配锚点、不入库**——这是这套口径的边界，也是它比 LLM 判分更硬的地方。

**Context Precision@5 用 AP@5 而不是裸占比。** 用户的诉求是「排序与占比是否合理」，而裸占比（相关块数 / 5）是位置无关的——第 1、3 位相关与第 4、5 位相关得分相同。AP@5 = Σ_{i: 相关} (前 i 个中相关数 / i) / R（R = Top-5 内相关块数，R=0 记 0），同时反映占比与位置：同样是 2 个相关块，排在第 1、3 位得 0.83，排在第 4、5 位只有 0.35。报告里同时给出**裸占比作参考列**，两者对照即可区分「相关块太少」与「相关块排得太靠后」。

**相关块的判定：2026-10-11 起改用 chunk 级 gold（真值）。** 题库里每道可跑题在 `facts` 之外多了一个 `gold` 字段——该题标准答案应落在哪些 chunk 上，主键是 `content_hash`（schema v3 字段，随文本稳定）。Top-k 里的 chunk「是否相关」直接查它是否在 gold 集里，**不再是「含任一锚点即相关」的近似**。产物与工具：

- 生成/校验：`python -m app.eval.build_gold [--check|--apply]`（`--check` 比对题库已有 gold 与当前索引，重切块后 hash 对不上会立刻报「gold 失效」→ 提示重标；并报 **WARN-REDUNDANT** = 其命中集合被其他块完全覆盖的冗余块；`--apply` 直接把最小覆盖集回写题库）。
- **gold = 最小覆盖集**（2026-10-11 修订）：能覆盖该题全部锚点的**最少块集合**。不直接把「所有含锚点的块」收进 gold，是因为切块本身有重叠（节内超长保留 80 字 `CHUNK_OVERLAP`、超大表格分块重复表头行）——同一句话会同时落进相邻两块，两块都算 gold 会让分母虚大、相关块计数虚高。实测 v3 上 39 个 gold 里有 12 个（30.8%）属纯冗余，最小覆盖集下为 27 个。
- 冗余修正让**两个方向相反的指标同时回归诚实**：chunk Rec@5 63.2% → **76.5%**（分母去重）；AP/MRR/NDCG 0.747/0.739/0.803 → **0.689/0.686/0.760**（同一个答案不再被重复计为多个相关命中）。
- 锚点校准：`python -m app.eval.calibrate_retrieval [--write]`（铁律 #5 的自动化版本：FAIL-0 / WARN-BROAD / WARN-CROSS / WARN-SPACE）。
- 记 `content_hash` 而非 `seq` 的理由：`seq` 是 jsonl 行号、重切块必变；hash 只在文本变化时变，正好用来**显式暴露「gold 需要重标」**这件事。

**为什么把主指标从 fact 级 micro Recall@5 换成 chunk 级 gold Recall@5。** fact 级的分母是「该题 fact 数」、命中又靠子串包含，**切块粒度一变数字整体漂移**：v2（400 字滑窗 + 80 字重叠）→ v3（按节切块 + 表格独立成块）同题同口径从 87.0% 掉到 79.6%。**该下降的归因已查明**（见 `docs/eval-attribution-v3.md`）：−7.4pt 全部集中在 3 条查询，其中 −4.6pt 是**解析层在贝莱德 p14–p16 的阅读顺序回归**（并列双表被读成列间交错流，`group_lines` 只按 y 合并、不按 x 间隙切分）、−2.8pt 是精排排序差异；**不是**「v2 重叠导致指标虚高」——该说法已证伪并更正（`docs/pdf-parsing-step7.md` §6.3）。
gold 口径以「标准答案的出处」为分母，语义稳定得多，且经最小覆盖集去重后**不受「一个答案被切成几块」影响**。代价是需要标注（现由 `build_gold.py` 生成 + 人工抽查），且**换解析器 / 改切块后必须重跑 `--check`**——这是它换来可比性的成本。

**跨解析器对照必须用 `--no-gold`。** gold 主键是 `content_hash`，换个解析器块边界全变、hash 必然对不上——那不是「召回差」，是「标准答案换了一套切法」，直接跑会得到一排 0 分的假象。评测脚本提供 `--no-gold`：强制作废 gold、所有题回落「含任一锚点」的近似口径，报告头会显式标注口径。**仅用于解析器 / 切块策略的 A/B 对照**（如 Docling vs 自研，见 `docs/eval-docling-vs-layout.md`），生产评测不要开。
配套地，`config` 提供 `RAG_INDEX_ROOT` / `RAG_KB_SUFFIX` 两个环境变量把对照臂的索引与后端 kb 值隔离到独立空间，两臂互不污染。

**P95 延迟有两个必须说明的口径边界**：① 当前题库 39 条查询，P95 ≈ 第 2 慢的那条，统计上不稳健——报告同时给出 P50 / max 与样本数，不单看 P95；要真正谈 P95 需要 ≥100 条查询。② 这是**检索延迟**（两路检索 + RRF 融合 + query embedding，百毫秒量级）；**端到端延迟**（含 LLM 生成，秒到分钟量级）在生成层统计，两者不能混为一表。

**生成层两个指标留到第二阶段**（2026-10-10 决策）。理由：Faithfulness 需要真实生成的答案，Answer Correctness 还需要**参考答案（gold answer）**这一全新的标注资产——两者都依赖 LLM judge 的落地（judge 协议见 3.2，代码尚未实现）。分阶段的顺序是：先把检索层指标与口径稳定下来（含之后要加的 rerank），再做生成层。届时还需决定评测对象：评「给定检索结果 → 单轮 RAG 回答」（可控、能隔离检索质量与编排质量）还是「agent 最终产出的研究报告」（贴产品形态、但需跑完整 agent 且参考答案要写到报告粒度）。

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
- 评测期间 `docker start ds-mysql`（DB 类任务硬依赖）；知识库类任务需 embedding / Qdrant / Elasticsearch 服务在线
- 判分模型与被测模型同源，存在自评偏向——如实写入 eval 报告的方法论一节，有条件时换 judge 模型交叉抽检

## 7. 结果记录（评测完成后回填）

| 日期 | 配置 | 任务成功率 | 事实命中率 | 幻觉率 | 平均轮数 | 平均 token | 备注 |
|------|------|------|------|------|------|------|------|
| | A | | | | | | |
| | B | | | | | | |

---

## 10. 生成层评测（2026-10-15 落地，runner = `eval_agent.py`）

第 3~7 节是**检索层**协议与历史结论。本节是**生成层**：完整跑一次 agent 任务才拿得到的 7 个指标。

### 10.1 为什么单臂（2026-10-15 用户拍板）

暂不设「原始版」对照臂，只评测**最新版**配置（反思 3 轮 + 引用规范 + 检索 `hybrid_rerank`）。因此产出的是**绝对值**，不含「提升了多少」。代价要如实说：拿不到 A→B 的提升幅度；收益是省掉一半跑批成本，且不必维护一个「关掉各种优化」的对照配置。

每题跑 **3 次**（LLM 非确定性，单次不可信），3 次中 ≥2 次 `pass` 记该题 pass。

### 10.2 七个指标的口径

| 指标 | 口径 | 判分层 | 备注 |
|------|------|--------|------|
| **任务成功率** | `verdict=pass` 的题数比例（judge，3 次多数投票） | LLM judge | 未开 judge 时退化为「三次都正常产出结果」 |
| **答案正确率** | DB 类：`gold_answer` 数值是否出现在报告（±1% 容差，规则层）；其余：`gold_facts` 覆盖率（judge） | 规则 + judge | 规则层优先，有 `gold_answer` 就不看 judge |
| **Faithfulness（规则层）**<br>claim 级锚点核验 | 分母 = `required_citations` 里 `anchor_match: true` 的**关键 claim 数**；分子 = 该 claim **自己标注的那些编号**里，存在「编号在本次采集映射中真实存在、且其 chunk 原文含该 claim 的 `anchors` 任一项」的 claim 数 | 规则层（编号配对由 judge 提供） | 2026-10-15 改口径：上一版拿报告里**全部**编号做存在性匹配（实测 157 个编号 vs 5 条 claim），任何一条含锚点即放行，等于测「报告里存不存在含这些词的证据」；挂错编号与编造编号都判不过，分别记 `off_topic` / `bogus`。锚点从整题共用改为**跟着 claim 走**，宽词（「数字人」「白皮书」）会让指标失去区分度 |
| **Faithfulness（judge 层）**<br>语义支撑率 | judge 逐条判「该 claim 标注的来源是否**真的支撑**这条论断」，分母固定为题库 `must_cite` claim 数（judge 漏答按不支持计） | LLM judge | 与规则层**并列呈现**（不合并）。judge 必须同时看见「引用编号 → 片段原文」才能判，否则它的 `supported` 只是先验猜测；上一版实现只把 supported 的记 1 分再求均值，**恒等于 1.0**，一条都不支撑时又整条跳过不记 0 |
| **claim 编号配对率** | 分子 = judge 成功抄回「这句话挂的是哪个编号」的 claim 数；分母 = 参与核验的 claim 数 | 元指标 | Faithfulness 规则层的**成立前提**：配对率不足 1 说明 judge 没照做，读数必须结合它看。`--no-judge` 时规则层不产出（不是 0） |
| **引用准确率** | 拆两列：**带引用** = `must_cite` 的 claim 中合规标注的比例（分母 = 必引 claim 数）；**编号可解析** = 报告里出现的去重引用编号中，能在采集映射里找到的比例（分母 = 报告编号总数，防凭空编编号） | 规则层 | 依赖 `EVAL_CAPTURE=1`；未开时②为空 |
| **平均工具调用次数** | `tool_start` 事件计数均值 | 过程层 | 超过题库 `max_tool_calls` 记「跑偏」并在报告标出 |
| **P50 / P95 延迟** | `POST /api/task` → `task_result` 墙钟时间 | 过程层 | 30 题×3 次=90 样本，P95 统计上不稳，**同时看 P50** |
| **单任务 API 成本** | monitor `token_usage` 事件累加的 usage，按 DeepSeek 峰谷分时价折算（元） | 过程层 | 缓存命中部分按 `cached_in` 单价；该事件由 `main_agent._run_agent_round` 逐次上报（**不上报则成本指标恒为空**，2026-10-15 首跑踩过） |

**cite_kind 三种判据**（题库 `required_citations` 字段）：

| cite_kind | 判据 | 适用 |
|-----------|------|------|
| `chunk` | 报告出现【引用BN-I】且该编号在本次采集映射里真实存在；带 `anchor_match: true` 时还要求该 chunk 含该 claim 自己的 `anchors` | KB / mixed 的 KB 部分 |
| `query_cond` | 报告写明表名（`drugs` / `inventory` / `sales_records`） | DB 类 |
| `url` | 报告出现 http(s) 链接 | web 类 |

**哪些 claim 不进 Faithfulness**：`anchor_match` 只标在「证据是原文里一句话」的 fact 型 claim 上。跨文档对比、共同判断、背离点这类**综合型 claim**（kb-05 / kb-13 / mix-02 第二条 / mix-06）不给锚点——它们的证据不是一句可逐字比对的话，硬配锚点只会退化成送分；这类 claim 由 judge 单判。锚点写了要自检：

```bash
python -m app.eval.eval_agent --check-anchors   # 逐字命中校验，不跑 agent、不花 API 钱
```

MISS（语料里根本没这个串）直接失败——切块改版后锚点会静默失效，与其跑完一遍发现 Faithfulness 全 0，不如当场暴露；全库命中占比 >3% 记 BROAD（判据与检索层 `calibrate_retrieval` 同源）。题级 `faithfulness_anchors` 字段现在只作为语料存在性的补充清单，**不再参与判分**。

### 10.3 DeepSeek 峰谷分时计价（deepseek-v4-flash）

**高峰时段 = 北京时间 09:00–12:00 与 14:00–18:00**；其余（夜间、周末、法定节假日）为空闲时段，**闲时价 = 高峰价的一半**。

| 项（元 / 百万 tokens） | 空闲时段 | 高峰时段 |
|---|---|---|
| 输入（缓存命中） | 0.02 | 0.04 |
| 输入（未命中） | 1 | 2 |
| 输出 | 4 | 8 |

单任务成本估算（输入 40k + 输出 6k）：

| 场景 | 单次 | 90 次（30 题 × 3） |
|---|---|---|
| 闲时 · 缓存未命中 | 0.064 | 约 5.8 元 |
| 闲时 · 缓存命中 60% | 0.040 | 约 3.6 元 |
| 高峰 · 缓存未命中 | 0.128 | 约 11.5 元 |
| 高峰 · 缓存命中 60% | 0.081 | 约 7.3 元 |

**跑批建议放夜间（00:30 之后）**，成本直接砍半。真正的成本是 token 与时长：90 次 × 2~6 分钟 ≈ 4~9 小时 GPU 占用，且会占住 embedding 容器，**不要和正常使用同时跑**。

### 10.4 采集开关（`EVAL_CAPTURE`，默认关闭）

引用准确率的「编号可解析」与 Faithfulness 都需要知道**每个引用编号背后到底是哪段原文**。为此 monitor 新增采集入口：

- **默认 `EVAL_CAPTURE=0`**：采集代码完全不执行，线上零磁盘开销、零行为变化；
- `EVAL_CAPTURE=1 EVAL_TRACE_DIR=<dir>`：知识库检索把「引用编号 → chunk（doc/heading/score/原文）」落盘；网络搜索只登记命中标题与 URL（不存正文，避免 trace 体积失控）；DB 工具落 `db_query` / `db_table_preview`（SQL 语句、列名、行数、前 100 行结果）——这一段是 2026-10-15 补的，此前 `db_calls` **恒为空数组**，DB 类题的「引用是否真实」退化成在报告里 grep 表名。

落盘路径 `<dir>/<thread_id>/<工具名>.<序号>.json`，runner 用 `--trace-dir` 指向同一目录读取。

**另一个必须知道的前提：执行层检索预算熔断**（ADR-021）。服务端 `TOOL_CALL_BUDGET`（默认 30）超限后，检索类工具**不再执行**，改为把「请基于已有材料收尾」的指令返回给模型。它直接改变延迟、工具调用次数与内容厚度这三个指标的读数——**跑批必须记录当时生效的预算值**，否则「P50 从 90 秒降到 60 秒」这种话分不清是优化还是被闸门掐断了。设为 0 即关闭。

### 10.5 跑法

```bash
# 1) 带采集开关起服务（关掉时 Faithfulness 与「编号可解析」为空）
EVAL_CAPTURE=1 EVAL_TRACE_DIR=app/eval/results/<run_id>/traces TOOL_CALL_BUDGET=30 \
  python -m uvicorn app.api.server:app --port 8001

# 1b) 题库锚点自检（不跑 agent、不花 API 钱；切块改版后必跑）
python -m app.eval.eval_agent --check-anchors

# 2) 小规模冒烟（建议先跑这个：2 题 × 1 次）
python -m app.eval.eval_agent --ids db-03,kb-07 --repeat 1 --trace-dir app/eval/results/<run_id>/traces

# 3) 正式跑批（挑夜间；单题超时默认 420s，旧值 900s）
python -m app.eval.eval_agent --repeat 3 --trace-dir app/eval/results/<run_id>/traces

# 只看过程指标、不花 judge 钱
python -m app.eval.eval_agent --repeat 1 --no-judge
```

**先冒烟再全量**（2026-10-15 的教训：直接上 30 题，20 分钟内就撞上「一题 903 秒超时」和「一题 49 次检索不收尾」，而 runner 只在全部跑完才写报告，中途停等于全部作废）。

产物：`app/eval/results/<run_id>/report.md`（人看的表）+ `summary.json`（逐题明细）。
