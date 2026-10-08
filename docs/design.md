# 技术设计记录（ADR）

> 本文档记录 deepsearch-agents 的关键技术决策：为什么这么做、放弃了什么方案。
> 面试时可直接引用每条决策的「背景 → 决策 → 放弃方案 → 理由」。

---

## ADR-001 反思循环：外层 Python 循环，不改框架

**背景**：deepsearch 类系统的核心是「搜索 → 反思信息是否充分 → 针对缺口补搜」。deepagents 框架只提供单次规划执行（planning → task 调度 → 汇总），没有信息充分性评估和迭代补搜机制。

**决策**：在 `run_deep_agent` 外层包一个反思循环，不 fork、不修改 deepagents 源码：

```text
for round in 1..MAX_ROUNDS:
    main_agent.astream(消息, thread_id)      # 复用同一条 thread，agent 自带上文
    ↓ 本轮最终回答 + token 消耗
    评估器（独立 LLM 结构化调用）
    ↓ SufficiencyEvaluation{sufficient, missing_dimensions, follow_up_queries}
    sufficient / 预算耗尽 → 终止，输出最后一轮结果
    不充分 → 把缺口维度 + 补充查询拼成新消息，进入下一轮
```

**放弃的方案**：
1. ~~改写主智能体 system_prompt，让模型自己反思~~——本质是提示词工程，循环不可控、无法强制预算约束，面试无法展示工程设计能力
2. ~~fork deepagents 在图内加反思节点~~——耦合框架内部结构，升级即碎；且上下文隔离机制（`_EXCLUDED_STATE_KEYS`）会让图内反思节点拿不到完整子智能体中间产物
3. ~~LangGraph 外面再套一层 graph~~——多一层抽象无实际收益，Python 循环已足够清晰

**为什么这个位置可行（源码依据）**：
- `create_deep_agent` 返回 LangGraph `CompiledStateGraph`，`checkpointer + thread_id` 持久化全部消息状态——同一 thread_id 二次 invoke，模型天然看到第一轮全部内容（`graph.py` L708-729）
- 子智能体经 `task` 工具调用，结果以 ToolMessage 回流主循环（`middleware/subagents.py`），外层循环拿最终 AIMessage 即可，无需感知内部结构

**终止条件（三选一）**：
1. 评估器判定 `sufficient=true`
2. 达到最大轮数 `REFLECTION_MAX_ROUNDS`（默认 3）
3. 累计 token 超预算 `REFLECTION_TOKEN_BUDGET`（默认 150k，取自 AIMessage.usage_metadata，无元数据时按字符数估算）

**成本权衡**：每轮多一次评估器 LLM 调用（输入=问题+本轮答案，输出=结构化 JSON，约 1k token）。换来的是可量化的质量提升（评测数据见 `docs/eval.md`，阶段四产出）。

---

## ADR-002 评估器与执行模型复用同一 model 实例

**背景**：反思评估器需要 LLM 做结构化输出。

**决策**：复用 `app/agent/llm.py` 的 `model`，但**不使用默认的 json_schema 结构化输出**，改为 `model.with_structured_output(SufficiencyEvaluation, method="json_mode")`，并保留纯文本 + 宽容 JSON 解析作为回退。

**理由**：
- 评估任务是轻量判断任务，不值得引入第二个模型配置
- 结构化输出保证 `missing_dimensions` / `follow_up_queries` 是可枚举的列表，可直接拼进补搜消息，不依赖文本解析
- **实测踩坑**：默认的 `json_schema` 方式在部分 OpenAI 兼容端点上直接返回 `400 This response_format type is unavailable now`。这个异常被降级逻辑吞掉后表现为「每轮都判定信息充分」，反思循环静默失效——**看起来在跑，其实永远不补搜**。改用兼容性更好的 `json_object` 模式后恢复正常

**三层兜底**（评估器故障不应让整个任务失败）：
1. `json_mode` 结构化输出（首次失败后置位标记，后续轮次不再重试，省一次无谓请求）
2. 纯文本调用 + 宽容 JSON 解析（剥 Markdown 围栏、截取首尾花括号、兼容布尔写成字符串）
3. 仍失败 → 判定「已充分」终止循环，保留本轮结果，并在 `reasoning` 里记下异常

**放弃的方案**：~~用便宜的小模型做评估器~~——合理方向，但当前只有一套 API 配置；待阶段四评测时若发现评估器成本占比显著，再引入分级模型（记为 TODO）。

---

## ADR-003 中间轮结果与最终结果的推送语义分离

**背景**：前端以 `task_result` 事件作为任务结束信号（`useDeepAgentSession.ts`），若反思循环的每轮都发 `task_result`，前端会提前结束。

**决策**：事件分级——
- `round_result`：每轮执行结果，前端仅展示在时间线
- `reflection_evaluation`：评估器结论（充分/不充分 + 缺口 + 理由）
- `reflection_supplement`：进入补搜轮，附补充查询列表
- `task_result`：仅整个反思循环结束后发送最终结果

这组事件本身也是产品能力的可视化：用户在时间线上能看到「搜了 → 反思 → 不够 → 再搜」的思考链。

---

## ADR-004 执行事件的投递模型：多连接广播 + 事件缓冲回放

**背景**：现象是「后端日志显示任务已经跑完，页面却一直停在生成中」，刷新也不一定恢复。定位过程与结论：

1. **写探针实测后端投递链路**——起独立服务、挂 WebSocket 客户端、POST 任务，确认事件确实发得出去，排除「后端没推」
2. 定位到两个投递层缺陷，且都与「页面收不到事件」直接对应：
   - `active_connections` 用 `thread_id → 单个 WebSocket` 存连接。前端 thread_id 存在 localStorage，**同一浏览器的多个标签页共享同一个 thread_id**，后建连的页面会把先建连的覆盖掉，事件只发给最后一个页面，其余页面永远空白
   - **事件即发即弃，没有缓冲**。页面刷新、断线重连、或 WebSocket 假死（对端进程被 `--reload` 重启，浏览器侧 TCP 未及时感知，以为还连着）时，错过的事件永久丢失，前端只能靠下一次任务重置

**决策**：把投递层从「单连接、即发即弃」改成「广播 + 可回放」：

```text
active_connections: dict[thread_id, set[WebSocket]]   # 一个 thread 允许多个页面
每个事件分配 seq（thread 内自增）并写入该 thread 的环形缓冲（默认 200 条 / 最多 50 个 thread）
emit  → 入缓冲（无论有没有连接）→ 广播给该 thread 下所有连接，发送失败的连接就地剔除
WS 建连 → 先回放该 thread 的缓冲事件，再接收新事件
begin_task() → 新任务开始时清空缓冲，避免回放到上一轮任务的轨迹
前端 → 按 seq 去重（seq <= 已处理最大值则丢弃），刷新/重连后补齐但不重复
```

**为什么缓冲放在后端而不是前端**：真实状态源在后端。前端重连后无法「问」后端要历史，除非后端存了。放在后端还有一个副作用是好的——**任务跑完后才打开的页面也能看到完整执行轨迹**，这恰好是简历项目 Demo 时最有用的行为。

**放弃的方案**：
1. ~~只改前端去重~~——治不了根因，事件在前端根本收不到
2. ~~用 Redis Pub/Sub 或消息队列存事件~~——单进程教学项目引入外部依赖，收益不匹配；缓冲上限写死 200 条已够覆盖一次研搜轨迹（进程重启后缓冲即失效，这是刻意的取舍，与阶段三的持久化 checkpointer 是两件事）
3. ~~给每个标签页单独分配连接键~~——会让「任务绑定的 thread_id」和「连接键」分裂成两套标识，反而更复杂；广播 + 回放已覆盖同一浏览器的所有页面

**前端配套的假死检测**：心跳定时器原本只 `send("ping")` 但从不检查 pong，连接假死时前端会一直以为自己在线。改为记录「最近一次收到任何服务端消息的时刻」，超过 70s 完全静默就主动 `close()` 触发重连，重连后由后端回放补齐。

**另一个连带修复**：`App.tsx` 里把事件挂到「最后一轮对话」时，若本地 `turns` 为空（刷新页面、跨标签页收到事件）会直接 return，导致事件收到了却不显示。现在会按事件补建一条轮次，并借助 `session_created` 事件里新增的 `query` 字段还原用户原始提问。

---

## ADR-005 外部调用容错：失败降级而非向上抛异常

**背景**：线上出现任务中途中断，前端报 `('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))`。截图显示中断前已经成功执行过两次网络搜索，第三次才失败——间歇性，不是被稳定阻断。

**定位过程**：

1. 从异常文本形态判断归属：该元组格式是 `requests`/urllib3 的签名，`httpx`（OpenAI 客户端）不会长这样；且项目内 LLM 走 httpx、搜索走 `tavily` 库，而 `tavily/tavily.py` 顶部就是 `import requests`。**结论：断的是 Tavily 搜索链路，不是大模型**
2. 实测网络：`api.tavily.com` DNS 正常、TCP 可连（1~9 秒，偏慢）；用 `requests` 分别走环境代理和绕过代理直连，均 2 秒内返回 200；官方 `TavilyClient` 同样正常。**排除「服务不可用」与「被墙」，确认是间歇性故障**
3. 结合「前两次成功、第三次失败」的现象定位机制：`TavilyClient` 内部持有 `requests.Session`，默认复用 keep-alive 连接池，而 `requests` 默认 `max_retries=0`。池中一条已被代理或服务端关闭的连接被复用时，直接抛 `ConnectionError` 且**不重试**

**决策**：把「一次网络抖动 = 整轮任务作废」改成「分层重试 + 失败降级」：

```text
传输层：requests.Session 挂 Retry
        connect/read/status 各 3 次、backoff 0.8 * 2^(n-1)、status_forcelist 含 429/5xx
        ⚠ allowed_methods 必须显式加 POST —— urllib3 默认只重试幂等方法，搜索是 POST
应用层：传输层重试耗尽后整体再试 2 次（覆盖代理抖动）
失败降级：仍失败 → 不抛异常，返回 JSON 错误说明 + 上报 tool_error 事件
```

**为什么失败要降级成返回值而不是继续抛**：工具异常会顺着 LangGraph 冒到图顶层，触发 `run_deep_agent` 的兜底分支，**把本轮已搜集的资料一并作废**。返回一段结构化错误（含 `reason` / `detail` / `guidance`）则让模型有机会换关键词重搜、改用数据库或知识库助手，或如实说明该部分缺失——这正是 agent 该有的自愈行为，也避免模型在检索失败时凭记忆编造内容。

**配套的三处改动**：

1. `monitor` 新增 `tool_error` 事件，与任务级 `error` 区分：前者表示「这次工具没取到结果，任务仍在继续」，后者表示「整轮中断」。同时新增 `report_error` 公开方法（原先外部直接调用私有的 `_emit`，属于坏味道）
2. `run_deep_agent` 的异常兜底：错误信息带上异常类型与原因；若此前已有产出，用 `report_task_result(partial=True)` 交付**部分结果**，前端标注「已中断」并提示信息不完整，避免用户把半成品当成完整交付
3. 不可重试的错误（密钥无效 / 配额耗尽 / 权限不足 / 参数错误）按类名识别后立即失败，不浪费重试次数和配额

**放弃的方案**：

1. ~~只在 `main_agent` 外层加重试重新执行整轮~~——会重复消耗已有上下文和配额，且 LangGraph 图已按线程打点，重入容易产生重复消息；根因在单次 HTTP 调用，应在最靠近故障处修复
2. ~~给模型再包一层「搜索失败自动换词重试」的包装工具~~——把决策权从模型手里拿走，且换词质量不如模型本身；降级为返回 `guidance` 让模型自己决策
3. ~~换用 tavily 的 `AsyncTavilyClient`（走 httpx）~~——httpx 对连接池失效的处理同样依赖显式重试配置，换库不解决根因，反而引入同步/异步工具混用问题

---

## ADR-006 知识库信源：用自建 RAG 整体替换 RAGFlow

**背景**：项目一（电商问数）的 Qdrant / Elasticsearch 是 NL2SQL 的 schema 召回，不是文档 RAG；本项目此前接入 RAGFlow 的方式是「向 RAGFlow Chat 助手提问」——提取、切分、embedding、检索、生成全部发生在 RAGFlow 内部，自己的代码里没有一条 RAG 链路。两个项目摆在一起，简历技能栏中「文档切分、Embedding、检索与上下文构建」没有任何代码证据，面试追问「RAG 你自己搭过吗」无法自证。

**决策**：整体移除 RAGFlow 依赖，自建轻量 RAG 模块（`app/rag/`），架构上做两个关键改变：

1. **检索与生成分离**。原方案把问题转发给 RAGFlow Chat 拿现成答案（R 和 G 都在外部服务里）；新方案工具只返回检索到的**原文片段**（含来源文档与章节），由知识库子智能体亲自阅读、筛选、综合——检索（R）是自建模块，生成（G）是子智能体，这才是标准的 RAG 架构，也把「知识库助手」从传话筒变成了真正的专家。
2. **双路检索 + 优雅降级**。BM25（jieba 分词）始终可用；embedding 端点独立可插拔（`EMBEDDING_*` 三项环境变量），配置齐全时走「BM25 + 向量余弦」双路召回，用 RRF（Reciprocal Rank Fusion）融合排序——两路得分量纲不同（词频统计 vs 余弦），RRF 只看排名不看分值，免调参。embedding 未配置时自动降级纯 BM25，不报错不阻塞。

**端点事实（实测）**：项目当前 LLM 端点仅有两个对话模型（`/v1/models` 实测），无 `/v1/embeddings`（404）。因此 embedding 不能复用 LLM 配置，必须独立接入 OpenAI 兼容的 embedding 服务（硅基流动 / DashScope 等），并通过降级设计保证零额外配置时系统仍可运行。

**切分策略**：结构感知——Markdown 按 `#` 标题、DOCX 按 heading 样式、PDF 按页提取，块内超长再按中文句读下钻，目标 600 字符 + 100 重叠。切分质量是 RAG 检索质量的第一决定因素，按标题/句子边界切出的块才能独立成义。

**摄入与增量**：`python -m app.rag.ingest` 扫描 `docs/knowledge_base/`（目录即知识库），按文件 sha256 增量摄入，索引落盘为 `chunks.jsonl + vectors.npy + meta.json`。

**放弃的方案**：
1. ~~保留 RAGFlow，只给附件信源加轻量 RAG~~——工程量最小，但知识库主链路仍是调外部服务，两个项目依然没有一条完整自建 RAG 链路（评估后用户拍板整体替换）
2. ~~引入 Qdrant / FAISS 等向量库~~——知识库规模是 5 份 PDF、数百 chunk，numpy 全内存余弦检索毫秒级返回，引入向量库是为简历堆名词，违背「最小够用」原则
3. ~~本地跑 embedding 模型（bge-m3）~~——2GB+ 模型下载、CPU 推理秒级延迟，教学项目不值；远程 API 一行配置即用
4. ~~加权拼接两路得分~~——BM25 与余弦分数量纲不可比，融合权重需要评测集校准，RRF 按排名融合天然免调参

**面试考点**：切分为什么按结构不按固定长度；BM25 与向量检索各自覆盖什么 case（精确术语 vs 语义改写）；RRF 为什么免调参；检索生成为什么要分离（可换生成模型、可观测、子智能体可多角度检索）。

---

---

## 框架选型（面试必答）

**为什么用 deepagents 而不是：**
- **手写 supervisor**——规划循环、todo 管理、上下文裁剪、子智能体隔离都是成熟轮子；我的价值增量在反思循环、持久化、可观测、安全，而非重复造调度层
- **AutoGen**——对话式多智能体（agent 间聊天）与「主智能体单向调度专家」的模型不匹配，对话轮次不可控
- **CrewAI**——角色固定、流程偏线性，不适合「主智能体按任务动态规划」的 deepsearch 场景
- **纯 LangGraph 手写**——deepagents 本身就是 LangGraph + middleware 的标准化封装（见 `graph.py`），用它等于站在标准结构上做扩展
