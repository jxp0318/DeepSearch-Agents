# Docling vs 自研版面解析：两臂端到端对照（2026-10-11）

## 0. 一句话结论

两条链路**各有胜负，且胜负分别落在不同类型的页面上**——不是「谁全面更好」：

| | 赢在哪 | 具体 |
|---|---|---|
| **Docling 胜** | 无框线**并列双表**的阅读顺序 | 贝莱德 p14–p16：`长期投资级别信用债我们持低配观点` 行内相邻，fact 4/4（自研 3/4） |
| **自研 v3 胜** | **图表混合页**的表格数据提取 | ICCT p12：Docling 整张车型数据表丢失，fact 0/4（自研 4/4） |
| **自研 v3 胜** | 正文**文本连续性** | Docling 保留行尾断行（`人口结构是印 度最核心`），断行处锚点检索失败 |

端到端 fact 级 Rec@5：**自研 83.7% vs Docling 64.4%**，但这 19.3pt 差距**高度集中在 2 道题**
（icct-q8/q9 共 8/45 个锚点，即 17.8pt）——Docling 的失分是一个**单点缺陷**（丢表），不是全面落后。

---

## 1. 对照设计（为什么要控制变量）

只对照**解析层**，不掺切块策略：两臂的正文切块**都用自研** `app.rag.chunker.split_text`
（Docling 臂在 `docling_chunks.py` 里 import 同一个函数）。两臂唯一的差异是：

    阅读顺序 / 标题识别与层级 / 表格结构还原 / 页眉页脚剔除 / 页码轨

这样归因才落在「解析器」上。若连切块器一起换，出了差异无法判断是谁造成的。

**为什么不用 chunk 级 gold 做主指标**：评测主指标的 gold 主键是 `content_hash`（文本 sha256）——
换个解析器块边界全变、hash 必然对不上。那不是「召回差」，是「标准答案换了一套切法」。故本次对照
统一用 `--no-gold`（本次为评测脚本新增的开关），相关块一律按「含任一锚点」近似判定，**两臂同口径**。

## 2. 两臂产物

| 臂 | 解析器 | ICCT（36 页） | 贝莱德（17 页） |
|---|---|---|---|
| **自研 v3** | PyMuPDF span→行→结构 + 自研分栏 / 表格 / 页码双轨 | 310 块（body 289 / table 21） | 79 块（body 78 / table 1） |
| **Docling** | docling 2.137.0 + `docling-layout-heron` 版面模型 + TableFormer(accurate) | 521 块（body 264 / table 257） | 97 块（body 69 / table 28） |

Docling 把大量原本走正文的内容**归为表格**，且表格按行切后块数膨胀（ICCT 15 张表 → 257 块）。
块长度中位数两臂接近（325 vs 327），说明切块尺度一致，差异来自**内容归属**。

> 注：Docling 首次跑通了 TableFormer（权重 212MB+145MB，走 hf-mirror）。此前该权重缺失，
> 只能降级 `--no-tables` 跑，表格会退化成「一个大单元格」的长文本。

## 3. 解析层指标：锚点覆盖率

**纯解析层口径**（不看排序、不看 top-k，只问「这句话解析完还在不在库里」），45 个可跑锚点：

| 口径 | 自研 v3 | Docling |
|---|---|---|
| **原样子串** | **100.0%** | 77.8% |
| **归一化后**（CJK 间空格/换行 + `\|` 分隔符拉齐） | **100.0%** | 82.2% |

- 原样口径的 10 个丢失锚点：icct-q8/q9 各 4 个（表格数值）+ blk-q6 2 个（正文断行）。
- 归一化后**只剩 8 个**（全是 icct-q8/q9）——说明 blk-q6 是**表示问题**（断行），可修；
  icct-q8/q9 是**内容真丢**，不可修。
- 自研 v3 两个口径都 100%，即它的产物没有「信息在但表示对不上」的情况。

## 4. 端到端检索（hybrid_rerank，同口径 `--no-gold`，48 条查询）

| 指标 | 自研 v3 | Docling | 差 |
|---|---|---|---|
| AP@5（Context Precision） | **0.747** | 0.623 | −0.124 |
| MRR@5 | **0.739** | 0.652 | −0.087 |
| NDCG@5 | **0.803** | 0.671 | −0.132 |
| 相关块占比@5 | **30.8%** | 28.3% | −2.5pt |
| **fact Rec@5** | **83.7%** | 64.4% | **−19.3pt** |
| 查询延迟 P50 | 818ms | **655ms** | Docling 快 163ms |

> 参照：自研 v3 的**生产口径**（含 chunk 级 gold）为 Rec@5 76.5% / AP 0.689 / MRR 0.686 / NDCG 0.760 /
> 池召回(gold) 88.9%，见 `app/eval/results/2026-10-11_21.25.20__retrieval__hybrid_rerank.md`。

Docling 延迟更低，与它把内容切得更碎（top-5 更短）有关，不是效率优势。

## 5. 逐题归因：差距到底出在哪

### 5.1 Docling 赢：无框线并列双表（贝莱德 p14–p16）

| 题 | 自研 v3 | Docling |
|---|---|---|
| blk-q4/zh | fact 3/4，AP 0.48 | **fact 4/4，AP 0.89** |
| 漏掉的 | `中国债券可提供稳定性与分散投资价值` | 无 |

自研链路的已知缺陷：`group_lines` 只按 y 合并、不按 x 间隙切分 ⇒ 左右两表的行被并成同一
`Line` ⇒ `analyze_columns` 判 `no_gap` ⇒ 退回全局 y 排序 ⇒ 两列交错，资产名与评论错位。
Docling 的版面模型把它正确识别并线性化输出，`长期投资级别  信用债  我们持低配观点。相比长期债券…`
行内相邻 —— **这正是自研链路最需要补的那一类页面**。

### 5.2 自研赢：图表混合页把表格整张丢掉（ICCT p12）

| 题 | 自研 v3 | Docling |
|---|---|---|
| icct-q8（欧洲市场，zh/en） | **fact 4/4，AP 1.00** | fact 0/4，AP 0.00 |
| icct-q9（美国市场，zh/en） | fact 3–4/4 | fact 0/4 |

**证据链**：ICCT p12 的 `67.3%` / `41.5%` 在 PDF 文本层里**确实存在**（PyMuPDF 直读可得，该页
图片数 = 0）。但 Docling 的产物里，p12 **没有任何 table 块**，正文块也只有图标题
（`Light-duty EV sales share, technology mix, and market share by manufacturer in Europe`）
—— 整张车型数据表被判为 **picture** 跳过，数值一个都没进库。

Docling 的版面模型对「带标题的图表 + 内嵌数据表」这类混合版面误判为图片；自研链路的
`detect_tables` 是基于文本层几何信号（框线 / 空白带 / 数值密度）判的，不依赖模型分类，所以能拿到。

### 5.3 自研赢：正文文本连续性（贝莱德 blk-q6）

Docling 的原生输出用**空格**代替行尾换行：`人口结构是印 度最核心`（原文一行放不下被折行）。
锚点 `人口结构是印度最核心` 因此匹配不到，检索 fact 0/2。
自研链路的 `_normalize_layout` 会删掉 CJK 之间的换行 ⇒ 句子连续。

归一化实验证明这是**表示问题而非内容缺失**：拉齐表示后覆盖率 77.8% → 82.2%，blk-q6 恢复。
但对**检索**它是真实损害——BM25/子串匹配在断行处断词，`印度` 被切成 `印`+`度`。

## 6. 结论与建议

1. **不要整体替换解析器**。Docling 与自研各有一块明确的能力边界，直接换掉会从「丢双表」变成
   「丢图表页表格」，两处都是真实数据。
2. **自研链路该补的是「表格结构还原」**，目标是 p14–p16 那类无框线并列双表。Docling 已验证
   「版面模型识别 + 线性化输出」这条路走得通，候选方案：
   - 只在 `detect_tables` 判定为「无框线且多列并列」的页面区域上，**旁路调用 Docling** 做局部还原
     （它是项目里的对照臂，已装好独立环境 `.venv-docling`，不污染主环境的 onnxruntime-gpu）；
   - 或引 `pymupdf_layout`（PyMuPDF 运行时提示的官方版面包）做同类结构输出。
   验收标准现成：`长期投资级别信用债我们持低配观点` 能作为**连续串**命中。
3. **Docling 若要用，必须先加一层输出归一化**（删 CJK 间换行/空格），否则正文断行会持续
   损害 BM25；这一层成本极低、收益已验证（覆盖率 +4.4pt）。
4. **顺带发现的自研侧问题**：v3 的 ICCT 表格块最长 915 字符，超过 `EMBEDDING_MAX_CHARS=480`
   的上限 ⇒ 会被截断一半（Docling 臂因按行切表头重复，最长 467，未超限）。建议给
   `TableBlock.split_for_chunks` 也加 480 字符硬约束。这条与 Docling 无关，是本次对照的副产品。

## 7. 复现

```bash
# 1) Docling 臂解析（独立环境，产出 schema v3 chunk JSONL）
.venv-docling/Scripts/python.exe .workbuddy/scripts/docling_chunks.py \
    --pdf "docs/knowledge_base/金融行业/贝莱德...pdf" --kb 金融行业 \
    --out-dir .workbuddy/scripts/out/docling_chunks --dump-md

# 2) 入库到独立索引根（不污染生产 app/rag/indexes/）
RAG_INDEX_ROOT=F:/Projects/deepsearch-agents/app/rag/indexes_docling RAG_KB_SUFFIX=__docling \
    .venv/Scripts/python.exe .workbuddy/scripts/ingest_docling.py \
    --kb 金融行业 --pdf "docs/knowledge_base/金融行业/贝莱德...pdf"

# 3) 两臂同口径评测
.venv/Scripts/python.exe -m app.eval.eval_retrieval --mode hybrid_rerank --no-gold
RAG_INDEX_ROOT=... RAG_KB_SUFFIX=__docling \
    .venv/Scripts/python.exe -m app.eval.eval_retrieval --mode hybrid_rerank --no-gold

# 4) 解析层覆盖率对照
.venv/Scripts/python.exe .workbuddy/scripts/anchor_coverage.py            # 原样
.venv/Scripts/python.exe .workbuddy/scripts/anchor_coverage.py --normalize # 归一化
```

| 资产 | 路径 |
|---|---|
| Docling 解析臂 | `.workbuddy/scripts/docling_chunks.py` |
| Docling 入库臂 | `.workbuddy/scripts/ingest_docling.py` |
| 覆盖率诊断 | `.workbuddy/scripts/anchor_coverage.py` |
| 索引隔离开关 | `app/rag/config.py`（`RAG_INDEX_ROOT` / `RAG_KB_SUFFIX` + `backend_kb()`） |
| 同口径评测开关 | `app/eval/eval_retrieval.py`（`--no-gold`） |
| Docling 臂索引 | `app/rag/indexes_docling/`（已 gitignore） |
| 评测报告 | `app/eval/results/2026-10-11_21.30.28__*.md`（v3）/ `..._21.31.19__*.md`（Docling） |
| 覆盖率明细 | `.workbuddy/scripts/out/anchor_coverage_docling_vs_v3.md`、`anchor_coverage_norm.md` |
