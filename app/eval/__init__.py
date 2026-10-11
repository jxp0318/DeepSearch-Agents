"""
评测模块

检索层与端到端评测的代码、题库与判分协议集中于此：

- eval_retrieval.py      检索层量化评测：★Context Recall@5 / ★Context Precision@5(AP@5)
                        / ★MRR@5 + 查询延迟 P50/P95/max，四模式对照
                        （bm25 / keyword / vector / hybrid）
- reporting.py           结果落盘工具：每次运行生成带时间戳的独立文件，不覆盖历史
- retrieval-tasks.yaml   检索层题库（13 道双语题 + 5 道单语题，事实锚点法可程序化判定）
- eval-tasks.yaml        端到端评测集（web / database / knowledge_base / mixed 四类，
                        gold_facts 为描述式、需 LLM judge 判定，属第二阶段）
- eval.md                评测协议设计（指标定义、判分三层、对比实验协议）
- results/               运行产物目录：历次评测结果（{时间戳}__{测试名}__{方式}.md/.json）

执行（在项目根目录）：
    python -m app.eval.eval_retrieval              # 全部题库 × 四模式
    python -m app.eval.eval_retrieval --diag       # 附带 Recall@20 诊断列
    python -m app.eval.eval_retrieval --list       # 只列题库与各库入库状态

尚未实现的第二阶段（生成层）：Faithfulness / Answer Correctness——需要真实生成的答案、
参考答案与 LLM judge，方案见 eval.md 3.4 节。
"""
