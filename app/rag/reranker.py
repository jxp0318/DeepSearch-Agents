"""
精排模块（cross-encoder，进程内 ONNX 推理）——见 ADR-012

**为什么需要它**：RRF 融合只看得票名次、丢掉了原始分值，双路都命中的块必赢单路
rank-1，跨语言时还会被「假公共 token」污染。离线评测已量化这个缺口——候选池
（depth 15，池约 21 条）本身覆盖 85.2% 的锚点，而 RRF 的输出 Recall@5 只有 59.3%，
也就是**约 26 个百分点的锚点躺在池子里没被排上来**。cross-encoder 把 query 与
候选拼成一条序列逐对打分，正好吃这块「排序缺口」：实测 Recall@5 59.3% → 84.3%，
吃满候选池上限的 99%。

**为什么是进程内而不是 TEI 容器**：TEI 那条路有两个与模型无关的遗留问题——
`queue_time` 排队（4 条批请求要在服务端干等约 1s）与 candle/ORT 后端的
`queue.rs:87` 并发 permit off-by-one（SIGSEGV 崩溃，且会拖垮 WSL2 写 ~3GB 转储）。
进程内直调不经过 TEI，**这两个问题连同代码路径一起消失**。

**设计要点**
- 推理走 `onnxruntime`，ONNX 模型由 `optimum-cli` 从 `model.safetensors` 导出，
  与原始权重数值一致（fp32 下 logits 最大绝对差 0.003）。
- 执行提供器：CUDA 优先、CPU 兜底（见 config.RERANK_ENABLED 三态开关）。
- 分词用 `tokenizers` 直接读 `tokenizer.json`——与 `transformers.AutoTokenizer`
  编码逐位一致（pad id 取 `<pad>`），但导入 0.03s vs 14.8s，不值得为一行便利拖
  一个 transformers+torch 进来。
- **批量推理 + 按长度排序切批**：候选一次性编码后**先按序列长度升序排再按
  `RERANK_BATCH_SIZE` 切批**，每批一次 `session.run`（不是 Python 循环逐篇调用）。
  排序的理由是 padding：`_encode` 把一批补齐到该批最长，而 cross-encoder 的注意力
  是 O(L²)，长短混排（900 字符表格块 + 200 字符正文块）时大批会把短块一起抬长——
  实测 K=25 下 bs=25 比 bs=4 慢 1.5×。排序**不改变任何分数与排名**，只换批内邻居。
- **懒加载**：模型 2.2GB，只有第一次真正调用精排时才建 session。
- **可降级**：模型缺失、依赖未装、CUDA 不可用或推理抛错，`rerank()` 一律返回
  `None`，调用方（Retriever）保持 RRF 原序并告警一次——检索永远不会因为精排而失败。
"""

import importlib.util
import logging
import os
import sys
import threading
from pathlib import Path

import numpy as np

from app.rag import config

logger = logging.getLogger(__name__)


class Reranker:
    """cross-encoder 精排器（进程内 ONNX 推理，模块级单例）"""

    def __init__(self) -> None:
        self._session = None
        self._tokenizer = None
        self._pad_id = 0
        self._provider = ""
        self._loaded = False
        self._load_failed = False
        self._fail_reason = ""
        self._lock = threading.Lock()

    # ---------- 路径与可用性（都不建 session，零成本） ----------

    @property
    def model_dir(self) -> Path:
        return config.RERANK_MODEL_DIR

    @property
    def onnx_path(self) -> Path:
        """ONNX 权重入口（optimum-cli 导出产物；大权重在旁边的 model.onnx_data）

        子目录由 ``config.rerank_onnx_subdir()`` 决定：``auto``（默认）在存在
        ``onnx_fp16/`` 时优先半精度（快一倍、指标一致），否则回退 ``onnx/``（fp32）。
        """
        return self.model_dir / config.rerank_onnx_subdir() / "model.onnx"

    def _precheck_fail(self) -> str:
        """不建 session 的前置检查，返回失败原因（空串 = 通过）"""
        if config.RERANK_DISABLED:
            return "已被 RERANK_ENABLED=0 关闭"
        if not self.onnx_path.exists():
            return f"ONNX 模型不存在：{self.onnx_path}"
        if not (self.model_dir / "tokenizer.json").exists():
            return f"tokenizer.json 不存在：{self.model_dir}"
        for mod in ("onnxruntime", "tokenizers"):
            if importlib.util.find_spec(mod) is None:
                return f"依赖未安装：{mod}"
        return ""

    @property
    def available(self) -> bool:
        """
        精排当前是否可用（**不做加载**，只看配置 / 文件 / 依赖是否齐）

        注意它不保证 CUDA 真能起来——`RERANK_ENABLED=auto` 下这一点要等第一次
        调用建 session 才知道。真正的判据是 `rerank()` 的返回值。
        """
        if self._loaded:
            return True
        if self._load_failed:
            return False
        return not self._precheck_fail()

    @property
    def provider(self) -> str:
        """已加载时返回实际生效的执行提供器（空串 = 尚未加载）"""
        return self._provider

    def status(self) -> str:
        """一行状态，供评测 / 日志打印（不会触发加载）"""
        if self._loaded:
            return f"已加载（{self._provider}）"
        if self._load_failed:
            return f"不可用（{self._fail_reason}）"
        reason = self._precheck_fail()
        if reason:
            return f"不可用（{reason}）"
        if config.RERANK_FORCED:
            return "就绪（未加载，强制模式：允许 CPU）"
        return "就绪（未加载，auto 模式：要求 CUDA 执行提供器）"

    def warmup(self) -> bool:
        """
        提前建 session 并跑一次哑推理

        模型加载（2.2GB）与 CUDA 首次 kernel 编译都在第一次前向里，合计数秒。评测计时
        前调一次，把这两笔一次性开销从首条查询的延迟里摘出去。
        """
        if not self._ensure_loaded():
            return False
        try:
            names = {i.name for i in self._session.get_inputs()}
            encs = self._tokenizer.encode_batch([("预热", "预热")])
            self._session.run(None, self._encode(encs, [0], names))
        except Exception as e:  # noqa: BLE001 —— 哑推理只为预热，失败不影响可用性
            logger.debug("精排预热推理失败（不影响可用性）：%s", e)
        return True

    # ---------- 加载 ----------

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._loaded:
                return True
            if self._load_failed:
                return False
            # 前置检查（配置关闭 / 模型缺失 / 依赖未装）是**预期内的正常状态**，直接
            # 落定原因、用 info 记录；只有真正的加载异常才走 warning。
            reason = self._precheck_fail()
            if reason:
                self._load_failed = True
                self._fail_reason = reason
                logger.info("精排未启用：%s", reason)
                return False
            try:
                self._load()
            except Exception as e:  # noqa: BLE001 —— 任何加载失败都只降级，不抛给调用方
                self._load_failed = True
                self._fail_reason = f"{type(e).__name__}: {e}"
                logger.warning("精排加载失败，检索将保持 RRF 原序：%s", self._fail_reason)
        return self._loaded

    def _load(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        ort.set_default_logger_severity(3)
        if sys.platform == "win32":
            self._link_cuda_dlls()

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = self._provider_chain()
        session = ort.InferenceSession(str(self.onnx_path), so, providers=providers)
        active = session.get_providers()

        # auto 模式要求 CUDA 真正生效：`get_available_providers()` 只说明 wheel
        # 编译时带了这个 EP，运行期缺 DLL 时 ORT 会**静默 fallback 到 CPU**——
        # 那样精排会从毫秒级掉到每候选数百毫秒，必须在启用前拦下。
        if not config.RERANK_FORCED and (not active or active[0] != "CUDAExecutionProvider"):
            raise RuntimeError(
                f"auto 模式要求 CUDA 执行提供器，实际生效 {active}；"
                "无 GPU 环境请显式设 RERANK_ENABLED=1（允许 CPU，但每查询会到秒级）"
            )

        tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=config.RERANK_MAX_LENGTH)
        tokenizer.no_padding()
        pad_id = tokenizer.token_to_id("<pad>")
        if pad_id is None:
            pad_id = tokenizer.token_to_id("</s>") or 0

        self._session = session
        self._tokenizer = tokenizer
        self._pad_id = int(pad_id)
        self._provider = active[0] if active else "?"
        self._loaded = True
        logger.info("精排就绪：provider=%s，权重=%s",
                    self._provider, f"{config.rerank_onnx_subdir()}/model.onnx")

    @staticmethod
    def _provider_chain() -> list[str]:
        """按配置给出执行提供器优先级（ORT 会把 CPU EP 作为最终兜底自动追加）"""
        if config.RERANK_PROVIDER == "cpu":
            return ["CPUExecutionProvider"]
        if config.RERANK_PROVIDER == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # auto：CUDA 优先，CPU 兜底；是否接受 CPU 结果由 RERANK_ENABLED 决定
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @staticmethod
    def _link_cuda_dlls() -> None:
        """
        让 ORT 在 Windows 上找到 pip 安装的 CUDA 运行库（nvidia-*-cu12）

        Windows 的坑：ORT 用 `LOAD_WITH_ALTERED_SEARCH_PATH` 加载
        `onnxruntime_providers_cuda.dll`，它的依赖只在「该 DLL 自身所在目录」+
        系统目录 + PATH 里解析；`os.add_dll_directory()` 登记的用户目录属于
        `LOAD_LIBRARY_SEARCH_USER_DIRS` 那条路，**不在**这个搜索范围内——实测单靠
        它无效，ORT 会静默 fallback 到 CPU。

        可靠解：把依赖 DLL **硬链接**到 `onnxruntime/capi/` 同目录。同盘硬链接零拷贝、
        不占额外空间，必然命中「被加载 DLL 所在目录」这条规则。重装 onnxruntime 会
        冲掉链接，此函数每次加载都会把缺失的补回来。

        另一个坑：最新 `onnxruntime-gpu` 要 CUDA 13，而 `nvidia-cublas`（cu13）在
        PyPI 上没有 Windows wheel → 必须用 1.22.0（CUDA 12 世代）+ `nvidia-*-cu12`。
        """
        import onnxruntime

        try:
            capi_dir = Path(onnxruntime.__file__).parent / "capi"
            site_packages = Path(sys.prefix) / "Lib" / "site-packages"
            bin_dirs = [
                str(p)
                for p in sorted((site_packages / "nvidia").glob("*/bin"))
                if p.is_dir()
            ]
            for d in bin_dirs:
                try:
                    os.add_dll_directory(d)
                except OSError:
                    pass
            if bin_dirs:
                os.environ["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + os.environ.get(
                    "PATH", ""
                )
            for src in sorted((site_packages / "nvidia").glob("*/bin/*.dll")):
                dst = capi_dir / src.name
                if dst.exists():
                    continue
                try:
                    os.link(src, dst)
                except OSError:
                    pass  # 链接失败不致命——若 DLL 已在 PATH 上 ORT 仍能找到
        except Exception as e:  # noqa: BLE001 —— DLL 布置失败留给建 session 时报错
            logger.debug("CUDA DLL 布置跳过：%s", e)

    # ---------- 打分 ----------

    def rerank(
        self,
        query: str,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[tuple[int, float]] | None:
        """
        对候选项逐对打分并按分数降序返回

        :param query: 用户查询
        :param texts: 候选 chunk 原文列表（顺序即池内下标）
        :param batch_size: 单次前向的候选数，缺省 config.RERANK_BATCH_SIZE
        :return: [(池内下标, 归一化分数∈(0,1))] 按分数降序；**不可用或失败时返回 None**
                 （调用方据此保持原序，精排永远不能让检索失败）
        """
        if not texts:
            return []
        if not self._ensure_loaded():
            return None

        bs = batch_size or config.RERANK_BATCH_SIZE
        scores = np.empty(len(texts), dtype=np.float32)
        input_names = {i.name for i in self._session.get_inputs()}

        try:
            # 先整体编码一次，再按**长度升序**分批（见 _batch_plan 的说明）。
            encodings = self._tokenizer.encode_batch([(query, t) for t in texts])
            for idxs in self._batch_plan(encodings, bs):
                feed = self._encode(encodings, idxs, input_names)
                logits = self._session.run(None, feed)[0]
                scores[idxs] = self._sigmoid(logits).reshape(-1)
        except Exception as e:  # noqa: BLE001 —— 单次推理失败同样降级
            if not self._load_failed:
                logger.warning("精排推理失败，本次查询保持 RRF 原序：%s: %s", type(e).__name__, e)
                self._load_failed = True
                self._fail_reason = f"推理失败 {type(e).__name__}"
            return None

        order = np.argsort(scores)[::-1]
        return [(int(i), float(scores[i])) for i in order]

    @staticmethod
    def _batch_plan(encodings: list, bs: int) -> list[list[int]]:
        """
        分批方案：**先按序列长度升序排，再按 bs 切**（返回每批的原下标）

        为什么必须按长度排：`_encode` 会把一批里所有序列**补齐到该批最长**，而
        cross-encoder 是 transformer，单序列注意力是 O(L²)——padding 不是免费的。
        原下标顺序切批时，一条 260 token 的长块能把同批其他 7 条短块一起抬到 260；
        候选池恰好是「长短混杂」（表格块 900+ 字符 / 正文块 200 字符），实测
        K=25 下 bs=25（一批补齐全部）比 bs=4（每批各自补齐）慢 **1.5×**
        （492ms vs 330ms，RTX 4060 Laptop，见 app/eval/bench_rerank.py）。
        排序后每批内的长度高度接近，padding 浪费被压到最小；**分数与排名不受影响**
        （逐对打分，与计算顺序无关），只是同一批里的邻居换了。

        注意它**不改变 batch 数**（仍是 ceil(n/bs) 批）——收益全部来自 padding 减少。
        """
        if config.RERANK_LENGTH_SORTED:
            order = sorted(range(len(encodings)), key=lambda i: len(encodings[i].ids))
        else:
            order = list(range(len(encodings)))
        return [order[i : i + bs] for i in range(0, len(order), bs)]

    def _encode(self, encodings: list, idxs: list[int], input_names: set[str]) -> dict:
        """把选中的 (query, chunk) 对排成规整矩阵（按本批最长 padding）"""
        picked = [encodings[i] for i in idxs]
        width = max(len(e.ids) for e in picked)
        input_ids = np.full((len(picked), width), self._pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(picked), width), dtype=np.int64)
        for row, enc in enumerate(picked):
            n = len(enc.ids)
            input_ids[row, :n] = enc.ids
            attention_mask[row, :n] = 1
        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        return {k: v for k, v in feed.items() if k in input_names}

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        """二分类 logit → (0,1) 分数；与 TEI /rerank 的输出口径一致"""
        x = np.asarray(logits, dtype=np.float32).reshape(-1)
        return 1.0 / (1.0 + np.exp(-x))


# 模块级单例：模型与 session 全进程复用一份
reranker = Reranker()
