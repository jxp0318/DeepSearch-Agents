"""
把 bge-reranker-v2-m3 的 fp32 ONNX 转成 fp16 —— 精排的第二个加速杠杆

**为什么做**：精排是检索链路里最贵的一环（占端到端 P50 的 ~60%）。fp32 的
`model.onnx_data` 有 2.27GB，全部走 fp32 CUDA core；Ada（RTX 4060）上 fp16 的
矩阵乘吞吐远高于 fp32，且权重减半后显存与带宽压力同步下降。

**为什么不直接 optimum-cli 重导**：`optimum-cli export onnx` 没有 fp16 开关，
且重导要重新跑一遍图优化。ORT 自带 `onnxruntime.transformers.float16`（从
onnxconverter-common 移植的版本）可以直接在已有 ONNX 上做权重降精度，**图结构不变**。

**风险与验证**：降精度会改变 logits 的末位数值，理论上分数会微动、排序可能换位。
所以：
- 默认 `--keep-io-types`，输入输出仍是 fp32（`_encode` 传 int64/fp32，无需改动）；
- 转换后必须用同一份题库跑 A/B 对比 Recall@5 / NDCG@5 / Top-5 是否换位——
  见 docs/eval-testset-v3.md 的 fp16 小节；
- 产物写到 `onnx_fp16/`，**不动 fp32 的 `onnx/`**；切换只需
  `RERANK_ONNX_SUBDIR=onnx_fp16`，回滚清空该环境变量即可。

用法::

    python docker/reranker/export_onnx_fp16.py            # 就地导出（默认路径）
    python docker/reranker/export_onnx_fp16.py --check    # 只跑一次数值对比，不写文件
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

MODEL_DIR = Path(__file__).parent / "bge-reranker-v2-m3"


def main() -> None:
    ap = argparse.ArgumentParser(description="bge-reranker-v2-m3 ONNX fp32 → fp16")
    ap.add_argument("--src", default=str(MODEL_DIR / "onnx" / "model.onnx"))
    ap.add_argument("--dst-dir", default=str(MODEL_DIR / "onnx_fp16"))
    ap.add_argument("--check", action="store_true", help="只做数值对比，不写文件")
    ap.add_argument("--max-diff", type=float, default=2e-2,
                    help="可接受的 logits 最大绝对差（超了就别用 fp16）")
    args = ap.parse_args()

    import numpy as np
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    from tokenizers import Tokenizer

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"找不到 fp32 ONNX：{src}")

    print(f"读取 {src}（外部权重在 model.onnx_data，约 2.27GB 需整份读进内存）…")
    # 必须 load_external_data=True：转换器要读权重的实际数值，只留引用会报
    # 「Data of TensorProto ... should be stored in model.onnx_data, but it is not regular file」。
    # 峰值内存 ≈ fp32 权重的 1.5 倍（原份 + fp16 副本），本机 8GB 下能过。
    model = onnx.load(str(src), load_external_data=True)
    fp16_model = convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)
    del model  # 尽早释放 fp32 原份，给下面的保存腾内存
    print("已转换为 fp16（keep_io_types=True：输入输出仍是 fp32，调用侧无需改动）")

    dst_dir = Path(args.dst_dir)
    dst = dst_dir / "model.onnx"
    if not args.check:
        dst_dir.mkdir(parents=True, exist_ok=True)
        # 权重降半后体积约 1.1GB，仍可能超过 protobuf 2GB 上限 ⇒ 一律外置
        onnx.save_model(
            fp16_model, str(dst),
            save_as_external_data=True, all_tensors_to_one_file=True,
            location="model.onnx_data", size_threshold=1024,
        )
        print(f"已写入 {dst}（+ model.onnx_data）")

    # ---- 数值对比：同一批 (query, doc) 对，fp32 与 fp16 的 logits ----
    import onnxruntime as ort

    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tok.enable_truncation(max_length=512)
    tok.no_padding()
    pairs = [
        ("具身智能的核心技术路线有哪些？", "具身智能强调通过物理身体与环境的交互来获取智能，主流路线包括端到端视觉-语言-动作模型。"),
        ("新能源汽车出海面临哪些关税壁垒？", "欧盟对中国电动汽车加征反补贴关税，税率因企业配合程度而不同。"),
        ("2024 年人工智能产业规模是多少？", "报告显示，我国人工智能核心产业规模接近六千亿元。"),
    ]
    encs = tok.encode_batch(pairs)
    width = max(len(e.ids) for e in encs)
    ids = np.zeros((len(encs), width), dtype=np.int64)
    mask = np.zeros((len(encs), width), dtype=np.int64)
    for r, e in enumerate(encs):
        ids[r, : len(e.ids)] = e.ids
        mask[r, : len(e.ids)] = 1
    feed = {"input_ids": ids, "attention_mask": mask}

    # Windows 上 ORT 的 CUDA EP 要求 nvidia-*-cu12 的 DLL 出现在
    # onnxruntime/capi/ 同目录，否则建 session 时会「Invalid handle. Cannot load
    # symbol cudnnCreate」——项目里这套布置在 Reranker._link_cuda_dlls()，这里直接复用，
    # 免得脚本自己建 session 时踩同一个坑。
    from app.rag.reranker import Reranker

    Reranker._link_cuda_dlls()

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # fp16 算子在 CPU EP 上支持不全且极慢，数值对比优先用 CUDA（没有才退回 CPU）
    prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()]
    s32 = ort.InferenceSession(str(src), so, providers=prov)
    logits32 = s32.run(None, feed)[0].reshape(-1)
    print(f"数值对比执行提供器：{s32.get_providers()[0]}")

    if args.check:
        # 只做对比时不落盘：把转换结果存到临时目录再加载
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "model.onnx"
            onnx.save_model(
                fp16_model, str(tmp_path),
                save_as_external_data=True, all_tensors_to_one_file=True,
                location="model.onnx_data", size_threshold=1024,
            )
            s16 = ort.InferenceSession(str(tmp_path), so, providers=prov)
            logits16 = s16.run(None, feed)[0].reshape(-1)
    else:
        s16 = ort.InferenceSession(str(dst), so, providers=prov)
        logits16 = s16.run(None, feed)[0].reshape(-1)

    diff = np.abs(logits32 - logits16)
    print("\nfp32 vs fp16 logits：")
    for (q, _), a, b in zip(pairs, logits32, logits16):
        print(f"  {q[:16]:<18} fp32={a:+.5f}  fp16={b:+.5f}  |Δ|={abs(a - b):.5f}")
    print(f"  最大绝对差 {diff.max():.5f}（阈值 {args.max_diff}）")
    same_order = list(np.argsort(-logits32)) == list(np.argsort(-logits16))
    print(f"  这三条样本内排序是否一致：{'是' if same_order else '否 ← 需要看评测集 A/B'}")
    if diff.max() > args.max_diff:
        print("\n⚠ 数值差超阈值——不要直接切 fp16，先用评测集跑 A/B 再决定。")
        sys.exit(1)
    print("\n数值差在阈值内。仍需用评测集跑 A/B（质量 + 延迟）后再切生产。")


if __name__ == "__main__":
    main()
