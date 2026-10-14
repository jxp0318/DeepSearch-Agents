"""
扫描页 OCR（C4 的实现端）

背景
----
``pdf_layout.detect_scanned_pages`` 只负责**检出**「无文本层 / 图片主导」的页并打
``needs_ocr`` 标记；本模块负责**实现**——把这类页渲染成位图后做 OCR，把原本提取不到
的文字（架构图、信息图、位图表格里的文本）补回检索链路。

典型样本（本项目实测）：

- 具身智能报告 p15：整页一张架构图，PyMuPDF 只提出 38 个字符（页眉+页码），
  图内的「数据/模型/云-边-端服务平台」「LLM/VLM/VLA/世界模型」等全部丢失；
  OCR 后提出 56 行、置信度 0.97~1.00。
- 斯坦福 AI 指数 p152：整页一张位图数据表（国家 × 协议矩阵），同样提不出来。

引擎选型（``rapidocr-onnxruntime``）
-----------------------------------
- 纯 onnx，**复用已有 onnxruntime**，不引入 torch/paddle 这类重依赖；
- 模型随 wheel 内置，**首次运行不联网**（无 HF/GitHub 下载风险，离线可用）；
- 中文识别质量好、中英混排开箱即用；CPU 单页 ~3s（150dpi），够用；
- Windows 无编译依赖。

设计要点
--------
1. **懒加载 + 可降级**：引擎只在第一次真的用到时初始化；未安装 ``rapidocr-onnxruntime``
   时 ``available()`` 返回 False，调用方按「无 OCR」继续，绝不因缺依赖整体失败。
2. **按页渲染 → OCR**：走 PyMuPDF 的 ``get_pixmap`` 出 PNG 字节直接喂引擎；dpi 可调
   （``OCR_DPI``），默认 150（A4 → 1240×1754，是识别率与耗时的折中）。
3. **低分过滤**：低于 ``OCR_MIN_SCORE`` 的行丢弃，避免水印/装饰噪声进索引。
4. **输出纯文本**：扫描页本就是「一页位图」，没有可靠的行/列结构可复用，故不入
   ``pdf_layout`` 的行管线，而是由 ``pdf_parser`` **整页作为一个 ``chunk_type="ocr"``
   的块**产出。
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.rag import config

logger = logging.getLogger(__name__)

_ENGINE = None  # 缓存的引擎实例
_ENGINE_TRIED = False  # 是否已尝试初始化（失败也记，避免反复重试）


def _load_engine():
    """懒加载 OCR 引擎；不可用时返回 None（只尝试一次）"""
    global _ENGINE, _ENGINE_TRIED
    if _ENGINE is not None:
        return _ENGINE
    if _ENGINE_TRIED:
        return None
    _ENGINE_TRIED = True
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception as exc:  # noqa: BLE001 —— 缺依赖时静默降级
        logger.info("OCR 引擎不可用（未安装 rapidocr-onnxruntime）：%s", exc)
        return None
    try:
        _ENGINE = RapidOCR()
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR 引擎初始化失败：%s", exc)
        return None
    return _ENGINE


def available() -> bool:
    """引擎是否可用（已安装且能初始化）"""
    return _load_engine() is not None


def ocr_image(image) -> list[tuple[list, str, float]]:
    """
    对一张图片做 OCR

    :param image: ``bytes``（PNG/JPEG）、文件路径或 ``pathlib.Path``
    :return: ``[(box, text, score), ...]``，按引擎给出的阅读顺序；引擎不可用或识别
             失败时返回空列表（调用方据此降级）。
    """
    engine = _load_engine()
    if engine is None:
        return []
    try:
        result, _ = engine(image)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR 识别失败：%s", exc)
        return []
    return list(result or [])


def ocr_page_text(image, *, min_score: float | None = None) -> str:
    """OCR 并只保留置信度 ≥ ``min_score`` 的行，返回换行拼接的纯文本"""
    threshold = config.OCR_MIN_SCORE if min_score is None else min_score
    lines = [text for _box, text, score in ocr_image(image) if float(score) >= threshold]
    return "\n".join(lines)


def ocr_document(
    pdf_path: Path | str,
    pages: list[int],
    *,
    dpi: int | None = None,
) -> dict[int, str]:
    """
    对指定页（1-based）批量 OCR

    :return: ``{page_index: text}``；引擎不可用、页号越界或识别为空时该页不出现在结果里。
    """
    if not pages or _load_engine() is None:
        return {}
    import pymupdf

    render_dpi = config.OCR_DPI if dpi is None else dpi
    out: dict[int, str] = {}
    document = pymupdf.open(str(pdf_path))
    try:
        for page_no in pages:
            if page_no < 1 or page_no > document.page_count:
                continue
            pix = document[page_no - 1].get_pixmap(dpi=render_dpi)
            text = ocr_page_text(pix.tobytes("png"))
            if text.strip():
                out[page_no] = text
    finally:
        document.close()
    return out
