"""
评测结果落盘：每次运行生成独立的指标文件，绝不覆盖历史，且文件名可辨识测试方式。

命名约定：

    {时间戳}__{测试名}__{方式}.{扩展名}

例：
    2026-10-09_12.59.33__retrieval__bm25+keyword+vector+hybrid.md
    2026-10-09_12.59.33__retrieval__bm25+keyword+vector+hybrid.json

- 时间戳：%Y-%m-%d_%H.%M.%S，秒级精度——同一分钟内重复跑也不会互相覆盖
- 测试名：retrieval（检索层量化评测）/ compare（四模式定性对比）等
- 方式  ：本次运行覆盖的检索模式，多个用 + 连接
- 每种运行同时写 .md（人读报告）与 .json（机器读指标），两者同前缀、同时间戳

落盘目录：app/eval/results/（脚本自动创建）。该目录是运行产物，是否纳入版本控制
由使用者决定（默认未写入 .gitignore）。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def timestamp(dt: datetime | None = None) -> str:
    """文件名安全的时间戳，例：2026-10-09_12.59.33

    传入 dt 可让文件名与报告内的时间取自同一时刻，避免两处时间不一致。
    """
    return (dt if dt is not None else datetime.now()).strftime("%Y-%m-%d_%H.%M.%S")


def _stem(kind: str, modes, ts: str) -> str:
    """不含扩展名的文件名主干：{时间戳}__{测试名}__{方式}"""
    label = modes if isinstance(modes, str) else "+".join(modes)
    return f"{ts}__{kind}__{label}"


def save_run(kind: str, modes, markdown: str, payload: dict, ts: str | None = None) -> list[Path]:
    """把一次运行的结果写成 {stem}.md 与 {stem}.json，返回写出的路径列表。

    文件名带秒级时间戳，天然不覆盖既有文件；极端情况下同秒撞名时追加 -2、-3 后缀，
    保证既有文件永不被覆盖。
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = _stem(kind, modes, ts if ts is not None else timestamp())
    written: list[Path] = []
    for ext, content in (
        ("md", markdown),
        ("json", json.dumps(payload, ensure_ascii=False, indent=2)),
    ):
        path = RESULTS_DIR / f"{stem}.{ext}"
        n = 2
        while path.exists():
            path = RESULTS_DIR / f"{stem}-{n}.{ext}"
            n += 1
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written
