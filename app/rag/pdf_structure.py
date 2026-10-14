"""
PDF 结构识别：④ 标题（多信号 + 置信度）与 ⑤ 表格（独立成块）

本模块建立在 ``app/rag/pdf_layout.py`` 的几何层之上（行 / 字号 / 粗体 / 坐标 +
页眉页脚剔除结果），**只做结构判定，不做切块**——切块在 ``app/rag/pdf_parser.py``。

④ 标题：为什么不能只靠字号
---------------------------
字号只是弱线索：大字号可能是**报告大标题、图表标题或页眉**；真正的章节标题也
可能**与正文同字号、仅加粗**。故采用多信号加权：

| 信号 | 判据 |
|---|---|
| 编号模式 | ``1.`` ``1.1`` ``第一章`` ``Appendix A`` ``PART III`` ``主题3`` |
| 字号偏离 | 相对**文档正文众数字号**的比值（非绝对 pt） |
| 字体样式 | 加粗 / 全大写（全大写对中文无效，按文档语言启用） |
| 位置特征 | 独占一行、上下留白大于正文行距、页首 |
| 跨页重复 | **反向证据**：同一文本在多页反复出现 ⇒ 是页眉，不是标题 |

编号是**最强的单条线索**（比字号可靠），因而权重最高；图表标题（``Figure 1`` /
``表 3``）带**惩罚**——它们字号经常很大，但属于图表附件而非章节。
输出带 ``score`` 与 ``confidence``：只有 high/medium 进入 ``section_path``，
**low 候选全部保留**供人工抽查（宁可在候选表里多留，也不在层级里错判）。

⑤ 表格：为什么要独立成块
-------------------------
表格混进正文流会同时毁掉两边（正文被行列碎片打断、表格失去表头归属）。故表格
独立成块，且必须还原**表头 / 行列关系 / 单位 / 表格标题**——单位与表头是表格
检索的命门（``%）`` ``万辆`` ``GWh`` 这类量纲往往只在表头或标题里出现一次）。

检测用 PyMuPDF 的几何表格检测（``lines`` 策略）；**结构化是本模块自己做的**：
标题剥离（表格标题常被检测器并进首行）、多行表头判定、单位抽取、跨页表拼接、
超长表分块时重复表头。``text`` 策略只在页面以表格为主体时启用——实测在文本密集
页上它会把整页凑成一张 62×8 的假表，故加了「填充率 + 面积」过滤。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.rag.pdf_layout import FurnitureAnalysis, Line, PageLayout

# =============================================================================
# ④ 标题识别
# =============================================================================

# 权重之和为 1；编号权重最高（C1：编号比字号更可靠）
H_W_NUMBER = 0.34
H_W_SIZE = 0.26
H_W_BOLD = 0.18
H_W_ISOLATED = 0.14
H_W_CAPS = 0.08

H_SIZE_SPAN = 0.60  # 字号比值达到 正文×1.6 时，字号信号记满分
H_SCORE_HIGH = 0.60
H_SCORE_MEDIUM = 0.42
H_REPEAT_MIN_PAGES = 3  # 跨页重复页数达到此值即视作页眉（强抑制）
H_CAPTION_PENALTY = 0.35  # 图表标题惩罚系数
H_REPEAT_PENALTY = 0.15  # 跨页重复惩罚系数
H_PARAGRAPH_PENALTY = 0.25  # 段落续行惩罚系数
H_SENTENCE_PENALTY = 0.35  # 「读起来是句子」惩罚系数
H_DIGIT_PENALTY = 0.50  # 数字占比高的短行惩罚系数（图表数值标签）
H_LONG_LINE_PENALTY = 0.30  # 超长行惩罚系数（标题不会很长）
H_LONG_LINE_CHARS = 120
H_NUMBER_MIN_SIZE_RATIO = 0.95  # 编号线索要求字号不低于正文（挡脚注编号）
H_NUMBER_LOOSE_MAX_CHARS = 60  # 「数字 + 空格」形式只认短行
H_NUMBER_MAX_CHARS = 90  # 任何编号形式的行都不该超过这个长度（挡「Appendix A provides ...」）
H_LEVEL1_RATIO = 1.45  # 无编号标题定级：相对正文字号的比值带
H_LEVEL2_RATIO = 1.25
H_MAX_HEADER_ROWS = 3
H_MAX_LEVEL = 3

# 编号模式族：``(名称, 正则, 固定层级或 None)``
_NUMBER_PATTERNS: list[tuple[str, "re.Pattern[str]", int | None]] = [
    # 数字编号必须「编号 + 分隔符」且分隔符后紧跟文字；宽松形式（数字 + 空格）
    # 另设短行约束。两条例外都来自实测：ICCT 图表页的 ``2015 2016 2017 …``、
    # ``10.0``、``0 250,000`` 与脚注 ``1 Anh Bui and Peter Slowik, ...`` 都曾被
    # 单条宽松正则误判成「编号标题」。
    ("decimal", re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\s*[\.、．)）]"), None),
    ("decimal_loose", re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\s+(?=[A-Z\u4e00-\u9fff])"), None),
    ("cn_chapter", re.compile(r"^\s*第\s*[一二三四五六七八九十百零〇\d]+\s*[章篇部]"), 1),
    ("cn_section", re.compile(r"^\s*第\s*[一二三四五六七八九十百零〇\d]+\s*[节]"), 2),
    ("appendix", re.compile(r"^\s*(?:Appendix|APPENDIX|Annex|ANNEX)\s+[A-Z0-9]\b"), 1),
    ("part", re.compile(r"^\s*(?:PART|Part)\s+(?:[IVXLC]+|\d+)\b"), 1),
    ("cn_num", re.compile(r"^\s*[一二三四五六七八九十]+\s*[、.．]"), 1),
    ("topic", re.compile(r"^\s*主题\s*[一二三四五六七八九十\d]+"), 1),
    ("chapter_word", re.compile(r"^\s*(?:Chapter|CHAPTER|Section|SECTION)\s+\d+\b"), 1),
]

# 图表标题：字号往往很大，但不是章节标题 → 惩罚（C1 明确点名的反例）
_CAPTION_PATTERNS: list["re.Pattern[str]"] = [
    re.compile(r"^\s*(?:Figure|FIGURE|Fig\.|Table|TABLE|Exhibit|EXHIBIT|Box|BOX)\s*\d+"),
    re.compile(r"^\s*(?:图|表)\s*\d+"),
    re.compile(r"^\s*(?:Source|Sources|Note|Notes)\s*[:：]", re.I),
]

_LATIN_RE = re.compile(r"[A-Za-z]")
_UPPER_RE = re.compile(r"[A-Z]")


@dataclass
class HeadingCandidate:
    """一条标题候选（含全部信号与置信度；low 也要落盘）"""

    page_index: int
    order: int
    text: str
    score: float
    confidence: str
    level: int | None
    size: float
    bold: bool
    numbered: bool
    number_depth: int | None
    x0: float = 0.0
    y_center: float = 0.0
    signals: dict = field(default_factory=dict)

    def as_dict(self, doc: str = "") -> dict:
        return {
            "doc": doc,
            "page_index": self.page_index,
            "order": self.order,
            "text": self.text,
            "score": round(self.score, 3),
            "confidence": self.confidence,
            "level": self.level,
            "signals": self.signals,
        }


@dataclass
class HeadingResult:
    body_size: float = 0.0
    candidates: list[HeadingCandidate] = field(default_factory=list)
    headings: list[HeadingCandidate] = field(default_factory=list)  # 已定级（high/medium）
    jumps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def by_page(self) -> dict[int, list[HeadingCandidate]]:
        out: dict[int, list[HeadingCandidate]] = {}
        for item in self.headings:
            out.setdefault(item.page_index, []).append(item)
        return out


def _weighted_mode(values: list[tuple[float, int]]) -> float:
    """按权重（字符数）取众数；值先归到 0.5pt 一档"""
    if not values:
        return 0.0
    buckets: dict[float, int] = {}
    for value, weight in values:
        key = round(value * 2) / 2
        buckets[key] = buckets.get(key, 0) + weight
    return max(buckets.items(), key=lambda kv: kv[1])[0]


def _is_word_start(ch: str) -> bool:
    """编号分隔符之后必须紧跟文字（字母或汉字），否则不是标题编号"""
    return ch.isalpha() or "\u4e00" <= ch <= "\u9fff"


def _match_number(text: str) -> tuple[str, int | None] | None:
    """
    匹配编号模式；编号层级由点分层数决定（``1.1.1`` → 3）

    ``decimal``（数字 + 分隔符）之后必须**紧跟文字**——``10.0`` 这种图表坐标会被
    这一步挡掉（分隔符后是数字而非文字）。
    """
    for name, pattern, fixed in _NUMBER_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        if not name.startswith("decimal"):
            return name, fixed
        rest = text[match.end():].lstrip(" .、．)）")
        if not rest or not _is_word_start(rest[0]):
            continue
        token = match.group(1)
        return name, min(H_MAX_LEVEL, token.count(".") + 1)
    return None


def _in_paragraph_block(siblings: list[Line], index: int, line: Line) -> bool:
    """
    判断该行是否是**两端对齐段落中的续行**

    判据：与相邻行**右边界严格对齐**（±2pt）+ 左边界接近（±24pt，允许悬挂缩进）
    + 字号相同（±0.6pt）。两端对齐的正文段落每行右边界都对齐，这是它与「标题栈」
    的区别——标题各行长短不一，右边界不会一致。

    这是**辅助信号**（主力是 ``_looks_like_sentence``）：早期版本用「左边界一致
    + 字号相同」判定，结果把 ICCT 的两行大标题（``SPECIFICATIONS OF EVS SOLD BY``
    / ``MAJOR AUTOMAKERS``，左对齐且同字号）也当成了段落续行，故收紧到右边界。
    """
    for other_index in (index - 1, index + 1):
        if not (0 <= other_index < len(siblings)):
            continue
        other = siblings[other_index]
        if (
            abs(other.x1 - line.x1) <= 2.0
            and abs(other.x0 - line.x0) <= 24.0
            and abs(other.size - line.size) <= 0.6
        ):
            return True
    return False


def _looks_like_sentence(text: str) -> bool:
    """
    判断一行文字是否**读起来是句子**（而非标题短语）

    三类：中文逗号（标题几乎不用逗号）、以句末标点结尾的长行、以小写字母开头
    （英文句子的续行）。中文报告里的正文断行最容易被误判成标题，这是主要护栏。
    """
    stripped = text.rstrip()
    if "，" in stripped or "；" in stripped:
        return True
    if stripped.endswith(("。", "．")):
        return True
    if len(stripped) > 12 and stripped.endswith((".", ";", "；")):
        return True
    return bool(stripped[:1].islower())


def _is_caption_like(text: str) -> bool:
    return any(pattern.match(text) for pattern in _CAPTION_PATTERNS)


def _median_line_gap(layouts: list[PageLayout]) -> float:
    """全文档正文行距中位数（用于「上下留白是否异常」）"""
    gaps: list[float] = []
    for layout in layouts:
        lines = [ln for ln in layout.lines if ln.text.strip()]
        for prev, cur in zip(lines, lines[1:]):
            gap = cur.bbox[1] - prev.bbox[3]
            if gap > 0:
                gaps.append(gap)
    if not gaps:
        return 0.0
    gaps.sort()
    return gaps[len(gaps) // 2]


def detect_headings(
    layouts: list[PageLayout],
    furniture: FurnitureAnalysis | None = None,
    *,
    doc_lang: str = "",
) -> HeadingResult:
    """
    多信号识别章节标题并按层级排序

    :param layouts: **已剔除页眉页脚**的逐页版面（``pdf_layout.clean_pages`` 的输出）
    :param furniture: 页眉页脚分析结果，用于「跨页重复 ⇒ 不是标题」的反向证据
    :param doc_lang: 文档级语言（``en`` 时才启用「全大写」信号）
    """
    result = HeadingResult()
    if not layouts:
        result.notes.append("no_pages")
        return result

    content: list[tuple[int, int, Line]] = []  # (page, 行内序号, line)
    for layout in layouts:
        for index, line in enumerate([ln for ln in layout.lines if ln.text.strip()]):
            content.append((layout.page, index, line))
    if not content:
        result.notes.append("no_content")
        return result

    result.body_size = _weighted_mode(
        [(line.size, len(line.text)) for _, _, line in content]
    )
    median_gap = _median_line_gap(layouts)
    appearances = furniture.appearances if furniture is not None else {}
    heights = {layout.page: layout.height for layout in layouts}

    # 每页的内容宽度（相对宽度信号用）
    page_lines: dict[int, list[Line]] = {}
    for layout in layouts:
        page_lines[layout.page] = [ln for ln in layout.lines if ln.text.strip()]

    raw: list[tuple[int, int, Line, dict, float, tuple | None]] = []
    for _order, (page, index, line) in enumerate(content):
        text = line.text.strip()
        size_ratio = line.size / result.body_size if result.body_size else 1.0

        numbered = _match_number(text)
        if numbered is not None:
            # 编号是强线索，但要过三道护栏：
            # ① 字号不得明显小于正文——挡掉脚注「1 Anh Bui and ...」（7pt vs 正文 9pt）；
            # ② 行不能太长 —— 挡掉「Appendix A provides a detailed breakdown of ...」；
            # ③ 「数字 + 空格」的宽松形式还须是短行且无句末标点。
            if size_ratio < H_NUMBER_MIN_SIZE_RATIO or len(text) > H_NUMBER_MAX_CHARS:
                numbered = None
            elif numbered[0] == "decimal_loose" and (
                len(text) > H_NUMBER_LOOSE_MAX_CHARS or text.rstrip().endswith((".", "。"))
            ):
                numbered = None

        caps_ratio = 0.0
        letters = _LATIN_RE.findall(text)
        if letters:
            caps_ratio = len(_UPPER_RE.findall(text)) / len(letters)

        siblings = page_lines.get(page, [])
        width = line.x1 - line.x0
        page_width = max((ln.x1 for ln in siblings), default=0.0) - min(
            (ln.x0 for ln in siblings), default=0.0
        )
        above = line.bbox[1] - siblings[index - 1].bbox[3] if index > 0 else 999.0
        below = (
            siblings[index + 1].bbox[1] - line.bbox[3]
            if index + 1 < len(siblings)
            else 999.0
        )
        in_paragraph = _in_paragraph_block(siblings, index, line)
        isolated = (
            (not page_width or width <= 0.80 * page_width)
            and not in_paragraph
            and median_gap > 0
            and (above > 1.5 * median_gap or below > 1.5 * median_gap)
        )
        repeat_pages = appearances.get(re.sub(r"\d+", "#", text.lower()), 0)
        sentence_like = _looks_like_sentence(text)
        digit_heavy = (
            sum(ch.isdigit() for ch in text) / max(1, len(text)) >= 0.25
            and len(text) <= 16
        )

        signals = {
            "size_ratio": round(size_ratio, 3),
            "bold": line.bold,
            "numbered": bool(numbered),
            "number_kind": numbered[0] if numbered else "",
            "caps_ratio": round(caps_ratio, 3),
            "isolated": bool(isolated),
            "in_paragraph": bool(in_paragraph),
            "sentence_like": sentence_like,
            "digit_heavy": digit_heavy,
            "caption_like": _is_caption_like(text),
            "repeat_pages": repeat_pages,
            "page_top": index == 0,
        }

        score = (
            H_W_NUMBER * (1.0 if numbered else 0.0)
            + H_W_SIZE * max(0.0, min(1.0, (size_ratio - 1.0) / H_SIZE_SPAN))
            + H_W_BOLD * (1.0 if line.bold else 0.0)
            + H_W_ISOLATED * (1.0 if isolated else 0.0)
            + H_W_CAPS
            * (1.0 if (doc_lang == "en" and caps_ratio >= 0.80 and len(letters) >= 3) else 0.0)
        )
        if signals["caption_like"]:
            score *= H_CAPTION_PENALTY
        if in_paragraph:
            score *= H_PARAGRAPH_PENALTY
        if sentence_like:
            score *= H_SENTENCE_PENALTY
        if digit_heavy:
            score *= H_DIGIT_PENALTY
        if len(text) > H_LONG_LINE_CHARS:
            score *= H_LONG_LINE_PENALTY
        if repeat_pages >= H_REPEAT_MIN_PAGES:
            score *= H_REPEAT_PENALTY
        raw.append((page, index, line, signals, score, numbered))

    # ---- 定级：有编号按编号层级；无编号按「相对正文字号的比值」分档 ----
    #
    # 不用「字号分档聚类」：报告里的递进字号（23.3 / 23.2 / 22.9 …）会把每一档都
    # 拆成一个层级，结果几乎所有标题都落到 L3。改用相对正文的固定比值带，含义
    # 稳定且跨文档可解释：正文的 1.45 倍以上为一级，1.25 倍以上为二级。
    candidates: list[HeadingCandidate] = []
    previous_level = 0
    for order, (page, _index, line, signals, score, numbered) in enumerate(raw):
        text = line.text.strip()
        if score >= H_SCORE_HIGH:
            confidence = "high"
        elif score >= H_SCORE_MEDIUM:
            confidence = "medium"
        else:
            confidence = "low"

        level: int | None = None
        if confidence != "low":
            if numbered is not None and numbered[1] is not None:
                level = numbered[1]
            else:
                ratio = line.size / result.body_size if result.body_size else 1.0
                if ratio >= H_LEVEL1_RATIO:
                    level = 1
                elif ratio >= H_LEVEL2_RATIO:
                    level = 2
                else:
                    level = 3

        candidate = HeadingCandidate(
            page_index=page,
            order=order,
            text=text,
            score=score,
            confidence=confidence,
            level=level,
            size=round(line.size, 1),
            bold=line.bold,
            numbered=bool(numbered),
            number_depth=numbered[1] if numbered else None,
            x0=round(line.x0, 1),
            y_center=round(line.y_center, 1),
            signals=signals,
        )
        candidates.append(candidate)
        if confidence == "low":
            continue

        if previous_level and level is not None and level - previous_level > 1:
            result.jumps.append(
                f"p{page} 「{text[:24]}」层级 {previous_level} → {level}（跳级）"
            )
        previous_level = level or previous_level

    result.candidates = candidates
    result.headings = _merge_wrapped_headings(
        [c for c in candidates if c.confidence != "low"]
    )
    if result.headings and not result.jumps:
        result.notes.append("no_level_jump")
    if any(c.signals.get("merged_wrapped") for c in result.headings):
        result.notes.append("merged_wrapped_headings")
    return result


def _merge_wrapped_headings(
    headings: list[HeadingCandidate],
) -> list[HeadingCandidate]:
    """
    合并「被折行的大标题」

    报告里的大标题常被折成两行（ICCT 的 ``SPECIFICATIONS OF EVS SOLD BY`` /
    ``MAJOR AUTOMAKERS``）。两行在阅读顺序里**相邻**、字号相同、层级相同 ⇒ 并成
    一条，否则同一标题会在 ``section_path`` 里出现两次，把整节内容切成两半。

    注意：``result.candidates`` 保留**未合并**的原始候选（作为证据），
    ``result.headings`` 是合并后的层级来源，两者定位不同。
    """
    merged: list[HeadingCandidate] = []
    for item in headings:
        if merged:
            previous = merged[-1]
            if (
                previous.page_index == item.page_index
                and item.order - previous.order == 1
                and previous.level == item.level
                and abs(previous.size - item.size) <= 0.6
                and previous.confidence != "low"
            ):
                previous.text = f"{previous.text} {item.text}".strip()
                previous.score = max(previous.score, item.score)
                if item.confidence == "high":
                    previous.confidence = "high"
                previous.order = item.order
                previous.signals = {**previous.signals, "merged_wrapped": True}
                continue
        merged.append(item)
    return merged


# =============================================================================
# ⑤ 表格抽取
# =============================================================================

T_MIN_ROWS = 2
T_MIN_COLS = 2
T_MIN_FILL_RATIO = 0.35  # 非空单元格占比下限（滤掉图表误检出的空洞「表」）
T_MIN_AREA_RATIO = 0.02  # 表面积占页面面积下限
T_CAPTION_MAX_DIST = 60.0  # 表格标题与本表的最大纵向距离（pt）
T_MERGE_EDGE_MARGIN = 0.15  # 跨页判定：上页表底 / 下页表顶 距页边 ≤15% 页高
T_MAX_SPLIT_ROWS = 12  # 单块最多数据行（超长表按行组切，每块重复表头）

_CAPTION_RE = re.compile(
    r"^\s*(?:Table|TABLE|Figure|FIGURE|Exhibit|EXHIBIT|Box|BOX|图|表)\s*[\dA-Za-z]+"
)
# 负号族必须收全：PDF 里的负值很少是 ASCII '-'，实测多为 U+2212（−）、
# U+2013（–）、U+2014（—）或全角 U+FF0D（－）。只认 ASCII 会把 `−3%` 判成「非数据」，
# 进而在「表头首行须含文字单元格」的假表过滤里把真表格误杀。
_SIGN = "[+\\-\u2212\u2013\u2014\uff0d]"
_DATA_CELL_RE = re.compile(
    rf"^\s*{_SIGN}?[\d,]+(?:\.\d+)?\s*"
    r"(?:%|％|pp|pt|辆|亿|万|billion|million|bn|mn|gwh|mwh|twh|kwh|usd|eur|rmb|cny|\$|€|¥)?\s*$",
    re.I,
)
_UNIT_RE = re.compile(
    r"(?:%|％|\bpp\b|GWh|MWh|TWh|kWh|万辆|千辆|亿辆|亿美元|万元|亿元|百万吨|"
    r"million|billion|bn\b|USD|US\$|EUR|RMB|CNY|gCO2|\$|€|¥)",
    re.I,
)


@dataclass
class TableBlock:
    """一个（可能已跨页拼接的）表格块"""

    page_index: int
    bbox: tuple[float, float, float, float]
    rows: list[list[str]]
    header_rows: int
    caption: str = ""
    units: list[str] = field(default_factory=list)
    cell_bboxes: list[list[tuple[float, float, float, float] | None]] = field(
        default_factory=list
    )
    merged_from: list[int] = field(default_factory=list)  # 参与拼接的物理页
    extra_boxes: list[tuple[int, tuple[float, float, float, float]]] = field(
        default_factory=list
    )  # 拼接进来的后续页表区域（切块时同样要排除）
    source: str = "lines"  # lines | text

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    @property
    def header(self) -> list[str]:
        if not self.rows:
            return []
        return ["" if c is None else c for c in self.rows[0]]

    def header_block(self) -> list[list[str]]:
        return [list(r) for r in self.rows[: self.header_rows]]

    def data_rows(self) -> list[list[str]]:
        return [list(r) for r in self.rows[self.header_rows :]]

    def as_dict(self) -> dict:
        return {
            "page_index": self.page_index,
            "bbox": [round(v, 1) for v in self.bbox],
            "caption": self.caption,
            "units": self.units,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "header_rows": self.header_rows,
            "merged_from": self.merged_from,
            "source": self.source,
            # 逐格 bbox：C3 的验收要求是「核对单元格值 **+ 原始位置**」，
            # 只落块级 bbox 无法定位到具体格子（哪个 "71%" 是哪一格），故一并落盘。
            # 仅供诊断产物使用；进 chunks.jsonl 的 table 元数据仍是精简版（不含逐格坐标）。
            "cell_bboxes": [
                [None if b is None else [round(v, 1) for v in b] for b in row]
                for row in self.cell_bboxes
            ],
            # 跨页拼接时，续页上属于本表的区域（不在本页 bbox 里，需单独记录）
            "extra_boxes": [
                {"page_index": p, "bbox": [round(v, 1) for v in b]} for p, b in self.extra_boxes
            ],
        }

    def to_text(self, rows: list[list[str]] | None = None, *, with_caption: bool = True) -> str:
        """线性化成可检索文本：标题 + 单位 + 表头 + 数据行（每行以 ``|`` 分隔）"""
        parts: list[str] = []
        if with_caption and self.caption:
            parts.append(self.caption)
        if self.units:
            parts.append("单位: " + " ".join(self.units))
        for row in self.header_block():
            parts.append(" | ".join(row))
        body = self.data_rows() if rows is None else rows
        for row in body:
            parts.append(" | ".join(row))
        return "\n".join(p for p in parts if p.strip())

    def split_for_chunks(self) -> list[str]:
        """超长表按行组切块，**每块重复表头**（表头是表格检索的命门）"""
        body = self.data_rows()
        if len(body) <= T_MAX_SPLIT_ROWS:
            return [self.to_text()] if self.rows else []
        groups = [
            body[i : i + T_MAX_SPLIT_ROWS] for i in range(0, len(body), T_MAX_SPLIT_ROWS)
        ]
        return [self.to_text(rows=group) for group in groups]


def _cell_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").replace("\n", " ")).strip()


def _is_data_like(text: str) -> bool:
    """判断单元格像不像「数据」：年份不算（它是表头里的常见成员）"""
    token = text.strip()
    if not token or not _DATA_CELL_RE.match(token):
        return False
    digits = re.sub(r"\D", "", token)
    if (
        len(digits) == 4
        and 1900 <= int(digits) <= 2100
        and "%" not in token
        and "," not in token
    ):
        return False
    return True


def _row_is_data_like(row: list[str]) -> bool:
    filled = [c for c in row if c.strip()]
    if not filled:
        return False
    return sum(1 for c in filled if _is_data_like(c)) * 2 > len(filled)


def _fill_ratio(rows: list[list[str]], n_cols: int) -> float:
    if not rows or n_cols <= 0:
        return 0.0
    total = len(rows) * n_cols
    used = sum(1 for row in rows for cell in row if cell.strip())
    return used / total


def _extract_units(text: str) -> list[str]:
    seen: list[str] = []
    for match in _UNIT_RE.finditer(text):
        token = match.group(0)
        if token not in seen:
            seen.append(token)
    return seen


def _cell_bbox_matrix(table) -> list[list[tuple[float, float, float, float] | None]]:
    """
    取出与 ``rows`` 对齐的 cell 坐标矩阵（拿不到时返回空列表）

    **PyMuPDF ≥1.26 的 ``TableRow.cells`` 元素已经是解包后的 bbox 元组**
    （``(x0, y0, x1, y1)``，跨列合并处为 ``None``），不是带 ``.bbox`` 属性的对象。
    早期写法 ``cell.bbox`` 会抛 ``AttributeError`` 并被本函数的兜底 ``except`` 吞掉，
    结果是坐标矩阵**恒为空**——不报错、不告警，只是所有 ``cell_bboxes`` 都拿不到，
    C3 要求的「核对单元格值 + 原始位置」实际无从核对。故此处直接迭代元组。
    """
    try:
        matrix: list[list] = []
        for row in table.rows:
            matrix.append(
                [
                    tuple(round(float(v), 1) for v in cell) if cell is not None else None
                    for cell in row.cells
                ]
            )
        return matrix
    except Exception:  # noqa: BLE001 —— 坐标拿不到不影响文本抽取，降级即可
        return []


def _align_bbox_rows(
    cell_bboxes: list[list], n_rows: int
) -> list[list]:
    """
    让坐标矩阵与文本行矩阵行数对齐；对不上就整个丢弃（宁缺勿错位）

    行错位的坐标矩阵比没有坐标更危险：逐格核验会拿到「邻格的框」，把正确的解析
    判成错误。所以只要无法确认对齐，就返回空列表降级为「无坐标」。
    """
    if not cell_bboxes:
        return []
    if len(cell_bboxes) == n_rows:
        return cell_bboxes
    return []


def _peel_caption(rows: list[list[str]]) -> tuple[str, list[list[str]]]:
    """
    表格标题常被检测器并进第一行（ICCT p23 的 ``China light-duty vehicles, 2025``）。

    规则：首行非空单元格 ≤1 且不是数据 → 视为标题剥离。最多剥 2 行，且至少留 2 行。
    """
    captions: list[str] = []
    while len(rows) > 2:
        filled = [c for c in rows[0] if c.strip()]
        if len(filled) > 1 or (filled and _is_data_like(filled[0])):
            break
        if not filled:
            rows = rows[1:]
            continue
        candidate = filled[0]
        # 标题应当像标题：不能是整句正文（过长或以句末标点结尾）
        if len(candidate) > 120 or candidate.rstrip().endswith(("。", ".")):
            break
        captions.append(candidate)
        rows = rows[1:]
        if len(captions) >= 2:
            break
    return " ".join(captions), rows


def _detect_header_rows(rows: list[list[str]]) -> int:
    """前若干「不像数据」的行即表头（至少 1 行，至多 H_MAX_HEADER_ROWS 行）"""
    count = 0
    for row in rows[: H_MAX_HEADER_ROWS + 1]:
        if _row_is_data_like(row):
            break
        count += 1
    return max(1, min(H_MAX_HEADER_ROWS, count))


def _find_caption_above(
    layout: PageLayout,
    top: float,
    x0: float,
    x1: float,
) -> str:
    """在表格上方 60pt 内、横向有重叠的行里找 ``Table N`` 之类的表格标题"""
    best: tuple[float, str] | None = None
    for line in layout.lines:
        text = line.text.strip()
        if not text:
            continue
        bottom = line.bbox[3]
        distance = top - bottom
        if distance < 0 or distance > T_CAPTION_MAX_DIST:
            continue
        if line.x1 < x0 - 20 or line.x0 > x1 + 20:
            continue
        if not _CAPTION_RE.match(text):
            continue
        if best is None or distance < best[0]:
            best = (distance, text)
    return best[1] if best else ""


def _build_table(table, page_index: int, layout: PageLayout, source: str) -> TableBlock | None:
    """把 PyMuPDF 的检测结果转成自研 TableBlock（含合法性过滤）"""
    rows_raw = [[_cell_text(c) for c in row] for row in table.extract()]
    if not rows_raw:
        return None

    caption, rows = _peel_caption(rows_raw)
    # 标题剥离只从**头部**吃行，坐标矩阵（对齐原始 table.rows）要同步切掉同样的行数，
    # 否则逐格核验会整表错位一行。
    peeled_rows = len(rows_raw) - len(rows)
    if len(rows) < T_MIN_ROWS:
        return None

    bbox = tuple(round(float(v), 1) for v in table.bbox)
    page_area = (layout.width * layout.height) or 1.0
    area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    n_cols = max((len(row) for row in rows), default=0)
    if n_cols < T_MIN_COLS or area / page_area < T_MIN_AREA_RATIO:
        return None
    if _fill_ratio(rows, n_cols) < T_MIN_FILL_RATIO:
        return None

    header_rows = _detect_header_rows(rows)
    if not any(c.strip() and not _is_data_like(c) for c in rows[0]):
        # 表头行没有任何「文字」单元格 ⇒ 多半是图表网格线围出来的假表
        # （实测 ICCT p8/p9 的坐标网格就是这样被检出的：整表只有数字与空格）
        return None
    if not caption:
        caption = _find_caption_above(layout, bbox[1], bbox[0], bbox[2])
    units = _extract_units(
        " ".join(" ".join(r) for r in rows[:header_rows]) + " " + caption
    )
    cell_bboxes = _cell_bbox_matrix(table)
    if cell_bboxes and peeled_rows:
        cell_bboxes = cell_bboxes[peeled_rows:]
    return TableBlock(
        page_index=page_index,
        bbox=bbox,
        rows=rows,
        header_rows=header_rows,
        caption=caption,
        units=units,
        cell_bboxes=_align_bbox_rows(cell_bboxes, len(rows)),
        source=source,
    )


def _merge_cross_page(
    tables: list[TableBlock], heights: dict[int, float]
) -> list[TableBlock]:
    """
    拼接跨页表：上一页表底贴页脚 + 本页表顶贴页眉 + 列数相同 ⇒ 同一张表

    若本页首行与上一块表头相同（检测器把重复表头也检进来了），**去掉重复表头**。
    """
    merged: list[TableBlock] = []
    for table in tables:
        if not merged:
            merged.append(table)
            continue
        previous = merged[-1]
        prev_height = heights.get(previous.page_index, 0.0)
        cur_height = heights.get(table.page_index, 0.0)
        contiguous = previous.page_index + 1 == table.page_index
        prev_at_bottom = (
            prev_height > 0
            and previous.bbox[3] >= prev_height * (1 - T_MERGE_EDGE_MARGIN)
        )
        cur_at_top = cur_height > 0 and table.bbox[1] <= cur_height * T_MERGE_EDGE_MARGIN
        if (
            contiguous
            and prev_at_bottom
            and cur_at_top
            and previous.n_cols == table.n_cols
            and not table.caption
        ):
            rows = list(table.rows)
            cell_rows = list(table.cell_bboxes)
            if rows and previous.rows and _same_row(rows[0], previous.rows[0]):
                rows = rows[1:]
                if cell_rows:  # 去重复表头时，坐标矩阵同步切头
                    cell_rows = cell_rows[1:]
            prev_n = len(previous.rows)
            previous.rows.extend(rows)
            previous.merged_from.append(table.page_index)
            previous.extra_boxes.append((table.page_index, table.bbox))
            # 只有当「已有的坐标矩阵恰好覆盖 previous 原行数」且「本页坐标覆盖拼进来的行数」
            # 时才延续；任何一边对不上就整体丢弃坐标（保留文本），避免错位。
            if len(previous.cell_bboxes) == prev_n and len(cell_rows) == len(rows):
                previous.cell_bboxes.extend(cell_rows)
            else:
                previous.cell_bboxes = []
            continue
        merged.append(table)
    return merged


def _same_row(left: list[str], right: list[str]) -> bool:
    norm = lambda cells: [c.strip().lower() for c in cells if c.strip()]  # noqa: E731
    return bool(norm(left)) and norm(left) == norm(right)


def _overlaps(a: tuple[float, ...], b: tuple[float, ...], threshold: float = 0.7) -> bool:
    """两个 bbox 的交集是否达到较小者的 70%（用于跨策略去重）"""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return False
    inter = (x1 - x0) * (y1 - y0)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return smaller > 0 and inter / smaller >= threshold


def detect_tables(
    pdf_path: Path | str,
    layouts: list[PageLayout],
    *,
    use_text_strategy: bool = False,
) -> dict[int, list[TableBlock]]:
    """
    逐页抽取表格，并把跨页表拼接起来

    :param pdf_path: PDF 路径
    :param layouts: 第①步的逐页版面（提供页面尺寸与「表格上方的标题行」）
    :param use_text_strategy: **默认关闭**。``text`` 策略本意是兜无框线表，但实测在
        文本密集页上它会把整页正文凑成一张 62×8 的假表（ICCT p1/p13/p20 全中招），
        且这批语料的真表都有框线。故只在明确需要时打开，并用「面积 ≥25% 页 +
        页面其余文本行 ≤4」双重约束收紧；无框线表的正确性另由合成用例覆盖。
    :return: ``{page_index: [TableBlock]}``（跨页表挂在**首页**）
    """
    import pymupdf

    by_page: dict[int, list[Line]] = {}
    heights: dict[int, float] = {}
    layout_by_page: dict[int, PageLayout] = {}
    for layout in layouts:
        by_page[layout.page] = [ln for ln in layout.lines if ln.text.strip()]
        heights[layout.page] = layout.height
        layout_by_page[layout.page] = layout

    flat: list[TableBlock] = []
    document = pymupdf.open(str(pdf_path))
    try:
        for page_no, page in enumerate(document, start=1):
            layout = layout_by_page.get(page_no)
            if layout is None:
                continue
            found: list[TableBlock] = []
            for table in page.find_tables(strategy="lines").tables:
                block = _build_table(table, page_no, layout, "lines")
                if block is not None:
                    found.append(block)

            if use_text_strategy:
                for table in page.find_tables(strategy="text").tables:
                    block = _build_table(table, page_no, layout, "text")
                    if block is None:
                        continue
                    # 无框线兜底只在「页面以该表为主体」时接受：表面积够大，
                    # 且页面其余文本行很少（否则文本密集页会被凑成一张假表）
                    page_area = (layout.width * layout.height) or 1.0
                    area = (block.bbox[2] - block.bbox[0]) * (block.bbox[3] - block.bbox[1])
                    outside = [
                        ln
                        for ln in by_page[page_no]
                        if not (
                            block.bbox[0] - 2 <= ln.x0
                            and ln.x1 <= block.bbox[2] + 2
                            and block.bbox[1] - 2 <= ln.y_center <= block.bbox[3] + 2
                        )
                    ]
                    if area < 0.25 * page_area or len(outside) > 4:
                        continue
                    if any(_overlaps(block.bbox, existing.bbox) for existing in found):
                        continue
                    found.append(block)

            # 跨策略去重：``lines`` 是主链路，凡与之重叠的 ``text`` 检出直接丢弃；
            # 同一策略内部的重复也按先来后到去重
            deduped: list[TableBlock] = []
            for block in found:
                if any(_overlaps(block.bbox, kept.bbox) for kept in deduped):
                    continue
                deduped.append(block)
            found = deduped
            if found:
                flat.extend(found)
    finally:
        document.close()

    merged = _merge_cross_page(flat, heights)
    result: dict[int, list[TableBlock]] = {}
    for table in merged:
        result.setdefault(table.page_index, []).append(table)
    return result
