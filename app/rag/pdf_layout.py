"""
PDF 版面解析模块（自研几何层）——从 span 重建「行 + 字号 + 粗体 + 坐标」

为什么需要它
------------
pypdf 的 ``page.extract_text()`` 只给一串字符，拿不到字号、粗体、坐标，
于是解析层无法做下面任何一件：

- 章节标题识别（#1）：标题只能靠「字号大于正文 + 加粗 + 独占一行」推
- 表格结构（#2）：列关系是几何信息，必须从坐标或线条反推
- 页眉页脚剔除（#3）：特征是「跨页重复 + 位置固定在页面顶部/底部窄带」
- 多栏阅读顺序（#4）：需要按块的 x 区间找垂直空白带（gutter）
- 页码字段化（#6）：页码只能当文本塞进 ``heading``，不是数值字段

本模块用 PyMuPDF 的 span 级输出（text / size / flags / bbox）重建出「可判定
结构」的最小单元——**行（Line）**。后续各步都建立在这一层之上：

    ① span 提取 + 行重建              ← 已实现
    ② 分栏检测与阅读顺序              ← 已实现（analyze_columns / reading_order_lines）
    ③ 页眉页脚剔除 + 页码双轨          ← 已实现（analyze_furniture / strip_furniture）
    ④ 标题识别 → section_path          ← 见 app/rag/pdf_structure.py
    ⑤ 表格抽取                          ← 见 app/rag/pdf_structure.py
    ⑥ 组装 Section 树 → 按节切块        ← 见 app/rag/pdf_parser.py

扫描页检测（C4，只检测不实现 OCR）也在本模块（``detect_scanned_pages``）。

设计原则
--------
- **只做几何，不做语义**。谁算标题、谁算页眉留给上层，便于逐步验收
- **不替换生产链路**。本模块先旁路产出，验证通过后才接进 ``chunker``
- **复用现有清洗语义**（``_sanitize_text`` / CJK 区间），保证与当前链路一致
- **懒加载 PyMuPDF**，未安装时不影响其余模块导入

已知边界（第①步的固有局限）
----------------------------
行重建只按 y 坐标聚类。**双栏页面上左右两栏同一水平线的文字会被并成一行**——
这是预期的，第②步的分栏检测负责拆解。因此本步在单栏页面上直接可用，
在多栏页面上与 pypdf 表现相当（pypdf 同样是"按视觉行"输出），不会更差。
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from app.rag.chunker import _CJK_RANGE, _sanitize_text

# PyMuPDF span flags 的 bold 位（bit 4 = 16）
_BOLD_FLAG = 1 << 4

# 句内不需要加空格的符号（避免 "word ." / "( x" 这类噪声）
_NO_SPACE_BEFORE = set(",.;:!?%)]}、，。；：！？）】》」』…’”—-")
_NO_SPACE_AFTER = set("([{（【《「『“‘")

# 两个 span 之间的水平间隙超过「字号 × 该系数」才补空格；
# 典型空格宽度约 0.25em，取 0.24 与之对齐
_SPACE_GAP_RATIO = 0.24

_CJK_CHARS = re.compile(rf"[{_CJK_RANGE}]")


@dataclass
class Span:
    """一段同字体同字号的连续文本（PyMuPDF 的最小文本单元）"""

    text: str
    size: float
    bold: bool
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Line:
    """重建后的行：同一水平线上、按 x 排序的 span 集合"""

    page: int
    spans: list[Span] = field(default_factory=list)

    @property
    def text(self) -> str:
        return _sanitize_text(_join_spans(self.spans))

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            min(s.x0 for s in self.spans),
            min(s.y0 for s in self.spans),
            max(s.x1 for s in self.spans),
            max(s.y1 for s in self.spans),
        )

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y_center(self) -> float:
        y0, y1 = self.bbox[1], self.bbox[3]
        return (y0 + y1) / 2

    @property
    def size(self) -> float:
        """按字符数加权的字号——行内混排大小时以「主体字号」为准"""
        return _weighted(self.spans, lambda s: s.size)

    @property
    def bold(self) -> bool:
        """加粗字符占比过半即判该行加粗"""
        total = sum(len(s.text) for s in self.spans)
        if not total:
            return False
        bold_chars = sum(len(s.text) for s in self.spans if s.bold)
        return bold_chars * 2 > total


@dataclass
class PageLayout:
    """单页版面：行列表 + 原始页面尺寸（宽高用于判定位置带与分栏阈值）"""

    page: int
    width: float
    height: float
    lines: list[Line] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text)


def extract_page_layouts(
    pdf_path: Path | str, *, y_tol_ratio: float = 0.5
) -> list[PageLayout]:
    """
    解析整份 PDF，逐页返回重建后的行结构

    :param pdf_path: PDF 路径
    :param y_tol_ratio: 行聚类容差系数，判定同一行时允许的纵向偏移 = 系数 × 字号
    :return: PageLayout 列表（按页序）
    """
    import pymupdf

    document = pymupdf.open(str(pdf_path))
    try:
        layouts: list[PageLayout] = []
        for page_no, page in enumerate(document, start=1):
            spans = _collect_spans(page)
            layouts.append(
                PageLayout(
                    page=page_no,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    lines=group_lines(spans, page_no, y_tol_ratio=y_tol_ratio),
                )
            )
        return layouts
    finally:
        document.close()


def _collect_spans(page) -> list[Span]:
    """从 PyMuPDF 的 dict 输出里拍平出全部 span（跳过图片等非文本块）"""
    raw = page.get_text("dict")
    spans: list[Span] = []
    for block in raw.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text:
                    continue
                x0, y0, x1, y1 = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
                spans.append(
                    Span(
                        text=text,
                        size=float(span.get("size", 0.0)),
                        bold=bool(int(span.get("flags", 0)) & _BOLD_FLAG),
                        x0=float(x0),
                        y0=float(y0),
                        x1=float(x1),
                        y1=float(y1),
                    )
                )
    return spans


def group_lines(
    spans: list[Span], page: int, *, y_tol_ratio: float = 0.5
) -> list[Line]:
    """
    把 span 聚成行

    算法：按 y 中心升序遍历，逐个尝试并入已有的行（与行的参考 y 中心之差
    不超过「系数 × 字号」即视为同一行）；并进去之后按字符数加权更新参考中心，
    避免行内基线缓慢漂移导致漏并。找不到归属就新开一行。

    不信任解析器给出的行分组——PyMuPDF 的 line 分组在多栏页面上并不总是可靠，
    自己重建才能保证第②步的分栏重排有干净的输入。
    """
    if not spans:
        return []

    ordered = sorted(spans, key=lambda s: (s.y_center, s.x0))
    groups: list[list[Span]] = []
    refs: list[float] = []

    for span in ordered:
        target = None
        for index, ref in enumerate(refs):
            tolerance = y_tol_ratio * max(span.size, _weighted(groups[index], lambda s: s.size))
            if abs(span.y_center - ref) <= tolerance:
                target = index
                break

        if target is None:
            groups.append([span])
            refs.append(span.y_center)
            continue

        group = groups[target]
        weight = sum(len(s.text) for s in group)
        new_weight = len(span.text)
        refs[target] = (refs[target] * weight + span.y_center * new_weight) / (
            weight + new_weight
        )
        group.append(span)

    lines = [Line(page=page, spans=sorted(g, key=lambda s: s.x0)) for g in groups]
    lines.sort(key=lambda line: (line.y_center, line.x0))
    return lines


def _join_spans(spans: list[Span]) -> str:
    """
    按 x 顺序拼接一行内的 span，并在必要时补空格

    补空格规则（三选一命中就补）：
    - 前一个 span 尾部或后一个 span 头部本身已有空白 → 不补
    - 任一侧紧邻字符是 CJK → 不补（中文行内不需要空格）
    - 后一个字符是「不该出现在前面的标点」/ 前一个字符是开括号 → 不补
    - 其余情况：水平间隙 > 字号 × 0.24（约一个空格宽）才补

    只按间隙补空格而不无脑拼接，是因为 PDF 里的词间距常常**不体现为空格字符**，
    而是纯粹的坐标差；反过来也有 PDF 把空格写进 span 文本。两条都要兜住。
    """
    buffer = ""
    ordered = [s for s in sorted(spans, key=lambda s: s.x0) if s.text]
    for index, span in enumerate(ordered):
        if index:
            previous = ordered[index - 1]
            if _need_space(previous, span):
                buffer += " "
        buffer += span.text
    # 折叠重复空白（span 自带空格 + 补的空格可能叠加）
    return re.sub(r"[ \t\u3000]{2,}", " ", buffer).strip()


def _need_space(previous: Span, current: Span) -> bool:
    left, right = previous.text, current.text
    if left.endswith((" ", "\u3000")) or right.startswith((" ", "\u3000")):
        return False
    if _CJK_CHARS.match(left[-1]) or _CJK_CHARS.match(right[0]):
        return False
    if right[0] in _NO_SPACE_BEFORE or left[-1] in _NO_SPACE_AFTER:
        return False
    gap = current.x0 - previous.x1
    return gap > _SPACE_GAP_RATIO * max(previous.size, current.size)


def _weighted(spans: list[Span], getter) -> float:
    """按字符数加权求均值；空列表返回 0"""
    total = sum(len(s.text) for s in spans)
    if not total:
        return 0.0
    return sum(getter(s) * len(s.text) for s in spans) / total


def detect_column_gaps(
    spans: list[Span], page_width: float, *, min_gap_ratio: float = 0.05
) -> list[tuple[float, float]]:
    """
    【第①步预研版，已被 ``analyze_columns`` 取代，勿用于新代码】

    检测页面上的垂直空白带（多栏 gutter）。

    做法：把整页 span 的 [x0, x1] 投影到 x 轴，取补集即为「没有任何文本横跨」
    的区间；宽度超过页面宽度的 ``min_gap_ratio`` 的区间视为候选分栏线。

    注意这只看「有没有文字横跨」，不看左右两侧是否都有内容，所以标题页、
    大留白的图表页会产生假阳性——第②步落地时需要叠加「两侧各有足够文本」
    与「栏宽相近」两条约束后再决定是否真的按栏重排。

    → 该约束已由 ``analyze_columns`` 完整实现（四条约束 + 区带切分 + 兜底），
    本函数保留仅供第①步报告复现历史结论。
    """
    if not spans:
        return []
    intervals = sorted((s.x0, s.x1) for s in spans)
    gaps: list[tuple[float, float]] = []
    cursor = intervals[0][0]
    for x0, x1 in intervals:
        if x0 > cursor:
            gaps.append((cursor, x0))
        cursor = max(cursor, x1)
    threshold = page_width * min_gap_ratio
    return [(g0, g1) for g0, g1 in gaps if g1 - g0 >= threshold]


# =============================================================================
# 第②步：分栏检测（检测 + 兜底 + 诊断）
# =============================================================================
#
# 设计约束（详见 docs/pdf-parsing-plan.md §C2）——这四条是**必须同时满足**的，
# 缺一条就会把封面、图表页、大留白页误判成双栏：
#
#   ① 空白带够宽（相对内容宽度）
#   ② 空白带**两侧各有足够数量的文本行**（不是空白）
#   ③ 两侧文字**竖向上真正并排**（不是「上半页偏左、下半页偏右」的错觉）
#   ④ 两侧**栏宽相近**
#   ⑤ 栏内**行左边界一致**（同一栏的行应当从同一个 x 起排）
#
# 并且判定粒度不是「整页」而是**区带（region）**：报告常见「单栏大标题 +
# 双栏正文 + 跨栏图表 + 页脚」，整页二分必然出错。做法是**先找分栏线、再由它
# 反推区带**（见 `_split_columns_region`）——不能在单栏页上拿"跨栏行"切页面，
# 那会把通栏排版的页面切成一堆碎片。
#
# 判不准时的行为：**退回单栏**（保持当前行为，绝不比重排前更差），并把
# 候选 gutter 及其各条约束的实测值写进诊断，供人工复核。
# → 「没看出来」无害，「看错成双栏」有害，两者必须能区分。

_EPS = 0.5  # 坐标比较容差（pt）

COL_FULL_WIDTH_RATIO = 0.72  # 「长行」门槛：宽度 ≥ 内容宽 × 此值即可能是跨栏行
COL_MIN_GAP_RATIO = 0.020  # 候选 gutter 最小宽度 = 内容宽 × 此值
COL_MIN_GAP_ABS = 6.0  # 候选 gutter 最小绝对宽度（pt），兜住窄页面
COL_MIN_SIDE_LINES = 3  # 约束②：gutter 两侧各需的最少行数
COL_GUTTER_POSITION = (0.25, 0.80)  # 约束①：gutter 中心须落在内容区的这个相对区间
COL_MIN_PERSISTENCE = 0.50  # 约束③：两侧文字的纵向交叠比例下限
COL_MAX_WIDTH_DIFF = 0.35  # 约束④：两侧栏宽相对差上限
COL_MAX_LEFT_EDGE_CV = 0.08  # 约束⑤：栏内行左边界离散度（标准差/栏宽）上限
COL_HIGH_GAP_RATIO = 0.025  # 判 high 的 gutter 宽度门槛（≈12pt@490pt 正文宽，真双栏的典型值）
COL_HIGH_SIDE_LINES = 5  # 判 high 的每侧行数门槛
COL_HIGH_WIDTH_DIFF = 0.20  # 判 high 的栏宽差门槛
COL_HIGH_LEFT_EDGE_CV = 0.04  # 判 high 的左边界离散度门槛
COL_MIN_REGION_LINES = 6  # 短行总数低于此值不做分栏判定（直接单栏）


@dataclass
class ColumnCandidate:
    """一条候选分栏线（gutter）及其四条约束的实测明细"""

    x0: float
    x1: float
    left_lines: int
    right_lines: int
    width_diff: float
    left_edge_cv: float
    persistence: float
    ok: bool
    reject: str = ""

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    def as_dict(self) -> dict:
        return {
            "x": [round(self.x0, 1), round(self.x1, 1)],
            "width": round(self.width, 1),
            "left_lines": self.left_lines,
            "right_lines": self.right_lines,
            "width_diff": round(self.width_diff, 3),
            "left_edge_cv": round(self.left_edge_cv, 3),
            "persistence": round(self.persistence, 3),
            "ok": self.ok,
            "reject": self.reject,
        }


@dataclass
class Region:
    """
    区带：一段纵向连续、栏数一致的版面条带

    只在**确认存在分栏线**的页面上才会切出区带；单栏页面整页就是一个 ``body``
    区带——这是为了避免「用跨栏行切区带」在通栏排版页面上把整页切碎
    （实测贝莱德单页曾被切成 17 个区带，噪声淹没信号）。

    ``kind``：
    - ``body``      单栏区带（分栏区段之外的正文，如跨栏标题以上的部分）
    - ``separator`` 跨栏行（横跨 gutter 的大标题 / 跨栏图表标题 / 页脚）
    - ``columns``   确认的双栏区带
    """

    y0: float
    y1: float
    lines: list[Line] = field(default_factory=list)
    kind: str = "body"
    n_columns: int = 1
    confidence: str = "single"
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "y": [round(self.y0, 1), round(self.y1, 1)],
            "kind": self.kind,
            "lines": len(self.lines),
            "columns": self.n_columns,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass
class PageColumns:
    """单页分栏分析的结论 + 证据（可直接序列化为诊断日志的一行）"""

    page: int
    regions: list[Region]
    content_x0: float
    content_x1: float
    fallback: bool = False
    gutter: ColumnCandidate | None = None  # 采纳的分栏线（判为双栏时）
    rejected: ColumnCandidate | None = None  # 未通过约束的最佳候选（判为单栏时）
    notes: list[str] = field(default_factory=list)

    @property
    def content_width(self) -> float:
        return self.content_x1 - self.content_x0

    @property
    def multi_regions(self) -> list[Region]:
        return [r for r in self.regions if r.n_columns > 1]

    @property
    def verdict(self) -> str:
        """single（全页单栏）/ multi（全部区带双栏）/ mixed（混排）"""
        multi = self.multi_regions
        if not multi:
            return "single"
        return "multi" if len(multi) == len(self.regions) else "mixed"

    def as_dict(self, doc: str = "") -> dict:
        return {
            "doc": doc,
            "page_index": self.page,
            "verdict": self.verdict,
            "fallback": self.fallback,
            "content_x": [round(self.content_x0, 1), round(self.content_x1, 1)],
            "gutter": self.gutter.as_dict() if self.gutter else None,
            "rejected": self.rejected.as_dict() if self.rejected else None,
            "regions": [r.as_dict() for r in self.regions],
            "notes": self.notes,
        }


def _x_gaps(intervals: list[tuple[float, float]], lo: float) -> list[tuple[float, float]]:
    """区间并集的**内部**空隙（不含右侧页边距），即候选 gutter"""
    if not intervals:
        return []
    gaps: list[tuple[float, float]] = []
    cursor = lo
    for a, b in sorted(intervals):
        if a > cursor:
            gaps.append((cursor, a))
        cursor = max(cursor, b)
    return gaps


def _cv(values: list[float], scale: float) -> float:
    """离散度 = 总体标准差 / 参照尺度（空或退化时返回 0，即「一致」）"""
    if len(values) < 2 or scale <= 0:
        return 0.0
    return statistics.pstdev(values) / scale


def _evaluate_gutter(
    gap: tuple[float, float], lines: list[Line], content_x0: float, content_width: float
) -> ColumnCandidate:
    """对一条候选 gutter 逐条核对四项约束，返回带拒绝原因的明细"""
    gx0, gx1 = gap
    left = [ln for ln in lines if ln.x1 <= gx0 + _EPS]
    right = [ln for ln in lines if ln.x0 >= gx1 - _EPS]

    left_w = right_w = 0.0
    left_cv = right_cv = 0.0
    if left:
        left_w = statistics.median(ln.x1 for ln in left) - statistics.median(
            ln.x0 for ln in left
        )
        left_cv = _cv([ln.x0 for ln in left], left_w)
    if right:
        right_w = statistics.median(ln.x1 for ln in right) - statistics.median(
            ln.x0 for ln in right
        )
        right_cv = _cv([ln.x0 for ln in right], right_w)

    width_diff = (
        abs(left_w - right_w) / max(left_w, right_w, 1e-6) if left and right else 1.0
    )
    edge_cv = max(left_cv, right_cv)

    # 竖向持续性：两侧文字必须在**同一段纵向区间**里并排（真有分栏时左右同高），
    # 否则只是「上半页偏左、下半页偏右」的排版错觉（如上下两块不同版的版式）
    persistence = 1.0
    if left and right:
        l_lo = min(ln.y_center for ln in left)
        l_hi = max(ln.y_center for ln in left)
        r_lo = min(ln.y_center for ln in right)
        r_hi = max(ln.y_center for ln in right)
        union = max(l_hi, r_hi) - min(l_lo, r_lo)
        if union > 0:
            persistence = max(0.0, min(l_hi, r_hi) - max(l_lo, r_lo)) / union

    candidate = ColumnCandidate(
        x0=gx0,
        x1=gx1,
        left_lines=len(left),
        right_lines=len(right),
        width_diff=width_diff,
        left_edge_cv=edge_cv,
        persistence=persistence,
        ok=False,
    )

    min_gap = max(COL_MIN_GAP_ABS, COL_MIN_GAP_RATIO * content_width)
    center_rel = ((gx0 + gx1) / 2 - content_x0) / content_width if content_width else 0.0

    if candidate.width < min_gap:
        candidate.reject = "gap_too_narrow"
    elif not (COL_GUTTER_POSITION[0] <= center_rel <= COL_GUTTER_POSITION[1]):
        candidate.reject = "gap_not_in_center_band"
    elif len(left) < COL_MIN_SIDE_LINES or len(right) < COL_MIN_SIDE_LINES:
        candidate.reject = "insufficient_text_beside_gap"
    elif persistence < COL_MIN_PERSISTENCE:
        candidate.reject = "columns_not_vertically_aligned"
    elif width_diff > COL_MAX_WIDTH_DIFF:
        candidate.reject = "column_width_mismatch"
    elif edge_cv > COL_MAX_LEFT_EDGE_CV:
        candidate.reject = "left_edge_inconsistent"
    else:
        candidate.ok = True
    return candidate


def _confidence(candidate: ColumnCandidate, content_width: float) -> str:
    """高置信门槛比「通过」更严——只有像真双栏的才敢标 high"""
    if (
        candidate.width >= COL_HIGH_GAP_RATIO * content_width
        and min(candidate.left_lines, candidate.right_lines) >= COL_HIGH_SIDE_LINES
        and candidate.width_diff <= COL_HIGH_WIDTH_DIFF
        and candidate.left_edge_cv <= COL_HIGH_LEFT_EDGE_CV
    ):
        return "high"
    return "medium"


def _candidates(
    lines: list[Line], content_x0: float, content_width: float
) -> list[ColumnCandidate]:
    """按宽度降序给出该区带内全部「够宽」的候选 gutter（附判定明细）"""
    intervals = [(ln.x0, ln.x1) for ln in lines]
    min_gap = max(COL_MIN_GAP_ABS, COL_MIN_GAP_RATIO * content_width)
    gaps = [g for g in _x_gaps(intervals, content_x0) if g[1] - g[0] >= min_gap]
    gapped = sorted(gaps, key=lambda g: -(g[1] - g[0]))  # 宽的在前
    return [_evaluate_gutter(g, lines, content_x0, content_width) for g in gapped]


def _split_columns_region(
    lines: list[Line], gutter: ColumnCandidate, confidence: str
) -> list[Region]:
    """
    由一条已采纳的分栏线反推该页的区带序列

    **顺序是「先找分栏线、再由它推区带」**，而不是拿「跨栏行」去切页面——
    后者在通栏排版页面上会把整页切成一堆碎片（早期版本实测把单页切成 17 个区带）。
    这里的分法：
    - 与分栏线两侧并排、且处在同一纵向区间的行 → ``columns``（双栏区带）
    - 纵向区间之内、但横跨分栏线的行 → ``separator``（跨栏大标题 / 跨栏图表 / 页脚）
    - 纵向区间之外的行 → ``body``（单栏区带，如跨栏标题以上的部分）
    """
    gx0, gx1 = gutter.x0, gutter.x1

    def on_left(line: Line) -> bool:
        return line.x1 <= gx0 + _EPS

    def on_right(line: Line) -> bool:
        return line.x0 >= gx1 - _EPS

    columnar = [ln for ln in lines if on_left(ln) or on_right(ln)]
    y_lo = min(ln.y_center for ln in columnar)
    y_hi = max(ln.y_center for ln in columnar)

    groups: list[tuple[str, list[Line]]] = []
    for line in sorted(lines, key=lambda ln: (ln.y_center, ln.x0)):
        if y_lo - _EPS <= line.y_center <= y_hi + _EPS:
            kind = "columns" if (on_left(line) or on_right(line)) else "separator"
        else:
            kind = "body"
        if groups and groups[-1][0] == kind:
            groups[-1][1].append(line)
        else:
            groups.append((kind, [line]))

    regions: list[Region] = []
    for kind, group in groups:
        region = Region(
            y0=min(ln.bbox[1] for ln in group),
            y1=max(ln.bbox[3] for ln in group),
            lines=group,
            kind=kind,
        )
        if kind == "columns":
            region.n_columns = 2
            region.confidence = confidence
            region.reason = "two_columns"
        else:
            region.reason = "separator" if kind == "separator" else "single"
        regions.append(region)
    return regions


def analyze_columns(layout: PageLayout) -> PageColumns:
    """
    分析单页的分栏结构（结论 + 证据，可直接落诊断日志）

    三步，顺序很关键：

    1. **只用短行求候选分栏线**（宽度 < 内容宽 × ``COL_FULL_WIDTH_RATIO``）。
       跨栏大标题会把分栏缝遮住，若算进来，混排页永远找不到分栏线；反过来，
       通栏排版的页面里几乎所有行都是长行，排除后自然没有候选，也就不会把
       一页切成一堆碎片。
    2. **逐条核对约束**：够宽 / 位置在页面中部 / 两侧各有足够文本 / 两侧竖向对齐 /
       栏宽相近 / 栏内左边界一致——取第一条全部通过的。
    3. 通过 → 由该分栏线反推区带；不通过 → **整页退回单栏**，并把「最佳但不合格」
       的候选留作证据（``fallback=True``，进人工复核队列）。

    :param layout: 第①步产出的单页版面（行已按 y 排序）
    """
    lines = [ln for ln in layout.lines if ln.text.strip()]
    if not lines:
        return PageColumns(
            page=layout.page, regions=[], content_x0=0.0, content_x1=0.0,
            notes=["empty_page"],
        )

    content_x0 = min(ln.x0 for ln in lines)
    content_x1 = max(ln.x1 for ln in lines)
    content_width = content_x1 - content_x0

    def single_page(reason: str, rejected: ColumnCandidate | None = None) -> PageColumns:
        region = Region(
            y0=min(ln.bbox[1] for ln in lines),
            y1=max(ln.bbox[3] for ln in lines),
            lines=list(lines),
            reason=reason,
        )
        return PageColumns(
            page=layout.page,
            regions=[region],
            content_x0=content_x0,
            content_x1=content_x1,
            fallback=rejected is not None,
            rejected=rejected,
        )

    short = [ln for ln in lines if (ln.x1 - ln.x0) < COL_FULL_WIDTH_RATIO * content_width]
    if len(short) < COL_MIN_REGION_LINES:
        # 短行都不够多 → 这是通栏排版，根本没有分栏缝可言，不算「判不准」
        return single_page("no_short_lines")

    found = _candidates(short, content_x0, content_width)
    if not found:
        return single_page("no_gap")

    best = next((c for c in found if c.ok), None)
    if best is None:
        return single_page(found[0].reject, rejected=found[0])

    return PageColumns(
        page=layout.page,
        regions=_split_columns_region(lines, best, _confidence(best, content_width)),
        content_x0=content_x0,
        content_x1=content_x1,
        gutter=best,
    )


# =============================================================================
# 第②步补充：无框线多列表的「列网格」重建
# =============================================================================
#
# 有一类页面，主体是一张**没有框线的宽表**——本项目实测最典型的是贝莱德
# 《2026 年全球投资展望》p14–p17 的「战术配置细分观点」：左右两栏并排，每栏
# 三列「资产｜观点｜评论」。这类页面同时踩中两个已有机制的盲区：
#
#   ① ``detect_tables`` 走 ``find_tables("lines")``——**表没有线条就抓不到**，
#      于是表格内容以「正文行」的身份混进第②步；
#   ② ``group_lines`` 按 y 合并同一水平线上的 span——于是**并排两栏中恰好同 y
#      的文字被拼成一行**。p16 实测：左栏「新兴市场…」与右栏「短期投资级别
#      信用债…」被并成一行，chunk 里出现 ``…供应链重构利好信用债 中性大。中性
#      墨西哥…`` 这种语义全毁的碎片，直接拖低 BM25 短语命中与向量质量。
#
# ``analyze_columns`` 也救不了：它按**行**找 gutter，而这类页面每一行都横跨
# 全部列（行级 x 投影永远是连通的），必然判 ``no_gap``。真正的空隙只存在于
# **span 级**（实测 p16 span 投影有 [459,494] 与 [516,543] 两条 30pt 上下的缝）。
#
# 所以把粒度下沉到 span：
#   1. span 级 x 投影找「覆盖率极低的窄缝」→ 切成若干**列带**（``_column_bands``）；
#   2. 长文本带（评论列）向左吸收同组相邻带，构成一个**大栏**（``_band_groups``）；
#   3. 大栏内以长文本带的段落为锚，把 y 重叠的其他带单元格并进来，得到「逻辑行」
#      ``资产名 观点 评论``；单元格内部的多行先纵向合并（``长期投资级别`` +
#      ``信用债`` → ``长期投资级别信用债``）。
#
# **门禁很严**（列带数 ≥3、至少一条长文本带、至少一条短语带，三条同时成立），
# 普通正文页 / 单栏页 / 双栏文本页一律不触发 ⇒ 对现有解析是恒等变换。这是它
# 能接在 ``reading_order_lines`` 前面的前提。

TABLE_GRID_ENABLED = True  # 一键回退开关
TABLE_GRID_MIN_SPANS = 30  # 页面 span 少于这个数不值得判定（封面/单行标题页）
# --- 门禁① 触发收窄：只在「页面真的被跨列误合并糟蹋过」时才动手 ---
# 判据不是「页面长得像表格」（那个判不出来），而是「``group_lines`` 真的干了坏事」：
# 同一行里出现相邻 span 的 x 间隙远大于正常词距 ⇒ 两个视觉上不相邻的文本块被拼进
# 一行。这是本模块要修的那个 bug 的**直接症状**，不是对版式的猜测。
GRID_TRIGGER_MIN_SPURIOUS = 4  # 页内这类「误合并行」至少这么多条才值得重建
SPURIOUS_GAP_RATIO = 3.0  # 间隙 > 此系数 × 字号 …
SPURIOUS_GAP_MIN = 24.0  # …且绝对间隙 > 此值（pt）才算误合并
# --- 门禁② 结构门禁 ---
COL_START_MIN_ROWS = 4  # 列起点：该簇的 span 至少覆盖这么多**不同行**
COL_START_MERGE = 6.0  # 列起点聚类的半径（pt）
BAND_MIN_WIDTH = 16.0  # 列带最小宽度（pt），更窄的带直接丢弃
BAND_TEXT_MED_LEN = 18.0  # 长文本带的判据：单元格字符数**中位数**下限
BAND_PHRASE_MED_LEN = 8.0  # 短语带的判据：单元格字符数中位数上限
BAND_MIN_TEXT_CELLS = 4  # 长文本带最少单元格数
BAND_MIN_PHRASE_CELLS = 3  # 短语带最少单元格数
GRID_MIN_PHRASE_PER_GROUP = 2  # 每个大栏至少这么多条短语列（资产名 + 观点档）
CELL_Y_TOL_RATIO = 0.55  # 「同一基线」聚类容差（× 字号）
PARA_GAP_RATIO = 1.6  # 段落切分：相邻单元格 y 间距 > 此系数 × 字号 → 另起一段
PAIR_TOL_RATIO = 0.6  # 锚段落与旁列单元格的 y 配对容差（× 字号）


@dataclass
class _Cell:
    """列带内的一个单元格：同一列、同一基线上的一组 span（可含多行）"""

    spans: list[Span]

    @property
    def y0(self) -> float:
        return min(s.y0 for s in self.spans)

    @property
    def y1(self) -> float:
        return max(s.y1 for s in self.spans)

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def x0(self) -> float:
        return min(s.x0 for s in self.spans)

    @property
    def x1(self) -> float:
        return max(s.x1 for s in self.spans)

    @property
    def size(self) -> float:
        return _weighted(self.spans, lambda s: s.size)

    @property
    def text(self) -> str:
        return _cell_text(self.spans)


@dataclass
class _Band:
    """一条列带：x 区间 + 落在其中的单元格"""

    x0: float
    x1: float
    cells: list[_Cell] = field(default_factory=list)

    @property
    def med_len(self) -> float:
        if not self.cells:
            return 0.0
        lengths = sorted(len(c.text) for c in self.cells)
        return float(lengths[len(lengths) // 2])

    @property
    def is_text(self) -> bool:
        """长文本带（评论列）：单元格多、文本长（用中位数，不被 ``中性`` 这类短格拉低）"""
        return (
            len(self.cells) >= BAND_MIN_TEXT_CELLS
            and self.med_len >= BAND_TEXT_MED_LEN
        )

    @property
    def is_phrase(self) -> bool:
        """短语带（资产名 / 观点档位列）：单元格多、文本短"""
        return (
            len(self.cells) >= BAND_MIN_PHRASE_CELLS
            and self.med_len <= BAND_PHRASE_MED_LEN
        )


def _cell_text(spans: list[Span]) -> str:
    """
    单元格文本：内部先按基线分行、行内按 x 拼接

    跨行时 CJK 直接相接（``长期投资级别`` + ``信用债`` → ``长期投资级别信用债``），
    非 CJK 补一个空格（英文表格的 ``North`` / ``America`` 不能粘成 ``NorthAmerica``）。
    """
    rows: list[list[Span]] = []
    for span in sorted(spans, key=lambda s: (s.y_center, s.x0)):
        placed = False
        if rows:
            tol = CELL_Y_TOL_RATIO * max(
                span.size, _weighted(rows[-1], lambda s: s.size)
            )
            base = _weighted(rows[-1], lambda s: s.y_center)
            if abs(span.y_center - base) <= tol:
                rows[-1].append(span)
                placed = True
        if not placed:
            rows.append([span])

    out = ""
    previous: Span | None = None
    previous_row = -1
    for row_index, row in enumerate(rows):
        for span in sorted(row, key=lambda s: s.x0):
            if previous is not None:
                cjk = bool(
                    _CJK_CHARS.match(previous.text[-1]) or _CJK_CHARS.match(span.text[0])
                )
                if row_index != previous_row:
                    if not cjk:
                        out += " "
                elif _need_space(previous, span):
                    out += " "
            out += span.text
            previous = span
            previous_row = row_index
    return _sanitize_text(re.sub(r"[ \t\u3000]{2,}", " ", out)).strip()


def _group_cells(spans: list[Span]) -> list[_Cell]:
    """把同一条列带内的 span 按基线聚成单元格（相邻行间距 > 容差 ⇒ 两个单元格）"""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s.y_center, s.x0))
    groups: list[list[Span]] = []
    refs: list[float] = []
    for span in ordered:
        target = None
        for index, ref in enumerate(refs):
            tol = CELL_Y_TOL_RATIO * max(
                span.size, _weighted(groups[index], lambda s: s.size)
            )
            if abs(span.y_center - ref) <= tol:
                target = index
                break
        if target is None:
            groups.append([span])
            refs.append(span.y_center)
            continue
        group = groups[target]
        weight = sum(len(s.text) for s in group)
        new_weight = len(span.text)
        refs[target] = (refs[target] * weight + span.y_center * new_weight) / (
            weight + new_weight
        )
        group.append(span)
    cells = [_Cell(sorted(g, key=lambda s: (s.y_center, s.x0))) for g in groups]
    cells.sort(key=lambda c: (c.y_center, c.x0))
    return cells


def count_spurious_lines(layout: PageLayout) -> int:
    """
    数一数这一页有多少条「被 ``group_lines`` 跨列误合并」的行

    一条行被判为误合并：行内相邻 span 之间存在远大于正常词距的 x 间隙
    （``> SPURIOUS_GAP_RATIO × 字号`` 且 ``> SPURIOUS_GAP_MIN``）。真表格页
    （并排双栏表、无框线多列表）必然大量出现；普通单栏正文页因为行内 span
    本来就首尾相接，几乎为 0——这就是它比「像不像表格」可靠的**直接症状**。
    """
    count = 0
    for line in layout.lines:
        spans = sorted(
            (s for s in line.spans if s.text.strip()), key=lambda s: s.x0
        )
        for left, right in zip(spans, spans[1:]):
            gap = right.x0 - left.x1
            limit = max(SPURIOUS_GAP_RATIO * max(left.size, right.size), SPURIOUS_GAP_MIN)
            if gap > limit:
                count += 1
                break
    return count


def _column_starts(spans: list[Span]) -> list[float]:
    """
    span 起点（x0）聚成簇 → 列起点列表

    判据是「该起点附近起头的 span 覆盖了多少个**不同的 y**」，而不是「有多少个
    span」：表格里真正的列（资产名、观点档位、评论段）每列都有十几条从同一 x
    起头的 span，且分布在不同行；而副标题里的孤立数字、表头词只贡献一两个 y，
    必须排掉——实测它们会把左栏评论列切成三段。
    """
    entries = sorted(spans, key=lambda s: s.x0)
    clusters: list[list[Span]] = []
    for span in entries:
        if clusters and span.x0 - clusters[-1][-1].x0 <= 2 * COL_START_MERGE:
            clusters[-1].append(span)
        else:
            clusters.append([span])

    starts: list[float] = []
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        rows = {int(round(s.y_center)) for s in cluster}
        if len(rows) < COL_START_MIN_ROWS:
            continue
        ordered = sorted(s.x0 for s in cluster)
        starts.append(ordered[len(ordered) // 2])
    return starts


def _column_bands(spans: list[Span]) -> list[_Band] | None:
    """
    按列起点把页面切成列带；不构成「多列 + 长文本 + 短语」结构时返回 ``None``
    """
    low = min(s.x0 for s in spans)
    high = max(s.x1 for s in spans)
    if high - low < 60:
        return None

    starts = _column_starts(spans)
    if len(starts) < 3:
        return None

    edges = sorted({round(low, 1), round(high, 1)} | {round(s, 1) for s in starts})
    intervals = [
        (edges[i], edges[i + 1])
        for i in range(len(edges) - 1)
        if edges[i + 1] - edges[i] >= BAND_MIN_WIDTH
    ]
    if len(intervals) < 3:
        return None

    buckets: list[list[Span]] = [[] for _ in intervals]
    for span in spans:
        center = (span.x0 + span.x1) / 2
        best_index, best_score = 0, float("-inf")
        for index, (x0, x1) in enumerate(intervals):
            if x0 <= center <= x1:
                best_index = index
                break
            score = -min(abs(center - x0), abs(center - x1))
            if score > best_score:
                best_score, best_index = score, index
        buckets[best_index].append(span)

    bands = [
        _Band(x0, x1, _group_cells(bucket))
        for (x0, x1), bucket in zip(intervals, buckets)
        if bucket
    ]
    if len(bands) < 3:
        return None

    # 门禁②-1：每条列带都必须是「短语带」或「长文本带」。
    # 出现无法归类的带 ⇒ 这些「列」只是把整片版面按起点硬切出来的，不是表格的列
    # （贝莱德 p4 图表页实测有 4 条这样的带，重建后把图注 / 正文 / 英文原句揉成一行）。
    if any(not (band.is_text or band.is_phrase) for band in bands):
        return None

    # 门禁②-2：至少一条长文本带（评论列）+ 一条短语带（资产/观点列）
    if not any(band.is_text for band in bands):
        return None
    if not any(band.is_phrase for band in bands):
        return None
    return bands


def _band_groups(bands: list[_Band]) -> list[list[_Band]]:
    """
    按 x 从左到右把列带分组为「大栏」：**每个长文本带向右封口一个组**

    并排两栏各有独立的行 y（左右栏的行并不一一对应），必须按大栏隔离配对，
    否则左栏的评论会把右栏的资产名吸进来。以「评论列（长文本带）」为组尾，
    它左边的资产/观点列自然归入同一组，下一个评论列开启新组。
    """
    groups: list[list[_Band]] = []
    current: list[_Band] = []
    for band in bands:
        current.append(band)
        if band.is_text:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _split_paragraphs(cells: list[_Cell]) -> list[list[_Cell]]:
    """锚带（评论列）的单元格按纵向间距切段——一段 = 一个逻辑行"""
    paragraphs: list[list[_Cell]] = []
    for cell in cells:
        if paragraphs:
            previous = paragraphs[-1][-1]
            tol = PARA_GAP_RATIO * max(cell.size, previous.size)
            if cell.y_center - previous.y_center <= tol:
                paragraphs[-1].append(cell)
                continue
        paragraphs.append([cell])
    return paragraphs


def _synthetic(cells: list[_Cell]) -> list[Span]:
    """把单元格合成为 span，复用 ``Line`` 既有的拼接与补空格逻辑"""
    return [
        Span(
            text=cell.text,
            size=cell.size,
            bold=False,
            x0=cell.x0,
            y0=cell.y0,
            x1=cell.x1,
            y1=cell.y1,
        )
        for cell in cells
        if cell.text
    ]


def _grid_rows(group: list[_Band]) -> list[tuple[float, list[Span]]]:
    """一个大栏 → 若干「逻辑行」：``(y, spans)``"""
    anchor = group[-1] if group[-1].is_text else None
    if anchor is None:
        # 尾部残余（没有评论列）：每个单元格各自成行
        return [(cell.y_center, _synthetic([cell])) for band in group for cell in band.cells]

    used: set[int] = set()
    rows: list[tuple[float, list[Span]]] = []

    for paragraph in _split_paragraphs(anchor.cells):
        top = min(c.y0 for c in paragraph)
        bottom = max(c.y1 for c in paragraph)
        cells = list(paragraph)
        for cell in paragraph:  # 锚带自身的单元格也标已用，否则会被 leftovers 再输出一次
            used.add(id(cell))
        for band in group[:-1]:
            for cell in band.cells:
                tol = PAIR_TOL_RATIO * cell.size
                if top - tol <= cell.y_center <= bottom + tol:
                    cells.append(cell)
                    used.add(id(cell))
        cells.sort(key=lambda c: c.x0)
        rows.append(((top + bottom) / 2, _synthetic(cells)))

    # 没被任何段落吸收的单元格（表头、跨栏说明等）→ 按 y 相邻合并成行
    leftovers = [c for band in group for c in band.cells if id(c) not in used]
    leftovers.sort(key=lambda c: c.y_center)
    pending: list[_Cell] = []
    for cell in leftovers:
        if pending:
            tol = PAIR_TOL_RATIO * max(cell.size, pending[-1].size)
            if cell.y_center - pending[-1].y_center > tol:
                rows.append(
                    (_row_y(pending), _synthetic(sorted(pending, key=lambda c: c.x0)))
                )
                pending = []
        pending.append(cell)
    if pending:
        rows.append((_row_y(pending), _synthetic(sorted(pending, key=lambda c: c.x0))))
    return rows


def _row_y(cells: list[_Cell]) -> float:
    return sum(c.y_center for c in cells) / len(cells)


def table_grid_lines(layout: PageLayout) -> list[Line] | None:
    """
    无框线多列表的「列网格」重建，把「资产｜观点｜评论」还原成一个个逻辑行

    返回 ``None`` 表示本页不符合条件（普通正文页 / 单栏页 / 双栏文本页），
    调用方应退回既有的阅读顺序逻辑——**对非表格页是恒等变换**。
    """
    if not TABLE_GRID_ENABLED:
        return None

    # 前置条件：常规分栏判定必须先否决（``single``）。真正的双栏/多栏正文页由
    # ``analyze_columns`` 负责（「左栏到底 → 右栏到底」），本函数**绝不能插手**——
    # 否则会把并排两栏按 y 交错，比不修更糟。这里复用的判据是「同一水平线上
    # 每一行都横跨全列」（行级 x 投影连通）⇒ 分栏器找不到 gutter ⇒ ``no_gap``。
    if analyze_columns(layout).verdict != "single":
        return None

    spans = [s for ln in layout.lines for s in ln.spans if s.text.strip()]
    if len(spans) < TABLE_GRID_MIN_SPANS:
        return None

    # 门禁①：页面必须真的有「跨列误合并」症状才动手——判据是 bug 的直接症状，
    # 不是「页面长得像表格」（后者判不准，见模块头注释）。
    if count_spurious_lines(layout) < GRID_TRIGGER_MIN_SPURIOUS:
        return None

    bands = _column_bands(spans)
    if bands is None:
        return None

    groups = _band_groups(bands)
    # 门禁②-3：每个大栏都必须是「若干短语列 + 一个评论列」的完整结构。末栏不带
    # 评论列（残余带）说明页面右侧还有别的版式，不是规整网格 ⇒ 否决。
    for group in groups:
        if not group[-1].is_text:
            return None
        if sum(1 for band in group if band.is_phrase) < GRID_MIN_PHRASE_PER_GROUP:
            return None

    rows: list[tuple[float, list[Span]]] = []
    for group in groups:
        rows.extend(_grid_rows(group))
    if not rows:
        return None

    lines: list[Line] = []
    for _, row_spans in sorted(
        rows, key=lambda item: (item[0], min((s.x0 for s in item[1]), default=0.0))
    ):
        if row_spans:
            lines.append(Line(page=layout.page, spans=row_spans))
    return lines or None


def reading_order_lines(layout: PageLayout, analysis: PageColumns | None = None) -> list[Line]:
    """
    给出该页的**阅读顺序**行序列

    三条路，按「越特殊越先试」排序：

    1. **无框线多列表网格重建**（``table_grid_lines``）——并排双栏表 / 无框线多列表
       这类 ``group_lines`` 会跨列误合并的页面，交给它还原「资产｜观点｜评论」逻辑行；
    2. **双栏区带**——按「左栏到底 → 右栏到底」重排；
    3. **单栏**——保持原有 y 序，即恒等变换。

    第 1 条带严格门禁（见 ``table_grid_lines``），判不准就返回 ``None`` 落到 2/3，
    因此对绝大多数的普通页仍是恒等变换——**退错时只是没优化，不会更糟**。
    """
    grid = table_grid_lines(layout)
    if grid is not None:
        return grid

    analysis = analysis or analyze_columns(layout)
    gutter = analysis.gutter
    ordered: list[Line] = []
    for region in analysis.regions:
        if region.n_columns > 1 and gutter is not None:
            left = [ln for ln in region.lines if ln.x1 <= gutter.x0 + _EPS]
            right = [ln for ln in region.lines if ln.x0 >= gutter.x1 - _EPS]
            ordered.extend(sorted(left, key=lambda ln: (ln.y_center, ln.x0)))
            ordered.extend(sorted(right, key=lambda ln: (ln.y_center, ln.x0)))
        else:
            ordered.extend(sorted(region.lines, key=lambda ln: (ln.y_center, ln.x0)))
    return ordered


# =============================================================================
# 第③步：页眉页脚剔除 + 页码双轨识别（page_index / page_label）
# =============================================================================
#
# 页眉页脚的特征是「**跨页重复 + 位置固定**」：同一段文字在多数页面的同一高度
# 反复出现。判据因此有两条，**缺一不可**：
#
#   ① 规范化文本（数字统一换成 #，消除「Page 3 / Page 4」这类差异）在
#      ≥ 阈值比例的页面上出现，且每页至多出现一次；
#   ② 其相对 y 位置在各页高度一致（标准差 ÷ 页高 ≤ 阈值）。
#
# 只靠 ① 会误伤正文里的高频短句（如「资料来源」「Continued」）；只靠 ② 会误伤
# 正文段落。两条同时成立才认定为页面家具（page furniture）。
#
# 页码是页眉页脚里的一条**特殊模式**，单独识别为 ``page_label``（印刷页码），
# 与 ``page_index``（PDF 物理页序）分开保存——两者不一定相等：封面/目录可能用
# 罗马数字、正文重新从 1 开始、也可能整体偏移。文档级映射落 ``page_map``。
# **评测锚点一律使用 page_index**，印刷页码只用于展示与溯源（C5）。
#
# 这一步顺带产出的「跨页重复文本」正是 C1 的反向证据——重复出现 ⇒ 是页眉，
# 不是章节标题。故结果以 ``appearances`` 暴露给标题识别使用。

FURN_MIN_PAGES = 3  # 至少出现在这么多页上才算重复
FURN_MIN_PAGE_RATIO = 0.25  # 或出现在 ≥ 25% 的页面上（两者取大）
FURN_Y_STDEV_RATIO = 0.035  # 相对 y 位置的标准差上限（÷ 页高）
FURN_BAND_TOP = 0.20  # 平均相对 y ≤ 此值 → 页眉带
FURN_BAND_BOTTOM = 0.80  # 平均相对 y ≥ 此值 → 页脚带
FURN_LABEL_TOP_BAND = 0.12  # 页码候选只在这些窄带里找（页眉）
FURN_LABEL_BOTTOM_BAND = 0.88  # 页码候选只在这些窄带里找（页脚）
FURN_LABEL_MIN_CONFIRM_RATIO = 0.60  # 页码一致性确认比例（value − page_index 的一致率）

_NUMBER_RUN_RE = re.compile(r"\d+")
_WHITESPACE_RUN_RE = re.compile(r"\s+")

_ROMAN_RE = re.compile(r"^[ivxlcdm]{1,7}$", re.IGNORECASE)
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}

# 页码模式：按优先级排列——含上下文的窄模式在前，裸数字在后（C5）
_PAGE_LABEL_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    ("page_of", re.compile(r"^\s*(?:page|p\.)\s*(\d{1,4})\s*(?:of|/)\s*(\d{1,4})\s*$", re.I)),
    ("zh", re.compile(r"^\s*第\s*(\d{1,4})\s*页(?:\s*[，,/]?\s*共\s*(\d{1,4})\s*页)?\s*$")),
    ("n_of_m", re.compile(r"^\s*(\d{1,4})\s*/\s*(\d{1,4})\s*$")),
    ("dash", re.compile(r"^\s*[-–—]\s*(\d{1,4})\s*[-–—]\s*$")),
    ("arabic", re.compile(r"^\s*(\d{1,4})\s*$")),
]

# 行内嵌页码：很多报告把页码拼在页眉/页脚文字里（ICCT 页脚是
# ``5 ICCT RESEARCH BRIEF | ...``，贝莱德是 ``…不得公幵派发8``），整行模式匹配不到。
# 两侧都用「前/后不是数字字母与 ./−%°」约束：除文档编号（``MKTG1225-...``）外，
# 还要挡住**图表数值标签**——贝莱德图里散落 ``13%`` ``10%`` 这类小字，
# 若把百分号前的数字当页码，整行就会被误剔（实测踩到）。数值范围由调用方用页数护栏再卡一道。
_NUM_GUARD = "0-9A-Za-z./%°-"
_LEAD_NUM_RE = re.compile(rf"^\s*(\d{{1,4}})(?![{_NUM_GUARD}])")
_TRAIL_NUM_RE = re.compile(rf"(?<![{_NUM_GUARD}])(\d{{1,4}})\s*$")


@dataclass
class PageLabel:
    """单页识别出的印刷页码（``page_index`` 才是物理页序，两者分开存）"""

    page: int
    label: str  # 印刷出来的页码字符串（"5" / "iv"）
    value: int | None
    kind: str
    line: str = ""  # 页码所在整行原文（诊断/剔除用）

    def as_dict(self) -> dict:
        return {
            "page_index": self.page,
            "label": self.label,
            "value": self.value,
            "kind": self.kind,
            "line": self.line,
        }


@dataclass
class FurnitureAnalysis:
    """整份文档的页眉页脚 + 页码分析结果（可直接序列化）"""

    n_pages: int = 0
    repeated: dict[str, dict] = field(default_factory=dict)  # 认定家具的文本 → 证据
    appearances: dict[str, int] = field(default_factory=dict)  # 全部文本 → 出现页数（供 C1）
    page_keys: dict[int, set[str]] = field(default_factory=dict)  # 每页被判为家具的行 key
    labels: dict[int, PageLabel] = field(default_factory=dict)
    page_map: dict[int, str] = field(default_factory=dict)  # page_index → page_label
    offset: int | None = None  # page_label − page_index 的众数（线性偏移）
    confirm_ratio: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_pages": self.n_pages,
            # key → {pages, mean_rel_y, stdev_rel_y, band, is_furniture}：保留完整证据，
            # 便于人工复盘「为什么把这条判成页眉」（只有 is_furniture=True 的才会被剔除）
            "furniture": self.repeated,
            "page_map": {str(k): v for k, v in sorted(self.page_map.items())},
            "labels": {str(k): v.as_dict() for k, v in sorted(self.labels.items())},
            "offset": self.offset,
            "confirm_ratio": round(self.confirm_ratio, 3),
            "notes": self.notes,
        }


def _furniture_key(text: str) -> str:
    """跨页比较用的文本指纹：折叠空白、小写、数字统一换成 #"""
    return _NUMBER_RUN_RE.sub("#", _WHITESPACE_RUN_RE.sub(" ", text).strip().lower())


def _roman_to_int(text: str) -> int | None:
    """把罗马数字转成整数；非法或超出 1~300 返回 None"""
    if not _ROMAN_RE.match(text):
        return None
    total = 0
    previous = 0
    for ch in reversed(text.lower()):
        value = _ROMAN_VALUES[ch]
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    return total if 0 < total <= 300 else None


def _match_page_label(text: str) -> tuple[str, int] | None:
    """把一行文字匹配成页码；返回 (模式名, 数值) 或 None"""
    for kind, pattern in _PAGE_LABEL_PATTERNS:
        match = pattern.match(text)
        if match:
            return kind, int(match.group(1))
    roman = _roman_to_int(text)
    if roman is not None:
        return "roman", roman
    return None


def _label_candidates(text: str, n_pages: int) -> list[tuple[str, int, str]]:
    """
    从一行文字里抽出全部「像页码」的候选，返回 ``[(kind, value, label)]``

    两级：先试**整行模式**（`Page 3 of 36` / `第 5 页` / 裸数字 / 罗马数字），
    命中即止；否则再找**行内嵌数字**（前缀或后缀）。所有候选都要过「值域护栏」
    （1 ≤ 值 ≤ 页数，前缀护栏放宽 5 页），把年份、文档编号挡在外面。
    """
    whole = _match_page_label(text)
    if whole is not None:
        kind, value = whole
        if 1 <= value <= n_pages + 5:
            return [(kind, value, text.strip() if kind == "roman" else str(value))]
        return []

    found: list[tuple[str, int, str]] = []
    for kind, pattern in (("lead_num", _LEAD_NUM_RE), ("trail_num", _TRAIL_NUM_RE)):
        for match in pattern.finditer(text):
            value = int(match.group(1))
            if 1 <= value <= n_pages:
                found.append((kind, value, str(value)))
    return found


def analyze_furniture(layouts: list[PageLayout]) -> FurnitureAnalysis:
    """
    识别整份文档的页眉页脚与印刷页码

    步骤：① 统计每个文本指纹出现在多少页上；② 对高频指纹再核对「位置是否固定」
    且「落在页眉/页脚带」（两条同时成立才认定为家具）；③ 逐页找页码候选，用
    「page_label − page_index 的众数」做文档级一致性确认，产出 ``page_map``。

    :param layouts: 第①步产出的逐页版面
    :return: FurnitureAnalysis（repeated / page_keys / page_map / offset ...）
    """
    analysis = FurnitureAnalysis(n_pages=len(layouts))
    if not layouts:
        analysis.notes.append("no_pages")
        return analysis

    pages_by_key: dict[str, list[tuple[int, float]]] = {}
    for layout in layouts:
        seen: set[str] = set()
        for line in layout.lines:
            text = line.text.strip()
            if not text:
                continue
            key = _furniture_key(text)
            if key in seen:
                continue  # 同一页出现多次 → 不可能是页眉页脚（每页只该出现一次）
            seen.add(key)
            rel_y = line.y_center / layout.height if layout.height else 0.0
            pages_by_key.setdefault(key, []).append((layout.page, rel_y))

    analysis.appearances = {key: len(hits) for key, hits in pages_by_key.items()}

    threshold = max(FURN_MIN_PAGES, int(FURN_MIN_PAGE_RATIO * len(layouts) + 0.999))
    for key, hits in pages_by_key.items():
        if len(hits) < threshold:
            continue
        rel_ys = [y for _, y in hits]
        stdev = statistics.pstdev(rel_ys) if len(rel_ys) > 1 else 0.0
        mean_y = statistics.fmean(rel_ys)
        band = "top" if mean_y <= FURN_BAND_TOP else ("bottom" if mean_y >= FURN_BAND_BOTTOM else "body")
        entry = {
            "pages": len(hits),
            "mean_rel_y": round(mean_y, 4),
            "stdev_rel_y": round(stdev, 4),
            "band": band,
            "is_furniture": band != "body" and stdev <= FURN_Y_STDEV_RATIO,
        }
        analysis.repeated[key] = entry

    for layout in layouts:
        keys: set[str] = set()
        for line in layout.lines:
            text = line.text.strip()
            if not text:
                continue
            key = _furniture_key(text)
            entry = analysis.repeated.get(key)
            if entry and entry["is_furniture"]:
                keys.add(key)
        analysis.page_keys[layout.page] = keys

    # ---- 页码：候选只从「家具行 + 上下窄带」取，再用 page_label − page_index 众数确认 ----
    candidates: dict[int, list[tuple[str, str, str, int]]] = {}
    for layout in layouts:
        keys = analysis.page_keys.get(layout.page, set())
        lines = [
            ln
            for ln in layout.lines
            if ln.text.strip()
            and (
                _furniture_key(ln.text.strip()) in keys
                or ln.y_center / layout.height <= FURN_LABEL_TOP_BAND
                or ln.y_center / layout.height >= FURN_LABEL_BOTTOM_BAND
            )
        ]
        found: list[tuple[str, str, str, int]] = []
        for ln in lines:
            text = ln.text.strip()
            for kind, value, label in _label_candidates(text, len(layouts)):
                found.append((text, kind, label, value))
        if found:
            candidates[layout.page] = found

    total = sum(len(v) for v in candidates.values())
    if total:
        diffs: dict[int, int] = {}
        for page, found in candidates.items():
            for _, _, _, value in found:
                diffs[value - page] = diffs.get(value - page, 0) + 1
        best_offset, best_count = max(diffs.items(), key=lambda kv: kv[1])
        analysis.confirm_ratio = best_count / total
        if best_count >= 2 and analysis.confirm_ratio >= FURN_LABEL_MIN_CONFIRM_RATIO:
            analysis.offset = best_offset
        else:
            analysis.notes.append(
                f"page_label_inconsistent(candidates={total}, best_offset={best_offset}, "
                f"ratio={analysis.confirm_ratio:.2f})"
            )

    if analysis.offset is not None:
        for page, found in sorted(candidates.items()):
            for text, kind, label, value in found:
                if value - page == analysis.offset:
                    analysis.labels[page] = PageLabel(
                        page=page, label=label, value=value, kind=kind, line=text
                    )
                    analysis.page_map[page] = label
                    break

    if not analysis.page_map:
        analysis.notes.append("no_page_label_detected")
    return analysis


# 整行判定为页码时才整行剔除；**行内嵌页码**（lead_num / trail_num）只在去掉数字后
# 残留的有效字符很少时才认定「这行就是页码」，否则整行保留——ICCT 的脚注
# ``6 PHEVs were excluded from the analysis due to limited data availability.``
# 行首的 ``6`` 是脚注序号而非页码，早先版本把它当页码整行删掉（实测 p6/p7 各丢一行正文）。
FURN_LABEL_RESIDUE_MAX = 6


def _label_residue_len(text: str, kind: str) -> int:
    """去掉行内嵌页码数字后，剩余的有效（字母 / 数字 / CJK）字符数"""
    if kind == "lead_num":
        match = _LEAD_NUM_RE.match(text)
        stripped = text[match.end():] if match else text
    elif kind == "trail_num":
        match = _TRAIL_NUM_RE.search(text)
        stripped = text[: match.start()] if match else text
    else:
        return 0  # 整行模式（Page 3 of 36 / 第 5 页 / 裸数字 / 罗马数字）：整行本身就是页码
    return len(re.sub(r"[\W_]+", "", stripped))


def strip_furniture(layout: PageLayout, analysis: FurnitureAnalysis) -> list[Line]:
    """剔除该页被判为页眉/页脚/页码的行（其余行保持原序）"""
    keys = analysis.page_keys.get(layout.page, set())
    label = analysis.labels.get(layout.page)
    kept: list[Line] = []
    for line in layout.lines:
        text = line.text.strip()
        if not text:
            continue
        if keys and _furniture_key(text) in keys:
            continue
        if label is not None and text == label.line:
            if _label_residue_len(text, label.kind) > FURN_LABEL_RESIDUE_MAX:
                kept.append(line)  # 页码只是嵌在正文行里，整行是内容，不能删
            continue
        kept.append(line)
    return kept


def clean_pages(
    layouts: list[PageLayout],
) -> tuple[list[PageLayout], FurnitureAnalysis]:
    """
    批量剔除页眉页脚，返回清洗后的版面与家具分析

    行对象是**同一个实例**（只做筛选，不复制），因此下游仍可拿到原始坐标。
    """
    analysis = analyze_furniture(layouts)
    cleaned = [
        PageLayout(
            page=layout.page,
            width=layout.width,
            height=layout.height,
            lines=strip_furniture(layout, analysis),
        )
        for layout in layouts
    ]
    return cleaned, analysis


# =============================================================================
# C4：扫描页检测（只检测 + 告警，本阶段不接 OCR）
# =============================================================================
#
# 存在的意义：``chunker._extract_pdf`` 用 ``if text:`` **静默跳过**空文本页，
# 扫描页会「表面成功、实际零 chunk」且不报错。这里把「提取不到文字」变成一条
# 显式的 ``needs_ocr`` 标记，随 chunk 落到索引与摄入汇总里。
#
# 两个判据（任一命中即检出）：① 可提取文本字符数近乎为空；② 图片面积占比过高。
# ``ocr_hook`` 是**按页触发的接入点**，默认 no-op——等真的出现扫描件再实现。

SCAN_MIN_TEXT_CHARS = 20  # 少于这么多字符即视作「几乎没有文本层」
SCAN_MAX_IMAGE_RATIO = 0.60  # 图片面积占页面面积的比例上限
SCAN_IMAGE_MAX_TEXT_CHARS = 80  # 仅在文本也很少时才用「图片占比」判据（排除设计封面）


def ocr_hook(page, page_index: int) -> str:
    """OCR 接入点（默认 no-op）。实现时应返回该页的识别文本，此处恒返回空串。"""
    return ""


def detect_scanned_pages(
    pdf_path: Path | str,
    *,
    min_chars: int = SCAN_MIN_TEXT_CHARS,
    max_image_ratio: float = SCAN_MAX_IMAGE_RATIO,
    image_max_chars: int = SCAN_IMAGE_MAX_TEXT_CHARS,
) -> dict[int, dict]:
    """
    逐页检测「疑似扫描页 / 无文本层页」

    :return: ``{page_index: {"needs_ocr": bool, "text_chars": int, "image_area_ratio": float, "reason": str}}``
    """
    import pymupdf

    results: dict[int, dict] = {}
    document = pymupdf.open(str(pdf_path))
    try:
        for page_no, page in enumerate(document, start=1):
            page_area = float(page.rect.width * page.rect.height) or 1.0
            text_chars = 0
            image_area = 0.0
            for block in page.get_text("dict").get("blocks", []):
                x0, y0, x1, y1 = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
                if block.get("type", 0) == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_chars += len((span.get("text") or "").strip())
                else:
                    image_area += max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
            ratio = min(1.0, image_area / page_area)
            reasons = []
            if text_chars < min_chars:
                reasons.append("no_text_layer")
            elif ratio > max_image_ratio and text_chars < image_max_chars:
                # 只有「文本也很少」时才可能是「正文由位图占据」；设计类封面
                # 虽然图片占满整页，但标题文字是可提取的，不该误报成扫描页。
                reasons.append("image_dominant")
            results[page_no] = {
                "needs_ocr": bool(reasons),
                "text_chars": text_chars,
                "image_area_ratio": round(ratio, 4),
                "reason": ",".join(reasons),
            }
    finally:
        document.close()
    return results
