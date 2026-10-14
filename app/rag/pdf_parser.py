"""
自研 PDF 解析器（第⑥步）：Section 树 → 按节切块 → schema v3 chunk

把几何层（``app/rag/pdf_layout.py``：行 / 页眉页脚 / 分栏 / 页码 / 扫描页）与
结构层（``app/rag/pdf_structure.py``：标题 / 表格）组装成**可直接落 chunks.jsonl**
的 chunk 列表。

装配规则
--------
1. 以**阅读顺序**走页（``reading_order_lines``，双栏区带按「左栏到底 → 右栏到底」）：
   单栏文档是恒等变换，不会比重排前更差；
2. 页眉页脚与页码行先剔除（第③步），**表格区域内的行也剔除**——否则同一份内容
   既进表格块又进正文块，检索时互相稀释；
3. 遇到 high/medium 标题 → 收口当前正文块、更新**标题栈**（level 小的压栈，
   同级/更高级的弹栈），标题文本进 ``section_path``，**不进正文文本**（与旧链路
   一致：``heading`` 是独立字段）；
4. 表格**独立成块**（``chunk_type="table"``），超长表按行组切、**每块重复表头**；
5. 正文按 ``chunker.split_text`` 的句子边界切块，元数据由本模块补齐；
6. **每页末收口正文缓冲**——否则一个段落跨页累积会让 ``page_index`` 停在段落起始页，
   整节 chunk 都标成节首页；``page_index`` 是 C5 要求的溯源/锚点真值，必须逐页准确。
7. **扫描页走 OCR**（C4 实现端，见 ``app/rag/ocr.py``）：被 ``detect_scanned_pages``
   检出的页用 OCR 整页文本产出 ``chunk_type="ocr"`` 块，把架构图 / 信息图 / 位图表格里
   **提取不到**的文字补回链路；引擎缺失时该页照常标 ``needs_ocr`` 但不产块。

为什么 ``block_index`` 语义变了
-------------------------------
旧链路（pypdf）的 ``block_index`` 是**页码**；这里升级为**节序号**（每遇到一个标题
就 +1）。这是一次 schema 变更，故 ``INDEX_SCHEMA_VERSION`` 2 → 3，必须整库重建。

schema v3 新增字段：``page_index`` / ``page_label`` / ``section_path`` /
``chunk_type`` / ``table`` / ``content_hash`` / ``parser_version`` / ``needs_ocr``。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from app.rag.chunker import _normalize_layout, _sanitize_text, split_text
from app.rag.config import OCR_DISABLED, PARSER_VERSION
from app.rag.pdf_layout import (
    clean_pages,
    detect_scanned_pages,
    extract_page_layouts,
    reading_order_lines,
)
from app.rag.pdf_structure import (
    HeadingCandidate,
    TableBlock,
    detect_headings,
    detect_tables,
)

# 表格区域内侧收边（pt）：行中心落进去即视为表格内容，从正文里剔除
TABLE_MARGIN = 2.0


@dataclass
class ParsedDocument:
    """一份 PDF 的完整解析产物（旁路预览/落盘都用它）"""

    doc: str
    page_count: int
    page_map: dict[int, str] = field(default_factory=dict)
    headings: list[HeadingCandidate] = field(default_factory=list)
    heading_candidates: list[HeadingCandidate] = field(default_factory=list)
    furniture: dict = field(default_factory=dict)
    tables: dict[int, list[TableBlock]] = field(default_factory=dict)
    scanned_pages: dict[int, dict] = field(default_factory=dict)
    chunks: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def heading_rows(self) -> list[dict]:
        return [h.as_dict(self.doc) for h in self.heading_candidates]

    def table_rows(self) -> list[dict]:
        rows: list[dict] = []
        for blocks in self.tables.values():
            for block in blocks:
                rows.append({"doc": self.doc, **block.as_dict()})
        return rows


def _inside(line, boxes: list[tuple[float, float, float, float]]) -> bool:
    """行的中心点是否落在任一个表格区域内"""
    cx = (line.x0 + line.x1) / 2
    cy = line.y_center
    for x0, y0, x1, y1 in boxes:
        if (
            x0 - TABLE_MARGIN <= cx <= x1 + TABLE_MARGIN
            and y0 - TABLE_MARGIN <= cy <= y1 + TABLE_MARGIN
        ):
            return True
    return False


def _make_chunk(
    *,
    doc: str,
    text: str,
    section_path: list[str],
    section_index: int,
    seq: int,
    page_index: int,
    page_label: str | None,
    chunk_type: str,
    doc_lang: str,
    doc_cjk_ratio: float,
    table_meta: dict | None,
    needs_ocr: bool,
) -> dict:
    """
    组装单个 chunk（schema v3）

    ``id`` 沿用旧链路的 ``doc::序号::标题::块内序号`` 形态，但序号语义是**节序号**；
    ``heading`` 取 ``section_path`` 末级，供展示；``zh_text`` 与旧链路一致由 lang 派生。
    """
    # 组装层唯一出口：chunk 的全部字符串字段在这里统一清洗。
    # 表格单元格文本来自 PyMuPDF 的 ``Table.extract()``——这条取字路径**不经过**
    # ``page.get_text()`` 的字符映射，实测会残留 U+D800–U+DFFF 孤立代理码位
    # （直接扫 ``get_text()`` 一个都查不到，所以那个口径会漏掉）。此类码位无法编为
    # UTF-8，会在 ``content_hash`` / 落盘 / 写 Qdrant 时抛 UnicodeEncodeError，
    # 直接中断整库摄入（实测两份新增报告）。在出口兜底后，无论上游文本来自
    # 表格提取、OCR 还是正文行，chunk 都可安全编码。
    text = _sanitize_text(text)
    heading = _sanitize_text(section_path[-1]) if section_path else ""
    section_path = [_sanitize_text(part) for part in section_path]
    table_meta = (
        {k: (_sanitize_text(v) if isinstance(v, str) else v) for k, v in table_meta.items()}
        if table_meta
        else table_meta
    )
    return {
        "id": f"{doc}::{section_index}::{heading or 'body'}::{seq}",
        "doc": doc,
        "heading": heading,
        "text": text,
        "lang": doc_lang,
        "cjk_ratio": doc_cjk_ratio,
        "zh_text": text if doc_lang == "zh" else "",
        "page_index": page_index,
        "page_label": page_label,
        "section_path": list(section_path),
        "chunk_type": chunk_type,
        "table": table_meta,
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        "parser_version": PARSER_VERSION,
        "needs_ocr": needs_ocr,
    }


def parse_document(
    pdf_path: Path | str,
    rel_name: str,
    *,
    doc_lang: str = "",
    doc_cjk_ratio: float = 0.0,
    use_text_tables: bool = False,
) -> ParsedDocument:
    """
    解析单份 PDF，产出 schema v3 的 chunk 列表

    :param pdf_path: PDF 路径
    :param rel_name: 写入 chunk ``doc`` 字段的名字（相对知识库根目录）
    :param doc_lang: 文档级语言（``zh`` / ``en``），决定是否启用「全大写」标题信号与 ``zh_text``
    :param doc_cjk_ratio: 文档级汉字占比，随 lang 一起落盘
    :param use_text_tables: 是否启用无框线表格兜底（默认关闭，见 ``detect_tables``）
    """
    layouts = extract_page_layouts(pdf_path)
    cleaned, furniture = clean_pages(layouts)
    heading_result = detect_headings(cleaned, furniture, doc_lang=doc_lang)
    tables = detect_tables(pdf_path, cleaned, use_text_strategy=use_text_tables)
    scanned = detect_scanned_pages(pdf_path)

    # C4 实现端：对检出「无文本层 / 图片主导」的页做 OCR，把图内文字补回检索链路。
    # 引擎未安装时静默跳过（扫描页仍带 needs_ocr 标记入库，不影响其余页）。
    ocr_texts: dict[int, str] = {}
    if not OCR_DISABLED:
        scanned_idx = [p for p, v in scanned.items() if v["needs_ocr"]]
        if scanned_idx:
            from app.rag.ocr import ocr_document

            ocr_texts = ocr_document(pdf_path, scanned_idx)

    # 表格区域（含跨页表拼进来的后续页区域）→ 正文要排除
    boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    for blocks in tables.values():
        for block in blocks:
            boxes.setdefault(block.page_index, []).append(block.bbox)
            for page, box in block.extra_boxes:
                boxes.setdefault(page, []).append(box)

    heading_index = {(h.page_index, h.y_center): h for h in heading_result.headings}

    page_map = furniture.page_map
    chunks: list[dict] = []
    stack: list[HeadingCandidate] = []
    section_path: list[str] = []
    section_index = 0
    seq = 0
    buffer: list[str] = []
    buffer_page: int | None = None

    def section_reset() -> None:
        """进入新一节：正文收口，块内序号归零"""
        nonlocal seq
        seq = 0

    def flush() -> None:
        nonlocal buffer, buffer_page, seq
        if not buffer:
            return
        text = _normalize_layout("\n".join(buffer))
        page = buffer_page or 1
        buffer = []
        buffer_page = None
        if not text:
            return
        label = page_map.get(page)
        needs_ocr = bool(scanned.get(page, {}).get("needs_ocr"))
        for piece in split_text(text):
            seq += 1
            chunks.append(
                _make_chunk(
                    doc=rel_name,
                    text=piece,
                    section_path=section_path,
                    section_index=section_index,
                    seq=seq,
                    page_index=page,
                    page_label=label,
                    chunk_type="body",
                    doc_lang=doc_lang,
                    doc_cjk_ratio=doc_cjk_ratio,
                    table_meta=None,
                    needs_ocr=needs_ocr,
                )
            )

    for layout in cleaned:
        page = layout.page
        page_label = page_map.get(page)
        page_needs_ocr = bool(scanned.get(page, {}).get("needs_ocr"))

        # 扫描 / 图片主导页：可提取的行基本只剩页眉页码（噪声），直接用 OCR 整页文本，
        # 作为独立的 ``chunk_type="ocr"`` 块产出——扫描页是一张位图，没有可靠的行 / 列
        # 结构可复用，故不入行管线。OCR 文本仍走 ``split_text``，受 CHUNK_SIZE 约束。
        ocr_text = ocr_texts.get(page, "")
        if ocr_text:
            flush()
            # OCR 的每一「行」是独立的文本框（架构图 / 图谱里的不同标签），**不是**同一
            # 段落的视觉续行——故先用空格把行连起来再归一化。否则 ``_normalize_layout``
            # 会把「CJK 之间的换行」直接删掉，导致词与词粘连（``数据本体仿真合成数据``）。
            for piece in split_text(_normalize_layout(ocr_text.replace("\n", " "))):
                seq += 1
                chunks.append(
                    _make_chunk(
                        doc=rel_name,
                        text=piece,
                        section_path=section_path,
                        section_index=section_index,
                        seq=seq,
                        page_index=page,
                        page_label=page_label,
                        chunk_type="ocr",
                        doc_lang=doc_lang,
                        doc_cjk_ratio=doc_cjk_ratio,
                        table_meta=None,
                        needs_ocr=True,
                    )
                )
            continue

        page_boxes = boxes.get(page, [])
        page_tables = sorted(tables.get(page, []), key=lambda t: t.bbox[1])
        table_cursor = 0

        for line in reading_order_lines(layout):
            if not line.text.strip():
                continue
            if page_boxes and _inside(line, page_boxes):
                continue  # 表格内容走独立的 table chunk

            heading = heading_index.get((page, round(line.y_center, 1)))
            if heading is not None:
                flush()
                while stack and (stack[-1].level or 9) >= (heading.level or 9):
                    stack.pop()
                stack.append(heading)
                section_path = [h.text for h in stack]
                section_index += 1
                section_reset()
                continue

            # 表格按纵向位置插进阅读顺序：行已经越过表格顶部就先出表格块
            while table_cursor < len(page_tables) and line.y_center > page_tables[
                table_cursor
            ].bbox[1]:
                flush()
                _emit_table(
                    chunks,
                    page_tables[table_cursor],
                    rel_name,
                    section_path,
                    section_index,
                    doc_lang,
                    doc_cjk_ratio,
                    page_label,
                    page_needs_ocr,
                )
                table_cursor += 1

            if buffer_page is None:
                buffer_page = page
            buffer.append(line.text)

        # 页尾还没处理的表格（表格在正文之下）
        while table_cursor < len(page_tables):
            flush()
            _emit_table(
                chunks,
                page_tables[table_cursor],
                rel_name,
                section_path,
                section_index,
                doc_lang,
                doc_cjk_ratio,
                page_label,
                page_needs_ocr,
            )
            table_cursor += 1

        # **必须在这里收口**：正文缓冲跨页累积会让 `buffer_page` 停在段落起始页，
        # 于是整节的 chunk 都标成「节首页」——`page_index` 是锚点与溯源用的真值
        # （C5），必须落在文字真实所在页。代价是跨页段落被拆成两个无重叠的 chunk，
        # 对检索影响可忽略（跨页段落本就是少数）。
        flush()

    flush()

    # 块内序号是**节内**连续的，表块与正文块共享同一计数器，故最后统一重编号
    _renumber(chunks)

    stats = {
        "chunks": len(chunks),
        "body": sum(1 for c in chunks if c["chunk_type"] == "body"),
        "table": sum(1 for c in chunks if c["chunk_type"] == "table"),
        "ocr": sum(1 for c in chunks if c["chunk_type"] == "ocr"),
        "sections": section_index,
        "headings": len(heading_result.headings),
        "heading_candidates": len(heading_result.candidates),
        "tables": sum(len(v) for v in tables.values()),
        "scanned_pages": sorted(p for p, v in scanned.items() if v["needs_ocr"]),
    }
    return ParsedDocument(
        doc=rel_name,
        page_count=len(layouts),
        page_map=page_map,
        headings=heading_result.headings,
        heading_candidates=heading_result.candidates,
        furniture=furniture.as_dict(),
        tables=tables,
        scanned_pages=scanned,
        chunks=chunks,
        stats=stats,
    )


def _emit_table(
    chunks: list[dict],
    block: TableBlock,
    rel_name: str,
    section_path: list[str],
    section_index: int,
    doc_lang: str,
    doc_cjk_ratio: float,
    page_label: str | None,
    needs_ocr: bool,
) -> None:
    """把一个表格块落成 1..N 个 chunk（超长表按行组切，每块重复表头）"""
    pieces = block.split_for_chunks()
    for index, piece in enumerate(pieces, start=1):
        text = piece if len(pieces) == 1 else f"{piece}\n(第 {index}/{len(pieces)} 段)"
        chunks.append(
            _make_chunk(
                doc=rel_name,
                text=text,
                section_path=section_path,
                section_index=section_index,
                seq=0,  # 统一在 _renumber 里赋值
                page_index=block.page_index,
                page_label=page_label,
                chunk_type="table",
                doc_lang=doc_lang,
                doc_cjk_ratio=doc_cjk_ratio,
                table_meta={
                    "caption": block.caption,
                    "units": block.units,
                    "n_rows": block.n_rows,
                    "n_cols": block.n_cols,
                    "header_rows": block.header_rows,
                    "bbox": [round(v, 1) for v in block.bbox],
                    "part": index,
                    "parts": len(pieces),
                },
                needs_ocr=needs_ocr,
            )
        )


def _renumber(chunks: list[dict]) -> None:
    """按（节序号, 出现顺序）重编块内序号，保证 id 唯一且稳定"""
    counters: dict[int, int] = {}
    for chunk in chunks:
        section = chunk["id"].split("::")[1]
        counters[section] = counters.get(section, 0) + 1
        parts = chunk["id"].split("::")
        parts[-1] = str(counters[section])
        chunk["id"] = "::".join(parts)
