"""
文档提取与结构感知切分模块

摄入流水线的第一、二步：
1. extract_document: 按格式提取纯文本（PDF / DOCX / Markdown / TXT）
2. chunk_text: 结构感知切分——优先按标题边界切块，块内再按段落、句号
   逐级下钻，直到满足目标长度；相邻块保留 overlap 防止语义断裂。

切分策略是 RAG 质量的第一决定因素：按固定长度硬切会把完整论述拦腰截断，
按标题 + 句子边界切出来的块才能独立成义，检索命中后才真正可用。
"""

import re
from dataclasses import dataclass
from pathlib import Path

from app.rag.config import CHUNK_OVERLAP, CHUNK_SIZE

# 支持摄入的文档后缀（知识库目录里暂时只有 PDF，其余格式为前瞻支持）
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".txt"}


@dataclass
class TextBlock:
    """切分前的中间结构：带章节路径的文本块"""

    text: str
    heading: str  # 所属章节标题，用于检索结果的来源定位


def extract_document(file_path: Path) -> list[TextBlock]:
    """
    提取单个文档，返回按章节组织的文本块列表

    - PDF：按页提取，每页一个块（页是 PDF 能拿到的最可靠结构单元）
    - DOCX：按段落提取，heading 样式的段落视为章节标题
    - Markdown：按 # 标题切节
    - TXT：整体作为一个块
    :param file_path: 文档路径
    :return: TextBlock 列表；提取失败时抛出的异常由调用方（ingest）统一降级处理
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(file_path)
    if suffix == ".docx":
        return _extract_docx(file_path)
    if suffix == ".md":
        return _extract_markdown(file_path)
    if suffix == ".txt":
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return [TextBlock(text=text, heading="")]
    return []


def _extract_pdf(file_path: Path) -> list[TextBlock]:
    """按页提取 PDF 文本，每页一个块，heading 记录页码便于溯源"""
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    blocks = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            blocks.append(TextBlock(text=text, heading=f"第{page_no}页"))
    return blocks


def _extract_docx(file_path: Path) -> list[TextBlock]:
    """提取 DOCX：heading 样式段落作为章节边界，普通段落累积到当前章节"""
    import docx

    document = docx.Document(str(file_path))
    blocks = []
    current_heading = ""
    current_lines: list[str] = []

    def flush():
        text = "\n".join(current_lines).strip()
        if text:
            blocks.append(TextBlock(text=text, heading=current_heading))
        current_lines.clear()

    for para in document.paragraphs:
        style_name = (para.style.name or "").lower()
        if style_name.startswith("heading") or style_name.startswith("标题"):
            flush()
            current_heading = para.text.strip()
        elif para.text.strip():
            current_lines.append(para.text.strip())
    flush()
    return blocks


_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


def _extract_markdown(file_path: Path) -> list[TextBlock]:
    """提取 Markdown：按 #~#### 标题切节，节内文本（含表格）整体保留"""
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    matches = list(_HEADING_RE.finditer(content))
    if not matches:
        return [TextBlock(text=content.strip(), heading="")] if content.strip() else []

    blocks = []
    # 第一个标题之前的内容（如有）作为无章节块
    if matches[0].start() > 0 and content[: matches[0].start()].strip():
        blocks.append(
            TextBlock(text=content[: matches[0].start()].strip(), heading="")
        )
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section_text = content[start:end].strip()
        if section_text:
            blocks.append(TextBlock(text=section_text, heading=match.group(2).strip()))
    return blocks


# 中文句读符号：在这些符号后切分不会破坏语义
_SENTENCE_BREAK_RE = re.compile(r"(?<=[。！？；!?;])")


def chunk_text(block: TextBlock, source_doc: str) -> list[dict]:
    """
    把单个 TextBlock 切分为目标长度的 chunk 列表

    切分优先级：目标长度内直接成块 → 超长时按句号下钻 → 单句仍超长时硬切。
    相邻 chunk 保留 CHUNK_OVERLAP 重叠。
    :param block: 提取得到的文本块
    :param source_doc: 所属文档名（写入 chunk 元数据，供检索结果溯源）
    :return: chunk 字典列表，字段 id / doc / heading / text
    """
    text = block.text.strip()
    if not text:
        return []

    # 不超目标长度：整块作为一个 chunk，无需切分
    if len(text) <= CHUNK_SIZE:
        return [_make_chunk(source_doc, block.heading, text)]

    # 超长：按句子边界下钻
    sentences = [s for s in _SENTENCE_BREAK_RE.split(text) if s.strip()]
    chunks = []
    buffer = ""
    for sentence in sentences:
        # 单句超过目标长度（如长表格、URL 堆积）：按目标长度硬切
        if len(sentence) > CHUNK_SIZE:
            if buffer:
                chunks.append(_make_chunk(source_doc, block.heading, buffer))
                buffer = ""
            for i in range(0, len(sentence), CHUNK_SIZE - CHUNK_OVERLAP):
                piece = sentence[i : i + CHUNK_SIZE]
                if piece.strip():
                    chunks.append(_make_chunk(source_doc, block.heading, piece))
            continue

        if len(buffer) + len(sentence) > CHUNK_SIZE and buffer:
            chunks.append(_make_chunk(source_doc, block.heading, buffer))
            # overlap：新块以旧块尾部开头，保证跨块语义连续
            buffer = buffer[-CHUNK_OVERLAP:] + sentence
        else:
            buffer += sentence
    if buffer.strip():
        chunks.append(_make_chunk(source_doc, block.heading, buffer))

    # 给每个 chunk 编上块内序号，保证 id 在文档内唯一且稳定
    for seq, chunk in enumerate(chunks, start=1):
        chunk["id"] = f"{source_doc}::{block.heading or seq}::{seq}"
    return chunks


def _make_chunk(source_doc: str, heading: str, text: str) -> dict:
    return {
        "id": "",
        "doc": source_doc,
        "heading": heading,
        "text": text.strip(),
    }
