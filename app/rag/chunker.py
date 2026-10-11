"""
文档提取与结构感知切分模块

摄入流水线的第一、二步：
1. extract_document: 按格式提取纯文本（PDF / DOCX / Markdown / TXT）；
   PDF 提取后额外做一次版面换行归一化（_normalize_layout），把 pypdf 的视觉换行
   还原成句子流，避免行内换行污染 chunk
2. chunk_text: 结构感知切分——优先按标题边界切块，块内再按句子边界逐级下钻，
   直到满足目标长度；相邻块保留 overlap 防止语义断裂。
   句子边界同时覆盖中英文（中文句读 + 英文 . ! ? ;），故中英文语料都能按句切块

切分策略是 RAG 质量的第一决定因素：按固定长度硬切会把完整论述拦腰截断，
按标题 + 句子边界切出来的块才能独立成义，检索命中后才真正可用。

除切分外，本模块还负责给每个 chunk 盖上语言字段（lang / cjk_ratio / zh_text），
这是关键词路「字段级语言路由」的摄入侧准备，见 ADR-010 与 app/rag/language.py。
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
        blocks = _extract_pdf(file_path)
    elif suffix == ".docx":
        blocks = _extract_docx(file_path)
    elif suffix == ".md":
        blocks = _extract_markdown(file_path)
    elif suffix == ".txt":
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        blocks = [TextBlock(text=text, heading="")]
    else:
        return []

    # 提取层是全文唯一的文本入口，在这里统一清洗，下游（chunks.jsonl / Qdrant / ES）都安全
    return [
        TextBlock(text=_sanitize_text(b.text), heading=_sanitize_text(b.heading))
        for b in blocks
    ]


def _sanitize_text(text: str) -> str:
    """
    剔除文本中的孤立代理字符（lone surrogates）

    pypdf 解析部分 PDF 的字体 CMap / ToUnicode 表时会解出 U+D800–U+DFFF 区间的
    孤立代理码位（不能构成合法 UTF-8）。这类文本若直接落盘会抛
    ``UnicodeEncodeError: surrogates not allowed``，**导致整个知识库的摄入中断**
    （实测出现于两份新增报告）。这里统一清除，使摄入对来源文档的编码瑕疵免疫。

    :param text: 原始文本
    :return: 可安全编码为 UTF-8 的文本
    """
    if not text:
        return text
    # errors="ignore" 会跳过所有无法编码的码位（含孤立代理），是这里最稳的兜底
    return text.encode("utf-8", "ignore").decode("utf-8")


# CJK 字符区间：中日韩统一表意文字 + 中文标点 + 全角符号。用于判断换行两侧是否都是中文
_CJK_RANGE = "\u3000-\u303f\u4e00-\u9fff\uff00-\uffef"


def _normalize_layout(text: str) -> str:
    """
    把 PDF 提取文本里的「视觉换行」还原成正常的句子流

    pypdf 按视觉行输出，整段文字里塞满了行内换行（实测如 ``'Global EV \\nOutlook 2026'``、
    ``'The IEA examines the \\nfull spectrum'``）。这些换行既让 chunk 里混进噪声、影响
    向量语义，也会干扰句子边界识别。这里做两步归一化：
    1. 两个 CJK 字符之间的换行直接删除（中文行内换行本就不需要空格）
    2. 其余换行统一压成一个空格，并把连续空白折叠为单个空格

    注意：段落分隔信息在此被抹平——当前切分只依赖句末标点，不依赖段落，故无影响。
    :param text: 原始提取文本
    :return: 句子连续、无行内换行的文本
    """
    if not text:
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 中文之间的换行：直接相连（避免在汉字词中间插入空格）
    text = re.sub(rf"(?<=[{_CJK_RANGE}])\n(?=[{_CJK_RANGE}])", "", text)
    # 其余换行（英文断行、页眉页脚与正文之间等）：替换为空格
    text = re.sub(r"\s*\n\s*", " ", text)
    # 折叠重复空白（含全角空格）
    text = re.sub(r"[ \t\u3000]{2,}", " ", text)
    return text.strip()


def _extract_pdf(file_path: Path) -> list[TextBlock]:
    """按页提取 PDF 文本，每页一个块，heading 记录页码便于溯源"""
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    blocks = []
    for page_no, page in enumerate(reader.pages, start=1):
        # 提取后立即做版面换行归一化，后续切分与向量化面对的都是干净的句子流
        text = _normalize_layout(page.extract_text() or "")
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


# 句子边界。两个分支：
# 1) 中文句读与英文 ! ? ; —— 出现在这些符号后即为句末，直接断开；
# 2) 英文句号 . —— 必须「后接连空白 + 大写字母/数字/引号/括号」才算句末。
#    这条限制是为了避开小数（735.43）、缩写（U.S.）、版本号（v1.2.3）：
#    它们的句号后面紧跟字符而非空格，故不会被误切。
# 早期版本只有分支 1，缺英文句号，导致英文整页找不到句子边界、直接落进
# 「定长硬切」兜底分支（实测 82% 的英文 chunk 断在半句/半词）。
_SENTENCE_BREAK_RE = re.compile(
    r"(?<=[。！？；!?;])"
    r"|(?<=\.)(?=\s+[A-Z0-9\"'“”‘’(（])"
)


def chunk_text(
    block: TextBlock,
    source_doc: str,
    block_index: int = 1,
    doc_lang: str = "",
    doc_cjk_ratio: float = 0.0,
) -> list[dict]:
    """
    把单个 TextBlock 切分为目标长度的 chunk 列表

    切分优先级：目标长度内直接成块 → 超长时按句号下钻 → 单句仍超长时硬切。
    相邻 chunk 保留 CHUNK_OVERLAP 重叠。
    :param block: 提取得到的文本块
    :param source_doc: 所属文档名（写入 chunk 元数据，供检索结果溯源）
    :param block_index: 该块在文档内的序号，用于保证 id 在「同标题的多个块」之间也不冲突
    :param doc_lang: 所属文档的语言（"zh" / "en"），由调用方按**整份文档**判定后传入
    :param doc_cjk_ratio: 所属文档的汉字占比，随 lang 一起落盘便于观测与调阈值
    :return: chunk 字典列表，字段 id / doc / heading / text / lang / cjk_ratio / zh_text
    """
    text = block.text.strip()
    if not text:
        return []

    if len(text) <= CHUNK_SIZE:
        # 不超目标长度：整块作为一个 chunk，无需切分
        chunks = [_make_chunk(source_doc, block.heading, text)]
    else:
        chunks = _split_long_text(text, source_doc, block.heading)

    # 统一编号：id 必须对**所有** chunk 赋值。早期版本只在超长分支里编号，
    # 导致未切分的短块 id 为空字符串——写入 Qdrant 时这些块被 uuid5 折叠成
    # 同一个 point，向量严重丢失（617 个 chunk 只剩 539 个点）。
    for seq, chunk in enumerate(chunks, start=1):
        chunk["id"] = f"{source_doc}::{block_index}::{block.heading or 'body'}::{seq}"

    # 语言字段在**文档级**统一盖章，不在块内重算——块级判定会被英文报告里的
    # 中文碎片块（cjk≈0.15~0.17）带偏，把噪声块判成"中文块"（见 ADR-010）。
    #
    # zh_text 是关键词路的「中文可检索视图」：
    # - 中文文档 → 原文，中文 query 打 zh_text 时同语言关键词增益照常生效；
    # - 英文文档 → 空串，中文 query 打 zh_text 时该文档的块天生不在候选里，
    #   关键词路在跨语言场景下自然退出、检索退化为向量路（实测最优行为）；
    #   英文文档仍由向量路用 text 原文召回，召回能力不丢。
    # 这是**字段级隔离**，不依赖任何阈值过滤——对比「按块语言过滤结果」的近似做法，
    # 后者只能剔掉显眼的碎片块，实测只把 Recall@5 从 0% 恢复到 15%。
    for chunk in chunks:
        chunk["lang"] = doc_lang
        chunk["cjk_ratio"] = doc_cjk_ratio
        chunk["zh_text"] = chunk["text"] if doc_lang == "zh" else ""
    return chunks


def _split_long_text(text: str, source_doc: str, heading: str) -> list[dict]:
    """超长文本按句子边界下钻切分（返回尚未编号的 chunk 列表）"""
    sentences = [s for s in _SENTENCE_BREAK_RE.split(text) if s.strip()]
    chunks: list[dict] = []
    buffer = ""
    for sentence in sentences:
        # 单句超过目标长度（如长表格、URL 堆积）：按目标长度硬切
        if len(sentence) > CHUNK_SIZE:
            if buffer:
                chunks.append(_make_chunk(source_doc, heading, buffer))
                buffer = ""
            for i in range(0, len(sentence), CHUNK_SIZE - CHUNK_OVERLAP):
                piece = sentence[i : i + CHUNK_SIZE]
                if piece.strip():
                    chunks.append(_make_chunk(source_doc, heading, piece))
            continue

        if len(buffer) + len(sentence) > CHUNK_SIZE and buffer:
            chunks.append(_make_chunk(source_doc, heading, buffer))
            # overlap：新块以旧块尾部开头，保证跨块语义连续
            buffer = buffer[-CHUNK_OVERLAP:] + sentence
        else:
            buffer += sentence
    if buffer.strip():
        chunks.append(_make_chunk(source_doc, heading, buffer))
    return chunks


def _make_chunk(source_doc: str, heading: str, text: str) -> dict:
    return {
        "id": "",
        "doc": source_doc,
        "heading": heading,
        "text": text.strip(),
    }
