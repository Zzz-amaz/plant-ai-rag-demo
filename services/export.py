"""Markdown 导出为 Word 文档。

只覆盖本项目实际用到的语法子集：标题、段落、列表、引用、表格与图片。
不追求通用 Markdown 兼容性，但保证导出结果在 Word 中可直接阅读。
"""

from __future__ import annotations

import io
import os
import re
from typing import List

from docx import Document
from docx.shared import Inches, Pt

_IMAGE_RE = re.compile(r"!\[([^\]]*)\][（(]([^\r\n）)]+)[）)]")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.*)$")
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_RULE_RE = re.compile(r"^\s*(?:---+|\*\*\*+)\s*$")


def markdown_to_docx(markdown: str, *, title: str = "分析报告") -> bytes:
    """把 Markdown 转成 .docx 字节流。"""
    document = Document()
    document.add_heading(title, level=0)
    _write_blocks(document, markdown.splitlines())
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _write_blocks(document: Document, lines: List[str]) -> None:
    index = 0
    total = len(lines)
    while index < total:
        line = lines[index].rstrip()
        stripped = line.strip()

        if not stripped or _RULE_RE.match(stripped):
            index += 1
            continue

        image_match = _IMAGE_RE.search(stripped)
        if image_match and stripped.startswith("!"):
            _add_image(document, image_match.group(2), image_match.group(1))
            index += 1
            continue

        if stripped.startswith("|"):
            index = _add_table(document, lines, index)
            continue

        heading = _HEADING_RE.match(stripped)
        if heading:
            level = min(len(heading.group(1)), 4)
            document.add_heading(_plain(heading.group(2)), level=level)
            index += 1
            continue

        if stripped.startswith(">"):
            paragraph = document.add_paragraph()
            run = paragraph.add_run(_plain(stripped.lstrip("> ").strip()))
            run.italic = True
            index += 1
            continue

        bullet = _BULLET_RE.match(line)
        if bullet:
            document.add_paragraph(_plain(bullet.group(1)), style="List Bullet")
            index += 1
            continue

        numbered = _NUMBERED_RE.match(line)
        if numbered:
            document.add_paragraph(_plain(numbered.group(1)), style="List Number")
            index += 1
            continue

        paragraph = document.add_paragraph()
        for fragment, is_bold in _split_bold(stripped):
            run = paragraph.add_run(fragment)
            run.bold = is_bold
        index += 1


def _add_table(document: Document, lines: List[str], start: int) -> int:
    """把连续的 Markdown 表格行转成 Word 表格，返回下一行下标。"""
    rows: List[List[str]] = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        raw = lines[index].strip().strip("|")
        cells = [cell.strip() for cell in raw.split("|")]
        if not all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells if cell):
            rows.append([_plain(cell) for cell in cells])
        index += 1

    if not rows:
        return index

    column_count = max(len(row) for row in rows)
    table = document.add_table(rows=len(rows), cols=column_count)
    table.style = "Table Grid"
    for row_index, row in enumerate(rows):
        for column_index in range(column_count):
            text = row[column_index] if column_index < len(row) else ""
            cell = table.cell(row_index, column_index)
            cell.text = text
            if row_index == 0:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.bold = True
    return index


def _add_image(document: Document, path: str, caption: str) -> None:
    """插入图片；文件缺失时退化为一行说明文字。"""
    cleaned = str(path).strip().rstrip("）】》")
    if os.path.isfile(cleaned):
        try:
            document.add_picture(cleaned, width=Inches(6.0))
            return
        except Exception:  # noqa: BLE001 - 图片格式异常时退化为文字
            pass
    paragraph = document.add_paragraph()
    run = paragraph.add_run(f"[图片缺失] {caption or os.path.basename(cleaned)}")
    run.font.size = Pt(9)


def _plain(text: str) -> str:
    """去掉行内加粗标记，用于表格、标题等不支持富文本的位置。"""
    return _BOLD_RE.sub(r"\1", text).replace("`", "")


def _split_bold(text: str):
    """把 ``**加粗**`` 拆成 (片段, 是否加粗) 序列。"""
    position = 0
    for match in _BOLD_RE.finditer(text):
        if match.start() > position:
            yield text[position : match.start()], False
        yield match.group(1), True
        position = match.end()
    if position < len(text):
        yield text[position:], False
