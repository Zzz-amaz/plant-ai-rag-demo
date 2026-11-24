"""加载设备知识文档（.txt / .md / .pdf / .docx）。

原项目从加密资源包解密到内存；演示版直接读取 ``devices/<device>/``，
保留相同的分段、标题识别与 MD5 增量逻辑。
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from core.logging import logger

from core.config import settings

_SKIP_FILES = {
    "device.json",
    "system_prompt.txt",
    "sample_input.txt",
    "prompt_data_analysis.txt",
    "prompt_chart_guide.txt",
}
_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}
_DOCUMENT_LOADER_SCHEMA = "docx-sections-v2"


def _compute_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _get_md5_registry_path(device_name: str) -> Path:
    registry_dir = settings.data_dir / "md5_registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    return registry_dir / f"{device_name}.json"


def _load_md5_registry(device_name: str) -> dict[str, str]:
    path = _get_md5_registry_path(device_name)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning(f"MD5 注册表损坏，将全量重建: {device_name}")
    return {}


def _save_md5_registry(device_name: str, registry: dict[str, str]) -> None:
    path = _get_md5_registry_path(device_name)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_device_files(device_name: str) -> dict[str, bytes]:
    """读取设备知识文件（.txt/.md/.pdf/.docx）。

    原项目从加密的 ``secure_assets.bin`` 中解密；演示版直接读
    ``devices/<device_name>/`` 明文目录，行为与返回值保持一致。
    """
    result: dict[str, bytes] = {}
    device_dir = settings.devices_dir / device_name
    if not device_dir.exists():
        logger.warning(f"设备目录不存在: {device_dir}")
        return result
    for path in device_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name in _SKIP_FILES or path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        result[path.relative_to(device_dir).as_posix()] = path.read_bytes()
    return result


def load_device_documents(
    device_name: str | None = None,
    incremental: bool = True,
) -> tuple[list[Document], bool]:
    """Load changed knowledge files for a device."""
    device_name = device_name or settings.DEFAULT_DEVICE
    current_files = _get_device_files(device_name)
    if not current_files:
        return [], False

    current_md5s = {name: _compute_md5(data) for name, data in current_files.items()}

    if incremental:
        old_md5s = _load_md5_registry(device_name)
        schema_changed = old_md5s.get("_loader_schema") != _DOCUMENT_LOADER_SCHEMA
        if schema_changed:
            # The DOCX parser now preserves section boundaries and metadata.
            # Rebuild all device documents so old unsplit chunks are not kept.
            files_to_load = current_files
            effective_incremental = False
            logger.info(
                f"设备 [{device_name}] 文档解析器已更新，准备全量重建"
            )
        else:
            files_to_load = {
                name: data
                for name, data in current_files.items()
                if old_md5s.get(name) != current_md5s[name]
            }
            effective_incremental = True
        if not files_to_load:
            logger.info(f"设备 [{device_name}] 文档无变化，跳过索引")
            return [], True
        logger.info(
            f"设备 [{device_name}] 增量索引: {len(files_to_load)}/{len(current_files)} 个文件有变化"
        )
    else:
        files_to_load = current_files
        effective_incremental = False

    documents: list[Document] = []
    for filename, data in files_to_load.items():
        try:
            docs = _load_single_file(filename, data, device_name)
            documents.extend(docs)
            if docs:
                logger.debug(f"已加载文档: {filename} ({len(docs)} 段)")
        except Exception as exc:
            logger.warning(f"加载文档失败: {filename}, 错误: {exc}")

    logger.info(
        f"设备 [{device_name}] 加载 {len(documents)} 份文档（{'增量' if incremental else '全量'}）"
    )
    registry = dict(current_md5s)
    registry["_loader_schema"] = _DOCUMENT_LOADER_SCHEMA
    _save_md5_registry(device_name, registry)
    return documents, effective_incremental


def _metadata(device_name: str, filename: str, kind: str, **extra) -> dict:
    return {
        "source": f"devices/{device_name}/{filename}",
        "device": device_name,
        "filename": Path(filename).name,
        "type": kind,
        **extra,
    }


_NUMBERED_HEADING_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百]+[章节]|[一二三四五六七八九十]+、|"
    r"\d+(?:\.\d+){0,3}[.、]?)\s*[^。！？]{1,80}$"
)


def _docx_heading(text: str, style_name: str = "") -> tuple[int, str] | None:
    """Return (hierarchy level, heading) for a DOCX paragraph, if any."""
    style = (style_name or "").lower()
    if "heading" in style or "标题" in style:
        match = re.search(r"(\d+)", style)
        return (int(match.group(1)) if match else 1, text)
    if not _NUMBERED_HEADING_RE.match(text):
        return None
    token = text.split()[0]
    if token.startswith("第") or token[:1] in "一二三四五六七八九十":
        return (1, text)
    return (token.count(".") + token.count("、") + 1, text)


def _load_docx_sections(
    filename: str,
    data: bytes,
    device_name: str,
) -> list[Document]:
    """Load a DOCX as section-aware documents with traceable metadata."""
    from docx import Document as DocxDocument

    docx = DocxDocument(io.BytesIO(data))
    sections: list[tuple[str, str, str]] = []
    heading_stack: list[tuple[int, str]] = []
    current_section = "文档正文"
    current_heading = ""
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_parts
        content = "\n\n".join(part for part in current_parts if part).strip()
        if content:
            sections.append((current_section, current_heading, content))
        current_parts = []

    for paragraph in docx.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        heading = _docx_heading(text, getattr(paragraph.style, "name", ""))
        if heading:
            flush()
            level, heading_text = heading
            heading_stack[:] = heading_stack[: max(0, level - 1)]
            heading_stack.append((level, heading_text))
            current_section = " > ".join(item[1] for item in heading_stack)
            current_heading = heading_text
            continue
        current_parts.append(text)

    # Tables are attached to the current section instead of being flattened
    # into an unrelated tail document.
    for table in docx.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                current_parts.append(" | ".join(cells))
    flush()

    documents: list[Document] = []
    for section_path, heading, content in sections:
        page_content = f"章节：{section_path}\n\n{content}"
        documents.append(
            Document(
                page_content=page_content,
                metadata=_metadata(
                    device_name,
                    filename,
                    "docx",
                    source_type="device_manual",
                    manual_version="V2.0",
                    section_path=section_path,
                    heading=heading,
                ),
            )
        )
    return documents


def _load_single_file(filename: str, data: bytes, device_name: str) -> list[Document]:
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        documents: list[Document] = []
        for index, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata=_metadata(device_name, filename, "pdf", page=index),
                    )
                )
        return documents

    if suffix == ".docx":
        return _load_docx_sections(filename, data, device_name)

    if suffix in {".txt", ".md"}:
        text = data.decode("utf-8-sig").strip()
        return [
            Document(
                page_content=text,
                metadata=_metadata(
                    device_name,
                    filename,
                    "txt" if suffix == ".txt" else "markdown",
                    source_type=(
                        "operation_rules"
                        if Path(filename).name == "operation_rules.txt"
                        else "device_knowledge"
                    ),
                ),
            )
        ] if text else []

    return []
