"""论文加载器 —— 三层回退：JATS XML → pymupdf PDF → CSV 摘要。
从 paper_manifest.jsonl 读取论文清单，按 preferred_source 解析全文。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET

from langchain_core.documents import Document
from core.logging import logger

from core.config import settings

# 最低正文长度（低于此值判定为无效/扫描版，回退摘要）
MIN_TEXT_LENGTH = 300


def load_papers_from_manifest(
    manifest_path: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[Document]:
    """从论文清单加载所有 ready 状态的论文，解析为 Document 列表。

    Args:
        manifest_path: manifest 路径，默认从 data/papers/paper_manifest.jsonl
        limit: 限制加载数量（None = 全部）
        offset: 跳过前 N 篇 ready 论文（用于分批续传）

    Returns:
        Document 列表，每个块的 metadata 包含 paper_id, doi, title 等
    """
    if manifest_path is None:
        manifest_path = settings.project_root / "data" / "papers" / "paper_manifest.jsonl"

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        logger.warning(f"论文清单不存在: {manifest_path}，请先运行 scripts/build_paper_manifest.py")
        return []

    papers_dir = Path(settings.PAPERS_DIR)
    if not papers_dir.exists():
        logger.warning(f"论文目录不存在: {papers_dir}")
        return []

    # 读取清单
    papers = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))

    # 只处理 ready 状态的论文
    ready = [p for p in papers if p.get("status") == "ready"]
    total_ready = len(ready)
    if offset:
        ready = ready[offset:]
    if limit:
        ready = ready[:limit]

    logger.info(f"开始加载论文: {len(ready)} 篇（总计 {total_ready} 篇 ready，offset={offset}，limit={limit}）")

    documents: list[Document] = []
    stats = {"xml_fulltext": 0, "pdf_fulltext": 0, "csv_abstract": 0, "failed": 0}

    for i, paper in enumerate(ready):
        src = paper.get("preferred_source", "")
        if src == "xml_fulltext":
            docs = _load_xml_paper(paper, papers_dir)
        elif src == "pdf_fulltext":
            docs = _load_pdf_paper(paper, papers_dir)
        elif src == "csv_abstract":
            docs = _load_abstract_paper(paper)
        else:
            docs = []

        if docs:
            documents.extend(docs)
            stats[src] = stats.get(src, 0) + 1
        else:
            # 回退：XML 失败试 PDF，PDF 失败试摘要
            if src == "xml_fulltext" and paper.get("pdf_path"):
                logger.debug(f"XML 解析失败，回退 PDF: {paper['title'][:50]}...")
                docs = _load_pdf_paper(paper, papers_dir)
                if docs:
                    documents.extend(docs)
                    stats["pdf_fulltext"] = stats.get("pdf_fulltext", 0) + 1
                    continue
            if src in ("xml_fulltext", "pdf_fulltext") and paper.get("abstract"):
                logger.debug(f"全文提取失败，回退摘要: {paper['title'][:50]}...")
                docs = _load_abstract_paper(paper)
                if docs:
                    documents.extend(docs)
                    stats["csv_abstract"] = stats.get("csv_abstract", 0) + 1
                    continue
            stats["failed"] += 1

        if (i + 1) % 100 == 0:
            logger.info(f"论文加载进度: {i+1}/{len(ready)}")

    logger.info(
        f"论文加载完成: {len(documents)} 个文档块 "
        f"(XML={stats.get('xml_fulltext', 0)}, "
        f"PDF={stats.get('pdf_fulltext', 0)}, "
        f"摘要={stats.get('csv_abstract', 0)}, "
        f"失败={stats['failed']})"
    )
    return documents


def _base_metadata(paper: dict) -> dict:
    """构造基础 metadata 字典。"""
    return {
        "paper_id": paper.get("paper_id", ""),
        "doi": paper.get("doi", ""),
        "title": paper.get("title", ""),
        "authors": paper.get("authors", ""),
        "year": paper.get("year"),
        "journal": paper.get("journal", ""),
        "source_type": paper.get("preferred_source", ""),
        "type": "paper",
        "filename": "",
    }


def _load_abstract_paper(paper: dict) -> list[Document]:
    """从 CSV 摘要构造 Document。"""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    content = f"Title: {title}\n\nAbstract: {abstract}"

    if len(content.strip()) < 50:
        return []

    meta = _base_metadata(paper)
    meta["section"] = "abstract"
    meta["source_type"] = "csv_abstract"
    return [Document(page_content=content, metadata=meta)]


# --- PDF 加载 ---

def _load_pdf_paper(paper: dict, papers_dir: Path) -> list[Document]:
    """用 pymupdf 提取 PDF 全文。"""
    pdf_path = papers_dir / paper.get("pdf_path", "")
    if not pdf_path.exists():
        logger.debug(f"PDF 文件不存在: {pdf_path}")
        return []

    try:
        import fitz
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        logger.warning(f"PDF 打开失败: {pdf_path} — {e}")
        return []

    page_texts = []
    for page in doc:
        text = page.get_text("text", sort=True)
        page_texts.append(text)
    doc.close()

    full_text = "\n\n".join(page_texts)

    # 清洗
    full_text = _clean_pdf_text(full_text)

    if len(full_text) < MIN_TEXT_LENGTH:
        logger.debug(f"PDF 文本过短 ({len(full_text)} 字符): {paper['title'][:50]}...（可能是扫描版）")
        return []

    # 按段落分块
    paragraphs = _split_into_paragraphs(full_text)
    meta = _base_metadata(paper)
    meta["source_type"] = "pdf_fulltext"
    meta["filename"] = Path(paper.get("pdf_path", "")).name
    meta["section"] = "fulltext"

    docs = []
    for i, para in enumerate(paragraphs):
        if len(para) < 100:
            continue
        m = dict(meta)
        m["chunk_index"] = i
        docs.append(Document(page_content=para, metadata=m))

    return docs


def _clean_pdf_text(text: str) -> str:
    """基本 PDF 文本清洗：去页眉页脚、合并断行、去参考文献。"""
    # 去掉空行过多
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    # 合并被换行打断的句子（行尾非标点符号 → 与下行合并）
    lines = text.split('\n')
    merged = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            merged.append('')
            continue
        if merged and merged[-1] and not merged[-1].endswith(('.', ';', ':', '?', '!', ')', ']')):
            # 上一行不是句末 → 可能是断行，合并
            merged[-1] = merged[-1] + ' ' + stripped
        else:
            merged.append(stripped)

    text = '\n'.join(merged)

    # 去参考文献段（从单独成行的 References 开始截断）
    ref_match = re.search(
        r'\n(References|REFERENCES|Bibliography|LITERATURE CITED|Literature Cited)\s*\n',
        text,
    )
    if ref_match:
        cut_point = ref_match.start()
        # 确保参考文献在文档后半部分
        if cut_point > len(text) * 0.4:
            text = text[:cut_point]

    return text.strip()


def _split_into_paragraphs(text: str) -> list[str]:
    """将文本按段落分割（连续空行或章节标题标记）。"""
    # 先按空行分
    raw = re.split(r'\n\s*\n', text)
    paragraphs = []
    for block in raw:
        block = block.strip()
        if len(block) >= 80:
            paragraphs.append(block)
    return paragraphs


# --- XML 加载 ---

def _load_xml_paper(paper: dict, papers_dir: Path) -> list[Document]:
    """从 JATS XML 提取章节文本。"""
    xml_path = papers_dir / paper.get("xml_path", "")
    if not xml_path.exists():
        logger.debug(f"XML 文件不存在: {xml_path}")
        return []

    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except Exception as e:
        logger.warning(f"XML 解析失败: {xml_path} — {e}")
        return []

    docs = []

    # --- 标题 ---
    title = _extract_xml_text(root, "article-title")
    if title and len(title) > 20:
        meta = _base_metadata(paper)
        meta["section"] = "title"
        meta["source_type"] = "xml_fulltext"
        meta["filename"] = Path(paper.get("xml_path", "")).name
        docs.append(Document(page_content=title, metadata=meta))

    # --- 摘要 ---
    abstract = _extract_xml_text(root, "abstract")
    if abstract and len(abstract) > 50:
        meta = _base_metadata(paper)
        meta["section"] = "abstract"
        meta["source_type"] = "xml_fulltext"
        meta["filename"] = Path(paper.get("xml_path", "")).name
        docs.append(Document(page_content=abstract, metadata=meta))

    # --- 正文章节 ---
    body = root.find(".//{http://dtd.nlm.nih.gov/publishing/}body")
    if body is None:
        # 无命名空间回退
        for elem in root.iter():
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag == "body":
                body = elem
                break

    if body is not None:
        # 构建章节路径
        section_docs = _extract_sections(body, paper, parent_path="")
        docs.extend(section_docs)

    if not docs:
        return []

    # 全文总长度检查
    total_chars = sum(len(d.page_content) for d in docs)
    if total_chars < MIN_TEXT_LENGTH:
        logger.debug(f"XML 文本过短 ({total_chars} 字符): {paper['title'][:50]}...")
        return []

    return docs


def _extract_xml_text(root, tag_name: str) -> str:
    """提取 XML 元素的所有文本（处理命名空间）。"""
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag == tag_name:
            return "".join(elem.itertext()).strip()
    return ""


def _extract_sections(elem, paper: dict, parent_path: str) -> list[Document]:
    """递归提取 JATS body 下的章节。"""
    docs = []
    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "sec":
            # 获取章节标题
            sec_title = ""
            for sub in child:
                sub_tag = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                if sub_tag == "title":
                    sec_title = "".join(sub.itertext()).strip()
                    break

            # 获取章节全部文本
            sec_text = "".join(child.itertext()).strip()
            path = f"{parent_path} > {sec_title}" if parent_path else sec_title

            if len(sec_text) < 100:
                # 短章节：可能只有标题，递归子章节
                sub_docs = _extract_sections(child, paper, path)
                docs.extend(sub_docs)
                continue

            # 去除公式和引用标记（保留文本）
            sec_text = re.sub(r'<[^>]+>', '', sec_text)
            sec_text = re.sub(r'\s+', ' ', sec_text).strip()

            if len(sec_text) >= 100:
                meta = _base_metadata(paper)
                meta["section"] = sec_title or "(untitled)"
                meta["section_path"] = path
                meta["source_type"] = "xml_fulltext"
                meta["filename"] = Path(paper.get("xml_path", "")).name
                docs.append(Document(page_content=sec_text, metadata=meta))

            # 子章节
            sub_docs = _extract_sections(child, paper, path)
            docs.extend(sub_docs)

    return docs
