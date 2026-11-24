"""文本切分器 —— 中文优化的 RecursiveCharacterTextSplitter。

不同文档类型使用不同的切分策略：
  - 手册/规则文档: chunk_size=800, overlap=150
  - 提示词/短文档: chunk_size=2000, overlap=200（尽可能保持完整）
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.logging import logger


# 中文优化的分隔符序列
CN_SEPARATORS = [
    "\n\n\n",       # 章节间多个空行
    "\n\n",          # 段落间空行
    "\n## ",         # Markdown 二级标题
    "\n### ",        # Markdown 三级标题
    "\n#### ",       # Markdown 四级标题
    "\n",            # 换行
    "。",            # 中文句号
    "；",            # 中文分号
    "，",            # 中文逗号
    ".",             # 英文句号
    ";",             # 英文分号
    " ",             # 空格
    "",              # 逐字
]

# 不同类型文档的切分配置
SPLIT_CONFIGS = {
    "txt": {"chunk_size": 800, "chunk_overlap": 150},
    "markdown": {"chunk_size": 800, "chunk_overlap": 150},
    "short": {"chunk_size": 2000, "chunk_overlap": 200},  # 短文档尽量保持完整
    "pdf": {"chunk_size": 800, "chunk_overlap": 150},
}

def split_documents(documents: list[Document]) -> list[Document]:
    """将文档列表切分为合适的块。

    根据文档 metadata 中的 type 选择切分策略。
    短文档（< 3000 字符）不做切分。
    """
    if not documents:
        return []

    all_chunks = []
    short_docs = []
    regular_docs = []

    for doc in documents:
        if len(doc.page_content) < 3000:
            short_docs.append(doc)
        else:
            regular_docs.append(doc)

    # 短文档用大 chunk_size 或不做切分
    if short_docs:
        short_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=CN_SEPARATORS,
            length_function=len,
        )
        for doc in short_docs:
            chunks = short_splitter.split_documents([doc])
            all_chunks.extend(chunks)

    # 常规文档按配置切分
    if regular_docs:
        regular_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=CN_SEPARATORS,
            length_function=len,
        )
        chunks = regular_splitter.split_documents(regular_docs)
        all_chunks.extend(chunks)

    # Every chunk from the structured manual keeps its section context.  The
    # metadata remains the machine-readable source of truth; the short prefix
    # also helps the answering model distinguish similarly named parameters.
    for chunk in all_chunks:
        if chunk.metadata.get("source_type") != "device_manual":
            continue
        section_path = str(chunk.metadata.get("section_path") or "").strip()
        if section_path and not chunk.page_content.startswith("章节："):
            chunk.page_content = f"章节：{section_path}\n\n{chunk.page_content}"

    logger.info(f"文档切分完成: {len(documents)} 份文档 → {len(all_chunks)} 个块")
    return all_chunks
