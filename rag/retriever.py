"""检索器 —— 语义 + 关键词混合检索 + Embedding 余弦去重。

支持设备手册（plant_{device}）和论文库（plant_papers）双 collection 检索。
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from core.logging import logger

from .query_routing_policy import decide_query_route
from .vector_store import get_vector_store


class PlantRetriever(BaseRetriever):
    """混合检索器：语义向量检索 + 关键词补充 + 余弦去重 + 论文ID去重。

    检索流程：
      1. ChromaDB 语义检索 → 候选集（k*2 个，含 Embedding）
      2. 相关性阈值过滤
      3. paper_id 去重（论文库专用）
      4. Embedding 余弦相似度去重
      5. 可选 BM25 关键词补充
      6. 返回 top-k 结果
    """

    device_name: str = "Device-A"
    k: int = 4
    score_threshold: float = 0.3
    dedup_threshold: float = 0.85
    use_keyword_search: bool = True
    paper_score_threshold: float = 0.15

    class Config:
        arbitrary_types_allowed = True

    @property
    def vector_store(self):
        return get_vector_store(self.device_name)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """检索仪器手册知识库。"""
        semantic_failed = False
        try:
            candidates = self.vector_store.similarity_search(
                query, k=self.k * 2, include_embeddings=True
            )
        except Exception as exc:
            # Interactive embedding has a short timeout.  If the provider is
            # unavailable, continue with local lexical retrieval instead of
            # making the page wait through batch-style retries.
            logger.warning(f"手册语义检索失败，切换本地关键词检索: {exc}")
            candidates = []
            semantic_failed = True
        filtered = [
            doc for doc in candidates
            if doc.metadata.get("score", 0) >= self.score_threshold
        ]
        filtered = _dedup_by_cosine(filtered, self.dedup_threshold)

        if self.use_keyword_search:
            keyword_docs = self._keyword_search(query, skip_semantic=semantic_failed)
            seen = {doc.page_content for doc in filtered}
            for doc in keyword_docs:
                if doc.page_content not in seen and len(filtered) < self.k + 2:
                    filtered.append(doc)
                    seen.add(doc.page_content)

        source_priority = {
            "device_manual": 2,
            "operation_rules": 1,
            "device_knowledge": 0,
        }
        filtered.sort(
            key=lambda d: (
                d.metadata.get("score", 0),
                source_priority.get(str(d.metadata.get("source_type", "")), 0),
            ),
            reverse=True,
        )
        return filtered[:self.k]

    def search_papers(self, query: str, k: int = 3) -> list[Document]:
        """检索论文知识库（plant_papers collection），带 paper_id 去重。"""
        from .vector_store import get_paper_vector_store

        store = get_paper_vector_store()
        if store.chunk_count == 0:
            return []

        query = _expand_paper_query(query)
        # 论文可能包含很多相似章节，扩大候选池后再按 paper_id 去重，
        # 避免同一篇论文的多个块挤占全部推荐名额。
        candidate_k = min(max(k * 20, 40), store.chunk_count)
        candidates = store.similarity_search(
            query, k=candidate_k, include_embeddings=True
        )

        filtered = []
        for doc in candidates:
            vector_score = doc.metadata.get("score", 0)
            lexical_boost = _paper_lexical_boost(query, doc)
            # 标题直接命中专业词时，即使向量分略低也保留。
            if vector_score >= self.paper_score_threshold or lexical_boost >= 0.12:
                doc.metadata["vector_score"] = vector_score
                doc.metadata["score"] = vector_score + lexical_boost
                filtered.append(doc)
        filtered = _dedup_by_paper_id(filtered, max_per_paper=1)
        # 不对不同论文做语义余弦去重：同一主题的论文相似是正常且有价值的。
        for doc in filtered:
            doc.metadata.pop("embedding", None)
        filtered.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)
        return filtered[:k]

    def get_paper_by_doi(
        self,
        doi: str,
        max_chunks: int = 8,
        query: str = "",
    ) -> list[Document]:
        """按 DOI 精确读取正文块，并可按当前追问做论文内词法排序。

        该方法不调用 Embedding。``query`` 用于“推荐后追问某一篇论文的
        某个部分”场景，例如优先挑出光响应曲线、实验方法或结果相关块。
        """
        from .vector_store import get_paper_vector_store

        normalized_doi = doi.strip().lower().rstrip(".,;，。；")
        if normalized_doi.startswith("doi:"):
            normalized_doi = normalized_doi[4:]
        store = get_paper_vector_store()
        # Full-text chunks use ``paper_id=doi:<DOI>`` as their stable key;
        # older abstract collections used a separate ``doi`` field.  Query
        # both layouts so a paper recommended from the full corpus can be
        # opened immediately in the next turn.
        query_variants = (
            {"paper_id": f"doi:{normalized_doi}"},
            {"paper_id": normalized_doi},
            {"doi": normalized_doi},
            {"doi": f"doi:{normalized_doi}"},
        )
        data = {"documents": [], "metadatas": []}
        for where in query_variants:
            try:
                candidate = store.collection.get(
                    where=where,
                    include=["documents", "metadatas"],
                )
            except Exception as exc:
                logger.debug(f"按 DOI 读取论文块失败（{where}）：{exc}")
                continue
            if candidate.get("documents"):
                data = candidate
                break
        docs: list[Document] = []
        seen: set[str] = set()
        for content, metadata in zip(data["documents"], data["metadatas"]):
            text = (content or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            docs.append(Document(page_content=text, metadata=dict(metadata or {})))

        # Last-resort local fallback.  The frozen BM25 corpus is one JSON line
        # per paper and contains the same chunk text/metadata as the full
        # collection.  This path is still offline and is only used when a
        # collection query returns no rows (for example after a Chroma schema
        # migration or while an older page process is holding a stale handle).
        if not docs:
            try:
                from src.core.config import settings

                relative = Path("data/papers/scale_all_fulltext/retrieval_assets/bm25_corpus.jsonl")
                corpus_candidates = (
                    settings.runtime_root / relative,
                    Path(__file__).resolve().parents[2] / relative,
                    Path.cwd() / relative,
                )
                target_paper_id = f"doi:{normalized_doi}"
                corpus_path = next((path for path in corpus_candidates if path.is_file()), None)
                if corpus_path is not None:
                    with corpus_path.open("r", encoding="utf-8") as handle:
                        for raw_line in handle:
                            if not raw_line.strip():
                                continue
                            row = json.loads(raw_line)
                            if str(row.get("paper_id") or "").lower() != target_paper_id:
                                continue
                            for chunk in row.get("chunks") or ():
                                text = str(chunk.get("content") or "").strip()
                                if not text or text in seen:
                                    continue
                                seen.add(text)
                                metadata = {
                                    key: chunk.get(key)
                                    for key in (
                                        "chunk_id", "paper_id", "section_path", "page_start",
                                        "page_end", "title", "year", "doi", "source_type",
                                    )
                                    if chunk.get(key) is not None
                                }
                                docs.append(Document(page_content=text, metadata=metadata))
                            break
            except Exception as exc:
                logger.debug(f"本地 BM25 论文正文兜底读取失败：{exc}")

        if query:
            expanded_query = _expand_paper_query(query).casefold()
            terms = tuple(dict.fromkeys(
                term
                for term in re.findall(r"[a-z][a-z0-9-]{2,}|[一-鿿]{2,}|\d+", expanded_query)
                if term not in {"论文", "文章", "研究", "分析", "部分", "这个", "最后"}
            ))

            def query_priority(doc: Document) -> tuple[int, int]:
                haystack = (
                    f"{doc.metadata.get('title', '')} "
                    f"{doc.metadata.get('section_path', '')} "
                    f"{doc.page_content}"
                ).casefold()
                lexical_hits = sum(haystack.count(term) for term in terms)
                return (min(lexical_hits, 30), _paper_evidence_priority(doc))

            docs.sort(key=query_priority, reverse=True)
        else:
            docs.sort(key=_paper_evidence_priority, reverse=True)
        selected = docs[:max_chunks]
        for index, doc in enumerate(selected, 1):
            doc.metadata["evidence_order"] = index
        return selected

    @staticmethod
    def is_recommendation_intent(query: str) -> bool:
        """Detect a fresh recommendation request before applying chat context."""
        text = query or ""
        explicit_request = re.search(
            r"(?:推荐|请推荐|给我|列出|找|来|想看)[^。！？?]{0,40}"
            r"(?:论文|文献|文章|articles?|papers?|publications?)",
            text,
            re.IGNORECASE,
        )
        if explicit_request:
            return True
        needs_request = re.search(
            r"(?:需要|我要|想要|我想要)([^。！？?]{0,40})"
            r"(?:论文|文献|文章|articles?|papers?|publications?)",
            text,
            re.IGNORECASE,
        )
        if needs_request:
            # “需要做哪些调整，文献中有哪些注意事项” is a methodology
            # question, not a request to recommend papers.
            between = needs_request.group(1)
            if not re.search(r"做|调整|设置|测量|分析|解释|注意|方法|如何|怎么", between):
                return True
        return bool(re.search(
            r"(?:相关|关于)[^。！？?]{0,24}(?:论文|文献|文章|articles?|papers?)",
            text,
            re.IGNORECASE,
        ))

    @staticmethod
    def is_paper_analysis_intent(query: str) -> bool:
        """Detect a request for a structured reading of one selected paper."""

        return bool(re.search(
            r"(?:详细)?分析|解读|详解|介绍(?:一下)?|说明(?:一下)?|解释(?:一下)?|"
            r"详细(?:介绍|说明|解释)|"
            r"describe|explain|tell\s+me\s+about|"
            r"讲讲(?:这|该|第一|第\s*\d+)?篇|研究(?:内容|方法|结果)?",
            query or "",
            re.IGNORECASE,
        ))

    @staticmethod
    def is_context_dependent_followup(query: str) -> bool:
        """Return whether an answer needs conversational history to resolve it.

        Questions with an explicit subject, such as ``Fv/Fm是什么``, are
        independent even when they follow a long paper analysis.  Only short
        anaphoric turns should inherit that history; otherwise chat payloads
        grow on every turn and can accidentally change retrieval routing.
        """

        text = (query or "").strip()
        return bool(re.search(
            r"^(?:继续|接着|再详细(?:一点|些)?|再解释(?:一下)?|再说明(?:一下)?|"
            r"更详细(?:一点|些)?|为什么(?:呢)?\s*$|"
            r"(?:你知道|还记得)?我?(?:上+个?(?:问题|提问)|前(?:一|两|几)个问题)(?:是什么|吗)?\s*$|"
            r"(?:这个|那个|上述|前面(?:提到)?的|它|其|这些|那些|两者|二者)"
            r"(?:是什么|怎么样|如何|为什么|有什么|怎么|呢|吗|的)?\s*$|"
            r"(?:continue|go\s+on|more\s+detail|explain\s+more|why)\s*[?.!]*$)",
            text,
            re.IGNORECASE,
        ))

    @staticmethod
    def classify_query(query: str) -> str:
        """根据问题内容自动路由：manual / paper / mixed / general。

        只有明确出现设备/软件操作信号时才查手册，只有明确出现论文或
        研究证据信号时才查论文；普通概念解释（如“Fv/Fm是什么”）走通用
        模型，避免把每个短问题都变成一次昂贵的知识库检索。
        """
        return decide_query_route(
            query,
            recommendation_intent=PlantRetriever.is_recommendation_intent(query),
        ).route

    @staticmethod
    def requested_paper_count(query: str, default: int = 4) -> int:
        """从“要五篇/推荐 8 篇”等表达中提取论文数量。"""
        chinese_numbers = {
            "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        }
        # Ordinal references are selections, not requested result counts.
        # Remove them before parsing so “倒数第一篇” can never become
        # “推荐 1 篇” if an upstream intent classifier makes a mistake.
        count_text = re.sub(
            r"最后一篇|倒数\s*(?:第\s*)?[一两二三四五六七八九十\d]+\s*篇|"
            r"第\s*[一两二三四五六七八九十\d]+\s*篇",
            "",
            query or "",
        )
        digit_match = re.search(r"(?:要|推荐|给我|列出|再来)?\s*(\d{1,2})\s*篇", count_text)
        if digit_match:
            return max(1, min(int(digit_match.group(1)), 10))
        chinese_match = re.search(
            r"(?:要|推荐|给我|列出|再来)?\s*([一两二三四五六七八九十])\s*篇", count_text
        )
        if chinese_match:
            return chinese_numbers[chinese_match.group(1)]
        return default

    @staticmethod
    def manual_query_for_retrieval(query: str) -> str:
        """Expand only broad device-operation requests into verified manual terms.

        A question such as “仪器如何使用” is semantically broad and can miss a
        short chapter heading.  The expansion contains only the standard
        Device-A workflow labels that exist in the manual; detailed setting
        questions keep their original wording.
        """

        text = str(query or "")
        is_broad_operation = bool(re.search(
            r"(?:仪器|设备|Device-A)\s*(?:如何|怎么|怎样)?\s*(?:使用|操作)|"
            r"(?:完整|基本)?\s*(?:仪器)?(?:使用|操作)(?:说明|流程|步骤)?|"
            r"how\s+to\s+use.*(?:instrument|device-a)",
            text,
            re.IGNORECASE,
        ))
        if not is_broad_operation:
            return text
        return (
            "Device-A 使用流程 串口连接 相机开关 新建测量 "
            "样品放置 预览 去背景阈值 ROI 测量参数 开始测量 数据保存"
        )

    @staticmethod
    def is_paper_followup(query: str, previous_assistant: str) -> bool:
        """识别“我要五篇/再推荐几篇”等承接上一轮论文请求的短追问。"""
        if not previous_assistant or not re.search(r"DOI|论文|文献|Title", previous_assistant, re.I):
            return False
        followup_patterns = [
            r"(?:要|给我|列出|推荐|再来)\s*[一两二三四五六七八九十\d]+\s*篇",
            r"再(?:推荐|补充|换)\s*(?:几|一些|[一两二三四五六七八九十\d]+)篇",
            r"这些(?:论文|文献)",
            r"(?:第\s*)?[一二三四五六七八九十\d]+\s*篇",
            r"最后一篇|倒数第一篇|(?:这|该)篇(?:论文|文章)?",
            r"\b(?:this|the)\s+(?:paper|article|publication)\b",
            r"\b(?:first|second|third|fourth|fifth|last)\s+(?:paper|article|publication)\b",
        ]
        return any(re.search(pattern, query) for pattern in followup_patterns)

    @staticmethod
    def is_ordinal_paper_reference(query: str) -> bool:
        """Whether a turn asks for an ordinal paper from prior context."""

        return bool(re.search(
            r"(?:第\s*)?[一二三四五六七八九十\d]+\s*篇|"
            r"最后一篇|倒数\s*(?:第\s*)?[一二三四五六七八九十\d]+\s*篇|"
            r"(?:这|该)篇(?:论文|文章)?|"
            r"\b(?:this|the)\s+(?:paper|article|publication)\b|"
            r"\b(?:first|second|third|fourth|fifth|last)\s+(?:paper|article|publication)\b",
            query or "",
            re.IGNORECASE,
        ))

    @staticmethod
    def is_numbered_paper_reference(query: str) -> bool:
        """Whether the user explicitly selected a numbered list item.

        This intentionally excludes “这篇论文/this paper”: those references
        should stay bound to the immediately preceding single-paper answer,
        while “第一篇/倒数第二篇” should bind to the latest full list.
        """

        return bool(re.search(
            r"最后一篇|倒数\s*(?:第\s*)?[一两二三四五六七八九十\d]+\s*篇|"
            r"(?:第\s*)?[一两二三四五六七八九十\d]+\s*篇|"
            r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\s+"
            r"(?:paper|article|publication)\b",
            query or "",
            re.IGNORECASE,
        ))

    @staticmethod
    def is_recommendation_refinement(query: str, previous_assistant: str) -> bool:
        """识别对上一轮论文推荐的筛选补充，如“年份新一点的”。"""

        if not previous_assistant or not re.search(r"参考文献|DOI|论文|文献", previous_assistant, re.I):
            return False
        # “分析最新一轮的倒数第二篇” uses 最新 to identify the current
        # list version, not to request newer publication years.  Explicit
        # ordinal/analysis turns always remain paper follow-ups.
        if PlantRetriever.is_numbered_paper_reference(query) or (
            PlantRetriever.is_paper_analysis_intent(query)
            and PlantRetriever.is_ordinal_paper_reference(query)
        ):
            return False
        return bool(re.search(
            r"(?:年份|年限|年代|较新|新一点|最新|近期|近年|近\s*[一二三四五六七八九十\d]+\s*年|"
            r"202\d\s*[-到至~]\s*20\d{2})",
            query or "",
            re.IGNORECASE,
        ))

    @staticmethod
    def resolve_referenced_paper_doi(query: str, previous_assistant: str) -> str:
        """将上一轮回答中的论文指代解析为 DOI。

        支持数字/中文序号、“最后一篇”以及在上一轮只有一篇论文时的
        “这篇论文/this paper”。重复出现在正文和参考来源中的 DOI 只计一次。
        """
        # An explicit DOI in the current turn is an unambiguous lock and does
        # not depend on chat history.
        direct_dois = re.findall(
            r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", query or "", re.I
        )
        if direct_dois:
            return direct_dois[0].rstrip(".,;，。；)").lower()
        if not previous_assistant:
            return ""
        dois = re.findall(
            r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", previous_assistant, re.I
        )
        cleaned = [doi.rstrip(".,;，。；)") for doi in dois]
        unique_dois = list(dict.fromkeys(item.lower() for item in cleaned if item))
        if not unique_dois:
            return ""

        reverse_match = re.search(
            r"倒数\s*(?:第\s*)?([一两二三四五六七八九十\d]+)\s*篇",
            query or "",
        )
        if reverse_match:
            raw_reverse = reverse_match.group(1)
            chinese_numbers = {
                "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
            }
            reverse_index = int(raw_reverse) if raw_reverse.isdigit() else chinese_numbers.get(raw_reverse, 0)
            if 1 <= reverse_index <= len(unique_dois):
                return unique_dois[-reverse_index]
            return ""
        if re.search(r"最后一篇|\bthe\s+last\s+(?:paper|article|publication)\b|"
                     r"\blast\s+(?:paper|article|publication)\b", query or "", re.I):
            return unique_dois[-1]

        # “这篇论文” is safe only when the immediately preceding answer
        # contains one unique DOI.  A recommendation list must not be guessed.
        if re.search(r"(?:这|该)篇(?:论文|文章)?|\b(?:this|the)\s+(?:paper|article|publication)\b", query or "", re.I):
            return unique_dois[0] if len(unique_dois) == 1 else ""

        ordinal_match = re.search(
            r"(?:第\s*)?([一二三四五六七八九十\d]+)\s*篇",
            query,
        )
        english_ordinals = {
            "first": 1, "second": 2, "third": 3, "fourth": 4,
            "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
            "ninth": 9, "tenth": 10,
        }
        english_match = re.search(
            r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+"
            r"(?:paper|article|publication)\b",
            query,
            re.IGNORECASE,
        )
        if not (ordinal_match or english_match):
            return ""
        raw_ordinal = ordinal_match.group(1) if ordinal_match else ""
        chinese_numbers = {
            "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        }
        index = (
            int(raw_ordinal)
            if raw_ordinal.isdigit()
            else chinese_numbers.get(raw_ordinal, 0)
        ) if ordinal_match else english_ordinals.get(english_match.group(1).lower(), 0)
        if index < 1:
            return ""
        return unique_dois[index - 1] if index <= len(unique_dois) else ""

    @staticmethod
    def build_paper_context(docs: list[Document]) -> str:
        """构造论文检索上下文（含引用格式）。"""
        groups: dict[str, list[Document]] = {}
        for doc in docs:
            paper_id = str(doc.metadata.get("paper_id") or doc.metadata.get("doi") or "unknown")
            groups.setdefault(paper_id, []).append(doc)
        parts = [
            "[PAPER_RESULT_POLICY]",
            f"Available unique papers in this response: {len(groups)}",
            "All papers in the local library are curated, authentic, and formally published. Do not question or re-verify their publication status.",
            "Only the papers listed below may be cited or recommended. ",
            "If they are fewer than the user requested, state that the current retrieval only found this many; never invent entries to reach the requested count.",
            "[/PAPER_RESULT_POLICY]",
        ]
        for i, paper_docs in enumerate(groups.values(), 1):
            d = paper_docs[0]
            title = d.metadata.get("title", "Unknown")
            year = d.metadata.get("year", "")
            journal = d.metadata.get("journal", "")
            authors = d.metadata.get("authors", "")
            doi = d.metadata.get("doi", "")
            section = d.metadata.get("section", "") or d.metadata.get("section_path", "")
            src_type = d.metadata.get("source_type", "")

            header = f"[Paper P{i}]"
            header += f"\nTitle: {title}"
            if year:
                header += f"\nYear: {year}"
            if journal:
                header += f"\nJournal: {journal}"
            if authors:
                header += f"\nAuthors: {authors}"
            if doi:
                header += f"\nDOI: {doi}"
            if section:
                header += f"\nSection: {section}"
            if src_type == "csv_abstract":
                header += "\nEvidence scope: abstract only"
            else:
                header += "\nEvidence scope: local full text"
            header += f"\nSource type: {src_type}"

            evidence = "\n\n".join(
                f"[Evidence {j}]\n{paper_doc.page_content}"
                for j, paper_doc in enumerate(paper_docs, 1)
            )
            parts.append(f"{header}\n\n{evidence}")

        return "\n\n---\n\n".join(parts)

    def _keyword_search(self, query: str, *, skip_semantic: bool = False) -> list[Document]:
        """简单的关键词匹配检索。"""
        if self.vector_store.chunk_count == 0:
            return []

        # When the embedding service is unavailable, English operation
        # questions still need to find the Chinese manual.  Add only stable
        # operation synonyms; this is a retrieval fallback, not an answer.
        expanded_query = query
        if re.search(r"\b(?:use|operate|start|measure|instrument|camera|software)\b", query, re.I):
            expanded_query += " 仪器 使用 操作步骤 测量 串口连接 相机开关 新建测量 保存路径"
        keywords = re.findall(r'[一-鿿]+|[A-Za-z/]+|\d+', expanded_query)
        if not keywords:
            return []

        try:
            if skip_semantic:
                raise RuntimeError("semantic retrieval already failed")
            all_docs = self.vector_store.similarity_search(query, k=50)
        except Exception as exc:
            # Do not call the embedding endpoint a second time after a query
            # timeout.  The manual collection is small enough for a local
            # metadata/document scan, which is the final offline fallback.
            logger.warning(f"关键词候选语义补召回失败，使用本地文档扫描: {exc}")
            try:
                data = self.vector_store.collection.get(
                    include=["documents", "metadatas"]
                )
                all_docs = [
                    Document(
                        page_content=text or "",
                        metadata=dict(metadata or {}),
                    )
                    for text, metadata in zip(
                        data.get("documents") or [], data.get("metadatas") or []
                    )
                    if text
                ]
            except Exception:
                return []

        scored = []
        for doc in all_docs:
            content = doc.page_content
            hits = sum(1 for kw in keywords if kw.lower() in content.lower())
            if hits > 0:
                doc.metadata["keyword_score"] = hits / len(keywords)
                doc.metadata["score"] = doc.metadata.get("score", 0) + hits * 0.1
                scored.append(doc)
        return scored


# --- 去重工具函数 ---

def _dedup_by_paper_id(documents: list[Document], max_per_paper: int = 2) -> list[Document]:
    """按 paper_id 去重：同一篇论文最多保留 N 个最高分块。"""
    if not documents:
        return documents

    groups: dict[str, list[Document]] = {}
    no_id: list[Document] = []
    for d in documents:
        pid = d.metadata.get("paper_id", "")
        if pid:
            groups.setdefault(pid, []).append(d)
        else:
            no_id.append(d)

    result = []
    for pid, docs in groups.items():
        docs.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)
        result.extend(docs[:max_per_paper])
    result.extend(no_id)
    return result


def _dedup_by_cosine(documents: list[Document], threshold: float = 0.85) -> list[Document]:
    """基于 Embedding 余弦相似度去重。"""
    if len(documents) <= 1:
        return documents

    has_emb = [d for d in documents if d.metadata.get("embedding") is not None]
    no_emb = [d for d in documents if d.metadata.get("embedding") is None]

    if len(has_emb) <= 1:
        return documents

    has_emb.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)

    keep = []
    for i, doc in enumerate(has_emb):
        emb_i = np.array(doc.metadata["embedding"], dtype=np.float32)
        is_dup = False
        for kept_doc in keep:
            emb_j = np.array(kept_doc.metadata["embedding"], dtype=np.float32)
            dot = np.dot(emb_i, emb_j)
            norm = np.linalg.norm(emb_i) * np.linalg.norm(emb_j)
            if norm > 0 and (dot / norm) > threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(doc)

    for doc in keep:
        doc.metadata.pop("embedding", None)

    return keep + no_emb


# --- 工厂函数 ---

def create_retriever(
    device_name: str | None = None,
    k: int = 4,
    score_threshold: float = 0.3,
) -> PlantRetriever:
    """创建检索器实例。"""
    return PlantRetriever(
        device_name=device_name or "Device-A",
        k=k,
        score_threshold=score_threshold,
    )


def _expand_paper_query(query: str) -> str:
    """为常用中文术语补充英文同义词，提升英文论文库召回。"""
    expansions = {
        "叶绿素荧光": "chlorophyll fluorescence",
        "叶绿素含量": "chlorophyll content",
        "荧光成像": "fluorescence imaging",
        "光响应曲线": "light response curve",
        "AQ曲线": "A-Q curve photosynthetic CO2 response curve",
        "A-Q曲线": "A-Q curve photosynthetic CO2 response curve",
        "CO2响应曲线": "A-Ci curve photosynthetic CO2 response",
    }
    additions = [english for chinese, english in expansions.items() if chinese in query]
    if "叶绿素" in query and not any("chlorophyll" in item for item in additions):
        additions.append("chlorophyll")
    return f"{query} {' '.join(additions)}".strip()


def _paper_lexical_boost(query: str, doc: Document) -> float:
    """按标题和正文的专业词命中进行轻量重排。"""
    stopwords = {
        "paper", "papers", "study", "research", "recommend", "title",
        "论文", "文献", "研究", "推荐", "给我", "列出", "一篇", "五篇",
    }
    english_terms = {
        term.lower() for term in re.findall(r"[A-Za-z][A-Za-z-]{3,}", query)
        if term.lower() not in stopwords
    }
    chinese_terms = {
        term for term in re.findall(r"[\u4e00-\u9fff]{2,}", query)
        if term not in stopwords and "篇" not in term
    }
    terms = english_terms | chinese_terms
    if not terms:
        return 0.0

    title = str(doc.metadata.get("title", "")).lower()
    content = doc.page_content.lower()
    boost = sum(0.08 for term in terms if term.lower() in title)
    boost += sum(0.015 for term in terms if term.lower() in content)
    if "chlorophyll fluorescence" in query.lower() and "chlorophyll fluorescence" in title:
        boost += 0.3
    return min(boost, 0.6)


def _paper_evidence_priority(doc: Document) -> tuple[int, int]:
    """优先选择摘要、方法、结果和结论块，降低参考文献块优先级。"""
    text = doc.page_content.lower()
    score = 0
    if "abstract" in text[:120]:
        score += 100
    if any(term in text for term in (
        "measurements were made", "experiments were", "methodology",
        "chlorophyll fluorescence measurements", "materials and methods",
    )):
        score += 60
    if any(term in text for term in (
        "results showed", "significantly", "experimental results",
        "in the lower range", "in the higher range", "rlc exhibited",
        "rtermax", "retrmax", "fv/fm",
    )):
        score += 50
    if any(term in text for term in (
        "in conclusion", "therefore", "we conclude", "discussion",
    )):
        score += 40
    if "references" in text[:80]:
        score -= 100
    citation_count = len(re.findall(r"(?:19|20)\d{2}|et al\.", text))
    if citation_count >= 5:
        score -= 100
    elif citation_count >= 3:
        score -= 40
    journal_reference_hits = sum(
        text.count(term) for term in (
            "mar ecol prog ser", "j phycol", "aquat bot", "photosynth res",
            "plant physiol", "limnol oceanogr",
        )
    )
    if journal_reference_hits >= 2:
        score -= 100
    if text.lstrip().startswith((". mar ", ". j ", ". plant ", ". aquat ")):
        score -= 120
    if len(text) >= 300:
        score += 10
    return score, len(text)
