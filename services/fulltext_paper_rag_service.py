"""Safe application-facing adapter for the full-corpus paper RAG grey path."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Callable, Sequence

from langchain_openai import ChatOpenAI

from core.config import get_llm_api_key, settings
from core.response_guard import compact_context
from rag.comparison_retrieval import ComparisonRetriever, deterministic_comparison_decompose, label_side_evidence
from rag.paper_identity_verifier import DeepSeekPaperIdentityVerifier
from rag.paper_citations import format_paper_reference, get_paper_citation
from rag.paper_conversation_state import PaperReference, normalize_doi
from rag.direct_evidence_matching import direct_profile_fallback, profile_matches_requested_entities
from rag.query_understanding import (
    EntityConstraints,
    build_query_plan,
    extract_entity_constraints,
)
from rag.provider_adapters import candidates_to_evidence
from rag.paper_research_tags import (
    PaperResearchTagIndex,
    read_research_tags,
    required_evidence_role,
    required_intervention_family,
)
from .paper_rag_factory import build_scalable_paper_rag_pipeline
from .paper_rag_pipeline import PaperRagPipeline


@dataclass(frozen=True)
class FulltextPaperAnswer:
    text: str
    sources: tuple[dict, ...]
    requires_paper_selection: bool = False
    recommendation_items: tuple[PaperReference, ...] = ()
    active_paper_id: str = ""
    # A source list can contain merely adjacent studies.  This flag is used by
    # the web-fallback gate for questions that explicitly require a species +
    # treatment pairing.
    direct_local_evidence: bool = False


@dataclass(frozen=True)
class FulltextPaperContext:
    """Evidence-only result for mixed paper + device questions."""

    context: str
    sources: tuple[dict, ...]
    active_paper_id: str = ""
    requires_paper_selection: bool = False


@lru_cache(maxsize=1)
def _research_tag_index() -> PaperResearchTagIndex:
    """Load locally generated qualification tags; missing tags fail open."""

    target = settings.resolve_path(settings.PAPER_RESEARCH_TAG_INDEX_PATH)
    if not Path(target).is_file():
        return PaperResearchTagIndex(())
    try:
        return PaperResearchTagIndex(read_research_tags(target))
    except Exception:
        return PaperResearchTagIndex(())


def _sources(candidates: Sequence) -> tuple[dict, ...]:
    return tuple(
        {
            "filename": str(getattr(item, "metadata", {}).get("title") or getattr(item, "title", "") or "论文")[:120],
            "source": "paper",
            "score": float(getattr(item, "score", getattr(item, "retrieval_score", 0.0))),
            "section": getattr(item, "section", getattr(item, "section_path", "")),
            "page_start": getattr(item, "metadata", {}).get("page_start", getattr(item, "page_start", None)),
        }
        for item in candidates
    )


def _profile_matches_requested_entities(profile, constraints: EntityConstraints) -> bool:
    """Compatibility alias for the shared strict evidence matcher."""

    return profile_matches_requested_entities(profile, constraints)


def _normalise_synthesis_citations(text: str, reference_count: int) -> str:
    """Keep only valid inline source markers and place list markers at the end."""

    body = str(text or "").strip()
    if reference_count <= 0:
        return re.sub(r"\[(\d+)\]", "", body).strip()
    body = re.sub(
        r"\[(\d+)\]",
        lambda match: match.group(0) if 1 <= int(match.group(1)) <= reference_count else "",
        body,
    )
    body = re.sub(
        r"^(\s*(?:[-*]|\d+[.、])\s*)\[(\d+)\]\s+(.+?)\s*$",
        lambda match: (
            f"{match.group(1)}{match.group(3)}"
            if re.search(rf"\[{match.group(2)}\]\s*$", match.group(3))
            else f"{match.group(1)}{match.group(3)} [{match.group(2)}]"
        ),
        body,
        flags=re.M,
    )
    if not re.search(r"\[\d+\]", body):
        body += "\n\n以上结论依据本次检索到的研究整理。[1]"
    return body


def _selection_message(retrieval) -> FulltextPaperAnswer:
    titles = [item.profile.title for item in retrieval.papers[:3]]
    return FulltextPaperAnswer(
        "当前问题对应的论文身份不够明确。请补充论文标题、研究对象、处理条件或年份后再问，"
        "我将避免把相似研究的证据混在一起。\n\n候选论文：\n"
        + "\n".join(f"- {title}" for title in titles),
        _sources(retrieval.evidence),
        True,
    )


def _paper_detail_guidance() -> str:
    """Give a concrete next step when local evidence cannot support detail."""

    return (
        "如需核对论文中的具体实验细节，请使用论文 DOI 打开期刊页面或 DOI.org，"
        "也可以把 DOI 或完整题名复制到 Crossref、PubMed 或 Google Scholar 中检索；"
        "如果您提供论文 PDF，我可以继续按原文逐项核对。"
    )


def _with_detail_guidance(answer: str, attempt) -> str:
    """Add a verification path when some required paper slots are unsupported."""

    missing = getattr(getattr(attempt, "validation", None), "insufficient_slot_ids", ())
    if not missing:
        return answer
    return answer.rstrip() + "\n\n当前证据无法确认部分论文细节。" + _paper_detail_guidance()


def _verify_low_confidence(question: str, retrieval, pipeline, *, interactive: bool = False):
    """Return a single resolved evidence island, or None for safe refusal."""
    islands = [tuple(retrieval.evidence)]
    paper_ids = [retrieval.active_paper_id, *retrieval.alternative_paper_ids]
    for paper_id in paper_ids[1:3]:
        candidates = pipeline.retriever.chunk_search(question, [paper_id], 10)
        islands.append(tuple(pipeline.retriever._rerank(question, candidates)))
    # Identity selection is deliberately short and deterministic.  Reusing
    # the full answer model's 4k output budget can turn a P1/P2/P3 decision
    # into an unnecessary long-running request.
    identity_llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        base_url=settings.LLM_BASE_URL,
        api_key=get_llm_api_key(),
        temperature=0,
        max_tokens=150,
        timeout=(settings.LLM_UI_TIMEOUT if interactive else settings.LLM_TIMEOUT),
        max_retries=(settings.LLM_UI_MAX_RETRIES if interactive else settings.LLM_MAX_RETRIES),
        streaming=False,
    )
    decision = DeepSeekPaperIdentityVerifier(identity_llm).verify(question, islands)
    index = decision.selected_index
    if index is None or index >= len(islands) or not islands[index]:
        return None
    selected_paper = paper_ids[index]
    return replace(
        retrieval,
        evidence=islands[index],
        active_paper_id=selected_paper,
        alternative_paper_ids=(),
        paper_lock_confident=True,
    )


def _answer_resolved(question: str, retrieval, pipeline):
    attempt = pipeline.answerer.answer(question, retrieval.evidence, retrieval.plan.required_slots)
    missing = tuple(dict.fromkeys((*attempt.validation.missing_slot_ids, *attempt.validation.insufficient_slot_ids)))
    if missing:
        retrieval = pipeline.retriever.expand_for_missing_slots(retrieval, missing)
        attempt = pipeline.answerer.answer(question, retrieval.evidence, retrieval.plan.required_slots)
    return retrieval, attempt


def _is_recommendation_request(question: str) -> bool:
    import re
    # The UI may append a parenthetical conversation summary for retrieval.
    # Intent must be decided from the current user turn only.
    raw = question or ""
    text = re.split(r"（对话背景：", raw, maxsplit=1)[0].casefold()
    # An explicit ordinal analysis is a selection from the active list even
    # when it contains words such as “最新”.  It must never reopen the
    # recommendation pipeline.
    has_analysis = re.search(r"分析|解读|介绍|说明|解释|详情|详解", text, re.IGNORECASE)
    has_ordinal = re.search(
        r"最后一篇|倒数\s*(?:第\s*)?[一两二三四五六七八九十\d]+\s*篇|"
        r"(?:第\s*)?[一两二三四五六七八九十\d]+\s*篇",
        text,
        re.IGNORECASE,
    )
    if has_analysis and has_ordinal:
        return False
    if (
        re.search(
            r"(?:推荐|请推荐|给我|列出|找|来|需要|我要|想要|我想要|想看)[^。！？?]{0,40}(?:论文|文献|文章|articles?|papers?)",
            text,
            re.IGNORECASE,
        )
        or re.search(r"(?:相关|关于)[^。！？?]{0,24}(?:论文|文献|文章|articles?|papers?)", text, re.IGNORECASE)
    ):
        return True
    # A short refinement such as “年份新一点的” is a recommendation request
    # only when the attached conversation context contains the prior list.
    background = raw.split("（对话背景：", 1)[1].casefold() if "（对话背景：" in raw else ""
    return bool(background and re.search(r"参考文献|doi|论文|文献|文章|paper|article", background) and re.search(
        r"年份|年限|年代|较新|新一点|最新|近期|近年|近\s*[一二三四五六七八九十\d]+\s*年", text
    ))


def _recommendation_count(question: str) -> int:
    chinese = {
        "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    }
    import re
    count_text = re.sub(
        r"最后一篇|倒数\s*(?:第\s*)?[一两二三四五六七八九十\d]+\s*篇|"
        r"第\s*[一两二三四五六七八九十\d]+\s*篇",
        "",
        question or "",
    )
    match = re.search(r"(\d{1,2})\s*篇", count_text)
    if match:
        return max(1, min(int(match.group(1)), 10))
    match = re.search(r"([一两二三四五六七八九十])\s*篇", count_text)
    return chinese.get(match.group(1), 3) if match else 3


def _recommendation_year_bounds(question: str, current_year: int | None = None) -> tuple[int | None, int | None]:
    """Extract an optional publication-year constraint from a recommendation request."""

    current_year = current_year or date.today().year
    text = str(question or "")
    range_match = re.search(r"(20\d{2})\s*[-到至~]\s*(20\d{2})", text)
    if range_match:
        start, end = sorted((int(range_match.group(1)), int(range_match.group(2))))
        return start, end
    recent_match = re.search(
        r"(?:近|最近|过去)\s*([一两二三四五六七八九十\d]+)\s*年",
        text,
    )
    if recent_match:
        raw = recent_match.group(1)
        chinese = {
            "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        }
        years = int(raw) if raw.isdigit() else chinese.get(raw)
        if years:
            return current_year - years + 1, current_year
    if re.search(r"较新|新一点|最新|近期|近年", text):
        return current_year - 5, current_year
    return None, None


def _recommendation_topic(question: str) -> str:
    """Extract a compact topic for fallback retrieval and user-facing headings."""
    topic_text = re.split(r"（对话背景：", question or "", maxsplit=1)[0]
    topic_text = re.sub(
        r"(?:请|帮我|给我|我想要|我需要|我要|我想|推荐|查找|查一下|查|找|来|列出|几篇|"
        r"[一两二三四五六七八九十\d]+篇|一些|相关的?|关于|论文|文献|文章|articles?|papers?|直接|"
        r"最新|较新|新一点|年份|近\s*[一两二三四五六七八九十\d]+\s*年|[，,。！？?])",
        "",
        topic_text,
    )
    return re.sub(r"\s+", "", topic_text).strip("的")[:36]


def _clean_recommendation_text(text: str, limit: int = 900) -> str:
    """Keep recommendation context compact and free of internal identifiers."""

    cleaned = re.sub(r"\[P\d{1,3}-E\d{1,3}\]", "", str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:limit]


def _clean_recommendation_title(text: str) -> str:
    """Remove PDF/Markdown artifacts before wrapping a title in bold text."""

    cleaned = re.sub(r"[*_`~]+", "", str(text or ""))
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _recommendation_evidence(question: str, profile, pipeline) -> str:
    """Return a short corpus-derived context for one recommendation.

    Most paper cards have an abstract.  When a card does not, inspect a few
    chunks from that same paper locally so the reason is based on the paper's
    content rather than exposing an ingestion limitation to the user.
    """

    abstract = _clean_recommendation_text(profile.abstract)
    if len(abstract) >= 60:
        return f"摘要：{abstract}"
    chunk_search = getattr(getattr(pipeline, "retriever", None), "chunk_search", None)
    if chunk_search is not None:
        try:
            chunks = chunk_search(
                f"{question}\n论文标题：{profile.title}",
                [profile.paper_id],
                3,
            )
        except Exception:
            chunks = ()
        snippets = []
        for chunk in chunks or ():
            text = _clean_recommendation_text(getattr(chunk, "text", ""), 700)
            if text:
                snippets.append(text)
        if snippets:
            return "正文片段：" + "；".join(snippets)
    # This fallback is intentionally user-facing and does not reveal that a
    # particular metadata field was missing.  The model will only see this
    # when no corpus text is available for the candidate.
    return "标题与结构化主题信息：" + _clean_recommendation_text(
        "；".join(
            value
            for value in (
                profile.title,
                ", ".join(profile.organisms + profile.latin_names),
                ", ".join(profile.study_types),
                ", ".join(profile.treatments),
                ", ".join(profile.metrics),
            )
            if value
        ),
        700,
    )


def _parse_recommendation_reasons(raw: str, expected_ids: Sequence[str]) -> dict[str, str]:
    """Parse the small JSON response used to write recommendation reasons."""

    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    rows = payload.get("reasons") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    expected = set(expected_ids)
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        identifier = str(row.get("id") or "").strip()
        reason = _clean_recommendation_text(str(row.get("reason") or ""), 180)
        # This wording exposed an internal evidence-generation step in older
        # prompts.  Keep the optional model path compatible with the same
        # user-facing style as the deterministic fast path.
        reason = re.sub(
            r"^摘要显示[，,:：]?\s*(?:该研究)?(?:主要讨论|主要研究)?[：:]?\s*",
            "",
            reason,
        ).strip()
        if identifier in expected and reason and identifier not in result:
            result[identifier] = reason[:180]
    return result


def _generate_recommendation_reasons(question: str, papers: Sequence, pipeline) -> dict[str, str]:
    """Ask the answer model for one evidence-grounded reason per paper.

    All selected papers are sent in one bounded request.  This avoids one LLM
    call per paper while still making recommendations explain *why* each
    paper is relevant.  The model never receives internal paper/chunk IDs.
    """

    contexts = []
    identifiers = []
    for index, item in enumerate(papers, 1):
        identifier = f"C{index}"
        identifiers.append(identifier)
        profile = item.profile
        metadata = profile.metadata
        contexts.append(
            "\n".join(
                part
                for part in (
                    f"候选{identifier}",
                    f"题名：{profile.title}",
                    f"年份：{metadata.get('year') or '未知'}",
                    f"研究对象/主题：{', '.join(profile.organisms + profile.latin_names + profile.study_types + profile.treatments + profile.metrics)}",
                    _recommendation_evidence(question, profile, pipeline),
                )
                if part and not part.endswith("：")
            )
        )
    llm = getattr(getattr(pipeline, "answerer", None), "llm", None)
    if llm is None or not contexts:
        return {}
    system = (
        "你是科研论文推荐助手。只能根据给定的论文题名、结构化信息和原文片段，"
        "为每篇论文写一句简短、具体的中文推荐理由（20-60字）。"
        "说明它研究了什么以及为什么与用户问题相关；不得编造原文没有的结论、数值、"
        "实验结果或适用性，不要提及候选编号以外的内部ID。只输出JSON："
        '{"reasons":[{"id":"C1","reason":"..."}]}，每个候选恰好一条。'
    )
    user = f"用户需求：{question}\n\n候选论文：\n\n" + "\n\n---\n\n".join(contexts)
    # Ten-paper recommendations can otherwise create a second, very large
    # prompt merely to write one sentence per paper.  Keep the reason-writing
    # request bounded; if a candidate is clipped, the deterministic local
    # fallback below still supplies a conservative reason.
    user = compact_context(user, min(settings.RAG_MAX_CONTEXT_CHARS, 18000))
    try:
        response = llm.invoke(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)
        return _parse_recommendation_reasons(raw, identifiers)
    except Exception:
        return {}


_RECOMMENDATION_SPECIES_CN = (
    ("arabidopsis thaliana", "拟南芥"),
    ("arabidopsis", "拟南芥"),
    ("tomato", "番茄"),
    ("solanum lycopersicum", "番茄"),
    ("grapevine", "葡萄"),
    ("vitis vinifera", "葡萄"),
    ("grape", "葡萄"),
    ("basil", "罗勒"),
    ("ocimum basilicum", "罗勒"),
    ("soybean", "大豆"),
    ("glycine max", "大豆"),
    ("wheat", "小麦"),
    ("rice", "水稻"),
    ("maize", "玉米"),
    ("corn", "玉米"),
    ("cotton", "棉花"),
    ("kiwifruit", "猕猴桃"),
    ("trifolium pratense", "红三叶"),
    ("hieracium pilosella", "毛莲蒿"),
    ("microalgae", "微藻"),
)


def _recommendation_source_text(profile) -> str:
    """Build trustworthy local matching text without exposing the abstract.

    Entity-card fields are automatically extracted and can contain occasional
    cross-field noise.  Recommendation reasons therefore use only the paper's
    own title and abstract; unsupported organism/treatment labels are not
    allowed to leak into user-facing prose.
    """

    values = (
        getattr(profile, "title", ""),
        getattr(profile, "abstract", ""),
    )
    return " ".join(str(value or "") for value in values).casefold()


def _recommendation_species(text: str) -> str:
    for needle, label in _RECOMMENDATION_SPECIES_CN:
        if needle in text:
            return label
    return ""


_ORGANISM_LABELS = {
    "phalaenopsis": "蝴蝶兰",
    "sugarcane": "甘蔗",
    "perennial ryegrass": "多年生黑麦草",
    "switchgrass": "柳枝稷",
    "peanut": "花生",
    "coffee": "咖啡",
    "rice": "水稻",
    "wheat": "小麦",
    "maize": "玉米",
    "soybean": "大豆",
    "arabidopsis": "拟南芥",
    "tomato": "番茄",
    "tobacco": "烟草",
    "potato": "马铃薯",
    "pepper": "辣椒",
    "barley": "大麦",
    "grapevine": "葡萄",
    "arugula": "芝麻菜",
    "cotton": "棉花",
    "cucumber": "黄瓜",
    "strawberry": "草莓",
    "citrus": "柑橘",
    "banana": "香蕉",
    "avocado": "鳄梨",
    "tea": "茶树",
}

_TREATMENT_LABELS = {
    "drought": "干旱胁迫",
    "salt": "盐胁迫",
    "heat": "热胁迫",
    "cold": "冷胁迫",
    "nitrogen": "氮处理",
    "light": "光照处理",
}

_METRIC_LABELS = {
    "chlorophyll fluorescence": "叶绿素荧光",
    "fv/fm": "Fv/Fm",
    "phi_psii": "Y(Ⅱ)",
    "npq": "NPQ",
    "etr": "ETR",
}


def _species_constraints(constraints: EntityConstraints) -> EntityConstraints:
    """Keep only species-level constraints for recommendation filtering."""

    return EntityConstraints(
        organisms=constraints.organisms,
        latin_names=constraints.latin_names,
    )


def _recommendation_topic_label(question: str, constraints: EntityConstraints) -> str:
    """Build a clean topic label instead of echoing router-appended hints.

    The question reaching this layer can carry retrieval expansions, so using it
    verbatim produced headings such as “干旱下番茄的荧光番茄高光胁迫干旱胁迫”.
    """

    species = "、".join(
        _ORGANISM_LABELS.get(name, name) for name in constraints.organisms[:2]
    )
    treatment = "、".join(
        _TREATMENT_LABELS.get(name, name) for name in constraints.treatments[:1]
    )
    metric = "、".join(
        _METRIC_LABELS.get(name, name) for name in constraints.metrics[:1]
    )
    if not metric and "荧光" in (question or ""):
        metric = "叶绿素荧光"
    subject = ""
    if species:
        subject = f"{treatment}下{species}" if treatment else species
    elif treatment:
        subject = treatment
    return "的".join(part for part in (subject, metric) if part)


def _local_recommendation_reason(
    question: str,
    profile,
    *,
    species_label: str = "",
) -> str:
    """Create a short Chinese reason without a second model request.

    Recommendation cards are intentionally generated from titles and
    structured metadata, rather than copying an English abstract into the
    answer.  This keeps the first response fast and avoids turning metadata
    quality or language into a visible error message.  The wording stays
    conservative: it states the research focus, not unsupported numerical
    results or significance claims.
    """

    text = _recommendation_source_text(profile)
    title_text = str(getattr(profile, "title", "") or "").casefold()
    # A species named only in passing inside an abstract must not relabel the
    # paper: the grapevine dataset mentioned maize in a comparison sentence.
    species = (
        species_label
        or _recommendation_species(title_text)
        or _recommendation_species(text)
    )
    conditions = []
    if "fluctuating light" in text or "variable light" in text:
        conditions.append("波动光")
    if "chilling" in text or "cold stress" in text or "cold" in text:
        conditions.append("低温")
    if "drought" in text or "water stress" in text:
        conditions.append("干旱或水分胁迫")
    if "salt stress" in text or "salinity" in text:
        conditions.append("盐胁迫")
    if "reduced air pressure" in text or "low air pressure" in text:
        conditions.append("低气压")
    if "nitrogen" in text:
        conditions.append("氮素变化")
    if "phosphorus deficiency" in text or "phosphate" in text:
        conditions.append("磷素限制")
    # Mention LED/light-spectrum conditions only when the title explicitly
    # makes them part of the study topic; an incidental growth-light detail in
    # an abstract is not enough to label the paper as an LED experiment.
    if "led" in title_text or ("light intensity" in title_text and "spectrum" in title_text):
        conditions.append("LED光强和光谱")
    conditions = list(dict.fromkeys(conditions))

    # A few recurring article types have a more useful, still conservative
    # Chinese description than a generic keyword list.
    if "qa model" in text and ("fluorescence induction" in text or "photosystem ii" in text):
        return "重新讨论叶绿素荧光诱导的QA模型及其与PSII光化学活性和结构动态的关系，适合参考荧光机理研究。"
    if ("npq" in text or "non-photochemical quenching" in text) and (
        "multivariate" in text or "photoprotection" in text
    ):
        return "围绕非光化学淬灭（NPQ）的分析方法和光保护机制展开，适合参考光保护评价与叶绿素荧光研究。"
    if ("led" in title_text or "light intensity" in title_text) and "spectrum" in title_text:
        subject = f"{species}" if species else "植物"
        return f"研究LED光强和光谱对{subject}光合作用及次生代谢的影响，适合参考光环境调控实验。"

    focus = []
    if "photosynth" in text:
        focus.append("光合作用")
    if "light response" in text or "light-response" in text or "rapid light curve" in text:
        focus.append("光响应曲线")
    if "chlorophyll fluorescence" in text or "chlorophyll-a fluorescence" in text or "chlf" in text:
        focus.append("叶绿素荧光")
    if "biosynthesis" in text or "catabolism" in text:
        focus.append("叶绿素代谢")
    if "photosystem" in text or "psii" in text:
        focus.append("光系统功能")
    if "gas exchange" in text:
        focus.append("气体交换")
    if "secondary metabol" in text:
        focus.append("次生代谢")
    if "growth" in text or "yield" in text:
        focus.append("生长或产量")
    if "photoprotection" in text or "photoinhibition" in text:
        focus.append("光保护")
    if "non-photochemical quenching" in text or "npq" in text:
        focus.append("非光化学淬灭")
    if not focus:
        focus.append("光合生理")

    subject = species or "植物"
    if conditions:
        condition_text = "、".join(conditions[:2])
        reason = f"研究{subject}在{condition_text}条件下的{'、'.join(focus[:3])}"
    else:
        reason = f"围绕{subject}的{'、'.join(focus[:3])}展开研究"
    if re.search(r"光响应曲线|light\s*response|rapid\s*light\s*curve", question or "", re.IGNORECASE):
        topic_text = "光响应曲线"
    elif re.search(r"光合作用|photosynth", question or "", re.IGNORECASE):
        topic_text = "光合作用"
    elif re.search(r"叶绿素荧光|chlorophyll\s*fluorescence", question or "", re.IGNORECASE):
        topic_text = "叶绿素荧光"
    else:
        topic_text = "相关方向"
    return f"{reason}，适合参考{topic_text}研究中的实验设计和指标评价。"


def _recommend_papers(question: str, pipeline) -> FulltextPaperAnswer:
    """Recommend papers with one evidence-grounded reason per paper."""

    topic = _recommendation_topic(question)
    if not topic:
        return FulltextPaperAnswer(
            "您希望推荐哪个方向的论文？例如叶绿素荧光、光合作用、胁迫检测、"
            "仪器应用等。补充一个研究主题后，我再为您筛选。",
            (),
        )
    count = _recommendation_count(question)
    min_year, max_year = _recommendation_year_bounds(question)
    search_limit = max(count * 4, 20)
    primary = tuple(
        pipeline.retriever.paper_search(question, search_limit)
    ) if pipeline.retriever.paper_search else ()
    candidates = list(primary)
    primary_ids = {item.profile.paper_id for item in primary}

    # If the exact topic has too few papers (or too few papers in the requested
    # year range), broaden once to adjacent photosynthesis/fluorescence topics
    # and fill the list without duplicating papers.
    def eligible_count(items) -> int:
        if min_year is None:
            return len(items)
        return sum(
            min_year <= int(item.profile.metadata.get("year") or 0) <= (max_year or 9999)
            for item in items
        )

    if eligible_count(candidates) < count and pipeline.retriever.paper_search:
        variants = tuple(dict.fromkeys(
            variant for variant in (
                f"{topic} photosynthesis chlorophyll fluorescence",
                f"{topic} photosynthesis",
                "plant photosynthesis chlorophyll fluorescence",
            ) if variant.strip()
        ))
        for variant in variants:
            for item in pipeline.retriever.paper_search(variant, search_limit):
                if item.profile.paper_id not in {row.profile.paper_id for row in candidates}:
                    candidates.append(item)
            if eligible_count(candidates) >= count:
                break

    # A crop named by the user is a hard boundary, not a ranking hint: returning
    # a maize paper for a tomato request is worse than returning nothing.  The
    # vector window can miss the requested species entirely, so backfill from the
    # small paper-card index before giving up.
    constraints = extract_entity_constraints(question)
    treatment_gap = ""
    if constraints.organisms or constraints.latin_names:
        species_only = _species_constraints(constraints)
        species_text = "、".join(
            _ORGANISM_LABELS.get(name, name) for name in constraints.organisms
        ) or "该物种"
        candidates = [
            item for item in candidates
            if profile_matches_requested_entities(item.profile, species_only)
        ]
        if not candidates:
            candidates = list(
                direct_profile_fallback(
                    pipeline.retriever.paper_index,
                    species_only,
                    limit=max(search_limit, count * 6),
                )
            )
        if not candidates:
            return FulltextPaperAnswer(
                f"当前本地论文库中没有检索到与「{species_text}」相关的研究，"
                "因此不会用其他物种的论文代替推荐。",
                (),
                direct_local_evidence=False,
            )
        # A named stress treatment is the second hard boundary.  When the corpus
        # has the species but not that treatment, do not relabel a heat-stress
        # tomato paper as a drought study: keep the species matches and say so.
        if constraints.treatments:
            treatment_only = EntityConstraints(treatments=constraints.treatments)
            with_treatment = [
                item for item in candidates
                if profile_matches_requested_entities(item.profile, treatment_only)
            ]
            if with_treatment:
                candidates = with_treatment
            else:
                treatment_gap = "、".join(
                    _TREATMENT_LABELS.get(name, name) for name in constraints.treatments
                )
        if treatment_gap:
            metric = "叶绿素荧光" if "荧光" in (question or "") else ""
            topic = f"{species_text}的{metric}" if metric else species_text
        else:
            topic = _recommendation_topic_label(question, constraints) or topic
        # Every candidate now satisfies the named species (and treatment when the
        # corpus has one), so the heading can say "直接相关".
        primary_ids = {item.profile.paper_id for item in candidates}

    candidates = tuple(candidates)
    if min_year is not None:
        papers = tuple(
            item for item in candidates
            if min_year <= int(item.profile.metadata.get("year") or 0) <= (max_year or 9999)
        )[:count]
    else:
        papers = candidates[:count]
    if not papers:
        if min_year is not None:
            return FulltextPaperAnswer(
                f"当前知识库中暂未检索到{min_year}—{max_year or 2026}年间、与该主题证据充分的论文。",
                (),
            )
        return FulltextPaperAnswer("当前知识库中未找到足够相关的论文，建议补充研究对象、胁迫类型或指标。", ())
    selected = tuple(papers[:count])
    # A recommendation should not wait for a second LLM call after retrieval.
    # Local abstracts/metadata provide an immediate conservative reason; the
    # optional model enrichment is explicitly opt-in for offline/batch use.
    reasons = (
        _generate_recommendation_reasons(question, selected, pipeline)
        if settings.PAPER_RECOMMENDATION_REASON_LLM_ENABLED
        else {}
    )
    broadened = any(item.profile.paper_id not in primary_ids for item in selected)
    if treatment_gap:
        heading = (
            f"根据当前检索结果，为您推荐以下{len(selected)}篇与{topic}相关的论文"
            f"（本地库暂未检索到{treatment_gap}条件下的直接研究，以下为同物种研究）："
        )
    elif topic and broadened:
        heading = f"根据当前检索结果，为您推荐以下{len(selected)}篇与{topic}相关的论文（含相关方向延伸）："
    elif topic:
        heading = f"根据当前检索结果，为您推荐以下{len(selected)}篇与{topic}直接相关的论文："
    else:
        heading = f"根据当前检索结果，为您推荐以下{len(selected)}篇相关论文："
    if min_year is not None and len(selected) < count:
        heading += f"当前符合{min_year}—{max_year or 2026}年条件的论文仅找到{len(selected)}篇，未使用更早论文凑数。"
    lines = [heading]
    references = []
    sources = []
    recommendation_items = []
    for index, item in enumerate(selected, 1):
        profile = item.profile
        metadata = profile.metadata
        catalog_citation = get_paper_citation(profile.paper_id)
        # Some legacy entity cards contain a shortened title fragment.  The
        # frozen citation catalog is the canonical title source, so prefer it
        # whenever it is more complete instead of exposing a truncated title
        # such as “the efficiency of photoprotection.”.
        profile_title = str(profile.title or "").strip()
        catalog_title = str(catalog_citation.title or "").strip() if catalog_citation else ""
        display_title = catalog_title if len(catalog_title) > len(profile_title) else profile_title
        display_title = _clean_recommendation_title(display_title)
        citation = format_paper_reference(
            paper_id=profile.paper_id,
            title=display_title,
            year=int(metadata.get("year") or 0),
            doi=str(metadata.get("doi") or ""),
        )
        reason = reasons.get(f"C{index}")
        if not reason:
            # Local fallback still uses corpus-derived abstract/chunk text and
            # never exposes a metadata-quality warning to the user.
            reason = _local_recommendation_reason(
                question,
                profile,
                species_label=(
                    _ORGANISM_LABELS.get(constraints.organisms[0], "")
                    if len(constraints.organisms) == 1
                    else ""
                ),
            )
        year = metadata.get("year") or (catalog_citation.year if catalog_citation else None) or "年份未知"
        doi = str(metadata.get("doi") or (catalog_citation.doi if catalog_citation else "")).removeprefix("doi:")
        extension_note = "（相关方向延伸）" if profile.paper_id not in primary_ids else ""
        title_line = f"{index}. **{display_title}（{year}）**{extension_note}"
        if doi:
            title_line += f"。DOI: {doi}"
        lines.append(f"{title_line}\n\n{reason}")
        if citation:
            references.append(f"[{index}] {citation}")
        sources.append({"filename": profile.title[:120], "source": "paper", "score": float(item.score)})
        recommendation_items.append(PaperReference(
            ordinal=index,
            paper_id=profile.paper_id,
            doi=normalize_doi(doi),
            title=display_title,
            year=int(year) if str(year).isdigit() else 0,
            fulltext_available=True,
        ))
    if references:
        # The UI already presents this block as a collapsible source section;
        # avoid the redundant nested label “参考文献：”.
        lines.append("参考来源\n" + "\n".join(references))
    lines.insert(-1 if references else len(lines), "如果您有具体的实验方向（如胁迫检测、光合效率评估或品种筛选），可以告诉我，我再帮您进一步筛选。")
    return FulltextPaperAnswer(
        "\n\n".join(lines),
        tuple(sources),
        recommendation_items=tuple(recommendation_items),
    )


def answer_literature_synthesis(
    question: str,
    *,
    pipeline: PaperRagPipeline | None = None,
    interactive: bool = False,
    max_papers: int = 8,
    manual_context: str = "",
    progress_callback: Callable[[str], None] | None = None,
    callbacks: Sequence[object] | None = None,
) -> FulltextPaperAnswer:
    """Answer a broad literature question with one bounded synthesis request.

    This deliberately does not use the two-sided comparison pipeline.  The
    user is asking for patterns across research systems, not for a verdict on
    two pre-identified papers.  Each selected paper stays a separate evidence
    island and only a compact local excerpt is sent to the answer model.
    """

    pipeline = pipeline or build_scalable_paper_rag_pipeline(interactive=interactive)
    limit = max(1, min(int(max_papers), 8))
    if progress_callback is not None:
        progress_callback("正在检索与当前问题直接相关的本地论文…")
    try:
        candidates = list(
            pipeline.retriever.paper_search(question, max(limit * 4, 24))
            if pipeline.retriever.paper_search is not None
            else pipeline.retriever.paper_index.rank(question, limit=max(limit * 4, 24))
        )
    except Exception:
        candidates = []
    plan = build_query_plan(question)
    evidence_role = required_evidence_role(question)
    intervention_family = required_intervention_family(question)
    tag_index = _research_tag_index()
    requires_exact_pair = bool(plan.constraints.organisms and plan.constraints.treatments)
    direct_candidates = [
        item for item in candidates
        if _profile_matches_requested_entities(item.profile, plan.constraints)
    ]
    if requires_exact_pair and not direct_candidates:
        # A direct title/abstract match can be absent from the bounded vector
        # candidate window.  Search the small local paper-card index before
        # declaring that the corpus has no evidence; do not scan all chunks.
        direct_candidates = direct_profile_fallback(
            pipeline.retriever.paper_index, plan.constraints, limit=limit
        )
    if requires_exact_pair and not direct_candidates:
        organism_text = "、".join(plan.constraints.organisms)
        treatment_text = "、".join(plan.constraints.treatments)
        return FulltextPaperAnswer(
            f"当前本地论文库没有检索到同时满足“{organism_text} + {treatment_text}”的直接证据。"
            "为避免用其他物种或其他胁迫替代回答，正在改用联网补充可核验的文献来源。",
            (),
            direct_local_evidence=False,
        )
    # A named crop is also a hard scientific boundary for a broad question.
    # Prefer paper cards that explicitly name that crop, without demanding an
    # equally explicit treatment term when the user did not ask for one.
    if (plan.constraints.organisms or plan.constraints.latin_names) and not direct_candidates:
        direct_candidates = direct_profile_fallback(
            pipeline.retriever.paper_index, plan.constraints, limit=max(limit * 4, 24)
        )
    # "Successful case" and "locus" questions have a stricter evidence
    # qualification than a general mechanism query. The tag is only a
    # retrieval guardrail: final claims are still grounded in paper text.
    if evidence_role:
        candidate_pool = direct_candidates or candidates
        qualified = [
            item for item in candidate_pool
            if (
                tag_index.eligible(item.profile.paper_id, required_role=evidence_role)
                and (
                    not intervention_family
                    or bool(set(tag_index.get(item.profile.paper_id).intervention_types if tag_index.get(item.profile.paper_id) else ()) & set(intervention_family))
                )
            )
        ]
        if not qualified:
            qualified = [
                item for item in pipeline.retriever.paper_index.rank(
                    question, limit=max(limit * 4, 24), strict_entities=False
                )
                if (
                    tag_index.eligible(item.profile.paper_id, required_role=evidence_role)
                    and (
                        not intervention_family
                        or bool(set(tag_index.get(item.profile.paper_id).intervention_types if tag_index.get(item.profile.paper_id) else ()) & set(intervention_family))
                    )
                )
            ]
        if not qualified:
            phrase = (
                "可核验的正向遗传改造研究"
                if evidence_role == "positive_intervention_case_candidate"
                else "直接遗传定位研究"
            )
            return FulltextPaperAnswer(
                f"当前本地论文库未检索到与问题条件匹配的{phrase}。"
                "我不会用综述、跨物种研究或间接胁迫结果替代具体案例。",
                (),
                direct_local_evidence=False,
            )
        direct_candidates = qualified
        candidates = qualified
    selected = []
    seen_ids: set[str] = set()
    for item in (direct_candidates or candidates):
        paper_id = str(item.profile.paper_id or "")
        if paper_id and paper_id not in seen_ids:
            selected.append(item)
            seen_ids.add(paper_id)
        if len(selected) >= limit:
            break
    if not selected:
        return FulltextPaperAnswer(
            "当前库内没有检索到足够可比的论文，无法可靠完成跨体系文献综合。",
            (),
        )

    if progress_callback is not None:
        progress_callback(f"已锁定 {len(selected)} 篇直接相关论文，正在核对正文证据…")

    sections: list[str] = []
    sources: list[dict] = []
    references: list[str] = []
    if manual_context.strip():
        sections.append(
            "【Device-A 操作手册证据】\n" + compact_context(manual_context.strip(), 4500)
        )
        sources.append({
            "filename": "移动式叶绿素荧光成像仪 Device-A 操作手册 V2.0",
            "source": "device_manual",
            "score": 1.0,
        })
        references.append(
            "Plant Technology Co., Ltd.. "
            "移动式叶绿素荧光成像仪 Device-A 操作手册 V2.0."
        )
    for index, paper in enumerate(selected, 1):
        profile = paper.profile
        paper_id = str(profile.paper_id)
        chunks = tuple(pipeline.retriever.chunk_search(question, [paper_id], 1))[:1]
        citation = get_paper_citation(paper_id)
        title = (citation.title if citation and citation.title else profile.title).strip()
        evidence_text = "\n\n".join(
            f"证据片段{chunk_index}: {str(chunk.text or '').strip()[:850]}"
            for chunk_index, chunk in enumerate(chunks, 1)
            if str(chunk.text or '').strip()
        )
        if not evidence_text:
            evidence_text = f"摘要：{_clean_recommendation_text(profile.abstract, 1200)}"
        source_index = len(references) + 1
        sections.append(
            f"【论文 [{source_index}]】\n题名：{title}\n"
            f"年份：{profile.metadata.get('year') or (citation.year if citation else '')}\n"
            f"{evidence_text}"
        )
        if progress_callback is not None:
            progress_callback(f"已核对 {index}/{len(selected)}：{title[:80]}")
        sources.append({
            "filename": title[:120], "source": "paper", "score": float(paper.score),
            "paper_id": paper_id,
        })
        reference = format_paper_reference(
            paper_id=paper_id,
            title=title,
            year=int(profile.metadata.get("year") or 0),
            doi=str(profile.metadata.get("doi") or ""),
        )
        if reference:
            references.append(reference)

    from src.agents.llm_factory import create_llm

    system = (
        "你是兼具植物科学通识与本地文献证据的科研助手。"
        "回答应把通用科学解释与给出的论文证据自然结合：先直接解释用户问题，再用论文支撑其中可核验的具体结论。"
        "通用解释可以用于定义概念、说明公认的生理机制、组织研究框架和提出审慎的研究建议；"
        "但不得把通用解释伪装成某篇论文的发现，也不得用它补造具体基因/位点、处理条件、数值、显著性、品种排序或产量增益。"
        "凡是具体实验结果、遗传位点、论文作者的结论或定量比较，必须能够由给出的论文支持并在句末引用。"
        "直接进入问题回答，不要用“不是穷尽性系统综述”“论文直接发现”“可作方法学参考”"
        "或“证据不足的方面”等内部工作流措辞作标题或开场。"
        "用户问的是跨体系科学问题时，不要把单篇研究的结果说成所有植物、基因型或环境都成立；"
        "用自然、专业的限定语表达范围，例如“在该研究条件下”“现有检索到的研究尚未量化……”。"
        "按用户的编号逐项回答；若某个具体数值或位点未被给出的论文确认，只简洁说明目前无法据此确定，"
        "随后给出已能确认的结论，不要罗列“证据不足”清单。"
        "若提供了Device-A操作手册证据，只用它回答仪器操作、测量条件和软件流程；"
        "不得把操作手册当作病害生物学结论的来源。"
        "不同植物或病害体系只能作为可比案例，必须说明其与用户所述体系的差异。"
        "每个来自论文的关键事实都要在句末标注相应的[n]，n必须对应给定证据块的论文编号；"
        "不要编造编号、不要把[n]放在段首、不要另写参考文献标题或文献列表，系统会统一附在底部。"
        "控制在 1800 个中文字符以内，优先给出能帮助实验设计决策的结论。"
    )
    user = f"用户问题：\n{question}\n\n可比论文证据：\n\n" + "\n\n---\n\n".join(sections)
    user = compact_context(user, 22000)
    try:
        if progress_callback is not None:
            progress_callback("正在根据已核对的论文整理回答…")
        llm = create_llm(streaming=bool(callbacks), interactive=interactive)
        response = llm.bind(max_tokens=1500).invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], config={"callbacks": list(callbacks or ())})
        text = str(response.content or "").strip()
    except Exception:
        text = "本次跨体系文献综合未能在限定时间内完成。已锁定可比论文，但暂不混用其他论文的结论；请稍后重试。"
    text = _normalise_synthesis_citations(text, len(references))
    if references:
        numbered = [
            f"[{index}] {reference}"
            for index, reference in enumerate(references, 1)
        ]
        text += "\n\n参考来源\n" + "\n\n".join(numbered)
    return FulltextPaperAnswer(
        text,
        tuple(sources),
        direct_local_evidence=bool(selected),
    )


def answer_fulltext_paper_question(
    question: str,
    *,
    paper_scope_ids: Sequence[str] | None = None,
    pipeline: PaperRagPipeline | None = None,
    interactive: bool = False,
) -> FulltextPaperAnswer:
    """Run the enabled fulltext chain; ambiguous identities never become answers."""

    pipeline = pipeline or build_scalable_paper_rag_pipeline(interactive=interactive)
    if not paper_scope_ids and _is_recommendation_request(question):
        return _recommend_papers(question, pipeline)
    plan = build_query_plan(question)
    if not plan.is_comparison:
        outcome = pipeline.run(question, paper_scope_ids=paper_scope_ids)
        if outcome.requires_paper_selection:
            resolved = _verify_low_confidence(
                question, outcome.retrieval, pipeline, interactive=interactive
            )
            if resolved is None:
                return _selection_message(outcome.retrieval)
            retrieval, attempt = _answer_resolved(question, resolved, pipeline)
            if not attempt.validation.rendered_answer:
                return FulltextPaperAnswer(
                    "当前库内证据不足，无法可靠回答该论文的具体细节。\n\n" + _paper_detail_guidance(),
                    _sources(retrieval.evidence),
                    active_paper_id=retrieval.active_paper_id,
                )
            return FulltextPaperAnswer(
                _with_detail_guidance(attempt.validation.rendered_answer, attempt),
                _sources(attempt.evidence),
                active_paper_id=retrieval.active_paper_id,
            )
        if outcome.answer is None or not outcome.answer.validation.rendered_answer:
            return FulltextPaperAnswer(
                "当前库内证据不足，无法可靠回答该论文的具体细节。\n\n" + _paper_detail_guidance(),
                _sources(outcome.retrieval.evidence),
                active_paper_id=outcome.retrieval.active_paper_id,
            )
        return FulltextPaperAnswer(
            _with_detail_guidance(outcome.answer.validation.rendered_answer, outcome.answer),
            _sources(outcome.answer.evidence),
            active_paper_id=outcome.retrieval.active_paper_id,
        )

    # The two research sides are retrieved independently.  Do not call the
    # answer model when either side has an ambiguous paper identity.
    comparison = ComparisonRetriever(pipeline.retriever, deterministic_comparison_decompose)
    retrieval = comparison.retrieve_initial(question)
    resolved_sides = []
    for side in retrieval.side_results:
        resolved_sides.append(
            side
            if side.paper_lock_confident
            else _verify_low_confidence(question, side, pipeline, interactive=interactive)
        )
    ambiguous = [item for item in resolved_sides if item is None]
    if ambiguous:
        evidence = tuple(candidate for item in retrieval.side_results for candidate in item.evidence)
        titles = [
            paper.profile.title
            for original, resolved in zip(retrieval.side_results, resolved_sides)
            if resolved is None
            for paper in original.papers[:2]
        ]
        return FulltextPaperAnswer(
            "跨论文比较的至少一侧论文身份不够明确。请补充两侧研究的物种、处理或论文标题，"
            "以避免跨论文串证据。\n\n候选论文：\n" + "\n".join(f"- {title}" for title in titles),
            _sources(evidence),
            True,
        )
    retrieval = replace(
        retrieval,
        side_results=tuple(resolved_sides),
        evidence=tuple(
            candidate
            for side, item in zip(retrieval.sides, resolved_sides)
            for candidate in label_side_evidence(side, item.evidence)
        ),
    )
    attempt = pipeline.answerer.answer(question, retrieval.evidence, retrieval.plan.required_slots)
    missing = tuple(dict.fromkeys((*attempt.validation.missing_slot_ids, *attempt.validation.insufficient_slot_ids)))
    if {"study_a", "study_b"} & set(missing):
        retrieval = comparison.expand_missing_sides(retrieval, missing)
        attempt = pipeline.answerer.answer(question, retrieval.evidence, retrieval.plan.required_slots)
    if not attempt.validation.rendered_answer:
        return FulltextPaperAnswer(
            "当前库内证据不足，无法可靠完成两侧比较。\n\n" + _paper_detail_guidance(),
            _sources(retrieval.evidence),
        )
    return FulltextPaperAnswer(
        _with_detail_guidance(attempt.validation.rendered_answer, attempt),
        _sources(attempt.evidence),
    )


def retrieve_fulltext_paper_context(
    question: str,
    *,
    paper_scope_ids: Sequence[str] | None = None,
    pipeline: PaperRagPipeline | None = None,
    interactive: bool = False,
) -> FulltextPaperContext:
    """Retrieve one isolated paper evidence island without generating prose.

    Mixed questions still need the manual retriever and a joint synthesizer,
    but their paper evidence must come from the same scalable paper-first
    retriever used by paper-only answers.
    """

    pipeline = pipeline or build_scalable_paper_rag_pipeline(interactive=interactive)
    retrieval = pipeline.retriever.retrieve_initial(
        question,
        paper_scope_ids=paper_scope_ids,
    )
    if not paper_scope_ids and not retrieval.paper_lock_confident:
        return FulltextPaperContext(
            context="",
            sources=_sources(retrieval.evidence),
            requires_paper_selection=True,
        )
    if not retrieval.coverage.complete:
        retrieval = pipeline.retriever.expand_for_missing_slots(
            retrieval,
            retrieval.coverage.missing_slot_ids,
        )
    evidence = candidates_to_evidence(retrieval.evidence)
    if not evidence:
        return FulltextPaperContext("", (), retrieval.active_paper_id)
    rows = [
        "[SELECTED_PAPER_POLICY]",
        f"Locked paper ID: {retrieval.active_paper_id}",
        "The following evidence belongs to the selected paper only. "
        "Keep paper facts separate from device-manual instructions and application suggestions.",
        "[/SELECTED_PAPER_POLICY]",
    ]
    for item in evidence:
        header = [
            f"[{item.evidence_id}]",
            f"Title: {item.title}",
            f"DOI: {item.doi}",
            f"Section: {item.section_path}",
        ]
        rows.append("\n".join(header) + "\n" + item.content)
    return FulltextPaperContext(
        context="\n\n---\n\n".join(rows),
        sources=_sources(retrieval.evidence),
        active_paper_id=retrieval.active_paper_id,
    )
