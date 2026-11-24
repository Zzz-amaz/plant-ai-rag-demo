"""Lightweight, deterministic query understanding for the paper RAG pipeline.

The rules in this module describe scientific language.  They deliberately do
not contain evaluation question ids or expected paper ids, so the same logic
can be used when the collection grows beyond the pilot set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

from .text_normalization import normalize_scientific_query


_ORGANISM_ALIASES: dict[str, tuple[str, ...]] = {
    "phalaenopsis": ("phalaenopsis", "蝴蝶兰"),
    "sugarcane": ("sugarcane", "甘蔗", "saccharum"),
    "perennial ryegrass": ("perennial ryegrass", "多年生黑麦草", "lolium perenne"),
    "switchgrass": ("switchgrass", "柳枝稷", "panicum virgatum"),
    "peanut": ("peanut", "groundnut", "花生", "arachis hypogaea"),
    "coffee": ("coffee", "咖啡", "coffea"),
    "rice": ("rice", "水稻", "oryza sativa"),
    "wheat": ("wheat", "小麦", "triticum aestivum"),
    "maize": ("maize", "corn", "玉米", "zea mays"),
    "soybean": ("soybean", "大豆", "glycine max"),
    "arabidopsis": ("arabidopsis", "拟南芥", "arabidopsis thaliana"),
    "tomato": ("tomato", "番茄", "solanum lycopersicum", "lycopersicon esculentum"),
    "tobacco": ("tobacco", "烟草", "nicotiana", "本氏烟", "nicotiana benthamiana"),
    "potato": ("potato", "马铃薯", "solanum tuberosum"),
    "pepper": ("pepper", "辣椒", "capsicum", "capsicum annuum"),
    "barley": ("barley", "hulless barley", "青稞", "大麦", "hordeum vulgare"),
    "grapevine": ("grapevine", "grape", "葡萄", "vitis vinifera"),
    "arugula": ("arugula", "rocket", "芝麻菜", "eruca sativa"),
    "cotton": ("cotton", "棉花", "gossypium"),
    "cucumber": ("cucumber", "黄瓜", "cucumis sativus"),
    "strawberry": ("strawberry", "草莓", "fragaria"),
    "citrus": ("citrus", "柑橘", "orange", "citrus sinensis"),
    "banana": ("banana", "香蕉", "musa"),
    "avocado": ("avocado", "鳄梨", "牛油果", "persea americana"),
    "tea": ("tea plant", "camellia sinensis", "茶树"),
    "garden plants": ("garden plants", "ornamental plants", "园林植物", "观赏植物"),
}

_STUDY_TYPES: dict[str, tuple[str, ...]] = {
    "review": ("review", "综述", "meta-analysis", "元分析"),
    "field": ("field experiment", "field study", "田间", "大田"),
    "controlled": ("controlled experiment", "growth chamber", "greenhouse", "温室", "控制实验"),
    "imaging": ("imaging", "image", "成像", "图像"),
    "method": ("method", "protocol", "方法", "流程"),
}

_TREATMENTS: dict[str, tuple[str, ...]] = {
    "drought": ("drought", "water deficit", "干旱", "水分胁迫"),
    "salt": ("salt stress", "salinity", "盐胁迫", "盐处理"),
    "heat": ("heat stress", "high temperature", "热胁迫", "高温"),
    "cold": ("cold stress", "low temperature", "冷胁迫", "低温"),
    "nitrogen": ("nitrogen", "氮素", "氮处理"),
    "light": ("light treatment", "irradiance", "光照", "光强"),
}

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "fv/fm": ("fv/fm", "fᵥ/fₘ"),
    "phi_psii": ("φpsii", "Φpsii", "phi_psii", "yield of psii", "实际光化学效率"),
    "npq": ("npq", "non-photochemical quenching", "非光化学淬灭"),
    "y_no": ("y(no)", "y no", "non-regulated energy dissipation", "非调节性能量耗散"),
    "etr": ("etr", "electron transport rate", "电子传递速率"),
    "chlorophyll fluorescence": ("chlorophyll fluorescence", "叶绿素荧光"),
}


def _normalise(text: str) -> str:
    return normalize_scientific_query(text).casefold()


def _match_aliases(text: str, mapping: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    lowered = _normalise(text)
    return tuple(
        canonical
        for canonical, aliases in mapping.items()
        if any(_normalise(alias) in lowered for alias in aliases)
    )


_NON_TAXONOMIC_HEADS = {
    "a", "an", "the", "this", "these", "what", "which", "how", "why",
    "chlorophyll", "drought", "salt", "heat", "cold", "water", "light",
    "performance", "response", "responses", "effect", "effects", "results",
    "mechanisms", "evaluation", "analysis", "study", "method", "methods",
}
_NON_TAXONOMIC_TAILS = {
    "stress", "response", "responses", "fluorescence", "content", "index",
    "effect", "effects", "tolerance", "treatment", "study", "analysis",
    "method", "methods", "showed", "can", "was", "were", "for", "and",
}


def extract_latin_names(text: str) -> tuple[str, ...]:
    """Return conservative binomial-name candidates.

    Capitalisation alone creates many false positives at sentence boundaries,
    so common scientific prose heads/tails are excluded.
    """

    candidates: list[str] = []
    for match in re.finditer(
        r"\b([A-Z][a-z]{2,})\s+([a-z][a-z-]{2,})\b", normalize_scientific_query(text)
    ):
        head, tail = match.group(1).casefold(), match.group(2).casefold()
        if head in _NON_TAXONOMIC_HEADS or tail in _NON_TAXONOMIC_TAILS:
            continue
        candidates.append(f"{head} {tail}")
    return tuple(dict.fromkeys(candidates))


@dataclass(frozen=True)
class EntityConstraints:
    organisms: tuple[str, ...] = ()
    study_types: tuple[str, ...] = ()
    treatments: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    latin_names: tuple[str, ...] = ()
    quoted_terms: tuple[str, ...] = ()

    @property
    def has_hard_entity(self) -> bool:
        return bool(self.organisms or self.latin_names or self.quoted_terms)


@dataclass(frozen=True)
class RequiredSlot:
    slot_id: str
    description: str
    keywords: tuple[str, ...] = ()
    required: bool = True


@dataclass(frozen=True)
class QueryPlan:
    question: str
    query_type: str
    constraints: EntityConstraints
    required_slots: tuple[RequiredSlot, ...] = field(default_factory=tuple)
    is_comparison: bool = False


def extract_entity_constraints(question: str) -> EntityConstraints:
    """Extract reusable scientific constraints from a user question."""

    latin_names = extract_latin_names(question)
    quoted = tuple(
        dict.fromkeys(
            part.strip().casefold()
            for part in re.findall(r"[“\"']([^”\"']{2,80})[”\"']", question or "")
            if part.strip()
        )
    )
    return EntityConstraints(
        organisms=_match_aliases(question, _ORGANISM_ALIASES),
        study_types=_match_aliases(question, _STUDY_TYPES),
        treatments=_match_aliases(question, _TREATMENTS),
        metrics=_match_aliases(question, _METRIC_ALIASES),
        latin_names=latin_names,
        quoted_terms=quoted,
    )


def _slot(slot_id: str, description: str, keywords: Iterable[str] = ()) -> RequiredSlot:
    return RequiredSlot(slot_id, description, tuple(keywords))


def _direct_slots(question: str) -> tuple[RequiredSlot, ...]:
    """Keep the user's requested outcome as the invariant validation target."""

    # A comma-separated prefix usually supplies experimental context rather
    # than a second answerable subquestion (for example, "in study X, ...").
    # Keeping the full question as one atomic requirement prevents those
    # prefixes from becoming false missing slots. Explicit method/comparison
    # components are represented by dedicated slots in build_query_plan.
    return (_slot("direct_answer", f"直接、完整回答原问题：{question.strip()}"),)

def build_query_plan(question: str) -> QueryPlan:
    """Classify a question and create answer requirements before retrieval."""

    text = _normalise(question)
    # Paper titles often contain intent-like words such as "method", "effect"
    # or "differences". They remain available to entity extraction, but must
    # not decide what kind of answer the user is requesting.
    intent_text = re.sub(r"[\"“”‘’][^\"“”‘’]{2,240}[\"“”‘’]", " ", text)
    explicit_two_sides = (
        bool(re.search(r"[^，。？?]{2,60}(?:与|versus|vs\.?)\s*[^，。？?]{2,60}", intent_text))
        or bool(re.search(r"(?:比较|对比)\s*[^，。？?]{2,40}和[^，。？?]{2,40}", intent_text))
    ) and "比较实验" not in intent_text
    comparison = (
        any(token in intent_text for token in ("两项研究", "两篇论文", "两者", "异同", "difference", "compare", "versus", " vs "))
        or (explicit_two_sides and any(token in intent_text for token in ("不同", "差异", "比较", "对比", "分别", "都", "为何")))
    )
    numeric = bool(re.search(r"多少|数值|范围|比例|浓度|温度|时长|how many|value|range|percent", intent_text))
    method = any(
        token in intent_text
        for token in (
            "如何设置", "如何完成", "怎样设置", "采用什么", "使用什么", "什么仪器",
            "测量方法", "实验方法", "方法", "流程", "步骤", "暗适应", "采集设置",
            "protocol", "method", "how did", "instrument",
        )
    )
    mechanism = any(
        token in intent_text
        for token in (
            "为什么", "为何", "机制", "原因", "影响", "趋势", "如何变化", "怎样变化",
            "如何发展", "如何分化", "如何恢复", "恢复时间", "进程", "何时", "说明什么",
            "如何破坏", "什么途径", "如何耗散", "mechanism", "why", "effect", "trend",
        )
    )

    if comparison:
        query_type = "comparison"
        slots = (
            _slot("study_a", "第一项研究的材料、处理、方法和主要结果"),
            _slot("study_b", "第二项研究的材料、处理、方法和主要结果"),
            _slot("comparison", "两项研究可由证据支持的相同点与差异"),
        )
    elif numeric and not method:
        query_type = "numeric"
        slots = (
            *_direct_slots(question),
            _slot("condition", "数值对应的实验条件、材料或时间点"),
            _slot("unit", "原文给出的单位或指标定义"),
        )
    elif method:
        query_type = "method"
        slot_items: list[RequiredSlot] = []
        if any(token in text for token in ("材料", "对象", "种源", "品种", "植株", "叶片")):
            slot_items.append(_slot("material", "问题要求的研究材料、对象或取样位置"))
        if any(token in text for token in ("处理", "对照", "停水", "灌溉", "胁迫")):
            slot_items.append(_slot("treatment", "问题要求的处理与对照条件"))
        if any(token in text for token in ("测量", "仪器", "成像", "照明", "暗适应", "采集", "参数")):
            slot_items.append(_slot("measurement", "问题要求的测量方法、仪器和参数设置"))
        if any(token in text for token in ("时间", "重复", "设计", "持续", "第", "天")):
            slot_items.append(_slot("design", "问题要求的时间点、重复或实验设计信息"))
        if not slot_items:
            slot_items.extend(_direct_slots(question))
        slots = tuple(slot_items)
    elif mechanism:
        query_type = "mechanism"
        asks_explanation = any(
            token in text for token in ("为什么", "为何", "机制", "原因", "说明什么", "什么途径", "mechanism", "why")
        )
        slot_items = list(_direct_slots(question))
        if asks_explanation:
            slot_items.append(_slot("explanation", "论文明确提出且有证据支持的原因或机制解释"))
        slots = tuple(slot_items)
    else:
        query_type = "fact"
        slots = _direct_slots(question)

    return QueryPlan(
        question=question,
        query_type=query_type,
        constraints=extract_entity_constraints(question),
        required_slots=slots,
        is_comparison=comparison,
    )


def organism_aliases() -> dict[str, tuple[str, ...]]:
    """Return a copy for corpus indexing without exposing mutable globals."""

    return dict(_ORGANISM_ALIASES)
