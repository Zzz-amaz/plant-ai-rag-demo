"""Generic two-sided retrieval for cross-paper comparison questions."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
from typing import Any, Protocol, Sequence

from .provider_adapters import CoverageAnswerAttempt
from .query_understanding import QueryPlan, build_query_plan
from .scalable_retrieval import EvidenceCandidate, RetrievalResult, ScalablePaperRetriever


@dataclass(frozen=True)
class ComparisonSide:
    side_id: str
    subquery: str


@dataclass(frozen=True)
class ComparisonRetrievalResult:
    plan: QueryPlan
    sides: tuple[ComparisonSide, ComparisonSide]
    side_results: tuple[RetrievalResult, RetrievalResult]
    evidence: tuple[EvidenceCandidate, ...]
    fallback_used: bool


class ComparisonDecomposer(Protocol):
    def __call__(self, question: str) -> Sequence[ComparisonSide]: ...


def deterministic_comparison_decompose(question: str) -> tuple[ComparisonSide, ComparisonSide]:
    """Cheap path for explicit A-vs-B phrasing; no corpus/test mappings."""

    text = question.strip()
    quoted = re.match(
        r"^(?:比较|对比)\s*[“\"](.+?)[”\"]\s*(?:与|和|versus|vs\.?)\s*[“\"](.+?)[”\"]\s*(?:两项)?研究\s*[：:]?\s*(.*)$",
        text,
        flags=re.IGNORECASE,
    )
    if quoted:
        left, right, shared = (part.strip(" ，。？?：:") for part in quoted.groups())
        axis = f"；需要回答的共同比较维度：{shared}" if shared else ""
        return ComparisonSide("study_a", left + axis), ComparisonSide("study_b", right + axis)

    generic = re.match(
        r"^(?:比较|对比)\s*(?:一项)?(.+?研究)\s*(?:与|和|versus|vs\.?)\s*(?:一项)?(.+?研究)\s*[：:]?\s*(.*)$",
        text,
        flags=re.IGNORECASE,
    )
    if generic:
        left, right, shared = (part.strip(" ，。？?：:") for part in generic.groups())
        axis = f"；需要回答的共同比较维度：{shared}" if shared else ""
        return ComparisonSide("study_a", left + axis), ComparisonSide("study_b", right + axis)

    patterns = (
        r"^(?:比较|对比)?\s*(.{2,40}?)\s*(?:与|versus|vs\.?)\s*(.{2,40}?)(?=研究(?:都|中|如何|分别)?|为什么|为何|如何|有何|的异同|之间|$)",
        r"^(?:比较|对比)\s*(.{2,40}?)\s*和\s*(.{2,40}?)(?=研究|为什么|为何|如何|有何|的异同|之间|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            left, right = (part.strip(" ，。？?：:") for part in match.groups())
            if left and right:
                shared = text[match.end() :].strip(" ，。？?：:")
                # Each side gets the full comparison axis, while the other
                # named study is excluded. This prevents a side query from
                # drifting toward whichever study has more searchable text.
                axis = f"；需要回答的共同比较维度：{shared}" if shared else ""
                return ComparisonSide("study_a", left + axis), ComparisonSide("study_b", right + axis)
    raise ValueError("comparison sides are not explicit enough for deterministic decomposition")


class LlmComparisonDecomposer:
    """Optional small-model decomposer for complex comparison wording."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def __call__(self, question: str) -> Sequence[ComparisonSide]:
        prompt = (
            "把下面的跨论文比较问题拆成两个可独立检索的研究侧问题。"
            "只输出JSON：{\"study_a\":\"...\",\"study_b\":\"...\"}。"
            "保留物种、处理、指标、时间和实验条件，不回答问题，不提供论文ID。\n"
            f"问题：{question}"
        )
        response = self.llm.invoke(prompt)
        raw = response.content if isinstance(response.content, str) else str(response.content)
        start, end = raw.find("{"), raw.rfind("}")
        payload = json.loads(raw[start : end + 1])
        sides = tuple(
            ComparisonSide(side_id, str(payload.get(side_id) or "").strip())
            for side_id in ("study_a", "study_b")
        )
        if any(not side.subquery for side in sides):
            raise ValueError("model did not return two comparison sides")
        return sides


def label_side_evidence(
    side: ComparisonSide, evidence: Sequence[EvidenceCandidate]
) -> tuple[EvidenceCandidate, ...]:
    return tuple(
        replace(
            item,
            section=f"[{side.side_id}] {item.section}",
            metadata={**dict(item.metadata), "comparison_side": side.side_id},
        )
        for item in evidence
    )


class ComparisonRetriever:
    def __init__(self, retriever: ScalablePaperRetriever, decomposer: ComparisonDecomposer) -> None:
        self.retriever = retriever
        self.decomposer = decomposer

    def retrieve_initial(self, question: str) -> ComparisonRetrievalResult:
        plan = build_query_plan(question)
        if not plan.is_comparison:
            raise ValueError("question is not classified as a comparison")
        sides = tuple(self.decomposer(question))
        if len(sides) != 2 or {side.side_id for side in sides} != {"study_a", "study_b"}:
            raise ValueError("comparison decomposer must return study_a and study_b")
        results = tuple(self.retriever.retrieve_initial(side.subquery) for side in sides)
        evidence = tuple(
            item
            for side, result in zip(sides, results)
            for item in label_side_evidence(side, result.evidence)
        )
        return ComparisonRetrievalResult(plan, sides, results, evidence, False)  # type: ignore[arg-type]

    def expand_missing_sides(
        self,
        result: ComparisonRetrievalResult,
        missing_slot_ids: Sequence[str],
    ) -> ComparisonRetrievalResult:
        missing = set(missing_slot_ids)
        expanded_results = []
        for side, side_result in zip(result.sides, result.side_results):
            if side.side_id in missing:
                side_result = self.retriever.expand_for_missing_slots(
                    side_result,
                    tuple(slot.slot_id for slot in side_result.plan.required_slots),
                )
            expanded_results.append(side_result)
        evidence = tuple(
            item
            for side, side_result in zip(result.sides, expanded_results)
            for item in label_side_evidence(side, side_result.evidence)
        )
        return ComparisonRetrievalResult(
            result.plan,
            result.sides,
            tuple(expanded_results),  # type: ignore[arg-type]
            evidence,
            any(item.fallback_used for item in expanded_results),
        )
