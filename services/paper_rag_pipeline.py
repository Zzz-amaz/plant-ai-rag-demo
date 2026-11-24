"""Grey-release orchestration for the scalable paper RAG path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from rag.provider_adapters import CoverageAnswerAttempt
from rag.comparison_retrieval import ComparisonRetriever
from rag.scalable_retrieval import RetrievalResult, ScalablePaperRetriever


class CoverageAnswerer(Protocol):
    def answer(self, question: str, candidates: Any, slots: Any) -> CoverageAnswerAttempt: ...


@dataclass(frozen=True)
class PaperRagOutcome:
    retrieval: RetrievalResult
    answer: CoverageAnswerAttempt | None
    answer_attempts: int
    requires_paper_selection: bool = False
    suggested_paper_ids: tuple[str, ...] = ()


class PaperRagPipeline:
    """Answer Top5, then expand to Top10 once only when validation finds gaps."""

    def __init__(
        self,
        retriever: ScalablePaperRetriever,
        answerer: CoverageAnswerer,
        *,
        max_paper_attempts: int = 2,
    ) -> None:
        self.retriever = retriever
        self.answerer = answerer
        self.max_paper_attempts = max(1, max_paper_attempts)

    def run(
        self, question: str, *, paper_scope_ids: Sequence[str] | None = None
    ) -> PaperRagOutcome:
        retrieval = self.retriever.retrieve_initial(
            question, paper_scope_ids=paper_scope_ids
        )
        if not paper_scope_ids and not retrieval.paper_lock_confident:
            suggestions = tuple(
                dict.fromkeys(
                    (
                        retrieval.active_paper_id,
                        *retrieval.alternative_paper_ids,
                    )
                )
            )
            return PaperRagOutcome(
                retrieval,
                None,
                0,
                requires_paper_selection=True,
                suggested_paper_ids=tuple(value for value in suggestions if value),
            )
        answer = self.answerer.answer(question, retrieval.evidence, retrieval.plan.required_slots)
        missing = tuple(
            dict.fromkeys(
                (*answer.validation.missing_slot_ids, *answer.validation.insufficient_slot_ids)
            )
        )
        if not missing:
            return PaperRagOutcome(retrieval, answer, 1)

        retrieval = self.retriever.expand_for_missing_slots(retrieval, missing)
        answer = self.answerer.answer(question, retrieval.evidence, retrieval.plan.required_slots)
        attempts = 2
        missing = tuple(
            dict.fromkeys((*answer.validation.missing_slot_ids, *answer.validation.insufficient_slot_ids))
        )
        # A different paper is a separate attempt, never an evidence merge.
        if missing and self.max_paper_attempts > 1 and retrieval.alternative_paper_ids:
            retrieval = self.retriever.try_next_paper(retrieval)
            answer = self.answerer.answer(question, retrieval.evidence, retrieval.plan.required_slots)
            attempts += 1
        return PaperRagOutcome(retrieval, answer, attempts)


class ComparisonPaperRagPipeline:
    """Retrieve and rerank two research sides independently before synthesis."""

    def __init__(self, retriever: ComparisonRetriever, answerer: CoverageAnswerer) -> None:
        self.retriever = retriever
        self.answerer = answerer

    def run(self, question: str) -> Any:
        retrieval = self.retriever.retrieve_initial(question)
        answer = self.answerer.answer(question, retrieval.evidence, retrieval.plan.required_slots)
        missing = tuple(
            dict.fromkeys((*answer.validation.missing_slot_ids, *answer.validation.insufficient_slot_ids))
        )
        if not ({"study_a", "study_b"} & set(missing)):
            return retrieval, answer, 1
        retrieval = self.retriever.expand_missing_sides(retrieval, missing)
        answer = self.answerer.answer(question, retrieval.evidence, retrieval.plan.required_slots)
        return retrieval, answer, 2
