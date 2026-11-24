"""Paper-first retrieval with slot-aware, bounded fallback.

External vector search and rerank services are injected as callables.  This
keeps policy testable locally and avoids coupling it to one database or model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from core.logging import logger

from .paper_entities import PaperEntityIndex, RankedPaper
from .paper_title_resolver import PaperTitleResolver
from .paper_identity_consistency import assess_paper_identity_consistency
from .query_understanding import QueryPlan, RequiredSlot, build_query_plan
from .entity_alias_catalog import EntityAliasCatalog


@dataclass(frozen=True)
class EvidenceCandidate:
    chunk_id: str
    paper_id: str
    text: str
    score: float = 0.0
    section: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SlotCoverage:
    covered_slot_ids: tuple[str, ...]
    missing_slot_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_slot_ids


@dataclass(frozen=True)
class RetrievalResult:
    plan: QueryPlan
    papers: tuple[RankedPaper, ...]
    evidence: tuple[EvidenceCandidate, ...]
    coverage: SlotCoverage
    fallback_used: bool
    initial_evidence_count: int
    active_paper_id: str = ""
    alternative_paper_ids: tuple[str, ...] = ()
    paper_lock_confident: bool = True
    paper_score_gap: float | None = None
    paper_identity_mismatches: tuple[str, ...] = ()


@dataclass(frozen=True)
class PaperLockDecision:
    """Production-visible routing decision for an isolated paper evidence island."""

    confident: bool
    score_gap: float | None
    candidate_count: int


def assess_paper_lock_confidence(
    papers: Sequence[RankedPaper], *, minimum_score_gap: float = 0.10
) -> PaperLockDecision:
    """Accept Top1 only when its reranker score clearly beats alternatives."""

    if not papers:
        return PaperLockDecision(False, None, 0)
    if len(papers) == 1:
        return PaperLockDecision(True, None, 1)
    gap = float(papers[0].score) - max(float(item.score) for item in papers[1:])
    return PaperLockDecision(gap >= minimum_score_gap, gap, len(papers))


class ChunkSearch(Protocol):
    def __call__(
        self, question: str, paper_ids: Sequence[str], limit_per_paper: int
    ) -> Sequence[EvidenceCandidate]: ...


CoverageChecker = Callable[[Sequence[RequiredSlot], Sequence[EvidenceCandidate]], SlotCoverage]
ChunkReranker = Callable[[str, Sequence[EvidenceCandidate]], Sequence[EvidenceCandidate]]
PaperSearch = Callable[[str, int], Sequence[RankedPaper]]
PaperReranker = Callable[[str, Sequence[RankedPaper]], Sequence[RankedPaper]]


def conservative_slot_coverage(
    slots: Sequence[RequiredSlot], evidence: Sequence[EvidenceCandidate]
) -> SlotCoverage:
    """Cheap pre-answer check; slots without keywords require model validation.

    This checker is intentionally conservative. A production LLM validator can
    be injected, while this fallback ensures that an unverified slot triggers
    at most one bounded evidence expansion.
    """

    combined = " ".join(item.text for item in evidence).casefold()
    covered: list[str] = []
    missing: list[str] = []
    for slot in slots:
        if slot.keywords and any(keyword.casefold() in combined for keyword in slot.keywords):
            covered.append(slot.slot_id)
        else:
            missing.append(slot.slot_id)
    return SlotCoverage(tuple(covered), tuple(missing))


def build_missing_slot_query(question: str, slots: Iterable[RequiredSlot]) -> str:
    descriptions = "；".join(slot.description for slot in slots)
    return f"{question}\n补充检索目标：{descriptions}" if descriptions else question


class ScalablePaperRetriever:
    """Retrieve papers first, then chunks, with one Top5 -> Top10 fallback."""

    def __init__(
        self,
        paper_index: PaperEntityIndex,
        chunk_search: ChunkSearch,
        *,
        paper_search: PaperSearch | None = None,
        chunk_reranker: ChunkReranker | None = None,
        coverage_checker: CoverageChecker = conservative_slot_coverage,
        paper_candidate_limit: int = 10,
        locked_paper_limit: int = 3,
        initial_chunks_per_paper: int = 5,
        fallback_chunks_per_paper: int = 10,
        isolate_single_paper: bool = True,
        entity_lock_min_score: float = 3.0,
        alias_catalog: EntityAliasCatalog | None = None,
        paper_reranker: PaperReranker | None = None,
        rerank_comparison_papers: bool = False,
        paper_lock_min_score_gap: float | None = None,
    ) -> None:
        if fallback_chunks_per_paper < initial_chunks_per_paper:
            raise ValueError("fallback chunk limit must be >= initial chunk limit")
        self.paper_index = paper_index
        self.paper_title_resolver = PaperTitleResolver(paper_index.profiles)
        self.paper_search = paper_search
        self.chunk_search = chunk_search
        self.chunk_reranker = chunk_reranker
        self.coverage_checker = coverage_checker
        self.paper_candidate_limit = paper_candidate_limit
        self.locked_paper_limit = locked_paper_limit
        self.initial_chunks_per_paper = initial_chunks_per_paper
        self.fallback_chunks_per_paper = fallback_chunks_per_paper
        self.isolate_single_paper = isolate_single_paper
        self.entity_lock_min_score = entity_lock_min_score
        self.alias_catalog = alias_catalog
        self.paper_reranker = paper_reranker
        self.rerank_comparison_papers = rerank_comparison_papers
        self.paper_lock_min_score_gap = paper_lock_min_score_gap

    @staticmethod
    def _deduplicate(items: Sequence[EvidenceCandidate]) -> list[EvidenceCandidate]:
        best: dict[str, EvidenceCandidate] = {}
        for item in items:
            existing = best.get(item.chunk_id)
            if existing is None or item.score > existing.score:
                best[item.chunk_id] = item
        return sorted(best.values(), key=lambda item: (-item.score, item.chunk_id))

    def _rerank(self, question: str, evidence: Sequence[EvidenceCandidate]) -> list[EvidenceCandidate]:
        if self.chunk_reranker is None:
            return self._deduplicate(evidence)
        try:
            return self._deduplicate(tuple(self.chunk_reranker(question, evidence)))
        except Exception as exc:
            # A paper is already identity-locked at this point.  A transient
            # external reranker timeout must not turn a request for “第2篇”
            # into an empty answer or make the user wait through another
            # request. Keep the local paper-scoped order as a safe fallback.
            logger.warning(f"论文 chunk 重排序失败，保留本地候选顺序: {exc}")
            return self._deduplicate(evidence)

    def retrieve(
        self,
        question: str,
        *,
        plan: QueryPlan | None = None,
        paper_scope_ids: Sequence[str] | None = None,
    ) -> RetrievalResult:
        """Compatibility entry point with a cheap pre-answer coverage check."""

        initial_result = self.retrieve_initial(question, plan=plan, paper_scope_ids=paper_scope_ids)
        if initial_result.coverage.complete:
            return initial_result
        return self.expand_for_missing_slots(initial_result, initial_result.coverage.missing_slot_ids)

    def retrieve_initial(
        self,
        question: str,
        *,
        plan: QueryPlan | None = None,
        paper_scope_ids: Sequence[str] | None = None,
    ) -> RetrievalResult:
        """Run paper locking and the first Top5 retrieval without expanding."""

        plan = plan or build_query_plan(question)
        if paper_scope_ids:
            scoped = []
            for paper_id in dict.fromkeys(str(value) for value in paper_scope_ids if str(value)):
                profile = self.paper_index.get(paper_id)
                if profile is None:
                    continue
                score, matched = self.paper_index.score_constraints(profile, plan.constraints)
                scoped.append(RankedPaper(profile, 1.0, 0.0, score, matched))
            paper_candidates = scoped
        else:
            paper_candidates = list(
                self.paper_search(question, self.paper_candidate_limit)
                if self.paper_search is not None
                else self.paper_index.rank(question, limit=self.paper_candidate_limit)
            )
        title_resolution = self.paper_title_resolver.resolve(question)
        # Partial titles remain a verification signal. In a growing corpus
        # they cannot safely override semantic Top1; only a full title locks.
        if not paper_scope_ids and title_resolution.confident and title_resolution.exact:
            current = {item.profile.paper_id: item for item in paper_candidates}
            chosen = current.get(title_resolution.paper_id)
            if chosen is None:
                profile = self.paper_index.get(title_resolution.paper_id)
                if profile is not None:
                    score, matched = self.paper_index.score_constraints(profile, plan.constraints)
                    chosen = RankedPaper(profile, title_resolution.confidence, 0.0, score, matched)
            if chosen is not None:
                paper_candidates = [
                    chosen,
                    *(item for item in paper_candidates if item.profile.paper_id != chosen.profile.paper_id),
                ]
        # For one explicit entity, a strong matching profile can safely move
        # ahead of generic semantic candidates. Multiple entities remain soft
        # because a single paper may legitimately discuss both.
        hard_entity_count = len(plan.constraints.organisms) + len(plan.constraints.latin_names)
        if not paper_scope_ids and hard_entity_count == 1:
            confident = []
            for item in paper_candidates:
                current_score, current_match = self.paper_index.score_constraints(
                    item.profile, plan.constraints
                )
                if current_match and current_score >= self.entity_lock_min_score:
                    confident.append((item, current_score))
            if confident:
                confident.sort(key=lambda pair: (-pair[1], -pair[0].score, pair[0].profile.paper_id))
                chosen = confident[0][0]
                paper_candidates = [chosen, *(item for item in paper_candidates if item.profile.paper_id != chosen.profile.paper_id)]
        if not paper_scope_ids and self.alias_catalog is not None:
            alias_scores = self.alias_catalog.match(question)
            current = {item.profile.paper_id: item for item in paper_candidates}
            lexical_candidates = {
                item.profile.paper_id: item
                for item in self.paper_index.rank(question, limit=self.paper_candidate_limit, strict_entities=False)
            }
            alias_matches = []
            for paper_id in alias_scores:
                item = current.get(paper_id) or lexical_candidates.get(paper_id)
                if item is None:
                    profile = self.paper_index.get(paper_id)
                    if profile is not None:
                        score, matched = self.paper_index.score_constraints(profile, plan.constraints)
                        item = RankedPaper(profile, alias_scores[paper_id], 0.0, score, matched)
                if item is not None:
                    alias_matches.append(item)
            if alias_matches:
                alias_matches.sort(key=lambda item: (-alias_scores[item.profile.paper_id], -item.score))
                chosen = alias_matches[0]
                paper_candidates = [
                    chosen,
                    *(item for item in paper_candidates if item.profile.paper_id != chosen.profile.paper_id),
                ]
        if (
            not paper_scope_ids
            and
            self.paper_reranker is not None
            and paper_candidates
            and (not plan.is_comparison or self.rerank_comparison_papers)
        ):
            paper_candidates = list(self.paper_reranker(question, paper_candidates))
        locked = tuple(paper_candidates[: self.locked_paper_limit])
        paper_ids = [item.profile.paper_id for item in locked]
        if not paper_ids:
            coverage = SlotCoverage((), tuple(slot.slot_id for slot in plan.required_slots))
            return RetrievalResult(
                plan, (), (), coverage, False, 0, paper_lock_confident=False
            )

        lock_decision = (
            PaperLockDecision(True, None, len(locked))
            if paper_scope_ids
            else
            assess_paper_lock_confidence(
                locked, minimum_score_gap=self.paper_lock_min_score_gap
            )
            if self.paper_lock_min_score_gap is not None
            else PaperLockDecision(True, None, len(locked))
        )
        identity = assess_paper_identity_consistency(locked[0].profile, plan.constraints)
        if not paper_scope_ids and identity.blocks_top1:
            lock_decision = PaperLockDecision(False, lock_decision.score_gap, lock_decision.candidate_count)

        active_paper_ids = paper_ids[:1] if self.isolate_single_paper and not plan.is_comparison else paper_ids

        initial = self._rerank(
            question,
            self.chunk_search(question, active_paper_ids, self.initial_chunks_per_paper),
        )
        coverage = self.coverage_checker(plan.required_slots, initial)
        initial_count = len(initial)
        return RetrievalResult(
            plan=plan,
            papers=locked,
            evidence=tuple(initial),
            coverage=coverage,
            fallback_used=False,
            initial_evidence_count=initial_count,
            active_paper_id=active_paper_ids[0] if len(active_paper_ids) == 1 else "",
            alternative_paper_ids=tuple(paper_ids[1:]) if len(active_paper_ids) == 1 else (),
            paper_lock_confident=lock_decision.confident,
            paper_score_gap=lock_decision.score_gap,
            paper_identity_mismatches=identity.mismatched_dimensions,
        )

    def expand_for_missing_slots(
        self,
        initial_result: RetrievalResult,
        missing_slot_ids: Sequence[str],
    ) -> RetrievalResult:
        """Perform the one allowed Top10 expansion after answer validation."""

        if initial_result.fallback_used or not missing_slot_ids:
            return initial_result
        missing_set = set(missing_slot_ids)
        missing = [slot for slot in initial_result.plan.required_slots if slot.slot_id in missing_set]
        paper_ids = (
            [initial_result.active_paper_id]
            if initial_result.active_paper_id
            else [item.profile.paper_id for item in initial_result.papers]
        )
        targeted_query = build_missing_slot_query(initial_result.plan.question, missing)
        expanded = self.chunk_search(targeted_query, paper_ids, self.fallback_chunks_per_paper)
        evidence = self._rerank(targeted_query, (*initial_result.evidence, *expanded))
        coverage = self.coverage_checker(initial_result.plan.required_slots, evidence)
        return RetrievalResult(
            plan=initial_result.plan,
            papers=initial_result.papers,
            evidence=tuple(evidence),
            coverage=coverage,
            fallback_used=True,
            initial_evidence_count=initial_result.initial_evidence_count,
            active_paper_id=initial_result.active_paper_id,
            alternative_paper_ids=initial_result.alternative_paper_ids,
            paper_lock_confident=initial_result.paper_lock_confident,
            paper_score_gap=initial_result.paper_score_gap,
            paper_identity_mismatches=initial_result.paper_identity_mismatches,
        )

    def try_next_paper(self, previous: RetrievalResult) -> RetrievalResult:
        """Try one alternative paper as a separate evidence island.

        No chunk from the previous paper is carried over, preventing protocol
        or numeric facts from being blended across similar studies.
        """

        if not previous.alternative_paper_ids:
            return previous
        next_paper = previous.alternative_paper_ids[0]
        question = previous.plan.question
        evidence = self._rerank(
            question,
            self.chunk_search(question, [next_paper], self.initial_chunks_per_paper),
        )
        coverage = self.coverage_checker(previous.plan.required_slots, evidence)
        return RetrievalResult(
            plan=previous.plan,
            papers=previous.papers,
            evidence=tuple(evidence),
            coverage=coverage,
            fallback_used=False,
            initial_evidence_count=len(evidence),
            active_paper_id=next_paper,
            alternative_paper_ids=previous.alternative_paper_ids[1:],
            paper_lock_confident=False,
            paper_score_gap=previous.paper_score_gap,
            paper_identity_mismatches=previous.paper_identity_mismatches,
        )
