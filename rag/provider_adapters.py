"""Production-facing adapters for Qwen rerank and DeepSeek evidence answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .coverage_answering import build_coverage_messages, validate_coverage_answer
from .citation_validator import parse_answer_json
from .evidence import Evidence
from .paper_citations import get_paper_citation
from .query_understanding import RequiredSlot
from .scalable_retrieval import EvidenceCandidate
from .paper_entities import RankedPaper
from .paper_rank_fusion import ConstraintAwarePaperRanker


class QwenChunkReranker:
    """Adapt the existing DashScope client to the generic retrieval policy."""

    def __init__(self, client: Any, *, top_n: int | None = None) -> None:
        self.client = client
        self.top_n = top_n

    def __call__(
        self, question: str, candidates: Sequence[EvidenceCandidate]
    ) -> Sequence[EvidenceCandidate]:
        if not candidates:
            return ()
        documents = [
            "\n".join(
                part
                for part in (
                    f"Title: {item.metadata.get('title', '')}",
                    f"Section: {item.section}",
                    item.text,
                )
                if not part.endswith(": ")
            )
            for item in candidates
        ]
        top_n = min(self.top_n or len(documents), len(documents))
        response = self.client.rerank(question, documents, top_n=top_n)
        reranked: list[EvidenceCandidate] = []
        seen: set[int] = set()
        for result in response.results:
            index = int(result["index"])
            if index < 0 or index >= len(candidates) or index in seen:
                raise ValueError(f"invalid or duplicate rerank result index: {index}")
            seen.add(index)
            original = candidates[index]
            reranked.append(
                EvidenceCandidate(
                    chunk_id=original.chunk_id,
                    paper_id=original.paper_id,
                    text=original.text,
                    score=float(result["relevance_score"]),
                    section=original.section,
                    metadata={
                        **dict(original.metadata),
                        "rerank_request_id": response.request_id,
                        "rerank_tokens": response.total_tokens,
                        "rerank_latency_ms": response.latency_ms,
                    },
                )
            )
        return reranked


class QwenPaperCardReranker:
    """Cheap first rerank over paper cards before any full-text chunk fan-out."""

    def __init__(self, client: Any, *, top_n: int = 10) -> None:
        self.client = client
        self.top_n = top_n

    def __call__(self, question: str, candidates: Sequence[RankedPaper]) -> Sequence[RankedPaper]:
        if not candidates:
            return ()
        documents = []
        for item in candidates:
            profile = item.profile
            documents.append(
                "\n".join(
                    (
                        f"Title: {profile.title}",
                        f"Abstract: {profile.abstract}",
                        f"Organisms: {', '.join(profile.organisms + profile.latin_names)}",
                        f"Treatments: {', '.join(profile.treatments)}",
                        f"Metrics: {', '.join(profile.metrics)}",
                    )
                )
            )
        response = self.client.rerank(
            question, documents, top_n=min(self.top_n, len(documents))
        )
        result: list[RankedPaper] = []
        seen: set[int] = set()
        for rank in response.results:
            index = int(rank["index"])
            if index < 0 or index >= len(candidates) or index in seen:
                raise ValueError(f"invalid or duplicate paper rerank index: {index}")
            seen.add(index)
            original = candidates[index]
            result.append(
                RankedPaper(
                    original.profile,
                    float(rank["relevance_score"]),
                    original.lexical_score,
                    original.constraint_score,
                    original.hard_entity_match,
                )
            )
        return result


class ConstraintAwareQwenPaperRanker:
    """Fuse retrieval and Qwen ranks after corpus-derived entity filtering."""

    def __init__(self, client: Any, ranker: ConstraintAwarePaperRanker) -> None:
        self.client = client
        self.ranker = ranker

    def __call__(self, question: str, candidates: Sequence[RankedPaper]) -> Sequence[RankedPaper]:
        if not candidates:
            return ()
        documents = []
        for item in candidates:
            profile = item.profile
            documents.append(
                "\n".join(
                    part for part in (
                        f"Title: {profile.title}",
                        f"Abstract: {profile.abstract}",
                        f"Organisms: {', '.join(profile.organisms + profile.latin_names)}",
                        f"Study types: {', '.join(profile.study_types)}",
                        f"Treatments: {', '.join(profile.treatments)}",
                        f"Metrics: {', '.join(profile.metrics)}",
                        f"Instruments: {', '.join(profile.instruments)}",
                    ) if not part.endswith(": ")
                )
            )
        response = self.client.rerank(question, documents, top_n=len(documents))
        qwen_ids = []
        qwen_scores: dict[str, float] = {}
        seen: set[int] = set()
        for result in response.results:
            index = int(result["index"])
            if index < 0 or index >= len(candidates) or index in seen:
                raise ValueError(f"invalid or duplicate paper rerank result index: {index}")
            seen.add(index)
            paper_id = candidates[index].profile.paper_id
            qwen_ids.append(paper_id)
            qwen_scores[paper_id] = float(result["relevance_score"])
        by_id = {item.profile.paper_id: item for item in candidates}
        fused = self.ranker.rank(
            question, [item.profile.paper_id for item in candidates], qwen_ids
        )
        # Hybrid/RRF supplies recall and entity eligibility.  Once Qwen has
        # examined paper cards, retain its calibrated relevance scale for the
        # final order and confidence gate.  RRF is rank-only (tiny gaps), so
        # comparing it with a semantic-score threshold would reject nearly
        # every otherwise valid paper lock.
        eligible = set(fused.eligible_paper_ids) if fused.hard_entity_applied else None
        selected_ids = [paper_id for paper_id in qwen_ids if eligible is None or paper_id in eligible]
        return [
            RankedPaper(
                by_id[paper_id].profile,
                qwen_scores[paper_id],
                by_id[paper_id].lexical_score,
                by_id[paper_id].constraint_score,
                by_id[paper_id].hard_entity_match,
            )
            for paper_id in selected_ids
        ]


def candidates_to_evidence(candidates: Sequence[EvidenceCandidate]) -> list[Evidence]:
    paper_order = list(dict.fromkeys(item.paper_id for item in candidates))
    paper_numbers = {paper_id: index for index, paper_id in enumerate(paper_order, 1)}
    counts: dict[str, int] = {}
    evidence: list[Evidence] = []
    for item in candidates:
        counts[item.paper_id] = counts.get(item.paper_id, 0) + 1
        metadata = item.metadata
        citation = get_paper_citation(item.paper_id)
        evidence.append(
            Evidence(
                evidence_id=f"P{paper_numbers[item.paper_id]:02d}-E{counts[item.paper_id]:02d}",
                paper_id=item.paper_id,
                chunk_id=item.chunk_id,
                doi=str(metadata.get("doi") or ""),
                title=str(metadata.get("title") or ""),
                year=int(metadata.get("year") or 0),
                source_type=str(metadata.get("source_type") or ""),
                section_path=item.section,
                page_start=metadata.get("page_start"),
                page_end=metadata.get("page_end"),
                content=item.text,
                content_hash=str(metadata.get("content_hash") or ""),
                retrieval_score=item.score,
                authors=str(metadata.get("authors") or (citation.authors if citation else "")),
                journal=str(metadata.get("journal") or (citation.journal if citation else "")),
                landing_page_url=str(metadata.get("landing_page_url") or (citation.url if citation else "")),
            )
        )
    return evidence


@dataclass(frozen=True)
class CoverageAnswerAttempt:
    raw_response: str
    validation: Any
    evidence: tuple[Evidence, ...]


class DeepSeekCoverageAnswerer:
    """Call a ChatOpenAI-compatible DeepSeek model and validate every slot."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def answer(
        self,
        question: str,
        candidates: Sequence[EvidenceCandidate],
        slots: Sequence[RequiredSlot],
    ) -> CoverageAnswerAttempt:
        evidence = candidates_to_evidence(candidates)
        slot_rows = [{"slot_id": slot.slot_id, "requirement": slot.description} for slot in slots]
        messages = build_coverage_messages(question, slot_rows, evidence)
        response = self.llm.invoke(messages)
        raw = response.content if isinstance(response.content, str) else str(response.content)
        payload = parse_answer_json(raw)
        validation = validate_coverage_answer(payload, slot_rows, evidence, allow_partial_claims=True)
        return CoverageAnswerAttempt(raw, validation, tuple(evidence))
