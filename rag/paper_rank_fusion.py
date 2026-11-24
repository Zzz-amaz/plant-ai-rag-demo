"""Constraint-aware paper rank fusion for scalable scientific retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .entity_alias_catalog import EntityAliasCatalog


@dataclass(frozen=True)
class PaperFusionResult:
    paper_ids: tuple[str, ...]
    hard_entity_applied: bool
    eligible_paper_ids: tuple[str, ...]
    scores: Mapping[str, float]


def reciprocal_rank_fuse(
    retrieval_ids: Sequence[str],
    rerank_ids: Sequence[str],
    *,
    rerank_weight: float = 0.2,
    rank_constant: int = 30,
    eligible_ids: set[str] | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Fuse two complete rankings while preserving deterministic tie-breaking."""

    if not 0.0 <= rerank_weight <= 1.0:
        raise ValueError("rerank_weight must be between 0 and 1")
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    retrieval = list(dict.fromkeys(retrieval_ids))
    reranked = list(dict.fromkeys(rerank_ids))
    universe = list(dict.fromkeys((*retrieval, *reranked)))
    if eligible_ids is not None:
        universe = [paper_id for paper_id in universe if paper_id in eligible_ids]
    retrieval_rank = {paper_id: index for index, paper_id in enumerate(retrieval, 1)}
    rerank_rank = {paper_id: index for index, paper_id in enumerate(reranked, 1)}
    missing_rank = len(universe) + max(len(retrieval), len(reranked)) + 1
    scores = {
        paper_id: (
            (1.0 - rerank_weight)
            / (rank_constant + retrieval_rank.get(paper_id, missing_rank))
            + rerank_weight
            / (rank_constant + rerank_rank.get(paper_id, missing_rank))
        )
        for paper_id in universe
    }
    fused = sorted(
        universe,
        key=lambda paper_id: (
            -scores[paper_id],
            retrieval_rank.get(paper_id, missing_rank),
            paper_id,
        ),
    )
    return fused, scores


class ConstraintAwarePaperRanker:
    """Apply corpus-derived entity eligibility before conservative rank fusion.

    Entity aliases are created at ingestion time and never from evaluation IDs.
    If no matching paper is present in the candidate set, the constraint is
    relaxed rather than turning a recall miss into an empty result.
    """

    def __init__(
        self,
        alias_catalog: EntityAliasCatalog,
        *,
        rerank_weight: float = 0.2,
        rank_constant: int = 30,
    ) -> None:
        self.alias_catalog = alias_catalog
        self.rerank_weight = rerank_weight
        self.rank_constant = rank_constant

    def rank(
        self,
        query: str,
        retrieval_ids: Sequence[str],
        rerank_ids: Sequence[str],
        *,
        use_reranker: bool = True,
    ) -> PaperFusionResult:
        candidates = set(retrieval_ids) | set(rerank_ids)
        matched = set(self.alias_catalog.match(query)).intersection(candidates)
        eligible = matched or None
        fused, scores = reciprocal_rank_fuse(
            retrieval_ids,
            rerank_ids,
            rerank_weight=self.rerank_weight if use_reranker else 0.0,
            rank_constant=self.rank_constant,
            eligible_ids=eligible,
        )
        return PaperFusionResult(
            paper_ids=tuple(fused),
            hard_entity_applied=bool(matched),
            eligible_paper_ids=tuple(sorted(matched)),
            scores=scores,
        )
