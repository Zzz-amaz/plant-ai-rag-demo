"""Semantic paper-candidate adapter with entity-constrained fusion."""

from __future__ import annotations

from typing import Any, Sequence

from .paper_entities import PaperEntityIndex, RankedPaper


class ChromaPaperSearch:
    """Fuse deep semantic chunk candidates into ranked paper candidates."""

    def __init__(
        self,
        vector_store: Any,
        entity_index: PaperEntityIndex,
        *,
        semantic_depth: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.entity_index = entity_index
        self.semantic_depth = semantic_depth

    def __call__(self, question: str, limit: int) -> Sequence[RankedPaper]:
        documents = self.vector_store.similarity_search(
            question,
            k=min(self.semantic_depth, self.vector_store.chunk_count),
        )
        scores: dict[str, float] = {}
        for document in documents:
            paper_id = str(document.metadata.get("paper_id") or document.metadata.get("doi") or "")
            if not paper_id:
                continue
            scores[paper_id] = max(scores.get(paper_id, float("-inf")), float(document.metadata.get("score") or 0.0))
        # Hard named entities are enforced by PaperEntityIndex; generic terms
        # remain soft, while semantic similarity supplies recall.
        return self.entity_index.rank(
            question,
            limit=limit,
            base_scores=scores,
            base_score_weight=32.0,
            strict_entities=False,
        )
