"""Local paper-internal BM25 search used for dry runs and hybrid recall."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from evaluation.local_hybrid import BM25Index, expand_query_terms

from .scalable_retrieval import EvidenceCandidate


class JsonlPaperChunkSearch:
    """Dependency-free search over a chunk JSONL; 1,500 papers remain tractable."""

    def __init__(self, path: str | Path | Sequence[str | Path]) -> None:
        """Load one or more immutable local chunk corpora.

        The production corpus and user-imported evidence corpus are kept in
        separate files so refreshing imported PDFs cannot mutate the frozen
        benchmark assets.  They are nevertheless one logical search source at
        runtime.
        """
        raw_paths = [path] if isinstance(path, (str, Path)) else list(path)
        self.paths = tuple(Path(item) for item in raw_paths if Path(item).is_file())
        self.path = self.paths[0] if self.paths else Path("")
        self._by_paper: dict[str, list[dict]] = {}
        for source_path in self.paths:
            with source_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    # Pilot chunks are one JSONL row per chunk, while the
                    # full-corpus BM25 asset is one row per paper with a ``chunks``
                    # array.  Normalize both shapes before building the per-paper
                    # indexes so runtime retrieval sees every full-text chunk.
                    nested_chunks = row.get("chunks")
                    if isinstance(nested_chunks, list):
                        paper_id = str(row.get("paper_id") or "")
                        for chunk in nested_chunks:
                            if not isinstance(chunk, dict):
                                continue
                            normalized = {**row, **chunk}
                            normalized.pop("chunks", None)
                            normalized["paper_id"] = str(chunk.get("paper_id") or paper_id)
                            self._by_paper.setdefault(normalized["paper_id"], []).append(normalized)
                    else:
                        self._by_paper.setdefault(str(row.get("paper_id") or ""), []).append(row)
        self._indexes = {
            paper_id: BM25Index([str(item.get("content") or "") for item in rows])
            for paper_id, rows in self._by_paper.items()
        }

    def __call__(
        self, question: str, paper_ids: Sequence[str], limit_per_paper: int
    ) -> Sequence[EvidenceCandidate]:
        terms = expand_query_terms(question)
        result: list[EvidenceCandidate] = []
        for paper_id in paper_ids:
            rows = self._by_paper.get(paper_id, [])
            index = self._indexes.get(paper_id)
            if index is None:
                continue
            ranked = index.rank(terms, limit_per_paper)
            # Questions consisting mainly of Chinese words can have no BM25
            # terms. Keep representative chunks so a downstream reranker can
            # still evaluate the locked paper.
            if not ranked:
                ranked = [(position, 0.0) for position in range(min(limit_per_paper, len(rows)))]
            for position, score in ranked:
                row = rows[position]
                result.append(
                    EvidenceCandidate(
                        chunk_id=str(row["chunk_id"]),
                        paper_id=paper_id,
                        text=str(row.get("content") or ""),
                        score=float(score),
                        section=str(row.get("section_path") or ""),
                        metadata=row,
                    )
                )
        return result
