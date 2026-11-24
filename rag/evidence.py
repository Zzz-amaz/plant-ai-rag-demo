"""Canonical evidence objects for traceable paper answers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    paper_id: str
    chunk_id: str
    doi: str
    title: str
    year: int
    source_type: str
    section_path: str
    page_start: int | None
    page_end: int | None
    content: str
    content_hash: str
    retrieval_score: float
    authors: str = ""
    journal: str = ""
    landing_page_url: str = ""

    def to_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        value = asdict(self)
        if not include_content:
            value.pop("content")
        return value


def build_evidence_objects(
    ranked_papers: list[str],
    chunks_by_paper: dict[str, list[dict]],
    *,
    max_papers: int = 5,
    max_chunks_per_paper: int = 2,
) -> list[Evidence]:
    """Assign deterministic Pxx-Exx IDs to ranked, per-paper evidence chunks."""
    result: list[Evidence] = []
    for paper_index, paper_id in enumerate(ranked_papers[:max_papers], 1):
        seen_hashes: set[str] = set()
        evidence_index = 0
        for chunk in chunks_by_paper.get(paper_id, []):
            content_hash = str(chunk.get("content_hash") or "")
            if not content_hash or content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            evidence_index += 1
            result.append(
                Evidence(
                    evidence_id=f"P{paper_index:02d}-E{evidence_index:02d}",
                    paper_id=paper_id,
                    chunk_id=str(chunk["chunk_id"]),
                    doi=str(chunk.get("doi") or ""),
                    title=str(chunk.get("title") or ""),
                    year=int(chunk.get("year") or 0),
                    source_type=str(chunk.get("source_type") or ""),
                    section_path=str(chunk.get("section_path") or ""),
                    page_start=chunk.get("page_start"),
                    page_end=chunk.get("page_end"),
                    content=str(chunk.get("content") or ""),
                    content_hash=content_hash,
                    retrieval_score=float(chunk.get("retrieval_score") or 0.0),
                )
            )
            if evidence_index >= max_chunks_per_paper:
                break
    return result


def evidence_prompt_block(evidence: list[Evidence]) -> str:
    parts = []
    for item in evidence:
        location = item.section_path or "unknown section"
        if item.page_start is not None:
            location += f"; page {item.page_start}"
            if item.page_end not in (None, item.page_start):
                location += f"-{item.page_end}"
        parts.append(
            "\n".join(
                [
                    f"[Evidence {item.evidence_id}]",
                    # Keep the runtime prompt traceable through its anonymous
                    # evidence label without disclosing the internal paper ID.
                    f"Paper: {item.evidence_id.split('-', 1)[0]}",
                    f"Title: {item.title}",
                    f"Location: {location}",
                    item.content,
                ]
            )
        )
    return "\n\n---\n\n".join(parts)
