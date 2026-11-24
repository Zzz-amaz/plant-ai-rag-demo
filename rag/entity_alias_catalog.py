"""Versioned corpus entity aliases, built at ingestion rather than per test."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class AliasEntry:
    canonical: str
    aliases: tuple[str, ...]
    paper_ids: tuple[str, ...]
    source: str
    confidence: float


class EntityAliasCatalog:
    def __init__(self, entries: Iterable[AliasEntry]) -> None:
        self.entries = tuple(entries)

    def match(self, query: str, *, min_confidence: float = 0.8) -> dict[str, float]:
        text = query.casefold()
        scores: dict[str, float] = {}
        for entry in self.entries:
            if entry.confidence < min_confidence:
                continue
            if any(alias.casefold() in text for alias in entry.aliases if len(alias.strip()) >= 2):
                for paper_id in entry.paper_ids:
                    scores[paper_id] = max(scores.get(paper_id, 0.0), entry.confidence)
        return scores

    @classmethod
    def read_jsonl(cls, path: str | Path) -> "EntityAliasCatalog":
        entries = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                entries.append(
                    AliasEntry(
                        canonical=str(row["canonical"]),
                        aliases=tuple(row.get("aliases") or ()),
                        paper_ids=tuple(row.get("paper_ids") or ()),
                        source=str(row.get("source") or ""),
                        confidence=float(row.get("confidence") or 0.0),
                    )
                )
        return cls(entries)

    def write_jsonl(self, path: str | Path) -> int:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(
                    json.dumps(
                        {
                            "canonical": entry.canonical,
                            "aliases": entry.aliases,
                            "paper_ids": entry.paper_ids,
                            "source": entry.source,
                            "confidence": entry.confidence,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        return len(self.entries)


def build_alias_entries(
    paper_rows: Sequence[Mapping[str, object]],
    bilingual_aliases: Mapping[str, Sequence[str]],
) -> list[AliasEntry]:
    """Bind generic bilingual names to papers by title/abstract occurrence."""

    result: list[AliasEntry] = []
    for canonical, aliases in bilingual_aliases.items():
        english_aliases = [alias.casefold() for alias in aliases if alias.isascii()]
        paper_ids = []
        for paper in paper_rows:
            text = f"{paper.get('title') or ''} {paper.get('abstract') or ''}".casefold()
            if any(alias in text for alias in english_aliases):
                paper_id = str(paper.get("paper_id") or "")
                if paper_id:
                    paper_ids.append(paper_id)
        if paper_ids:
            result.append(
                AliasEntry(
                    canonical=canonical,
                    aliases=tuple(dict.fromkeys((canonical, *aliases))),
                    paper_ids=tuple(dict.fromkeys(paper_ids)),
                    source="catalog_title_abstract_bilingual_alias",
                    confidence=0.95,
                )
            )
    return result

