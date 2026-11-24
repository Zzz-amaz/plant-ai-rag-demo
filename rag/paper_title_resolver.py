"""Resolve explicit paper references from corpus-generated title prefixes."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .paper_entities import PaperEntityProfile
from .text_normalization import normalize_scientific_query

_REFERENCE_CUES = ("研究", "论文", "文章", "study", "paper", "article")
_STOPWORDS = {"a", "an", "the", "and", "of", "in", "on", "for", "to", "with", "under"}


def _tokens(text: str) -> tuple[str, ...]:
    normalized = normalize_scientific_query(text).casefold()
    return tuple(re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)*|[\u4e00-\u9fff]+", normalized))


@dataclass(frozen=True)
class PaperTitleResolution:
    paper_id: str = ""
    confidence: float = 0.0
    matched_prefix: str = ""
    reason: str = "no_unique_title_reference"
    exact: bool = False

    @property
    def confident(self) -> bool:
        return bool(self.paper_id and self.confidence >= 0.95)


class PaperTitleResolver:
    """Resolve only explicit title prefixes that are unique in this corpus."""

    def __init__(self, profiles: Iterable[PaperEntityProfile]) -> None:
        self._profiles = tuple(profiles)
        self._title_tokens = {profile.paper_id: _tokens(profile.title) for profile in self._profiles}

    def resolve(self, query: str) -> PaperTitleResolution:
        normalized = normalize_scientific_query(query).casefold()
        if not _tokens(query):
            return PaperTitleResolution()
        has_reference_cue = any(cue in normalized for cue in _REFERENCE_CUES)
        matches: list[tuple[int, int, str, str]] = []
        for profile in self._profiles:
            title_tokens = self._title_tokens[profile.paper_id]
            matched = 0
            for size in range(min(len(title_tokens), 16), 1, -1):
                phrase = " ".join(title_tokens[:size])
                if phrase in normalized:
                    matched = size
                    break
            if matched:
                content_count = sum(token not in _STOPWORDS for token in title_tokens[:matched])
                matches.append((matched, content_count, profile.paper_id, " ".join(title_tokens[:matched])))
        if not matches:
            return PaperTitleResolution()
        best_length = max(item[0] for item in matches)
        best = [item for item in matches if item[0] == best_length]
        if len(best) != 1:
            return PaperTitleResolution(reason="ambiguous_title_prefix")
        length, content_count, paper_id, phrase = best[0]
        if not has_reference_cue or length < 2 or content_count < 1:
            return PaperTitleResolution(reason="weak_title_prefix")
        exact = length == len(self._title_tokens[paper_id])
        confidence = 1.0 if exact else 0.99 if length >= 4 else 0.97 if length >= 3 else 0.95
        reason = "exact_title_reference" if exact else "unique_explicit_title_prefix"
        return PaperTitleResolution(paper_id, confidence, phrase, reason, exact)
