"""Paper-level entity profiles and constrained paper ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .query_understanding import (
    EntityConstraints,
    extract_entity_constraints,
    extract_latin_names,
    organism_aliases,
)


_INSTRUMENT_PATTERNS = (
    re.compile(r"\bLI[- ]?COR\s+(?:LI[- ]?)?\d{3,4}(?:[- ]?\d{1,3})?(?:XT)?\b", re.I),
    re.compile(r"\b(?:Dual[- ]?PAM[- ]?\d+|Imaging[- ]?PAM(?:\s+M[- ]?Series)?|MINI[- ]?PAM(?:[- ]?II)?|PAM[- ]?\d+)\b", re.I),
    re.compile(r"\b(?:Handy|Pocket)\s*PEA(?:\+)?\b", re.I),
    re.compile(r"\bPlant Efficiency Analy[sz]er(?:\s+PEA)?\b", re.I),
    re.compile(r"\bFluorPen(?:\s+FP[- ]?\d+)?\b", re.I),
    re.compile(r"\bMultispeQ(?:\s+v?\d+(?:\.\d+)*)?\b", re.I),
    re.compile(r"\bSPAD[- ]?\d{3}(?:Plus)?\b", re.I),
    re.compile(r"\b(?:OS1p|OS5p\+?|CF Imager|FluorCam)\b", re.I),
)


def extract_instrument_mentions(text: str) -> tuple[str, ...]:
    """Extract conservative instrument family/model mentions from full text.

    Patterns intentionally require a known scientific instrument family so
    years, treatment codes, and statistical model numbers are not promoted to
    paper-level identity constraints.
    """

    mentions: list[str] = []
    for pattern in _INSTRUMENT_PATTERNS:
        for match in pattern.finditer(text or ""):
            value = re.sub(r"\s+", " ", match.group(0)).strip(" ,.;:()[]")
            if value:
                mentions.append(value)
    return tuple(dict.fromkeys(value.casefold() for value in mentions))


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_/-]{1,}|[\u4e00-\u9fff]{2,}", text or "")
    }


def _tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = re.split(r"[,;|，；]", value)
    elif isinstance(value, (list, tuple, set)):
        parts = [str(item) for item in value]
    else:
        parts = [str(value)]
    return tuple(dict.fromkeys(part.strip().casefold() for part in parts if part.strip()))


@dataclass(frozen=True)
class PaperEntityProfile:
    paper_id: str
    title: str
    abstract: str = ""
    organisms: tuple[str, ...] = ()
    study_types: tuple[str, ...] = ()
    treatments: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    instruments: tuple[str, ...] = ()
    latin_names: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def search_text(self) -> str:
        return " ".join(
            (
                self.title,
                self.abstract,
                *self.organisms,
                *self.study_types,
                *self.treatments,
                *self.metrics,
                *self.instruments,
                *self.latin_names,
                *self.keywords,
            )
        )


def profile_from_paper(paper: Mapping[str, Any]) -> PaperEntityProfile:
    """Build a profile from title/abstract and optional extracted entity context."""

    title = str(paper.get("title") or paper.get("paper_title") or "")
    abstract = str(paper.get("abstract") or "")
    metadata_text = " ".join(
        str(paper.get(key) or "")
        for key in ("stress_type", "fluorescence_metrics", "instrument_model", "keywords", "study_type")
    )
    # entity_context is produced from method/results chunks during offline
    # ingestion. It improves structured fields but is not stored in the short
    # paper card or sent to a paper-level reranker.
    entity_context = str(paper.get("entity_context") or "")
    constraints = extract_entity_constraints(f"{title} {abstract} {metadata_text} {entity_context}")

    # Latin names are useful hard constraints even when a crop is not yet in
    # the small bilingual alias table.
    # Use the title only. Abstract sentence boundaries create too many apparent
    # "Capitalised lowercase" pairs to treat them as hard taxonomy constraints.
    latin_names = extract_latin_names(title)
    return PaperEntityProfile(
        paper_id=str(paper.get("paper_id") or paper.get("id") or paper.get("doi") or "").strip(),
        title=title.strip(),
        abstract=abstract.strip(),
        organisms=constraints.organisms,
        study_types=tuple(dict.fromkeys((*constraints.study_types, *_tuple(paper.get("study_type"))))),
        treatments=tuple(dict.fromkeys((*constraints.treatments, *_tuple(paper.get("stress_type"))))),
        metrics=tuple(dict.fromkeys((*constraints.metrics, *_tuple(paper.get("fluorescence_metrics"))))),
        instruments=_tuple(paper.get("instrument_model") or paper.get("instruments")),
        latin_names=latin_names,
        keywords=_tuple(paper.get("keywords")),
        metadata={key: value for key, value in paper.items() if key not in {"abstract", "entity_context"}},
    )


@dataclass(frozen=True)
class RankedPaper:
    profile: PaperEntityProfile
    score: float
    lexical_score: float
    constraint_score: float
    hard_entity_match: bool


class PaperEntityIndex:
    """A small in-memory paper-card index; 1,500 profiles fit comfortably."""

    def __init__(self, profiles: Iterable[PaperEntityProfile]):
        self.profiles = tuple(profile for profile in profiles if profile.paper_id)
        self._by_id = {profile.paper_id: profile for profile in self.profiles}
        self._tokens = {profile.paper_id: _tokens(profile.search_text) for profile in self.profiles}
        self._title_tokens = {profile.paper_id: _tokens(profile.title) for profile in self.profiles}
        counts: dict[str, int] = {}
        for terms in self._tokens.values():
            for term in terms:
                counts[term] = counts.get(term, 0) + 1
        total = max(1, len(self.profiles))
        self._idf = {term: math.log((total + 1) / (count + 1)) + 1 for term, count in counts.items()}

    def get(self, paper_id: str) -> PaperEntityProfile | None:
        return self._by_id.get(paper_id)

    @staticmethod
    def _constraint_score(profile: PaperEntityProfile, constraints: EntityConstraints) -> tuple[float, bool]:
        score = 0.0
        hard_match = not constraints.has_hard_entity

        if constraints.organisms:
            matched = set(constraints.organisms) & set(profile.organisms)
            hard_match = bool(matched)
            score += 4.0 * len(matched)
            if not matched:
                score -= 6.0
        if constraints.latin_names:
            matched = set(constraints.latin_names) & set(profile.latin_names)
            hard_match = hard_match or bool(matched)
            score += 5.0 * len(matched)
            if not matched:
                score -= 6.0
        if constraints.quoted_terms:
            haystack = profile.search_text.casefold()
            matched_count = sum(term in haystack for term in constraints.quoted_terms)
            hard_match = hard_match or matched_count > 0
            score += 3.0 * matched_count
        score += 1.5 * len(set(constraints.study_types) & set(profile.study_types))
        score += 1.5 * len(set(constraints.treatments) & set(profile.treatments))
        score += 1.5 * len(set(constraints.metrics) & set(profile.metrics))
        return score, hard_match

    def score_constraints(
        self, profile: PaperEntityProfile, constraints: EntityConstraints
    ) -> tuple[float, bool]:
        """Score current-query constraints for an externally ranked profile."""

        return self._constraint_score(profile, constraints)

    def rank(
        self,
        question: str,
        limit: int = 10,
        strict_entities: bool = True,
        *,
        base_scores: Mapping[str, float] | None = None,
        base_score_weight: float = 8.0,
    ) -> list[RankedPaper]:
        constraints = extract_entity_constraints(question)
        query_terms = _tokens(question)
        rows: list[tuple[PaperEntityProfile, float, float, bool, float]] = []
        for profile in self.profiles:
            if base_scores is not None and profile.paper_id not in base_scores:
                continue
            overlap = query_terms & self._tokens[profile.paper_id]
            lexical = sum(self._idf.get(term, 1.0) for term in overlap)
            lexical += sum(self._idf.get(term, 1.0) for term in query_terms & self._title_tokens[profile.paper_id])
            constraint_score, hard_match = self._constraint_score(profile, constraints)
            if strict_entities and constraints.has_hard_entity and not hard_match:
                continue
            base_score = float((base_scores or {}).get(profile.paper_id, 0.0))
            rows.append((profile, lexical, constraint_score, hard_match, base_score))

        ranked: list[RankedPaper] = []
        lexical_max = max((row[1] for row in rows), default=1.0) or 1.0
        base_values = [row[4] for row in rows]
        base_min = min(base_values, default=0.0)
        base_span = max(base_values, default=0.0) - base_min
        for profile, lexical, constraint_score, hard_match, base_score in rows:
            if base_scores is None:
                total = lexical + constraint_score
            else:
                # Semantic/hybrid retrieval supplies recall. Entity and title
                # overlap refine it, but cannot swamp it merely because their
                # raw scoring scales are larger.
                semantic = (base_score - base_min) / base_span if base_span else 0.0
                lexical_normalised = lexical / lexical_max
                total = base_score_weight * semantic + 2.0 * lexical_normalised + constraint_score
            ranked.append(
                RankedPaper(
                    profile,
                    total,
                    lexical,
                    constraint_score,
                    hard_match,
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.profile.paper_id))
        return ranked[: max(0, limit)]


def write_profiles(path: str | Path, profiles: Iterable[PaperEntityProfile]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for profile in profiles:
            handle.write(json.dumps(asdict(profile), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_profiles(path: str | Path) -> list[PaperEntityProfile]:
    profiles: list[PaperEntityProfile] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in (
                "organisms", "study_types", "treatments", "metrics", "instruments", "latin_names", "keywords"
            ):
                row[key] = tuple(row.get(key) or ())
            profiles.append(PaperEntityProfile(**row))
    return profiles


def known_organism_aliases() -> dict[str, tuple[str, ...]]:
    return organism_aliases()
