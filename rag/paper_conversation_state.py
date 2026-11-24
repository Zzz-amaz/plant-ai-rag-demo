"""Structured, conversation-scoped paper selection state.

Paper identity must not be reconstructed from assistant prose.  A generated
answer can cite several related works, which makes pronouns such as “这篇”
ambiguous even though the user has already selected one paper.  This module
keeps the recommendation list and active paper as application data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Iterable


_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def normalize_doi(value: str) -> str:
    raw = str(value or "").strip().casefold()
    raw = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", raw)
    match = _DOI_RE.search(raw)
    if not match:
        return ""
    doi = match.group(0).rstrip(".,;，。；)").casefold()
    return re.sub(r"(?:\.?svg)+$", "", doi, flags=re.IGNORECASE).rstrip(".")


def normalize_title(value: str) -> str:
    """Normalize a title for deterministic containment/equality matching."""

    text = str(value or "").casefold()
    text = re.sub(r"[*_`~]", "", text)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text).strip()


@dataclass(frozen=True)
class PaperReference:
    ordinal: int
    paper_id: str
    doi: str
    title: str
    year: int = 0
    fulltext_available: bool = True

    @classmethod
    def from_dict(cls, row: dict) -> "PaperReference":
        return cls(
            ordinal=int(row.get("ordinal") or 0),
            paper_id=str(row.get("paper_id") or "").strip(),
            doi=normalize_doi(str(row.get("doi") or "")),
            title=str(row.get("title") or "").strip(),
            year=int(row.get("year") or 0),
            fulltext_available=bool(row.get("fulltext_available", True)),
        )


@dataclass(frozen=True)
class PendingTask:
    """A small, deterministic task waiting for one user-supplied field.

    This is deliberately not a transcript summary.  It represents a concrete
    unfinished UI task such as "recommend 10 papers, waiting for a topic" so a
    terse follow-up ("叶绿素荧光方向") can complete the original request.
    """

    kind: str = ""
    requested_count: int = 0
    awaiting: tuple[str, ...] = ()

    @property
    def is_recommendation_topic_request(self) -> bool:
        return self.kind == "recommend_papers" and "topic" in self.awaiting

    @classmethod
    def recommendation_topic(cls, requested_count: int) -> "PendingTask":
        return cls(
            kind="recommend_papers",
            requested_count=max(1, min(int(requested_count or 4), 10)),
            awaiting=("topic",),
        )


@dataclass
class PaperConversationState:
    list_version: int = 0
    recommendation_items: list[PaperReference] = field(default_factory=list)
    active_paper_id: str = ""
    active_doi: str = ""
    active_title: str = ""
    pending_task: PendingTask | None = None

    def replace_recommendations(self, items: Iterable[PaperReference]) -> None:
        rows = list(items)
        self.list_version += 1
        self.recommendation_items = rows
        self.clear_active()

    def activate(self, item: PaperReference) -> None:
        self.active_paper_id = item.paper_id
        self.active_doi = normalize_doi(item.doi)
        self.active_title = item.title

    def activate_values(self, *, paper_id: str, doi: str = "", title: str = "") -> None:
        self.active_paper_id = str(paper_id or "").strip()
        self.active_doi = normalize_doi(doi)
        self.active_title = str(title or "").strip()

    def clear_active(self) -> None:
        self.active_paper_id = ""
        self.active_doi = ""
        self.active_title = ""

    def set_pending_recommendation_topic(self, requested_count: int) -> None:
        self.pending_task = PendingTask.recommendation_topic(requested_count)

    def clear_pending_task(self) -> None:
        self.pending_task = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "list_version": self.list_version,
                "recommendation_items": [asdict(item) for item in self.recommendation_items],
                "active_paper_id": self.active_paper_id,
                "active_doi": self.active_doi,
                "active_title": self.active_title,
                "pending_task": asdict(self.pending_task) if self.pending_task else None,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, raw: str | None) -> "PaperConversationState":
        if not raw:
            return cls()
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return cls()
            pending_raw = payload.get("pending_task")
            pending = None
            if isinstance(pending_raw, dict):
                awaiting = pending_raw.get("awaiting") or ()
                if isinstance(awaiting, (list, tuple)):
                    pending = PendingTask(
                        kind=str(pending_raw.get("kind") or ""),
                        requested_count=max(0, int(pending_raw.get("requested_count") or 0)),
                        awaiting=tuple(str(item) for item in awaiting if str(item)),
                    )
            return cls(
                list_version=max(0, int(payload.get("list_version") or 0)),
                recommendation_items=[
                    PaperReference.from_dict(row)
                    for row in payload.get("recommendation_items") or []
                    if isinstance(row, dict)
                ],
                active_paper_id=str(payload.get("active_paper_id") or "").strip(),
                active_doi=normalize_doi(str(payload.get("active_doi") or "")),
                active_title=str(payload.get("active_title") or "").strip(),
                pending_task=pending,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return cls()


@dataclass(frozen=True)
class PaperResolution:
    item: PaperReference | None
    source: str = ""


_CHINESE_NUMBERS = {
    "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _ordinal(query: str, count: int) -> int | None:
    text = str(query or "")
    reverse = re.search(r"倒数\s*(?:第\s*)?([一两二三四五六七八九十\d]+)\s*篇", text)
    if reverse:
        raw = reverse.group(1)
        value = int(raw) if raw.isdigit() else _CHINESE_NUMBERS.get(raw, 0)
        return count - value + 1 if 1 <= value <= count else None
    if re.search(r"最后一篇|\b(?:the\s+)?last\s+(?:paper|article|publication)\b", text, re.I):
        return count if count else None
    match = re.search(r"(?:第\s*)?([一两二三四五六七八九十\d]+)\s*篇", text)
    if match:
        raw = match.group(1)
        value = int(raw) if raw.isdigit() else _CHINESE_NUMBERS.get(raw, 0)
        return value if 1 <= value <= count else None
    english = {
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    }
    match = re.search(
        r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+"
        r"(?:paper|article|publication)\b",
        text,
        re.I,
    )
    value = english.get(match.group(1).casefold(), 0) if match else 0
    return value if 1 <= value <= count else None


def resolve_from_state(
    query: str,
    state: PaperConversationState,
    *,
    catalog_items: Iterable[PaperReference] = (),
) -> PaperResolution:
    """Resolve DOI > explicit title > ordinal > active-paper pronoun."""

    direct_doi = normalize_doi(query)
    available = [*state.recommendation_items, *catalog_items]
    if direct_doi:
        item = next((row for row in available if normalize_doi(row.doi) == direct_doi), None)
        return PaperResolution(
            item or PaperReference(0, f"doi:{direct_doi}", direct_doi, ""),
            "doi",
        )

    normalized_query = normalize_title(query)
    title_matches = [
        row for row in available
        if len(normalize_title(row.title)) >= 16
        and normalize_title(row.title) in normalized_query
    ]
    if title_matches:
        title_matches.sort(key=lambda row: len(normalize_title(row.title)), reverse=True)
        return PaperResolution(title_matches[0], "title")

    position = _ordinal(query, len(state.recommendation_items))
    if position is not None:
        item = next(
            (row for row in state.recommendation_items if row.ordinal == position),
            None,
        )
        return PaperResolution(item, "ordinal" if item else "")

    has_pronoun = bool(re.search(
        r"(?:这|该)(?:篇)?(?:论文|文章)|(?:该|这)(?:项)?研究|"
        r"(?:它|其)(?:与|和|的|中)|"
        r"\b(?:this|the)\s+(?:paper|article|publication|study)\b",
        query or "",
        re.I,
    ))
    if has_pronoun and state.active_paper_id:
        return PaperResolution(
            PaperReference(
                0,
                state.active_paper_id,
                state.active_doi,
                state.active_title,
            ),
            "active",
        )
    return PaperResolution(None, "")
