"""User-facing paper references backed by the frozen ingestion catalog."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from core.config import settings
from rag.paper_schema import normalize_doi


@dataclass(frozen=True)
class PaperCitation:
    paper_id: str
    authors: str = ""
    title: str = ""
    journal: str = ""
    year: int = 0
    doi: str = ""
    url: str = ""


@lru_cache(maxsize=1)
def _catalog() -> dict[str, PaperCitation]:
    path = settings.resolve_path(settings.PAPER_CITATION_CATALOG_PATH)
    if not Path(path).exists():
        return {}
    records: dict[str, PaperCitation] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            paper_id = str(row.get("paper_id") or "").strip()
            if paper_id:
                records[paper_id] = PaperCitation(
                    paper_id=paper_id,
                    authors=str(row.get("authors") or "").strip(),
                    title=str(row.get("title") or "").strip(),
                    journal=str(row.get("journal") or "").strip(),
                    year=int(row.get("year") or 0),
                    doi=str(row.get("doi") or "").strip(),
                    url=str(row.get("landing_page_url") or "").strip(),
                )
    return records


def get_paper_citation(paper_id: str) -> PaperCitation | None:
    raw = str(paper_id or "").strip()
    catalog = _catalog()
    # Metadata sources are not fully consistent about whether a DOI carries
    # the ``doi:`` prefix.  Normalize both forms so the canonical title and
    # citation are still recovered for recommendation cards.
    for key in (raw, raw.casefold(), raw.removeprefix("doi:").casefold(), f"doi:{raw.removeprefix('doi:').casefold()}"):
        item = catalog.get(key)
        if item is not None:
            return item
    return None


def iter_paper_citations() -> tuple[PaperCitation, ...]:
    """Return canonical catalog rows once each for title/DOI resolution."""

    unique: dict[str, PaperCitation] = {}
    for item in _catalog().values():
        unique.setdefault(item.paper_id, item)
    return tuple(unique.values())


def format_paper_reference(
    *, paper_id: str, title: str = "", year: int = 0, doi: str = "",
    authors: str = "", journal: str = "", url: str = "",
) -> str:
    """Render a compact GB/T-like reference without internal chunk locations."""
    catalog_item = get_paper_citation(paper_id)
    if catalog_item is not None:
        # The frozen citation catalog is the ingestion-verified bibliographic
        # record. Retrieval cards may carry abbreviated or OCR-derived fields,
        # so they must never overwrite a verified user-facing reference.
        authors = catalog_item.authors or authors
        title = catalog_item.title or title
        journal = catalog_item.journal or journal
        year = catalog_item.year or year
        doi = catalog_item.doi or doi
        url = catalog_item.url or url
    doi = normalize_doi(doi)
    # Some imported catalog rows contain the adjacent web icon text appended
    # to a DOI (for example ``...101svg``).  It is a scraper/UI artifact, not
    # part of the DOI.
    doi = re.sub(r"(?:\.?svg)+$", "", doi, flags=re.IGNORECASE).rstrip(".")

    raw_url = str(url or "").strip()
    parsed = urlparse(raw_url) if raw_url else None
    host = (parsed.hostname or "").casefold() if parsed else ""
    unsafe_host = any(token in host for token in ("sci-hub", "scihub"))
    icon_url = bool(re.search(r"(?:\.?svg)+(?:[?#].*)?$", raw_url, re.IGNORECASE))
    safe_landing_url = (
        raw_url
        if parsed and parsed.scheme in {"http", "https"} and host
        and not unsafe_host and not icon_url
        else ""
    )
    # A canonical DOI resolver is stable and publisher-neutral.  Prefer it to
    # copied PDF mirrors or third-party download addresses.  For records that
    # have no DOI, retain only a safe HTTP(S) landing page.
    url = f"https://doi.org/{doi}" if doi else safe_landing_url
    parts: list[str] = []
    if authors:
        parts.append(authors.rstrip(".。"))
    if title:
        parts.append(f"{title.rstrip('.。')} [J]")
    if journal:
        parts.append(journal.rstrip(".。"))
    if year:
        parts.append(str(year))
    reference = ". ".join(parts).strip()
    if url:
        reference += f". {url.rstrip('.')}"
    if doi:
        reference += f". doi: {doi.rstrip('.')}"
    return reference.rstrip(".") + "." if reference else ""
