"""Stable identifiers and file helpers for the paper knowledge base."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path


DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)


def clean_text(value: object) -> str:
    """Return a trimmed string while treating ``None`` as empty."""
    if value is None:
        return ""
    return str(value).strip()


def normalize_doi(value: object) -> str:
    """Normalize DOI URLs/prefixes, whitespace, case, and trailing punctuation."""
    doi = clean_text(value)
    if not doi:
        return ""
    doi = DOI_PREFIX_RE.sub("", doi).strip().lower()
    doi = re.sub(r"\s+", "", doi)
    doi = doi.rstrip(".,;)")
    # Remove a known web-scraping artifact where an adjacent SVG icon label
    # was concatenated to the DOI value.
    return re.sub(r"(?:\.?svg)+$", "", doi, flags=re.IGNORECASE).rstrip(".")


def canonical_doi(value: object) -> str:
    """Return a punctuation-free DOI form used only for filename matching."""
    return re.sub(r"[^a-z0-9]", "", normalize_doi(value))


def normalize_title(value: object) -> str:
    """Normalize a title for deduplication without changing its display form."""
    title = unicodedata.normalize("NFKD", clean_text(value)).casefold()
    title = re.sub(r"[^\w\s]", " ", title, flags=re.UNICODE)
    return re.sub(r"\s+", " ", title).strip()


def normalize_identifier(value: object) -> str:
    """Normalize PMID/PMCID-like identifiers."""
    return re.sub(r"\s+", "", clean_text(value)).lower()


def parse_year(value: object) -> int | None:
    """Extract a plausible four-digit publication year."""
    match = re.search(r"\b(19\d{2}|20\d{2})\b", clean_text(value))
    return int(match.group(1)) if match else None


def stable_paper_id(record: dict) -> str:
    """Build the documented DOI → PMCID/PMID → title+year identity."""
    doi = normalize_doi(record.get("doi"))
    if doi:
        return f"doi:{doi}"

    pmcid = normalize_identifier(record.get("pmcid"))
    if pmcid:
        return f"pmcid:{pmcid}"

    pmid = normalize_identifier(record.get("pmid"))
    if pmid:
        return f"pmid:{pmid}"

    title = normalize_title(record.get("title"))
    year = parse_year(record.get("year"))
    if not title:
        return ""
    digest = hashlib.sha1(f"{title}\n{year or ''}".encode("utf-8")).hexdigest()[:20]
    return f"titleyear:{digest}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file into SHA256 without loading the whole paper into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def relative_posix(path: Path, root: Path) -> str:
    """Return a portable path relative to the external paper root."""
    return path.resolve().relative_to(root.resolve()).as_posix()
