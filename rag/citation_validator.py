"""Deterministic validation and rendering for evidence-bound model answers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .evidence import Evidence
from .paper_citations import format_paper_reference


REFUSAL_TEXT = "当前知识库证据不足，无法可靠回答该问题。"
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-−–]?\d+(?:\.\d+)?%?")
_SHARED_PERCENT_RE = re.compile(
    r"(?<![A-Za-z])([-−–]?\d+(?:\.\d+)?)(?=\s*(?:,|and\b|和|及|与)[^.;。；\n]{0,40}?[-−–]?\d+(?:\.\d+)?%)",
    flags=re.IGNORECASE,
)
_SLASH_PERCENT_RE = re.compile(
    r"(?<![A-Za-z])[-−–]?\d+(?:\.\d+)?(?:\s*[/／]\s*[-−–]?\d+(?:\.\d+)?)+%"
)
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "january": "1", "february": "2", "march": "3", "april": "4", "may": "5", "june": "6",
    "july": "7", "august": "8", "september": "9", "october": "10", "november": "11", "december": "12",
}


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    status: str
    errors: tuple[str, ...]
    cited_evidence_ids: tuple[str, ...]
    cited_paper_ids: tuple[str, ...]
    rendered_answer: str
    accepted_claim_count: int = 0
    dropped_claim_count: int = 0
    partial: bool = False
    transformed_to_refusal: bool = False
    warnings: tuple[str, ...] = ()


def parse_answer_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Answer payload must be a JSON object")
    return value


def _numbers(text: str) -> set[str]:
    text = re.sub(r"(?<=\d)[−–](?=\d)", " to ", text)
    values = {match.group(0).replace("−", "-").replace("–", "-") for match in _NUMBER_RE.finditer(text)}
    # A compact series such as ``70/50/30%`` applies the trailing unit to all
    # values; do not also retain 70 and 50 as unitless numbers.
    for match in _SLASH_PERCENT_RE.finditer(text):
        series_numbers = [
            item.replace("−", "-").replace("–", "-")
            for item in re.findall(r"[-−–]?\d+(?:\.\d+)?", match.group(0))
        ]
        for item in series_numbers:
            values.discard(item)
            values.add(f"{item}%")
    # Scientific prose often shares a trailing percent sign across coordinated
    # values (for example, "93.3 and 99.1%"). Preserve that exact semantics.
    values.update(match.group(1).replace("−", "-").replace("–", "-") + "%" for match in _SHARED_PERCENT_RE.finditer(text))
    words = set(re.findall(r"[A-Za-z]+", text.lower()))
    values.update(value for word, value in _NUMBER_WORDS.items() if word in words)
    return values


_DECREASE_WORD_RE = re.compile(
    r"(?:下降|降低|减少|低于|较低|比[^。；;]{0,30}低|decreas(?:e|ed|es|ing)|"
    r"reduc(?:e|ed|es|ing|tion)|lower|less|declin(?:e|ed|es|ing))",
    re.IGNORECASE,
)


def _unsupported_numbers(claim_text: str, cited_text: str) -> set[str]:
    """Return claim numbers that are not supported by the cited prose.

    Scientific papers commonly encode a relative decrease as a signed value
    (``-13%``), while a Chinese answer naturally says ``降低13%``.  The sign
    and the directional word carry the same meaning, so accept that exact
    magnitude only when the claim explicitly states a decrease.  Other
    unmatched numbers remain strict.
    """
    claim_numbers = _numbers(claim_text)
    cited_numbers = _numbers(cited_text)
    unsupported = claim_numbers.difference(cited_numbers)
    if not unsupported or not _DECREASE_WORD_RE.search(claim_text):
        return unsupported
    return {
        value
        for value in unsupported
        if value.startswith("-") or f"-{value}" not in cited_numbers
    }


def validate_and_render_answer(
    payload: dict[str, Any],
    evidence: list[Evidence],
    *,
    allow_partial_claims: bool = False,
) -> ValidationResult:
    allowed = {item.evidence_id: item for item in evidence}
    status = str(payload.get("status") or "")
    claims = payload.get("claims")
    errors: list[str] = []
    warnings: list[str] = []
    cited_ids: list[str] = []

    if status not in {"answered", "insufficient_evidence"}:
        errors.append("invalid_status")
    if not isinstance(claims, list):
        errors.append("claims_must_be_list")
        claims = []
    if status == "insufficient_evidence":
        if claims:
            errors.append("refusal_must_not_contain_claims")
        return ValidationResult(
            valid=not errors,
            status=status,
            errors=tuple(errors),
            cited_evidence_ids=(),
            cited_paper_ids=(),
            rendered_answer=REFUSAL_TEXT if not errors else "",
        )
    if status == "answered" and not claims and not allow_partial_claims:
        errors.append("answered_requires_claims")

    rendered_claims: list[tuple[str, list[str]]] = []
    dropped_claim_count = 0
    for index, claim in enumerate(claims, 1):
        claim_errors: list[str] = []
        if not isinstance(claim, dict):
            claim_errors.append(f"claim_{index}_must_be_object")
            if allow_partial_claims:
                warnings.extend(claim_errors)
                dropped_claim_count += 1
            else:
                errors.extend(claim_errors)
            continue
        text = str(claim.get("text") or "").strip()
        ids = claim.get("evidence_ids")
        if not text:
            claim_errors.append(f"claim_{index}_empty")
        if not isinstance(ids, list) or not ids:
            claim_errors.append(f"claim_{index}_requires_citation")
            ids = []
        clean_ids = [str(item) for item in ids]
        invalid = [item for item in clean_ids if item not in allowed]
        if invalid:
            claim_errors.append(f"claim_{index}_invalid_citations:{','.join(invalid)}")
        valid_ids = [item for item in clean_ids if item in allowed]
        cited_text = " ".join(allowed[item].content for item in valid_ids)
        unsupported_numbers = sorted(_unsupported_numbers(text, cited_text))
        if unsupported_numbers:
            claim_errors.append(f"claim_{index}_unsupported_numbers:{','.join(unsupported_numbers)}")
        if claim_errors:
            if allow_partial_claims:
                warnings.extend(claim_errors)
                dropped_claim_count += 1
            else:
                errors.extend(claim_errors)
                cited_ids.extend(valid_ids)
            continue
        if text and valid_ids:
            cited_ids.extend(valid_ids)
            rendered_claims.append((text, valid_ids))

    if allow_partial_claims and status == "answered" and not rendered_claims and not errors:
        return ValidationResult(
            valid=True,
            status="insufficient_evidence",
            errors=(),
            cited_evidence_ids=(),
            cited_paper_ids=(),
            rendered_answer=REFUSAL_TEXT,
            accepted_claim_count=0,
            dropped_claim_count=dropped_claim_count,
            partial=bool(dropped_claim_count),
            transformed_to_refusal=True,
            warnings=tuple(warnings),
        )

    unique_ids = tuple(dict.fromkeys(cited_ids))
    cited_papers = tuple(dict.fromkeys(allowed[item].paper_id for item in unique_ids))
    if errors:
        rendered = ""
    else:
        references = []
        seen_papers: set[str] = set()
        paper_numbers: dict[str, int] = {}
        for evidence_id in unique_ids:
            item = allowed[evidence_id]
            if item.paper_id in seen_papers:
                continue
            seen_papers.add(item.paper_id)
            number = len(paper_numbers) + 1
            paper_numbers[item.paper_id] = number
            reference = format_paper_reference(
                paper_id=item.paper_id, authors=item.authors, title=item.title,
                journal=item.journal, year=item.year, doi=item.doi,
                url=item.landing_page_url,
            )
            if reference:
                references.append(f"[{number}] {reference}")
        rendered = "\n\n".join(text for text, _ids in rendered_claims)
        # Omitted unsupported claims are an internal safety action.  Exposing
        # a generic disclaimer after an otherwise complete answer confuses
        # users and invites them to treat it as a factual qualification.  The
        # answer service supplies a specific verification path only when a
        # required slot is actually unresolved.
        if references:
            rendered += "\n\n参考文献：\n" + "\n".join(references)
    return ValidationResult(
        valid=not errors,
        status=status,
        errors=tuple(errors),
        cited_evidence_ids=unique_ids,
        cited_paper_ids=cited_papers,
        rendered_answer=rendered,
        accepted_claim_count=len(rendered_claims),
        dropped_claim_count=dropped_claim_count,
        partial=bool(allow_partial_claims and dropped_claim_count),
        warnings=tuple(warnings),
    )
