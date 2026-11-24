"""Anonymous, isolated-evidence paper identity verification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .scalable_retrieval import EvidenceCandidate


@dataclass(frozen=True)
class IdentityDecision:
    selection: str
    reason: str = ""

    @property
    def selected_index(self) -> int | None:
        if re.fullmatch(r"P[123]", self.selection):
            return int(self.selection[1]) - 1
        return None


def parse_identity_decision(raw: str) -> IdentityDecision:
    """Treat malformed output as NONE instead of guessing a paper."""
    try:
        value = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I))
    except Exception:
        value = {}
    selection = str(value.get("selection") or "NONE").upper() if isinstance(value, dict) else "NONE"
    if selection not in {"P1", "P2", "P3", "NONE"}:
        selection = "NONE"
    return IdentityDecision(selection, str(value.get("reason") or "")[:300] if isinstance(value, dict) else "")


def build_identity_messages(question: str, islands: Sequence[Sequence[EvidenceCandidate]]) -> list[dict[str, str]]:
    blocks = []
    for position, evidence in enumerate(islands[:3], 1):
        label = f"P{position}"
        title = str(evidence[0].metadata.get("title") or "") if evidence else ""
        rows = [f"[{label}] Title: {title}"]
        for number, item in enumerate(evidence[:10], 1):
            page = f"; page {item.metadata.get('page_start')}" if item.metadata.get("page_start") is not None else ""
            rows.append(f"{label}-C{number} | {item.section}{page}\n{item.text}")
        blocks.append("\n\n".join(rows))
    prompt = (
        "Select the one paper whose isolated evidence directly matches the research query, or NONE if none is sufficient. "
        "Never combine evidence from papers. Return only JSON: {\"selection\":\"P1|P2|P3|NONE\",\"reason\":\"brief\"}.\n\n"
        f"Query:\n{question}\n\n" + "\n\n---\n\n".join(blocks)
    )
    return [
        {"role": "system", "content": "You are a cautious scientific paper identity verifier."},
        {"role": "user", "content": prompt},
    ]


class DeepSeekPaperIdentityVerifier:
    """Small adapter kept separate from answer generation for auditability."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def verify(self, question: str, islands: Sequence[Sequence[EvidenceCandidate]]) -> IdentityDecision:
        response = self.llm.invoke(build_identity_messages(question, islands))
        raw = response.content if isinstance(response.content, str) else str(response.content)
        return parse_identity_decision(raw)
