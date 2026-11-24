"""Slot-aware answer protocol for detecting omitted subquestions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .citation_validator import ValidationResult, _numbers, validate_and_render_answer
from .evidence import Evidence, evidence_prompt_block


COVERAGE_SYSTEM_PROMPT = """你是论文证据问答器。只能使用给定Evidence回答，不得使用外部知识，不得编造。
用户会给出必须逐项处理的回答槽位。输出必须是一个JSON对象，不要Markdown代码围栏：
{"status":"answered|insufficient_evidence","slots":[{"slot_id":"原样复制槽位ID","status":"answered|insufficient_evidence","claims":[{"text":"一个原子事实","evidence_ids":["P01-E01"]}]}]}
规则：
1. 每个给定slot_id必须且只能出现一次，不得新增、合并或省略槽位。
2. 每个槽位独立判断；证据不足时该槽位status=insufficient_evidence且claims=[]。
3. 每条claim必须被引用Evidence直接支持；数字、单位、处理条件、物种和仪器必须在证据中明确出现。
4. 只要至少一个槽位有答案，总status=answered；所有槽位均证据不足时才输出insufficient_evidence。
5. 使用中文，简洁回答，不输出论文标题、年份、DOI或参考文献；除非用户明确要求逐项列举，最多输出8条原子事实。"""


@dataclass(frozen=True)
class CoverageValidationResult:
    valid: bool
    status: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    required_slot_count: int
    answered_slot_count: int
    missing_slot_ids: tuple[str, ...]
    insufficient_slot_ids: tuple[str, ...]
    complete: bool
    claim_validation: ValidationResult | None
    rendered_answer: str


def build_coverage_messages(question: str, slots: list[dict], evidence: list[Evidence]) -> list[dict[str, str]]:
    slot_block = "\n".join(f"- {slot['slot_id']}: {slot['requirement']}" for slot in slots)
    user = f"问题：\n{question}\n\n必须逐项处理的回答槽位：\n{slot_block}\n\n可用证据：\n{evidence_prompt_block(evidence)}"
    return [{"role": "system", "content": COVERAGE_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def coverage_prompt_sha256(messages: list[dict[str, str]], model: str) -> str:
    serialized = model + "\n" + "\n".join(f"{m['role']}\n{m['content']}" for m in messages)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_coverage_answer(
    payload: dict[str, Any],
    slots: list[dict],
    evidence: list[Evidence],
    *,
    allow_partial_claims: bool = False,
) -> CoverageValidationResult:
    required = [str(slot["slot_id"]) for slot in slots]
    raw_slots = payload.get("slots")
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(raw_slots, list):
        raw_slots = []
        errors.append("slots_must_be_list")
    seen: set[str] = set()
    answered_ids: list[str] = []
    insufficient_ids: list[str] = []
    flattened_claims: list[dict] = []
    allowed = {item.evidence_id: item for item in evidence}
    for index, item in enumerate(raw_slots, 1):
        if not isinstance(item, dict):
            errors.append(f"slot_{index}_must_be_object")
            continue
        slot_id = str(item.get("slot_id") or "")
        if slot_id not in required:
            errors.append(f"unknown_slot:{slot_id}")
            continue
        if slot_id in seen:
            errors.append(f"duplicate_slot:{slot_id}")
            continue
        seen.add(slot_id)
        slot_status = str(item.get("status") or "")
        claims = item.get("claims")
        if not isinstance(claims, list):
            errors.append(f"slot_claims_must_be_list:{slot_id}")
            claims = []
        if slot_status == "answered":
            if not claims:
                errors.append(f"answered_slot_requires_claims:{slot_id}")
            else:
                answered_ids.append(slot_id)
                for claim in claims:
                    if isinstance(claim, dict):
                        repaired = dict(claim)
                        ids = [str(item) for item in repaired.get("evidence_ids", []) if str(item) in allowed]
                        claim_numbers = _numbers(str(repaired.get("text") or ""))
                        cited_numbers = set().union(*(_numbers(allowed[item].content) for item in ids)) if ids else set()
                        unsupported = claim_numbers.difference(cited_numbers)
                        if unsupported:
                            cited_papers = {allowed[item].paper_id for item in ids}
                            matches = [
                                item.evidence_id
                                for item in evidence
                                if (not cited_papers or item.paper_id in cited_papers)
                                and unsupported.issubset(_numbers(item.content))
                            ]
                            if matches:
                                repaired["evidence_ids"] = list(dict.fromkeys(ids + [matches[0]]))
                                warnings.append(f"numeric_citation_repaired:{slot_id}:{matches[0]}")
                        flattened_claims.append({**repaired, "_slot_id": slot_id})
        elif slot_status == "insufficient_evidence":
            if claims:
                errors.append(f"insufficient_slot_must_not_have_claims:{slot_id}")
            insufficient_ids.append(slot_id)
        else:
            errors.append(f"invalid_slot_status:{slot_id}")
    missing = [slot_id for slot_id in required if slot_id not in seen]
    errors.extend(f"missing_slot:{slot_id}" for slot_id in missing)

    claim_validation = None
    if not errors and flattened_claims:
        claim_validation = validate_and_render_answer(
            {"status": "answered", "claims": flattened_claims},
            evidence,
            allow_partial_claims=allow_partial_claims,
        )
        if not claim_validation.valid:
            errors.extend(claim_validation.errors)
        warnings.extend(claim_validation.warnings)
    payload_status = str(payload.get("status") or "")
    expected_status = "answered" if answered_ids else "insufficient_evidence"
    if payload_status != expected_status:
        errors.append("overall_status_mismatch")
    complete = not errors and len(answered_ids) == len(required) and bool(claim_validation and claim_validation.status == "answered")
    rendered = claim_validation.rendered_answer if claim_validation and claim_validation.valid else ""
    return CoverageValidationResult(
        valid=not errors,
        status=expected_status,
        errors=tuple(errors),
        warnings=tuple(warnings),
        required_slot_count=len(required),
        answered_slot_count=len(answered_ids),
        missing_slot_ids=tuple(missing),
        insufficient_slot_ids=tuple(insufficient_ids),
        complete=complete,
        claim_validation=claim_validation,
        rendered_answer=rendered,
    )


def slots_sha256(slots: list[dict]) -> str:
    return hashlib.sha256(json.dumps(slots, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
