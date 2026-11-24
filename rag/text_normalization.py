"""Conservative query text repair for legacy UTF-8/GB18030 mojibake."""

from __future__ import annotations

import re

_MOJIBAKE_MARKERS = ("锛", "銆", "鈥", "€", "鍙", "绱", "鑳", "儊", "杩", "閽", "鐨", "澶", "涓")


def _suspicion(text: str) -> int:
    private_use = sum(0xE000 <= ord(char) <= 0xF8FF for char in text)
    return private_use * 4 + sum(text.count(marker) for marker in _MOJIBAKE_MARKERS)


def repair_mojibake(text: str) -> str:
    """Repair one common double-decoding path, leaving normal text untouched."""

    value = str(text or "")
    before = _suspicion(value)
    if before == 0:
        return value
    try:
        candidate = value.encode("gb18030").decode("utf-8", errors="ignore")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    return candidate if candidate and _suspicion(candidate) < before else value


def normalize_scientific_query(text: str) -> str:
    repaired = repair_mojibake(text)
    repaired = re.sub(r"(?<=[\u4e00-\u9fff])\?(?=[\u4e00-\u9fff]|$)", "", repaired)
    return re.sub(r"\s+", " ", repaired).strip()
