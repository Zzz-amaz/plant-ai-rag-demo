"""Deterministic safeguards for long-running and over-sized RAG requests.

This module is deliberately provider-agnostic.  It does not retry a failed
model call blindly (which can double cost and latency); it keeps prompts and
replies within a bounded size and turns common transport/provider failures
into concise, actionable user messages.
"""

from __future__ import annotations

import re
from typing import Iterable


CONTEXT_MARKER = "\n\n【证据已压缩，保留了最相关的内容】\n"
REPLY_MARKER = "\n\n内容较长，已先保留核心结论；如需展开，请指定一个部分继续。"
_NUMERIC_RANGE_RE = re.compile(
    r"(?P<left>\d+(?:\.\d+)?)\s*(?:~~|~|〜)\s*(?P<right>\d+(?:\.\d+)?)"
)

# Some providers/models leak tool-call syntax into the answer text (for example
# when a model decides it wants a search tool that the request did not provide).
# The artifact is never meaningful to a reader, so it is removed deterministically
# before the text is rendered, stored, or copied.
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<\s*tool_calls?\s*>.*?(?:<\s*/\s*tool_calls?\s*>|\Z)", re.IGNORECASE | re.DOTALL
)
_TOOL_CALL_TAG_RE = re.compile(r"<\s*/?\s*tool_calls?\s*>", re.IGNORECASE)
_SPECIAL_TOOL_CALL_RE = re.compile(
    r"<[|｜]\s*/?\s*tool_calls?[^|｜>]*[|｜]>.*?"
    r"(?:<[|｜]\s*/?\s*tool_calls?[^|｜>]*[|｜]>|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_SPECIAL_TOKEN_RE = re.compile(r"<[|｜][^<>|｜]{0,60}[|｜]>")
_TOOL_CALL_JSON_RE = re.compile(
    r'^\s*\{\s*"(?:name|tool|function)"\s*:\s*"[^"]+"\s*,\s*'
    r'"(?:arguments|parameters|args)"\s*:.*\}\s*$',
    re.IGNORECASE,
)


def normalize_answer_text(text: str) -> str:
    """Normalize numeric ranges before Markdown rendering.

    Models occasionally emit ``1~~8`` when a source contains ``1~8``;
    Markdown interprets the double tilde as a strike-through delimiter.  Only
    digit-to-digit ranges are changed, so mathematical notation and prose are
    left untouched.
    """

    return _NUMERIC_RANGE_RE.sub(
        lambda match: f"{match.group('left')}–{match.group('right')}",
        str(text or ""),
    )


def _clip(text: str, max_chars: int, *, marker: str = CONTEXT_MARKER) -> str:
    """Clip text at a safe paragraph boundary while preserving both ends."""

    text = str(text or "")
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(marker) + 12:
        return text[:max_chars]

    available = max_chars - len(marker)
    # The beginning usually contains the paper identity/section, while the
    # ending often contains the conclusion or the last evidence block.
    head = max(1, int(available * 0.68))
    tail = max(1, available - head)
    head_text = text[:head].rsplit("\n", 1)[0].rstrip() or text[:head].rstrip()
    tail_text = text[-tail:].split("\n", 1)[-1].lstrip() or text[-tail:].lstrip()
    return f"{head_text}{marker}{tail_text}"


def compact_context(text: str, max_chars: int = 24000) -> str:
    """Bound retrieved evidence while retaining section/island boundaries.

    The paper and manual retrievers use ``---`` as a top-level separator.  We
    allocate space in order, clip an oversized island locally, and never mix
    two islands into a new sentence.  This is a prompt-only transformation;
    source metadata and the local index remain unchanged.
    """

    text = str(text or "")
    if len(text) <= max_chars:
        return text
    separators = re.split(r"\n\s*---\s*\n", text)
    if len(separators) <= 1:
        return _clip(text, max_chars)

    # Give every island a small minimum budget first, then distribute the
    # remainder to earlier/high-ranked islands.  The retriever already orders
    # these islands by relevance, so deterministic order is intentional.
    separator_cost = (len(separators) - 1) * 9
    usable = max(1, max_chars - separator_cost)
    minimum = min(1200, max(300, usable // max(len(separators), 1)))
    budgets = [minimum] * len(separators)
    remaining = max(0, usable - minimum * len(separators))
    for index, part in enumerate(separators):
        if remaining <= 0:
            break
        extra = min(len(part) - minimum, remaining)
        budgets[index] += max(0, extra)
        remaining -= max(0, extra)

    clipped = [_clip(part, budget) for part, budget in zip(separators, budgets)]
    return "\n\n---\n\n".join(clipped)


def compact_messages(messages: Iterable[dict], max_chars: int = 12000) -> list[dict]:
    """Keep the latest conversation turns within a hard character budget."""

    rows = [
        {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
        for item in (messages or [])
        if isinstance(item, dict) and item.get("content")
    ]
    if not rows or max_chars <= 0:
        return []
    if sum(len(item["content"]) for item in rows) <= max_chars:
        return rows

    selected: list[dict] = []
    used = 0
    # The current user turn is most important.  Walk backwards so the latest
    # user/assistant pair survives; restore chronological order afterwards.
    for item in reversed(rows):
        remaining = max_chars - used
        if remaining <= 0:
            break
        content = item["content"]
        if len(content) > remaining:
            content = _clip(content, remaining, marker="\n…【历史已压缩】…\n")
        selected.append({"role": item["role"], "content": content})
        used += len(content)
    return list(reversed(selected))


def strip_tool_call_artifacts(text: str) -> str:
    """Remove tool-call syntax that a model leaked into human-readable text.

    A model that wants a tool it was not given (for example ``web_search``)
    may emit ``<tool_call>{"name": ...}</tool_call>`` or ``<|tool_call|>``
    markers inside its answer.  Those markers must never reach the UI, the
    saved history, or an exported report, so they are stripped here rather
    than at each render site.
    """

    value = str(text or "")
    if not value:
        return ""
    value = _TOOL_CALL_BLOCK_RE.sub("", value)
    value = _SPECIAL_TOOL_CALL_RE.sub("", value)
    value = _TOOL_CALL_TAG_RE.sub("", value)
    value = _SPECIAL_TOKEN_RE.sub("", value)
    kept = [
        line
        for line in value.splitlines()
        if not _TOOL_CALL_JSON_RE.match(line.strip())
    ]
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept))
    return cleaned.strip()


def compact_reply(text: str, max_chars: int = 16000) -> str:
    """Bound generated output without cutting a citation block in half."""

    text = strip_tool_call_artifacts(normalize_answer_text(str(text or "").strip()))
    if len(text) <= max_chars:
        return text

    marker = "参考文献："
    if marker in text:
        body, references = text.split(marker, 1)
        # Bibliographies are useful even when the prose must be shortened.
        reference_budget = min(len(references), max(800, max_chars // 4))
        body_budget = max(200, max_chars - reference_budget - len(marker) - len(REPLY_MARKER))
        body = _clip(body.rstrip(), body_budget, marker=REPLY_MARKER)
        return f"{body}\n\n{marker}{references[:reference_budget].rstrip()}"

    budget = max(1, max_chars - len(REPLY_MARKER))
    clipped = text[:budget].rsplit("\n", 1)[0].rstrip() or text[:budget].rstrip()
    return clipped + REPLY_MARKER


def classify_llm_error(exc: BaseException) -> str:
    """Classify provider/network failures without depending on one SDK."""

    text = f"{type(exc).__name__}: {exc}".casefold()
    if any(
        term in text
        for term in (
            "context_length",
            "maximum context",
            "too many tokens",
            "input is too long",
            "token limit",
            "413",
        )
    ):
        return "over_limit"
    if any(term in text for term in ("timeout", "timed out", "read timeout", "deadline exceeded")):
        return "timeout"
    if any(term in text for term in ("rate limit", "too many requests", "429", "quota")):
        return "rate_limit"
    if any(
        term in text
        for term in ("connection", "connect error", "dns", "network", "502", "503", "504")
    ):
        return "network"
    return "unknown"


def user_fallback_message(
    kind: str,
    *,
    mode: str = "回答",
    language: str = "zh",
) -> str:
    """Return a safe message that never pretends an answer was generated."""

    if language == "en":
        if kind == "over_limit":
            return (
                f"This {mode.lower()} contains too much information for one request. "
                "Please split it into one focused question, such as the purpose, "
                "methods, or results."
            )
        if kind == "timeout":
            return (
                f"This {mode.lower()} is taking too long and was not completed. "
                "Please try again or narrow it to one paper, parameter, or operating step."
            )
        if kind == "rate_limit":
            return (
                "The model service is busy right now. Please try again later; "
                "no unreliable conclusion was generated."
            )
        if kind == "network":
            return (
                "The model service or network is temporarily unavailable. Please try again; "
                "no unreliable conclusion was generated."
            )
        return (
            "This answer could not be completed. Please try again or ask one more "
            "specific question."
        )

    if kind == "over_limit":
        return (
            f"这次{mode}包含的信息较多，超过了单次处理上限。"
            "请把问题拆成一个更具体的小问题（例如“研究目的”“实验方法”或“主要结果”），我再逐项回答。"
        )
    if kind == "timeout":
        return (
            f"这次{mode}处理时间较长，暂未完成。"
            "可以稍后重试，或先把问题缩小到一个论文、一个参数或一个操作步骤。"
        )
    if kind == "rate_limit":
        return "当前服务请求较多，暂时无法完成回答。请稍后重试；本次没有生成不可靠的结论。"
    if kind == "network":
        return "当前网络或模型服务暂时不可用。请检查网络后重试；本次没有生成不可靠的结论。"
    return "本次回答未能完成。请稍后重试，或把问题拆成更具体的一项。"
