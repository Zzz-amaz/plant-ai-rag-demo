"""Qwen text-rerank adapter and deterministic helpers for paper retrieval evaluation."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence


DEFAULT_HTTP_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


DEFAULT_INSTRUCTION = (
    "Given a scientific research question, rank passages by whether they contain direct evidence "
    "that answers the question. Prefer exact experimental conditions, measurements, results, and "
    "mechanistic conclusions over passages that only mention the same topic."
)


def rerank_document_text(chunk: dict) -> str:
    """Build the text sent to the reranker without adding unverified metadata."""
    parts = [
        f"Title: {str(chunk.get('title') or '').strip()}",
        f"Section: {str(chunk.get('section_path') or '').strip()}",
        str(chunk.get("content") or "").strip(),
    ]
    return "\n".join(part for part in parts if part and not part.endswith(": "))


def request_cache_key(model: str, instruction: str, query: str, candidate_ids: Sequence[str]) -> str:
    payload = json.dumps(
        {"model": model, "instruction": instruction, "query": query, "candidate_ids": list(candidate_ids)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_rerank_results(candidates: Sequence[dict], results: Sequence[dict]) -> list[dict]:
    """Map API result indexes back to candidates, rejecting malformed/duplicate indexes."""
    ranked: list[dict] = []
    seen: set[int] = set()
    for rank, result in enumerate(results, 1):
        index = int(result["index"])
        if index < 0 or index >= len(candidates) or index in seen:
            raise ValueError(f"Invalid rerank result index: {index}")
        seen.add(index)
        item = dict(candidates[index])
        item["rerank_score"] = float(result["relevance_score"])
        item["rerank_rank"] = rank
        ranked.append(item)
    return ranked


def select_top_per_paper(
    ranked: Sequence[dict],
    paper_order: Sequence[str],
    *,
    chunks_per_paper: int,
) -> list[dict]:
    """Keep reranker order within each paper while preserving the selected paper order."""
    selected: list[dict] = []
    for paper_id in paper_order:
        matches = [item for item in ranked if item.get("paper_id") == paper_id]
        selected.extend(matches[:chunks_per_paper])
    return selected


@dataclass(frozen=True)
class RerankResponse:
    results: list[dict]
    total_tokens: int
    request_id: str
    latency_ms: float


class QwenRerankClient:
    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-rerank",
        instruction: str = DEFAULT_INSTRUCTION,
        *,
        endpoint: str = DEFAULT_HTTP_ENDPOINT,
        timeout_seconds: float = 120.0,
        session: Any | None = None,
        force_http: bool = False,
        default_attempts: int = 3,
    ):
        if not api_key.strip():
            raise ValueError("DashScope API key is required")
        self.api_key = api_key.strip()
        self.model = model
        self.instruction = instruction
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.session = session
        self.force_http = force_http
        self.default_attempts = max(1, int(default_attempts))

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
        attempts: int | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> RerankResponse:
        try:
            import dashscope
        except ModuleNotFoundError:
            dashscope = None

        attempts = self.default_attempts if attempts is None else max(1, int(attempts))
        top_n = min(top_n or len(documents), len(documents))
        if top_n <= 0:
            raise ValueError("At least one document is required")
        last_error: Exception | None = None
        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                if dashscope is not None and not self.force_http:
                    response = dashscope.TextReRank.call(
                        api_key=self.api_key,
                        model=self.model,
                        query=query,
                        documents=list(documents),
                        top_n=top_n,
                        return_documents=False,
                        instruct=self.instruction,
                    )
                    status_code = int(getattr(response, "status_code", 0) or 0)
                    if status_code != 200:
                        raise RuntimeError(
                            f"DashScope rerank failed: status={status_code} "
                            f"code={getattr(response, 'code', '')} message={getattr(response, 'message', '')}"
                        )
                    output = dict(getattr(response, "output", {}) or {})
                    usage = dict(getattr(response, "usage", {}) or {})
                    request_id = str(getattr(response, "request_id", "") or "")
                else:
                    output, usage, request_id = self._rerank_http(query, documents, top_n)

                latency_ms = (time.perf_counter() - started) * 1000
                results = [dict(item) for item in output.get("results", [])]
                if len(results) != top_n:
                    raise RuntimeError(f"Expected {top_n} rerank results, received {len(results)}")
                return RerankResponse(
                    results=results,
                    total_tokens=int(usage.get("total_tokens") or 0),
                    request_id=request_id,
                    latency_ms=latency_ms,
                )
            except Exception as exc:  # network/API failures are retried and then surfaced
                last_error = exc
                if attempt + 1 < attempts:
                    sleeper(2**attempt)
        raise RuntimeError(f"qwen3-rerank failed after {attempts} attempts") from last_error

    def _rerank_http(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> tuple[dict, dict, str]:
        import requests

        session = self.session or requests
        response = session.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": {"query": query, "documents": list(documents)},
                "parameters": {
                    "top_n": top_n,
                    "return_documents": False,
                    "instruct": self.instruction,
                },
            },
            timeout=self.timeout_seconds,
        )
        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(f"DashScope rerank returned non-JSON status={response.status_code}") from exc
        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope rerank failed: status={response.status_code} "
                f"code={body.get('code', '')} message={body.get('message', '')}"
            )
        return (
            dict(body.get("output") or {}),
            dict(body.get("usage") or {}),
            str(body.get("request_id") or ""),
        )
