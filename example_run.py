"""命令行入口：直接跑一次真实的论文 RAG（与网页端完全同一条链路）。

用法::

    python example_run.py
    python example_run.py "干旱胁迫下 NPQ 与 Y(NO) 如何变化？"

链路：QueryPlan → Chroma 向量召回 + BM25 词法召回 → RRF 融合 →
qwen3-rerank 重排 → 收敛式作答 → 数字/证据校验 → 参考文献（含 DOI）。
"""

from __future__ import annotations

import sys
import time

from services.fulltext_paper_rag_service import answer_fulltext_paper_question

DEFAULT_QUESTION = "干旱胁迫下叶绿素荧光参数 Fv/Fm 的变化及其生理意义是什么？"


def main() -> int:
    question = sys.argv[1].strip() if len(sys.argv) > 1 else DEFAULT_QUESTION
    print(f"问题：{question}\n")

    started = time.perf_counter()
    answer = answer_fulltext_paper_question(question, interactive=True)
    elapsed = time.perf_counter() - started

    print(answer.text)
    print("\n" + "=" * 70)
    print(f"耗时 {elapsed:.1f}s ｜ 命中证据 {len(answer.sources)} 段")
    for index, item in enumerate(answer.sources, 1):
        score = item.get("score")
        score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "-"
        print(f"  [{index}] 重排 {score_text} ｜ {str(item.get('filename'))[:56]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
