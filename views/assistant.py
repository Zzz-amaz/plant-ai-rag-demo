"""科研问答助手页：真实论文 RAG 链路。

链路：QueryPlan 解析 → Chroma 向量召回 + BM25 词法召回 → RRF 融合 →
qwen3-rerank 二次重排 → 收敛式作答 → 数字与证据 ID 校验 → 渲染参考文献（含 DOI）。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List

import streamlit as st

from services.fulltext_paper_rag_service import answer_fulltext_paper_question
from services.llm_gateway import (
    MISSING_KEY_MESSAGE,
    MissingCredentialsError,
    credentials_ready,
    missing_credentials,
)
from ui import components, state, theme

logger = logging.getLogger(__name__)

_MAX_SOURCE_ROWS = 12

_SUGGESTED = (
    "干旱胁迫下叶绿素荧光参数 Fv/Fm 的变化及其生理意义是什么？",
    "水分胁迫下 PSII 光化学效率 Y(II) 如何响应？",
    "干旱条件下 NPQ 与 Y(NO) 的变化趋势有何差异？",
)


def render() -> None:
    theme.page_header(
        "科研问答助手",
        "问题经关键词与向量双路召回、RRF 融合与 qwen3-rerank 重排后作答，"
        "结论中的数字逐条回溯到原文，参考文献给出可跳转 DOI。",
        "💬",
    )
    state.load_chat_state()
    _render_sidebar()

    if not credentials_ready():
        st.error(f"{MISSING_KEY_MESSAGE}\n\n缺少：{'、'.join(missing_credentials())}")

    messages: List[dict] = st.session_state.chat_messages
    if not messages and not st.session_state.get("chat_pending"):
        _render_welcome()
    for message in messages:
        _render_message(message)

    _handle_pending_question()

    question = st.chat_input("请输入您的问题…", key="chat_box")
    if question and question.strip():
        # 先登记再重跑，避免重跑时读到旧输入重复作答
        st.session_state.chat_pending = question.strip()
        st.rerun()


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("**对话管理**")
        if st.button("＋ 新建对话", width="stretch"):
            state.archive_chat()
            state.clear_chat()
            st.rerun()

        threads = st.session_state.get("chat_threads") or []
        if threads:
            st.caption(f"存档 {len(threads)} 条")
            for thread in threads[:5]:
                label_column, delete_column = st.columns([0.82, 0.18])
                with label_column:
                    if st.button(
                        str(thread.get("title") or "未命名")[:18],
                        key=f"thread_{thread.get('id')}",
                        width="stretch",
                        help=str(thread.get("updated") or ""),
                    ):
                        state.load_thread(str(thread.get("id")))
                        st.rerun()
                with delete_column:
                    if st.button("🗑", key=f"thread_del_{thread.get('id')}"):
                        state.delete_thread(str(thread.get("id")))
                        st.rerun()
        else:
            st.caption("暂无存档")

        if st.session_state.chat_messages:
            st.divider()
            st.download_button(
                label="导出对话 (.md)",
                data=_transcript(),
                file_name=f"{datetime.now():%Y%m%d%H%M}_对话记录.md",
                mime="text/markdown",
                width="stretch",
            )
            if st.button("清空当前对话", width="stretch"):
                state.clear_chat()
                st.rerun()


def _render_welcome() -> None:
    theme.empty_state(
        "💬",
        "开始一次对话",
        "提问会实时检索论文库；回答中的数字若无法在原文定位，会被校验器剔除。",
    )
    st.caption("试试这些问题（都会命中真实文献）：")
    columns = st.columns(len(_SUGGESTED))
    for column, question in zip(columns, _SUGGESTED):
        with column:
            if st.button(question, key=f"suggest_{question}", width="stretch"):
                st.session_state.chat_pending = question
                st.rerun()


def _render_message(message: dict) -> None:
    with st.chat_message(message.get("role", "assistant")):
        if message.get("role") == "assistant":
            components.render_answer(message.get("content", ""))
            _render_sources(message.get("sources") or [])
        else:
            st.markdown(message.get("content", ""))


def _render_sources(sources: List[dict]) -> None:
    """折叠展示本次命中的证据片段（含重排分数）。"""
    if not sources:
        return
    with st.expander(f"检索证据（{len(sources)} 段）", expanded=False):
        for index, item in enumerate(sources[:_MAX_SOURCE_ROWS], 1):
            score = item.get("score")
            score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "-"
            title = str(item.get("filename") or "论文")
            section = str(item.get("section") or "").strip()
            st.markdown(f"**[{index}]** {title}")
            st.caption(f"重排得分 {score_text} ｜ 章节 {section or '未标注'}")


def _handle_pending_question() -> None:
    """消费待处理问题：回显提问并生成回答。"""
    pending = str(st.session_state.get("chat_pending") or "").strip()
    if not pending:
        return
    st.session_state.chat_pending = ""
    st.session_state.pop("chat_box", None)

    with st.chat_message("user"):
        st.markdown(pending)
    st.session_state.chat_messages.append({"role": "user", "content": pending})
    _answer(pending)


def _answer(question: str) -> None:
    """调用真实 RAG 服务并渲染回答，最后写入会话存档。"""
    if not credentials_ready():
        with st.chat_message("assistant"):
            st.error("请配置 Key 后再提问。")
        return

    with st.chat_message("assistant"):
        with st.status("正在检索论文库…", expanded=True) as status:
            st.write("① 解析问题意图与必需证据槽位")
            st.write("② Chroma 向量召回 + BM25 词法召回 → RRF 融合")
            st.write("③ qwen3-rerank 二次重排，收敛逐篇证据")
            try:
                answer = answer_fulltext_paper_question(question, interactive=True)
            except MissingCredentialsError as exc:
                status.update(label="缺少凭证", state="error", expanded=True)
                st.error(str(exc))
                return
            except Exception as exc:  # noqa: BLE001 - 真实失败如实呈现，不降级兜底
                logger.exception("论文检索失败")
                status.update(label="检索失败", state="error", expanded=True)
                st.error(f"检索失败：{exc}")
                return
            st.write("④ 校验数字与证据 ID，渲染参考文献")
            status.update(
                label=f"完成：命中 {len(answer.sources)} 段证据",
                state="complete",
                expanded=False,
            )

        components.render_answer(answer.text)
        _render_sources(list(answer.sources))

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": answer.text,
            "sources": list(answer.sources),
        }
    )
    state.save_chat_state()


def _transcript() -> str:
    """把当前会话导出为 Markdown 文本。"""
    lines = ["# 对话记录", ""]
    for message in st.session_state.chat_messages:
        role = "我" if message.get("role") == "user" else "助手"
        lines.extend([f"## {role}", "", message.get("content", ""), ""])
    return "\n".join(lines)
