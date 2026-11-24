"""实验方案生成页。

页面流程：
- 左栏输入需求、右栏展示结果的双栏布局；
- 生成过程用 ``st.spinner`` + ``st.write_stream`` 呈现流式效果；
- 正文与「参考来源」分离，来源折叠进 expander；
- 结果可导出 Word。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterator

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from services import export
from services.llm_gateway import (
    MISSING_KEY_MESSAGE,
    MissingCredentialsError,
    create_llm,
    credentials_ready,
)
from ui import components, state, theme

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """你是一名植物生理实验设计专家，服务于 Device-A 荧光成像平台。

请输出结构化的实验方案，章节顺序为：实验目的、材料与样品准备、仪器与参数设置、
操作步骤、数据记录与质控、注意事项，最后附加「## 参考来源」章节。

约束：
1. 参数档位不确定时必须写"以设备手册为准"，不得编造具体档位或按钮名称。
2. 不得推导仪器内部的 rETR/ETR 计算公式。
3. 全部使用中文，用 Markdown 标题组织章节，不要输出思考过程。
"""


def render() -> None:
    theme.page_header(
        "实验方案生成",
        "描述实验需求，自动生成可执行的测量方案、参数清单与数据记录规范。",
        "🧪",
    )
    components.history_panel("experiment_history", "exp_result", title="方案历史")

    left, right = st.columns([0.42, 0.58], gap="large")
    with left:
        _render_form()
    with right:
        _render_result()


def _render_form() -> None:
    st.subheader("实验需求输入")
    requirement = st.text_area(
        "请描述您的实验需求",
        height=210,
        key="exp_input",
        placeholder=(
            "例如：需要对三组番茄幼苗进行 4℃ 冷胁迫处理 0/3/5 天，"
            "测量暗适应 Fv/Fm 与光响应曲线，比较品种间耐冷性差异。"
        ),
    )

    if st.button("生成实验方案", type="primary", width="stretch"):
        if not requirement.strip():
            st.warning("请先输入实验需求。")
        elif not credentials_ready():
            st.error(MISSING_KEY_MESSAGE)
        else:
            st.session_state.exp_pending = True
            st.rerun()

    result = st.session_state.get("exp_result")
    if result:
        st.divider()
        st.subheader("导出")
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        try:
            payload = export.markdown_to_docx(result["markdown"], title="实验方案")
            st.download_button(
                label="下载 Word 文档 (.docx)",
                data=payload,
                file_name=f"{timestamp}_实验方案.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                width="stretch",
            )
        except Exception:  # noqa: BLE001 - 导出失败不影响阅读
            logger.exception("导出 Word 失败")
            st.caption("Word 导出失败，可直接复制正文使用。")


def _render_result() -> None:
    if st.session_state.get("exp_pending"):
        st.session_state.exp_pending = False
        requirement = str(st.session_state.get("exp_input") or "").strip()
        st.subheader("生成结果")
        try:
            markdown = _generate(requirement)
        except MissingCredentialsError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - 真实失败如实呈现
            logger.exception("实验方案生成失败")
            st.error(f"生成失败：{exc}")
            return
        if not markdown.strip():
            st.error("生成结果为空，请稍后重试。")
            return
        result = {
            "markdown": markdown,
            "requirement": requirement,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        st.session_state.exp_result = result
        state.save_history(
            {
                "title": requirement[:24] or "实验方案",
                "timestamp": result["timestamp"],
                "payload": result,
            },
            "experiment_history",
        )
        st.rerun()
        return

    result = st.session_state.get("exp_result")
    if not result:
        theme.empty_state(
            "🧪",
            "还没有生成方案",
            "在左侧描述实验需求后点击「生成实验方案」，结果会在这里展示。",
        )
        return

    st.subheader("生成结果")
    body, references = components.split_references(result["markdown"])
    st.markdown(body)
    components.render_references(references)


def _generate(requirement: str) -> str:
    """执行一次真实生成，返回 Markdown 文本。"""
    if not credentials_ready():
        raise MissingCredentialsError(MISSING_KEY_MESSAGE)
    llm = create_llm(streaming=True)
    with st.spinner("AI 正在生成实验方案，预计 20-40 秒…"):
        return str(st.write_stream(_stream_llm(llm, requirement)))


def _stream_llm(llm, requirement: str) -> Iterator[str]:
    """把模型输出转成纯文本片段流。"""
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"【用户需求】{requirement}\n\n请生成完整实验方案。"),
    ]
    for chunk in llm.stream(messages):
        content = getattr(chunk, "content", "")
        if content:
            yield str(content)
