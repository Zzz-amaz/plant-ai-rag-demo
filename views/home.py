"""首页：功能导航卡片。"""

from __future__ import annotations

from typing import Dict

import streamlit as st

from ui import theme
from services.llm_gateway import credentials_ready, model_label

# app.py 在注册页面后回填，供卡片按钮跳转
_PAGES: Dict[str, object] = {}

_CARDS = (
    (
        "experiment",
        "🧪",
        "实验方案生成",
        "输入实验需求，生成含参数设置、操作步骤与质控要点的完整方案，"
        "并可导出 Word 文档。",
    ),
    (
        "analysis",
        "📊",
        "智能数据分析",
        "上传 CSV / Excel，自动完成结构校验、统计图表与解读，"
        "每张图都能单独下载图片与复现脚本。",
    ),
    (
        "assistant",
        "💬",
        "科研问答助手",
        "围绕荧光参数、测量流程与文献证据多轮追问，回答附带可折叠的参考来源。",
    ),
)


def register_pages(pages: Dict[str, object]) -> None:
    """由入口注入 st.Page 映射。"""
    _PAGES.update(pages)


def render() -> None:
    theme.page_header(
        "Plant 荧光实验智能助手",
        "实验方案生成 · 智能数据分析 · 科研问答，三条链路共用同一套本地统计与检索内核。",
    )

    columns = st.columns(3, gap="large")
    for column, (key, icon, title, body) in zip(columns, _CARDS):
        with column, st.container(border=True):
            theme.feature_card(icon, title, body)
            if st.button("进入 →", key=f"nav_{key}", type="primary", width="stretch"):
                target = _PAGES.get(key)
                if target is not None:
                    st.switch_page(target)

    st.divider()
    ready = credentials_ready()
    chip = theme.model_chip(model_label(), not ready)
    st.markdown(
        f"{chip}&nbsp;&nbsp;"
        '<span class="plant-meta">数据与图表仅在本机处理，'
        "文献检索经由 DashScope 与 DeepSeek 接口完成。</span>",
        unsafe_allow_html=True,
    )
    if not ready:
        st.error("请配置 Key：在项目根目录的 .env 中填写 LLM_API_KEY 与 EMBEDDING_API_KEY。")
