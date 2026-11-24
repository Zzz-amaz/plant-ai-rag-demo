"""页面主题与通用视觉元素。

保持原项目的浅色渐变页头风格，但所有颜色都显式声明，
不依赖 Streamlit 默认主题，切换深色主题时也不会出现低对比文本。
"""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
:root {
    --plant-green: #2e7d32;
    --plant-green-soft: #e8f5e9;
    --plant-blue-soft: #e3f2fd;
    --plant-border: #c8e6c9;
    --plant-muted: #5f6b7a;
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

.plant-hero {
    background: linear-gradient(135deg, #e8f5e9 0%, #e3f2fd 50%, #f1f8e9 100%);
    border: 1px solid var(--plant-border);
    border-radius: 16px;
    padding: 26px 30px;
    margin-bottom: 22px;
}

.plant-hero h1 {
    margin: 0;
    font-size: 1.65rem;
    font-weight: 700;
    color: #1b5e20;
}

.plant-hero p {
    margin: 10px 0 0 0;
    font-size: 0.95rem;
    color: #37474f;
    line-height: 1.7;
}

.plant-card {
    border: 1px solid var(--plant-border);
    border-radius: 14px;
    background: #ffffff;
    padding: 22px 22px 18px 22px;
    height: 100%;
}

.plant-card h3 {
    margin: 0 0 8px 0;
    font-size: 1.05rem;
    color: #1b5e20;
}

.plant-card p {
    margin: 0;
    color: #455a64;
    font-size: 0.9rem;
    line-height: 1.65;
}

.plant-empty {
    border: 1px dashed var(--plant-border);
    border-radius: 14px;
    padding: 46px 24px;
    text-align: center;
    color: var(--plant-muted);
    background: #fafdfa;
}

.plant-empty .icon {
    font-size: 2.2rem;
    display: block;
    margin-bottom: 10px;
}

.plant-empty .title {
    font-size: 1rem;
    font-weight: 600;
    color: #33691e;
    margin-bottom: 6px;
}

.plant-chip {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    border: 1px solid var(--plant-border);
    background: var(--plant-green-soft);
    color: #2e7d32;
}

.plant-chip.demo {
    background: #fff8e1;
    border-color: #ffe082;
    color: #8d6e63;
}

.plant-meta {
    color: var(--plant-muted);
    font-size: 0.8rem;
}
</style>
"""


def inject() -> None:
    """注入全局样式，每次脚本运行调用一次。"""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, icon: str = "🍃") -> None:
    """渲染页面顶部渐变标题条。"""
    st.markdown(
        f"""
        <div class="plant-hero">
            <h1>{icon} {title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, hint: str) -> None:
    """渲染空状态占位，避免右侧长时间空白。"""
    st.markdown(
        f"""
        <div class="plant-empty">
            <span class="icon">{icon}</span>
            <div class="title">{title}</div>
            <div>{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_chip(label: str, demo: bool) -> str:
    """返回模型状态徽标的 HTML 片段。"""
    css_class = "plant-chip demo" if demo else "plant-chip"
    return f'<span class="{css_class}">{label}</span>'


def feature_card(icon: str, title: str, body: str) -> None:
    """首页功能卡片内容区。

    边框由外层 ``st.container(border=True)`` 提供，这里只负责排版，
    避免出现双层描边。
    """
    st.markdown(
        f"""
        <div style="padding:4px 2px 10px 2px;">
            <div style="font-size:1.9rem;line-height:1;margin-bottom:10px;">{icon}</div>
            <div style="font-size:1.05rem;font-weight:600;color:#1b5e20;
                        margin-bottom:8px;">{title}</div>
            <div style="color:#455a64;font-size:0.9rem;line-height:1.7;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
