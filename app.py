"""Plant 演示平台 —— 唯一入口。

替代原项目的 ``app_main.py`` + ``page_runner.py``：不再用 runpy 动态执行
页面模块，而是把每个页面注册成函数交给 ``st.navigation`` 统一路由；
全局状态全部走 ``st.session_state``，不依赖任何数据库或鉴权模块。

凭证统一来自项目根目录的 ``.env``：缺少 Key 时页面直接报错，不做任何假数据降级。

启动：``streamlit run app.py``
"""

from __future__ import annotations

import streamlit as st

from services.llm_gateway import (
    MISSING_KEY_MESSAGE,
    credentials_ready,
    embedding_label,
    missing_credentials,
    model_label,
)
from ui import state, theme
from views import analysis, assistant, experiment, home


def _render_model_panel() -> None:
    """侧边栏模型状态：只展示真实凭证状态，不提供会话级假 Key。"""
    if credentials_ready():
        st.success("✅ 已通过本地配置（.env）连接大模型")
        st.caption(f"对话 {model_label()} ｜ 向量 {embedding_label()}")
        st.caption("修改 .env 后重启应用即可生效。")
        return

    st.error(MISSING_KEY_MESSAGE)
    st.caption(f"缺少：{'、'.join(missing_credentials())}")


def _render_sidebar_brand() -> None:
    st.markdown("### 🍃 Plant 助手")
    st.caption("荧光实验智能辅助平台")


st.set_page_config(
    page_title="Plant 荧光实验智能助手",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

state.init_state()
theme.inject()

PAGES = {
    "home": st.Page(home.render, title="首页", icon="🏠", default=True, url_path="home"),
    "experiment": st.Page(
        experiment.render, title="实验方案生成", icon="🧪", url_path="experiment"
    ),
    "analysis": st.Page(
        analysis.render, title="智能数据分析", icon="📊", url_path="analysis"
    ),
    "assistant": st.Page(
        assistant.render, title="科研问答助手", icon="💬", url_path="assistant"
    ),
}
home.register_pages(PAGES)

with st.sidebar:
    _render_sidebar_brand()

navigation = st.navigation(list(PAGES.values()), position="sidebar")

with st.sidebar:
    _render_model_panel()
    st.divider()
    st.caption("当前设备：Device-A")
    st.caption("论文库：plant_papers_pilot_full1979_v1（75,779 段证据）")
    st.caption("文献与图表仅在本机索引，向量化经由 DashScope 接口完成。")

navigation.run()
