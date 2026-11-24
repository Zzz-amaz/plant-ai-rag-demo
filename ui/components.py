"""可复用的渲染组件。

三处共性逻辑集中在这里：
- 把报告 Markdown 按 ``![标题](路径)`` 切成段落，逐段用 ``st.image`` 内联图表；
- 把正文末尾的参考文献/参考来源折叠进 expander；
- 渲染单张图表的统计明细与可复现代码。
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

# 同时兼容半角/全角括号，且路径不允许跨行（防止吞掉后续正文）
_IMAGE_RE = re.compile(r"!\[([^\]]*)\][（(]([^\r\n）)]+)[）)]")
_MAX_REPORT_CHARS = 100_000

REFERENCE_HEADINGS = ("## 参考来源", "## 参考文献")
REFERENCE_MARKERS = ("参考文献：", "参考来源：", "参考来源")


def split_references(markdown: str) -> Tuple[str, str]:
    """把正文与参考文献块分开，返回 (正文, 参考文本)。"""
    for heading in REFERENCE_HEADINGS:
        marker = f"\n{heading}"
        if marker in markdown:
            body, references = markdown.split(marker, 1)
            return body.rstrip(), references.strip()
    for marker in REFERENCE_MARKERS:
        if marker in markdown:
            body, references = markdown.split(marker, 1)
            return body.rstrip(), references.strip()
    return markdown, ""


def render_references(text: str, label: str = "查看参考来源") -> None:
    """把参考文本渲染成默认折叠的面板。"""
    if not text.strip():
        return
    with st.expander(label, expanded=False):
        st.markdown(text.replace("\n", "\n\n"))


def render_report(markdown: str, chart_results: List[Dict[str, Any]]) -> None:
    """渲染分析报告：文本走 Markdown，图片走 st.image。

    单个图表或文本段渲染失败只影响该段，不阻断整篇报告。
    """
    body, references = split_references(markdown or "")
    if len(body) > _MAX_REPORT_CHARS:
        body = body[:_MAX_REPORT_CHARS] + "\n\n---\n*（报告过长，已截断显示）*"

    # 路径 → 图表结果，同时支持绝对路径与文件名匹配
    by_path: Dict[str, Dict[str, Any]] = {}
    for item in chart_results:
        image_path = str(item.get("image_path") or "")
        if not image_path:
            continue
        by_path[image_path] = item
        by_path[os.path.basename(image_path)] = item

    parts = _IMAGE_RE.split(body)
    for index, part in enumerate(parts):
        position = index % 3
        try:
            if position == 0:
                if part.strip():
                    st.markdown(part)
            elif position == 1:
                continue  # 图片的 alt 文本已由下方 caption 呈现
            else:
                _render_image_block(part.strip().rstrip("）】》"), by_path)
        except MemoryError:
            st.warning("该段内容过长，已跳过；可导出 DOCX 查看完整报告。")
        except Exception:  # noqa: BLE001 - 单段失败不影响其余内容
            st.warning(f"该段渲染失败，已跳过：{part[:40]}…")

    render_references(references if references else "")


def _render_image_block(path_text: str, by_path: Dict[str, Dict[str, Any]]) -> None:
    """渲染一张图表及其统计明细。"""
    chart = by_path.get(path_text) or by_path.get(os.path.basename(path_text))
    if chart and os.path.exists(str(chart.get("image_path") or "")):
        actual_path = str(chart["image_path"])
    elif os.path.exists(path_text):
        actual_path = path_text
        chart = None
    else:
        st.warning(f"图片未找到：{path_text}")
        return

    st.image(actual_path, width="stretch")
    if chart:
        render_chart_details(chart)


def render_chart_details(chart: Dict[str, Any]) -> None:
    """展示单张图表的统计结果、趋势事实与复现脚本。"""
    statistics = chart.get("statistics") or {}
    groups = statistics.get("groups")

    if isinstance(groups, list) and groups and isinstance(groups[0], dict):
        with st.expander("查看统计结果", expanded=False):
            table = pd.DataFrame(groups).rename(
                columns={"name": "分组", "mean": "均值", "std": "标准差", "n": "样本数"}
            )
            st.dataframe(table, hide_index=True)
            p_value = statistics.get("p_value")
            if p_value is not None:
                st.caption(f"Welch t 检验 p = {p_value}")
            elif len(groups) == 2:
                st.caption("样本量不足，未执行显著性检验。")
    elif statistics.get("par_levels"):
        with st.expander("查看曲线参数", expanded=False):
            st.caption(f"PAR 梯度：{statistics['par_levels']}")
            st.caption(f"分组：{statistics.get('varieties') or []}")

    trend_hint = chart.get("trend_hint")
    if trend_hint:
        st.caption(f"程序计算结果：{trend_hint}")

    _render_code(chart)


def _render_code(chart: Dict[str, Any]) -> None:
    """展示自动生成的复现脚本。"""
    code_path = str(chart.get("code_path") or "")
    if not code_path or not os.path.isfile(code_path):
        return
    with st.expander("查看 Python 代码", expanded=False):
        try:
            with open(code_path, "r", encoding="utf-8") as handle:
                st.code(handle.read(), language="python")
        except OSError:
            st.caption("复现脚本读取失败。")


def render_answer(text: str, references_label: str = "参考来源") -> None:
    """聊天回答渲染：正文在上，参考文献折叠。"""
    answer, references = split_references(text or "")
    st.markdown(answer or text)
    render_references(references, label=references_label)


def download_buttons(
    image_path: str,
    code_path: str,
    *,
    image_label: str,
    code_label: str,
    index: int,
) -> None:
    """并排渲染图片与脚本的下载按钮。"""
    column_image, column_code = st.columns(2)
    if image_path and os.path.isfile(image_path):
        with column_image, open(image_path, "rb") as handle:
            st.download_button(
                label=image_label,
                data=handle.read(),
                file_name=os.path.basename(image_path),
                mime="image/png",
                key=f"dl_img_{index}_{os.path.basename(image_path)}",
            )
    if code_path and os.path.isfile(code_path):
        with column_code, open(code_path, "r", encoding="utf-8") as handle:
            st.download_button(
                label=code_label,
                data=handle.read(),
                file_name=os.path.basename(code_path),
                mime="text/x-python",
                key=f"dl_code_{index}_{os.path.basename(code_path)}",
            )


def history_panel(
    kind: str,
    result_key: str,
    *,
    title: str = "历史记录",
    limit: int = 6,
) -> None:
    """侧边栏历史列表：点击载入，垃圾桶删除。

    记录结构约定为 ``{"id", "timestamp", "title", "payload"}``，
    载入时把 payload 写回 ``st.session_state[result_key]``。
    """
    from ui import state as state_store

    items = state_store.load_history_list(kind)
    with st.sidebar:
        st.markdown(f"**{title}**")
        if not items:
            st.caption("暂无记录")
            return
        st.caption(f"共 {len(items)} 条，保留最近 {limit} 条")
        for item in items[:limit]:
            label = str(item.get("title") or "未命名记录")
            row_label, row_delete = st.columns([0.82, 0.18])
            with row_label:
                if st.button(
                    label[:26],
                    key=f"hist_{kind}_{item.get('id')}",
                    width="stretch",
                    help=item.get("timestamp", ""),
                ):
                    st.session_state[result_key] = item.get("payload")
                    st.rerun()
            with row_delete:
                if st.button("🗑", key=f"hist_del_{kind}_{item.get('id')}"):
                    state_store.delete_history(str(item.get("id")), kind)
                    st.rerun()
        if len(items) > limit:
            st.caption(f"另有 {len(items) - limit} 条更早记录")
