"""智能数据分析页。

页面流程：
- 多文件上传 → 大小/列结构校验 → 按行合并；
- 参数配置区（物种、处理、时间、标签说明、绘图偏好）；
- 两段式确认：先分类并弹窗预览，确认后再执行完整分析；
- 报告按图片标记分段渲染，图片用 ``st.image`` 内联，可单独下载图片与脚本。
"""

from __future__ import annotations

import io
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from agents.analysis_agent import CHART_TYPE_CN_MAP, AnalysisAgent
from services import export
from services.llm_gateway import MISSING_KEY_MESSAGE, credentials_ready
from ui import components, state, theme

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = ("name", "PAR", "DateTime")
_MAX_FILE_MB = 20

_PALETTES = {
    "默认（蓝灰）": "#4C78A8",
    "暖色（砖红）": "#C0504D",
    "冷色（青绿）": "#2E8B7A",
}


def render() -> None:
    theme.page_header(
        "智能数据分析",
        "上传测量数据，自动完成结构校验、统计图表与解读，每张图表都可单独导出。",
        "📊",
    )
    components.history_panel("analysis_history", "an_result", title="分析历史")

    left, right = st.columns([0.44, 0.56], gap="large")
    with left:
        _render_uploader()
        _render_context_form()
        _render_run_controls()
    with right:
        _render_result()

    if st.session_state.get("an_preview"):
        _preview_dialog()


# --- 数据导入 ---


def _render_uploader() -> None:
    st.subheader("数据导入")
    files = st.file_uploader(
        "上传数据文件（支持多文件合并）",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="an_uploader",
    )
    if not files:
        if st.session_state.get("an_frame") is not None:
            _reset_dataset()
        st.caption("支持 CSV / Excel，多个文件将按行合并，列结构必须一致。")
        return

    oversized = [f"{item.name}（{item.size / 1024 / 1024:.1f}MB）" for item in files
                 if item.size / 1024 / 1024 > _MAX_FILE_MB]
    if oversized:
        st.error(f"以下文件超过 {_MAX_FILE_MB}MB 限制：" + "、".join(oversized))
        st.stop()

    merged_key = "|".join(sorted(item.name for item in files))
    if st.session_state.get("an_merged_key") == merged_key and st.session_state.get("an_csv_path"):
        _render_dataset_badge()
        return
    _merge_files(files, merged_key)


def _read_upload(handle) -> pd.DataFrame:
    """按扩展名读取上传文件；CSV 依次尝试常见编码。"""
    suffix = Path(handle.name).suffix.lower()
    raw = handle.getvalue()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(raw))
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"无法识别文件编码：{last_error}")


def _merge_files(files, merged_key: str) -> None:
    """校验并合并多个文件，结果写入 session_state。"""
    frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    first_columns: list[str] | None = None

    for handle in files:
        try:
            frame = _read_upload(handle)
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            st.error(f"{handle.name} 读取失败：{exc}")
            st.stop()

        if first_columns is None:
            first_columns = list(frame.columns)
        elif set(frame.columns) != set(first_columns):
            missing = set(first_columns) - set(frame.columns)
            extra = set(frame.columns) - set(first_columns)
            details = []
            if missing:
                details.append("缺少列：" + "、".join(sorted(missing)))
            if extra:
                details.append("多余列：" + "、".join(sorted(extra)))
            st.error(f"{handle.name} 列结构与首个文件不一致，无法合并。" + "；".join(details))
            st.stop()

        missing_required = [c for c in _REQUIRED_COLUMNS if c not in frame.columns]
        if missing_required:
            st.error(
                f"{handle.name} 缺少必需列：{'、'.join(missing_required)}。"
                "请上传仪器导出的标准数据文件。"
            )
            st.stop()

        frames.append(frame[frame["name"].notna()].copy())
        summaries.append(
            {"文件名": handle.name, "行数": len(frame), "列数": len(frame.columns)}
        )

    merged = pd.concat(frames, ignore_index=True)
    target_dir = Path(tempfile.mkdtemp(prefix="plant_demo_"))
    csv_path = target_dir / "merged_measurements.csv"
    merged.to_csv(csv_path, index=False)

    st.session_state.an_frame = merged
    st.session_state.an_csv_path = str(csv_path)
    st.session_state.an_merged_key = merged_key
    st.session_state.an_filenames = [item.name for item in files]
    st.session_state.an_summaries = summaries
    st.session_state.an_preview = None

    _render_dataset_badge()
    with st.expander("合并详情", expanded=False):
        st.dataframe(pd.DataFrame(summaries), hide_index=True)


def _render_dataset_badge() -> None:
    frame = st.session_state.get("an_frame")
    if frame is None:
        return
    varieties = frame["name"].nunique() if "name" in frame.columns else 0
    st.success(
        f"已合并 {len(st.session_state.get('an_filenames') or [])} 个文件 → "
        f"{len(frame)} 行 × {len(frame.columns)} 列　|　{varieties} 个品种/处理"
    )


def _reset_dataset() -> None:
    st.session_state.an_frame = None
    st.session_state.an_csv_path = None
    st.session_state.an_merged_key = None
    st.session_state.an_filenames = []
    st.session_state.an_summaries = []
    st.session_state.an_preview = None


# --- 参数配置 ---


def _render_context_form() -> None:
    if st.session_state.get("an_frame") is None:
        return

    st.subheader("实验信息")
    st.caption("这些信息用于解释图表含义，系统不会从样品标签中自行猜测。")
    st.text_input(
        "研究对象/物种（必填）",
        key="an_species",
        placeholder="例如：水稻、番茄、本氏烟；未知请填「未知样品」",
    )
    st.text_area(
        "处理与分组含义（必填）",
        key="an_treatment",
        height=88,
        placeholder="例如：CK 为 25℃ 对照；Cold 为 4℃ 冷胁迫 7 天",
    )
    st.text_input(
        "处理或测量时间（可选）",
        key="an_timing",
        placeholder="例如：处理第 0、3、5 天；每天 9:00–11:00 测量",
    )
    st.text_area(
        "数据标签说明（可选）",
        key="an_label_note",
        height=70,
        placeholder="例如：A_1–A_6 表示品种 A 的 6 个独立单株",
    )

    st.subheader("绘图偏好")
    st.selectbox("配色方案", list(_PALETTES), key="an_palette")
    st.text_area(
        "画图细节补充（可选）",
        key="an_desc",
        height=84,
        placeholder="例如：标题改为光响应曲线；横轴只保留时分秒；柱子细一点",
    )


def _build_context() -> str:
    """把表单内容拼成送给模型的实验背景描述。"""
    parts = [
        f"研究对象：{st.session_state.get('an_species', '').strip()}",
        f"处理与分组：{st.session_state.get('an_treatment', '').strip()}",
    ]
    timing = str(st.session_state.get("an_timing") or "").strip()
    if timing:
        parts.append(f"处理或测量时间：{timing}")
    labels = str(st.session_state.get("an_label_note") or "").strip()
    if labels:
        parts.append(f"数据标签说明：{labels}")
    detail = str(st.session_state.get("an_desc") or "").strip()
    if detail:
        parts.append(f"绘图偏好：{detail}")
    return "\n".join(parts)


def _render_run_controls() -> None:
    has_data = st.session_state.get("an_frame") is not None
    ready = has_data and bool(str(st.session_state.get("an_species") or "").strip()) \
        and bool(str(st.session_state.get("an_treatment") or "").strip())

    if st.button("开始智能分析", type="primary", width="stretch", disabled=not ready):
        _classify_for_preview()

    if not has_data and not st.session_state.get("an_result"):
        st.caption("请先上传数据文件。")
    elif not ready and not st.session_state.get("an_result"):
        st.caption("请填写物种与处理分组含义后再开始分析。")


def _classify_for_preview() -> None:
    """第一段：只做分类，结果放进弹窗等用户确认。"""
    csv_path = st.session_state.get("an_csv_path")
    if not csv_path:
        return
    with st.spinner("正在识别数据结构…"):
        try:
            agent = AnalysisAgent()
            preview = agent.classify(csv_path)
        except Exception:  # noqa: BLE001 - 分类失败给出可读提示
            logger.exception("数据分类失败")
            st.error("数据分类失败，请检查文件格式后重试。")
            return

    if preview.get("chart_type") == "error":
        st.error(str(preview.get("error_message") or "无法解析该数据文件。"))
        return

    st.session_state.an_preview = preview
    st.session_state.an_context = _build_context()
    st.rerun()


@st.dialog("分析预览", width="large")
def _preview_dialog() -> None:
    """确认弹窗：展示分类结果与数据质量警告。"""
    preview = st.session_state.get("an_preview") or {}
    chart_types = [
        CHART_TYPE_CN_MAP.get(name, name) for name in preview.get("chart_types", [])
    ]

    st.markdown("**将生成的图表**")
    st.markdown("\n".join(f"- {name}" for name in chart_types) or "- 无可生成图表")

    metrics = st.columns(3)
    metrics[0].metric("品种/处理数", preview.get("variety_count", 0))
    metrics[1].metric("PAR 梯度", preview.get("par_gradients", 0))
    metrics[2].metric("重复结构", "已识别" if preview.get("has_replicates") else "未识别")

    varieties = preview.get("varieties") or []
    if varieties:
        st.caption("参与分组的样品：" + "、".join(map(str, varieties)))

    bar_par = preview.get("bar_par_filter")
    if bar_par is not None:
        st.caption(
            f"柱状图将固定在 PAR = {bar_par} 切片比较（取覆盖品种最多的光强，"
            "跨光强求总平均没有意义）。"
        )

    for warning in preview.get("data_warnings") or []:
        st.warning(f"⚠ {warning}")

    confirm_column, cancel_column = st.columns(2)
    if confirm_column.button("确认并开始分析", type="primary", width="stretch"):
        st.session_state.an_pending = True
        st.session_state.an_preview = None
        st.rerun(scope="app")
    if cancel_column.button("取消", width="stretch"):
        st.session_state.an_preview = None
        st.rerun(scope="app")


# --- 执行与结果 ---


def _render_result() -> None:
    if st.session_state.get("an_pending"):
        st.session_state.an_pending = False
        _execute_analysis()
        return

    result = st.session_state.get("an_result")
    if not result:
        theme.empty_state(
            "📊",
            "还没有分析结果",
            "上传数据、填写实验信息后点击「开始智能分析」，报告与图表会在这里展示。",
        )
        return

    st.subheader("分析报告")
    try:
        components.render_report(
            result.get("markdown_report", ""), result.get("chart_results", [])
        )
    except Exception:  # noqa: BLE001 - 渲染异常不白屏
        logger.exception("报告渲染失败")
        st.warning("报告渲染失败，可通过下方按钮导出查看。")

    _render_exports(result)


def _execute_analysis() -> None:
    """第二段：真正执行分析管线。"""
    st.subheader("分析报告")
    if not credentials_ready():
        st.error(MISSING_KEY_MESSAGE)
        return
    st.warning("⏳ 正在生成统计图表与解读，请勿切换页面。")
    progress = st.empty()
    palette_color = _PALETTES.get(st.session_state.get("an_palette"), "#4C78A8")
    with st.spinner("AI 正在分析数据并生成统计图表，预计 1-2 分钟…"):
        try:
            agent = AnalysisAgent()
            result = agent.analyze(
                csv_path=st.session_state.get("an_csv_path"),
                user_desc=st.session_state.get("an_context", ""),
                progress_callback=lambda message: progress.info(message),
                chart_config={"color": palette_color},
            )
        except Exception as exc:  # noqa: BLE001 - 真实失败如实呈现
            logger.exception("数据分析失败")
            progress.empty()
            st.error(f"分析失败：{exc}")
            return
    progress.empty()

    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.an_result = result

    filenames = st.session_state.get("an_filenames") or []
    state.save_history(
        {
            "title": (", ".join(filenames) or "数据分析")[:26],
            "timestamp": result["timestamp"],
            "payload": result,
        },
        "analysis_history",
    )
    st.rerun()


def _render_exports(result: dict) -> None:
    """导出区：整篇报告 + 每张图表的图片与复现脚本。"""
    chart_results = result.get("chart_results") or []
    st.divider()
    st.subheader("导出")
    timestamp = datetime.now().strftime("%Y%m%d%H%M")

    try:
        payload = export.markdown_to_docx(
            result.get("markdown_report", ""), title="数据分析报告"
        )
        st.download_button(
            label="下载报告 (.docx)",
            data=payload,
            file_name=f"{timestamp}_数据分析报告.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            width="stretch",
        )
    except Exception:  # noqa: BLE001
        logger.exception("导出 Word 失败")
        st.caption("Word 导出失败，可直接复制正文使用。")

    if not chart_results:
        return
    st.caption(f"共 {len(chart_results)} 张图表，可单独下载图片与复现脚本：")
    for index, chart in enumerate(chart_results, 1):
        type_cn = CHART_TYPE_CN_MAP.get(chart.get("chart_type", ""), chart.get("chart_type", ""))
        components.download_buttons(
            str(chart.get("image_path") or ""),
            str(chart.get("code_path") or ""),
            image_label=f"图表{index} - {type_cn} (.png)",
            code_label=f"代码{index} - {chart.get('param', '')} (.py)",
            index=index,
        )
