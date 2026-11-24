"""数据分析 Agent —— LangGraph StateGraph 管线。

管线：读取 CSV → pandas 规则分类 → 调度图表工具 → LLM 撰写报告 → 结束

设计要点：
- 无 ReAct 循环、无中间件、无限流：分类由确定性规则完成，结果可复现；
- LLM 只负责把「程序算出的统计事实」组织成科研报告，不参与数值计算；
- 检索证据通过构造函数注入的 ``evidence_provider`` 挂载，默认不启用。
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Callable, Optional, TypedDict, cast

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .chart_tools import (
    generate_anova_bars,
    generate_aq_curves,
    generate_kinetics_curves,
    generate_single_fvfm_bar,
    generate_single_light_curve,
    generate_ttest_bars,
    to_csv,
)
from .kinetics_data import has_kinetics_cycles, kinetics_cycle_gaps
from .prompts import LANGUAGE_INSTRUCTION, build_system_prompt
from .sample_names import classify_sample_names

logger = logging.getLogger(__name__)

DEVICE_NAME = os.getenv("DEMO_DEVICE_NAME", "Device-A")
DEFAULT_LLM_MODEL = os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"

# 各图表类型的参数集；实际执行时会过滤掉数据中不存在的列
CHART_PARAMS_MAP = {
    "aq_curves": '["rETR"]',
    "anova_bars": '["Y(II)","Y(NPQ)","Y(NO)"]',
    "ttest_bars": '["Y(II)","Y(NPQ)","Y(NO)"]',
    # 六个指标覆盖光化学、调节性耗散与松弛动力学，避免把代数互补项
    # （如 1-Y(II)）当成独立生物学证据重复计数
    "kinetics": '["Y(II)","NPQ","qL","Y(NPQ)","Y(NO)","Rfd"]',
    "single_fvfm": '["Fv/Fm"]',
    "single_curve": '["rETR"]',
}

KINETICS_PARAM_PRIORITY = ("Y(II)", "NPQ", "qL", "Y(NPQ)", "Y(NO)", "Rfd")

CHART_TYPE_CN_MAP = {
    "anova_bars": "多组均值比较柱状图",
    "ttest_bars": "两组均值比较柱状图",
    "aq_curves": "光响应曲线",
    "kinetics": "荧光动力学时间序列曲线",
    "single_fvfm": "暗适应最大光化学效率柱状图",
    "single_curve": "单样品光响应曲线",
}

# 图表类型 → 工具函数
TOOL_MAP = {
    "anova_bars": generate_anova_bars,
    "ttest_bars": generate_ttest_bars,
    "aq_curves": generate_aq_curves,
    "kinetics": generate_kinetics_curves,
    "single_fvfm": generate_single_fvfm_bar,
    "single_curve": generate_single_light_curve,
}

# 理论取值范围在 [0, 1] 的荧光参数，±0.02 容差用于吸收测量噪声
UNIT_RANGE_PARAMS = ("Y(II)", "Y(NPQ)", "Y(NO)", "Fv/Fm", "Fv_Fm", "qP", "qL", "qN")

_IMAGE_MARKER_RE = re.compile(r"\[{1,2}\s*CHART_IMAGE_(\d+)\s*\]{1,2}", re.IGNORECASE)
# 模型偶发复述的内部提示标签
_INTERNAL_LABEL_RE = re.compile(
    r"(?:【)?(?:程序计算的)?趋势(?:描述|事实)(?:[^\n：:】]*)(?:】)?\s*[：:]\s*"
)


def build_default_llm() -> ChatOpenAI:
    """构造默认模型客户端（延迟到首次调用，复用统一工厂）。"""
    from .llm_factory import create_llm

    return create_llm(streaming=False)


class AnalysisState(TypedDict):
    """StateGraph 共享状态，各节点只返回增量字段。

    注意：TypedDict 的注解会被 langgraph 通过 ``get_type_hints`` 真实求值，
    因此这里必须使用 ``Optional[...]`` 而非 ``X | None``，以兼容 Python 3.9。
    """

    csv_path: str
    user_desc: str
    progress_callback: Optional[Callable[[str], None]]
    chart_config: dict
    # 分类结果
    chart_type: str
    chart_types: list[str]
    bar_par_filter: Optional[float]
    kinetics_par_filter: Optional[float]
    data_warnings: list[str]
    error_message: str
    varieties: list[str]
    variety_count: int
    par_gradients: int
    has_replicates: bool
    # 工具结果
    chart_result: dict
    chart_results_list: list[dict]
    image_paths: list[str]
    statistics_list: list[dict]
    trend_hints: list[str]
    # 最终输出
    markdown_report: str


class AnalysisAgent:
    """数据分析管线 Agent。

    用 LangGraph StateGraph 替代 ReAct Agent：
    - classify_csv: pandas 读取 CSV，按固定规则确定性分类
    - call_tools: 直接调用对应统计图表工具
    - generate_report: LLM 基于统计事实撰写报告
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        *,
        device_name: str = DEVICE_NAME,
        evidence_provider: Optional[Callable[[str], str]] = None,
        offline_report: Optional[Callable[[list], str]] = None,
    ) -> None:
        """
        Args:
            llm: 模型客户端；None 表示首次生成报告时按环境变量惰性创建。
            device_name: 展示用设备名。
            evidence_provider: 检索焦点 → 证据文本的可选挂载点。
            offline_report: 模型不可用时的报告兜底生成器，
                入参为有效图表结果列表，返回 Markdown。为 None 时异常向上抛出。
        """
        self._llm = llm
        self.device_name = device_name
        self.evidence_provider = evidence_provider
        self.offline_report = offline_report
        self.system_prompt = build_system_prompt(device_name)
        self.graph = self._build_graph()

    @property
    def llm(self) -> ChatOpenAI:
        """惰性创建模型客户端，保证只做分类时无需 API Key。"""
        if self._llm is None:
            self._llm = build_default_llm()
        return self._llm

    # --- 图构建 ---

    def _build_graph(self):
        builder = StateGraph(AnalysisState)
        builder.add_node("classify_csv", self._classify_csv)
        builder.add_node("call_tools", self._call_tools)
        builder.add_node("generate_report", self._generate_report)
        builder.add_node("handle_error", self._handle_error)

        builder.set_entry_point("classify_csv")
        builder.add_conditional_edges(
            "classify_csv",
            self._route_after_classify,
            {"call_tools": "call_tools", "handle_error": "handle_error"},
        )
        builder.add_edge("call_tools", "generate_report")
        builder.add_edge("generate_report", END)
        builder.add_edge("handle_error", END)
        return builder.compile()

    # --- 节点：分类 ---

    def _classify_csv(self, state: AnalysisState) -> dict:
        """用 pandas 读取 CSV，代码级分类（结果确定，不依赖 LLM）。"""
        progress = state.get("progress_callback")
        if progress:
            progress("正在读取数据并识别可生成的图表…")

        csv_path = to_csv(state["csv_path"])
        try:
            frame = pd.read_csv(csv_path)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            return self._error_state(f"无法读取数据文件: {exc}", csv_path)

        missing = {"DateTime", "name", "PAR"}.difference(frame.columns)
        if missing:
            return self._error_state(
                "数据缺少必需列 "
                f"{', '.join(sorted(missing))}，请上传 {self.device_name} 导出的标准数据文件。",
                csv_path,
            )

        # 剔除 Overall 汇总行与空名称行
        clean = frame[~frame["name"].astype(str).str.startswith("Overall")].copy()
        clean = clean[clean["name"].notna()].copy()
        single_sample_mode = False
        if clean.empty:
            # Overall 可能是唯一有效数据（单株整体测量），回退到全量
            clean = frame[frame["name"].notna()].copy()
            if clean.empty:
                return self._error_state("该数据不含可分析的有效测量行。", csv_path)
            single_sample_mode = True
            logger.info("Overall 是唯一数据源，进入单样品模式")

        clean, varieties, has_replicates = classify_sample_names(clean)
        variety_count = len(varieties)
        if variety_count < 2:
            single_sample_mode = True
            logger.info("品种数=1，进入单样品模式（不做组间统计检验）")

        data_warnings = self._collect_warnings(clean)
        par_values = self._par_levels(clean)
        chart_types = self._detect_chart_types(
            clean,
            par_values,
            single_sample_mode,
            variety_count,
            has_replicates,
        )
        if chart_types is None:
            return self._error_state(
                "单样品数据中未检测到 Fv/Fm 或光响应曲线数据，无法生成图表。",
                csv_path,
            )

        kinetics_added, kinetics_par_filter = self._detect_kinetics(
            clean, single_sample_mode, variety_count
        )
        if kinetics_added and "kinetics" not in chart_types:
            chart_types.append("kinetics")

        bar_par_filter = self._bar_par_filter(clean, chart_types, par_values)

        logger.info(
            "分类完成 chart_types=%s varieties=%d PAR梯度=%d 重复结构=%s "
            "柱状图PAR=%s 动力学PAR=%s",
            chart_types,
            variety_count,
            len(par_values),
            has_replicates,
            bar_par_filter,
            kinetics_par_filter,
        )
        return {
            "chart_types": chart_types,
            "chart_type": chart_types[0],
            "bar_par_filter": bar_par_filter,
            "kinetics_par_filter": kinetics_par_filter,
            "data_warnings": data_warnings,
            "varieties": varieties,
            "variety_count": variety_count,
            "par_gradients": len(par_values),
            "has_replicates": has_replicates,
            "error_message": "",
            "csv_path": csv_path,
        }

    @staticmethod
    def _error_state(message: str, csv_path: str) -> dict:
        """统一的错误状态，交由 handle_error 节点输出。"""
        return {
            "chart_type": "error",
            "error_message": message,
            "data_warnings": [],
            "varieties": [],
            "variety_count": 0,
            "par_gradients": 0,
            "has_replicates": False,
            "csv_path": csv_path,
        }

    def _collect_warnings(self, frame: pd.DataFrame) -> list[str]:
        """数据质量检查：非阻断，仅生成警告供报告引用。"""
        warnings: list[str] = []

        if "PAR" in frame.columns:
            par_numeric = pd.to_numeric(frame["PAR"], errors="coerce")
            negative = int((par_numeric < 0).sum())
            if negative:
                warnings.append(f"检测到 {negative} 行 PAR 为负值，已自动忽略")

        # 荧光参数理论上落在 [0,1]，超界通常意味着数据导出或单位异常
        for column in UNIT_RANGE_PARAMS:
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                continue
            out_of_range = int(((values < -0.02) | (values > 1.02)).sum())
            if out_of_range:
                warnings.append(f"{column} 有 {out_of_range} 个值明显超出 0-1 范围，请检查数据")
            break  # 只报第一个异常指标，避免刷屏

        duplicated = int(frame.duplicated().sum())
        if duplicated:
            warnings.append(f"检测到 {duplicated} 行完全重复的数据，统计结果可能受影响")

        # Cycle 缺项会影响动力学解释，必须显式暴露而不是静默当成连续数据
        if has_kinetics_cycles(frame):
            for phase, missing in kinetics_cycle_gaps(frame).items():
                labels = ", ".join(f"{phase.title()}_{index}" for index in missing)
                warnings.append(
                    f"动力学 Cycle 序列存在缺项：{labels}；"
                    "请确认是仪器命名规则、导出遗漏还是未完成测量"
                )
        return warnings

    @staticmethod
    def _par_levels(frame: pd.DataFrame) -> list[float]:
        """提取有效 PAR 梯度（过滤 0 与缺失值）。"""
        if "PAR" not in frame.columns:
            return []
        par_series = pd.to_numeric(frame["PAR"], errors="coerce")
        values = par_series[par_series > 1].dropna().unique().astype(float)
        return sorted(values.tolist())

    @staticmethod
    def _has_values(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
        """判断给定列中是否有任一有效测量值。"""
        for column in columns:
            if column in frame.columns and frame[column].notna().any():
                return True
        return False

    def _detect_chart_types(
        self,
        frame: pd.DataFrame,
        par_values: list[float],
        single_sample_mode: bool,
        variety_count: int,
        has_replicates: bool,
    ) -> Optional[list[str]]:
        """检测所有满足条件的图表类型（非排他，允许多类型共存）。"""
        chart_types: list[str] = []

        if single_sample_mode:
            # 单株也可能含完整的光诱导—暗恢复协议：它不能做群体推断，
            # 但也不能因为样品标签只有一个就被丢弃
            if has_kinetics_cycles(frame):
                chart_types.append("kinetics")
                logger.info("单样品数据中根据 Cycle 列检测到 Light/Dark 序列")
            if self._has_values(frame, ("Fv/Fm", "Fv_Fm")):
                chart_types.append("single_fvfm")
            if len(par_values) >= 3 and self._has_values(frame, ("rETR", "Y(II)")):
                chart_types.append("single_curve")
            if not chart_types:
                return None
            return chart_types

        if len(par_values) >= 3 and variety_count >= 2:
            chart_types.append("aq_curves")
        if has_replicates:
            if variety_count >= 3:
                chart_types.append("anova_bars")
            elif variety_count == 2:
                chart_types.append("ttest_bars")

        # 无任何可判定类型时退回动力学，后续由工具层报错兜底
        if not chart_types:
            chart_types.append("kinetics")
        return chart_types

    def _detect_kinetics(
        self,
        frame: pd.DataFrame,
        single_sample_mode: bool,
        variety_count: int,
    ) -> tuple[bool, Optional[float]]:
        """判定是否构成动力学序列。

        Returns:
            (是否追加动力学图表, 动力学应使用的 PAR 切片或 None)

            Cycle 标注存在时按 Cycle 划分阶段，不再按单一 PAR 切片；
            无 Cycle 标注时按「同一 PAR 下时间点足够多」判定，并选出
            时间点最密集的 PAR。
        """
        if single_sample_mode or variety_count < 2:
            return False, None
        if has_kinetics_cycles(frame):
            logger.info("根据 Cycle 列检测到 Light_*/Dark_* 动力学序列")
            return True, None
        if "DateTime" not in frame.columns:
            return False, None

        # 只统计有实际测量值的行
        metric_columns = [
            column
            for column in frame.columns
            if column not in {"DateTime", "name", "PAR", "Cycle", "Comment"}
        ]
        if metric_columns:
            has_value = frame[metric_columns].notna().any(axis=1)
        else:
            has_value = pd.Series(True, index=frame.index)
        measured = frame[has_value].copy()
        if measured.empty:
            return False, None

        time_counts = measured.groupby(["base_name", "PAR"])["DateTime"].nunique()
        max_time_points = int(time_counts.max())
        if max_time_points < 5:
            return False, None
        par_time_totals = time_counts.groupby("PAR").sum()
        return True, float(par_time_totals.idxmax())

    @staticmethod
    def _bar_par_filter(
        frame: pd.DataFrame,
        chart_types: list[str],
        par_values: list[float],
    ) -> Optional[float]:
        """柱状图存在时，选覆盖品种最多的 PAR 值；同时校正图型。"""
        has_bar = any(
            chart_type in {"anova_bars", "ttest_bars"} for chart_type in chart_types
        )
        if not has_bar or not par_values:
            return None

        best_par: Optional[float] = None
        best_count = 0
        for par in par_values:
            subset = frame[pd.to_numeric(frame["PAR"], errors="coerce") == par]
            count = int(subset["base_name"].nunique())
            # 覆盖品种更多者优先；同覆盖数取更高光强（胁迫对比更有意义）
            if count > best_count or (
                count == best_count and (best_par is None or par > best_par)
            ):
                best_count = count
                best_par = par

        if best_count == 2 and "anova_bars" in chart_types:
            chart_types.remove("anova_bars")
            if "ttest_bars" not in chart_types:
                chart_types.append("ttest_bars")
        elif best_count < 2:
            chart_types[:] = [
                chart_type
                for chart_type in chart_types
                if chart_type not in {"anova_bars", "ttest_bars"}
            ]
        return best_par

    # --- 节点：工具调用 ---

    def _call_tools(self, state: AnalysisState) -> dict:
        """把图表类型×参数展开为独立任务，逐个调用工具。"""
        chart_types = state.get("chart_types") or [state["chart_type"]]
        csv_path = state["csv_path"]
        bar_par_filter = state.get("bar_par_filter")
        kinetics_par_filter = state.get("kinetics_par_filter")

        planned = self._plan_chart_types(chart_types, csv_path, bar_par_filter)
        if not planned:
            return {
                "chart_result": {},
                "chart_results_list": [],
                "image_paths": [],
                "statistics_list": [],
                "trend_hints": [],
                "markdown_report": "**图表生成失败**：无可执行的图表任务。",
            }

        tasks = self._expand_tasks(planned, csv_path)
        progress = state.get("progress_callback")
        if progress:
            progress(f"正在生成 {len(tasks)} 张图表并计算统计结果…")

        chart_results: list[dict] = []
        for index, task in enumerate(tasks, 1):
            if progress:
                progress(f"正在生成图表和统计结果（{index}/{len(tasks)}）…")

            chart_type = task["chart_type"]
            param = task["param"]
            tool_fn = TOOL_MAP.get(chart_type)
            if tool_fn is None:
                logger.error("未知图表类型: %s", chart_type)
                continue

            kwargs: dict = {
                "csv_path": csv_path,
                "params_json": json.dumps([param]),
                "config_json": json.dumps(
                    state.get("chart_config") or {}, ensure_ascii=False
                ),
            }
            # 柱状图按 PAR 切片；动力学按最密集 PAR 切片（PAR 可为 0，
            # 因此不能用真值判断）
            if bar_par_filter is not None and chart_type in {"anova_bars", "ttest_bars"}:
                kwargs["par_filter"] = bar_par_filter
            if chart_type == "kinetics" and kinetics_par_filter is not None:
                kwargs["par_filter"] = kinetics_par_filter

            try:
                parsed = json.loads(tool_fn(**kwargs))
            except (json.JSONDecodeError, TypeError) as exc:
                logger.error("%s/%s 工具返回解析失败: %s", chart_type, param, exc)
                chart_results.append(
                    {"error": f"工具返回解析失败: {exc}",
                     "chart_type": chart_type, "param": param}
                )
                continue

            if "error" in parsed:
                logger.warning("%s/%s 工具返回错误: %s", chart_type, param, parsed["error"])
            parsed.setdefault("chart_type", chart_type)
            parsed.setdefault("param", param)
            chart_results.append(parsed)

        valid = [item for item in chart_results if item.get("image_path")]
        return {
            "chart_result": valid[0] if valid else {},
            "chart_results_list": chart_results,
            "image_paths": [item.get("image_path", "") for item in chart_results
                            if item.get("image_path")],
            "statistics_list": [item.get("statistics", {}) for item in chart_results
                                if item.get("image_path")],
            "trend_hints": [item.get("trend_hint", "") for item in chart_results
                            if item.get("image_path")],
        }

    @staticmethod
    def _plan_chart_types(
        chart_types: list[str],
        csv_path: str,
        bar_par_filter: Optional[float],
    ) -> list[str]:
        """预检降级：PAR 切片后品种不足时，anova 降为 t 检验或直接跳过。"""
        planned: list[str] = []
        for chart_type in chart_types:
            actual = chart_type
            if bar_par_filter is not None and chart_type in {"anova_bars", "ttest_bars"}:
                frame = pd.read_csv(csv_path)
                if {"PAR", "name"}.issubset(frame.columns):
                    subset = frame[pd.to_numeric(frame["PAR"], errors="coerce") == bar_par_filter]
                    subset, varieties, _ = classify_sample_names(subset)
                    count = len(varieties)
                    if chart_type == "anova_bars" and count == 2:
                        logger.info("降级 anova_bars → ttest_bars (PAR=%s)", bar_par_filter)
                        actual = "ttest_bars"
                    elif count < 2:
                        logger.info("跳过 %s (PAR=%s 品种数不足)", chart_type, bar_par_filter)
                        continue
            planned.append(actual)
        return planned

    @staticmethod
    def _expand_tasks(planned: list[str], csv_path: str) -> list[dict]:
        """展开为「图表类型 × 参数」的独立任务列表。"""
        tasks: list[dict] = []
        for chart_type in planned:
            params = json.loads(CHART_PARAMS_MAP.get(chart_type, "[]"))
            if chart_type == "kinetics":
                # 只为本次导出实际存在的参数出图，且每个生理维度只留一个代表
                try:
                    available = set(pd.read_csv(csv_path, nrows=2).columns)
                except (OSError, pd.errors.ParserError):
                    available = set()
                filtered = [name for name in KINETICS_PARAM_PRIORITY if name in available]
                params = filtered or params[:1] or ["Y(II)"]
            for param in params:
                tasks.append({"chart_type": chart_type, "param": param})
        return tasks

    # --- 节点：报告生成 ---

    def _generate_report(self, state: AnalysisState) -> dict:
        """LLM 基于统计事实撰写最终报告。"""
        results = state.get("chart_results_list") or [state.get("chart_result") or {}]
        valid = [item for item in results if item.get("image_path")]
        if not valid:
            errors = [str(item["error"]) for item in results if item.get("error")]
            detail = "；".join(errors) if errors else "无可用的图表结果"
            return {
                "markdown_report": f"**图表生成失败**：{detail}\n\n"
                                   "请检查数据格式或重新上传。"
            }

        csv_path = state["csv_path"]
        focus = self._build_focus(valid)
        facts = _build_dataset_facts(csv_path)
        warnings = state.get("data_warnings") or []
        quality_note = "\n".join(f"- {item}" for item in warnings)

        evidence_context = ""
        if self.evidence_provider is not None:
            try:
                evidence_context = self.evidence_provider(focus) or ""
            except Exception:  # noqa: BLE001 - 检索失败不应阻断报告
                logger.exception("证据检索失败，报告将仅基于本地数据")

        progress = state.get("progress_callback")
        if progress:
            progress("统计事实已就绪，正在撰写科研分析报告…")

        user_message = self._build_report_prompt(
            valid, focus, facts, quality_note, evidence_context,
            state.get("user_desc", ""),
        )
        messages = [
            SystemMessage(content=f"{self.system_prompt}\n\n{LANGUAGE_INSTRUCTION}"),
            HumanMessage(content=user_message),
        ]
        try:
            response = self.llm.invoke(messages)
            report = getattr(response, "content", str(response))
        except Exception:  # noqa: BLE001 - 模型不可用时交由兜底生成器
            if self.offline_report is None:
                raise
            logger.warning("模型调用失败，改用离线模板生成报告", exc_info=True)
            report = self.offline_report(valid)

        report = _sanitize_report_output(str(report))
        report = _normalize_image_markers(report, valid)
        return {"markdown_report": report}

    @staticmethod
    def _build_focus(results: list[dict]) -> str:
        """把图表类型与参数整理成检索焦点文本。"""
        parts = []
        for item in results:
            type_cn = CHART_TYPE_CN_MAP.get(item.get("chart_type", ""),
                                            item.get("chart_type", ""))
            parts.append(f"{type_cn}（{item.get('param', '')}）")
        return "；".join(dict.fromkeys(parts))

    @staticmethod
    def _build_report_prompt(
        results: list[dict],
        focus: str,
        facts: dict,
        quality_note: str,
        evidence_context: str,
        user_desc: str,
    ) -> str:
        """组装报告生成提示词（约 18 条核心约束）。"""
        blocks = []
        for index, item in enumerate(results, 1):
            type_cn = CHART_TYPE_CN_MAP.get(item.get("chart_type", ""),
                                            item.get("chart_type", ""))
            blocks.append(
                f"""【图表 {index}：{type_cn}】
图表标题：{item.get('title', '')}
[[CHART_IMAGE_{index}]]

统计数据：
{json.dumps(item.get('statistics', {}), ensure_ascii=False, indent=2)}

程序计算的趋势事实：
{item.get('trend_hint') or '（工具未返回趋势数据，请如实说明缺失，禁止编造走向）'}"""
            )
        charts_section = "\n\n---\n\n".join(blocks)

        return f"""【任务】基于已生成的 {len(results)} 张统计图表，撰写完整的数据分析报告。

【实验补充说明】{user_desc or '无'}
【检索焦点】{focus}
【本地论文证据】{evidence_context or '（本次未挂载证据检索，报告只能基于当前数据）'}
【程序计算的数据概览】
{json.dumps(facts, ensure_ascii=False, indent=2)}
【数据质量与协议提示】
{quality_note or '（未发现自动规则可确认的异常，这不等于测量条件已通过人工核查）'}

{charts_section}

【写作约束】
1. 直接以 "## 1. 数据概览" 开头，不要输出"好的""让我来分析"等过程性语句。
2. 每个图表单独成节，按编号顺序，标题形如 "## 2. 图表一：<主题>"。
3. 在每节相应位置输出 [[CHART_IMAGE_N]] 占位符（N 为图表编号），不要写文件路径。
4. 报告中的每个数值都必须能在上方「统计数据」或「趋势事实」中找到原文，禁止自行计算新数值。
5. 每节采用「观察到的现象 → 解释说明」两段式，不要停留于参数升降的逐列复述。
6. 单样品数据只能称为"描述性结果"，不得声称存在生物学重复或统计检验差异。
7. 不得推导仪器内部的 rETR/ETR 计算公式，只称"仪器导出的 rETR 值"。
8. 光响应曲线必须区分 PAR（横轴）与 rETR（纵轴），禁止把 PAR 数值写成 rETR。
9. 动力学曲线按 Cycle 标记划分光/暗阶段；Cycle 缺项必须在数据质量处如实指出。
10. 参数解释要限定强度：Y(II) 偏低不等于胁迫或损伤，Y(NO) 偏高不等于已发生光抑制。
11. 引用只能来自上方「本地论文证据」中给出的编号；没有证据就只写数据观察。
12. 文献只能解释参数含义或提出待验证机制，不得替代、修正或扩展当前数据。
13. 提出机制假设时必须使用"可能""待验证"等限定语，并说明仍需排除的因素。
14. 禁止输出"趋势事实""逐字照抄""程序计算"等内部标签或提示词原文。
15. 不要输出参考文献列表，系统会在文末统一追加实际使用的来源。
16. 在所有图表分析之后，单独输出 "## 数据解读边界与后续建议"，给出 2-4 条可执行建议。
17. 全文控制在约 3500-4800 个汉字，不要为压缩篇幅而遗漏关键趋势或统计结果。
18. 科学符号、参数名与单位保持原样（如 Y(II)、Fv/Fm、μmol·m⁻²·s⁻¹）。
"""

    def _handle_error(self, state: AnalysisState) -> dict:
        """数据不可用时的错误输出（不调用工具、不调用模型）。"""
        return {"markdown_report": state["error_message"]}

    # --- 路由 ---

    @staticmethod
    def _route_after_classify(state: AnalysisState) -> str:
        return "handle_error" if state["chart_type"] == "error" else "call_tools"

    # --- 对外接口 ---

    def classify(self, csv_path: str) -> dict:
        """只执行分类节点。

        便于在无 API Key、无网络的环境下调试分类规则与阈值。
        """
        state = cast(
            AnalysisState,
            {"csv_path": csv_path, "progress_callback": None},
        )
        return self._classify_csv(state)

    def analyze(
        self,
        csv_path: str,
        user_desc: str = "",
        progress_callback: Optional[Callable[[str], None]] = None,
        chart_config: Optional[dict] = None,
    ) -> dict:
        """执行完整分析管线。

        Args:
            csv_path: CSV 或 Excel 数据文件路径。
            user_desc: 用户的实验补充说明。
            progress_callback: 进度回调，接收阶段描述文本。
            chart_config: 传给图表工具的样式配置（如 ``{"color": "#hex"}``）。

        Returns:
            包含 ``markdown_report`` 与 ``chart_results`` 的字典。
        """
        initial_state: AnalysisState = {
            "csv_path": csv_path,
            "user_desc": user_desc or "",
            "chart_config": dict(chart_config or {}),
            "progress_callback": progress_callback,
            "chart_type": "",
            "chart_types": [],
            "bar_par_filter": None,
            "kinetics_par_filter": None,
            "data_warnings": [],
            "error_message": "",
            "varieties": [],
            "variety_count": 0,
            "par_gradients": 0,
            "has_replicates": False,
            "chart_result": {},
            "chart_results_list": [],
            "image_paths": [],
            "statistics_list": [],
            "trend_hints": [],
            "markdown_report": "",
        }
        # 递归上限防止条件路由意外成环
        result = self.graph.invoke(initial_state, {"recursion_limit": 10})

        chart_results = [
            item for item in result.get("chart_results_list") or []
            if item.get("image_path")
        ]
        return {
            "markdown_report": result["markdown_report"],
            "chart_results": chart_results,
        }


def _sanitize_report_output(report: str) -> str:
    """移除模型偶发复述的内部提示标签，不改动正文。"""
    report = _INTERNAL_LABEL_RE.sub("", report)
    return report.replace("（逐字照抄）", "").replace("(逐字照抄)", "")


def _normalize_image_markers(report: str, results: list[dict]) -> str:
    """把占位符收敛为唯一的 Markdown 图片标记，每张图只保留一处。"""
    for index, item in enumerate(results, 1):
        image_path = str(item.get("image_path") or "")
        if not image_path:
            continue
        marker = f"![图表{index}]({image_path})"
        # 模型可能已按路径或标准 alt 文本写出图片，此时无需补写
        if marker in report or image_path in report:
            continue
        pattern = re.compile(
            rf"\[{{1,2}}\s*CHART_IMAGE_{index}\s*\]{{1,2}}", re.IGNORECASE
        )
        if pattern.search(report):
            # 用 lambda 替换，避免 Windows 路径中的反斜杠被当作转义序列
            report = pattern.sub(lambda _match: marker, report, count=1)
            report = pattern.sub("", report)
        else:
            # 模型漏写占位符时，插到「数据解读边界」之前，
            # 避免图片落在参考文献或建议章节之后
            anchor = report.find("## 数据解读边界")
            if anchor < 0:
                report = report.rstrip() + f"\n\n{marker}"
            else:
                report = report[:anchor] + f"{marker}\n\n" + report[anchor:]
    return report


def _build_dataset_facts(csv_path: str) -> dict:
    """从原始数据确定性生成概览事实，避免 LLM 猜测行数与设计。"""
    frame = pd.read_csv(csv_path)
    if "name" in frame.columns:
        named = frame[frame["name"].notna()].copy()
        overall = named[named["name"].astype(str).str.startswith("Overall")]
        clean = named[~named["name"].astype(str).str.startswith("Overall")].copy()
        name_counts = clean["name"].astype(str).value_counts().sort_index().to_dict()
        _, varieties, has_replicates = classify_sample_names(clean)
    else:
        named = frame.copy()
        overall = frame.iloc[0:0].copy()
        clean = frame.copy()
        name_counts = {}
        varieties = []
        has_replicates = False

    par_levels: list[float] = []
    if "PAR" in clean.columns:
        par_numeric = pd.to_numeric(clean["PAR"], errors="coerce").dropna()
        par_levels = sorted(par_numeric.unique().astype(float).tolist())
    effective_par = [level for level in par_levels if level > 1]

    measurement_times: list[str] = []
    if "DateTime" in named.columns:
        raw_times = named["DateTime"].dropna().astype(str).str.strip()
        # 仪器导出格式为 YYYY-MM-DD-HH-MM-SS：显式指定格式，避免 pandas
        # 把末尾的时分秒误判为时区偏移
        parsed = pd.to_datetime(raw_times, format="%Y-%m-%d-%H-%M-%S", errors="coerce")
        unresolved = parsed.isna()
        if unresolved.any():
            parsed.loc[unresolved] = pd.to_datetime(
                raw_times[unresolved], format="mixed", errors="coerce", utc=True
            ).dt.tz_localize(None)
        measurement_times = [
            value.strftime("%Y-%m-%d %H:%M:%S")
            for value in sorted(parsed.dropna().drop_duplicates())
        ]

    return {
        "有效数据行数": len(named),
        "参与分析的数据行数（不含 Overall）": len(clean),
        "Overall 汇总行数": len(overall),
        "原始名称及行数": name_counts,
        "识别到的品种或处理": varieties,
        "识别到重复结构": has_replicates,
        "单样品单区域模式": len(name_counts) == 1 and len(overall) > 0,
        "测量时间点数量": len(measurement_times),
        "测量时间点": measurement_times,
        "有效 PAR 梯度数": len(effective_par),
        "有效 PAR 范围": (
            [round(min(effective_par), 2), round(max(effective_par), 2)]
            if effective_par else []
        ),
    }


__all__ = [
    "AnalysisAgent",
    "AnalysisState",
    "CHART_PARAMS_MAP",
    "CHART_TYPE_CN_MAP",
    "KINETICS_PARAM_PRIORITY",
    "TOOL_MAP",
    "build_default_llm",
]
