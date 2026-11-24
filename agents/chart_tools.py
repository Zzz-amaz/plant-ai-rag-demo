"""图表工具层：确定性统计 + 轻量渲染。

统计全部由 pandas 完成，绘图用 matplotlib，两者都不含随机性或模型调用，
同一份数据必然得到同一张图。工具层只负责「统计事实 → 图表落盘 → JSON 结果」，
可以整体替换为自有绘图后端而不影响 Agent 的状态机。

所有工具函数签名统一为::

    tool(csv_path, params_json, config_json, par_filter=None) -> str

返回 JSON 字符串，字段为 ``chart_type`` / ``param`` / ``image_path`` /
``statistics`` / ``trend_hint`` / ``title``；失败时返回含 ``error`` 的对象。
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (必须在设置后端之后导入)
import pandas as pd  # noqa: E402

from .kinetics_data import has_kinetics_cycles  # noqa: E402
from .sample_names import classify_sample_names  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHART_OUTPUT_DIR = Path(
    os.getenv("DEMO_CHART_OUTPUT_DIR") or _PROJECT_ROOT / "outputs" / "Demo_Charts"
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120


def to_csv(source: str | Path) -> str:
    """把 Excel 输入转成临时 CSV；CSV 直接返回原路径。"""
    path = Path(source)
    if path.suffix.lower() not in {".xlsx", ".xls"}:
        return str(path)
    frame = pd.read_excel(path)
    target = Path(tempfile.mkdtemp(prefix="demo_csv_")) / f"{path.stem}.csv"
    frame.to_csv(target, index=False)
    return str(target)


def _output_path(stem: str) -> Path:
    """生成唯一输出路径，目录不存在时自动创建。"""
    CHART_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    return CHART_OUTPUT_DIR / f"{safe_stem}_{stamp}.png"


def _read_frame(csv_path: str, par_filter: float | None = None) -> pd.DataFrame:
    """读取数据并按 PAR 切片（PAR 可为 0，不能用真值判断）。"""
    frame = pd.read_csv(to_csv(csv_path))
    if par_filter is not None and "PAR" in frame.columns:
        par_numeric = pd.to_numeric(frame["PAR"], errors="coerce")
        frame = frame[par_numeric == float(par_filter)]
    if "name" not in frame.columns:
        raise KeyError("数据缺少 name 列，无法分组")
    frame, _varieties, _has_rep = classify_sample_names(frame)
    return frame


_REPRO_HEADER = '''"""可复现脚本：{title}

由分析管线随图表一同生成，脱离本项目也能独立运行。
用法：python {filename}
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path({csv_literal})
OUTPUT_PATH = Path({output_literal})
PARAM = {param_literal}
PAR_FILTER = {par_literal}
TITLE = {title_literal}

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def load_frame() -> pd.DataFrame:
    """读取数据并按光强切片，保留 name 列用于分组。"""
    frame = pd.read_csv(CSV_PATH)
    if PAR_FILTER is not None and "PAR" in frame.columns:
        par_values = pd.to_numeric(frame["PAR"], errors="coerce")
        frame = frame[par_values == float(PAR_FILTER)]
    return frame
'''

_REPRO_BAR_BODY = '''

def main() -> None:
    frame = load_frame()
    tidy = pd.DataFrame(
        {"group": frame["name"], "value": pd.to_numeric(frame[PARAM], errors="coerce")}
    ).dropna()
    summary = tidy.groupby("group")["value"].agg(["mean", "std", "count"]).sort_index()

    figure, axes = plt.subplots(figsize=(max(4.0, 1.1 * len(summary) + 2), 3.6))
    axes.bar(
        range(len(summary)),
        summary["mean"],
        yerr=summary["std"].fillna(0.0),
        capsize=4,
        color="#4C78A8",
        alpha=0.85,
    )
    axes.set_xticks(range(len(summary)))
    axes.set_xticklabels(summary.index)
    axes.set_ylabel(PARAM)
    axes.set_title(TITLE)
    axes.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(OUTPUT_PATH)
    print("已保存: " + str(OUTPUT_PATH))


if __name__ == "__main__":
    main()
'''

_REPRO_CURVE_BODY = '''

def main() -> None:
    frame = load_frame()
    figure, axes = plt.subplots(figsize=(6.4, 4.0))
    for name, chunk in frame.groupby("name", sort=True):
        tidy = pd.DataFrame(
            {
                "par": pd.to_numeric(chunk["PAR"], errors="coerce"),
                "value": pd.to_numeric(chunk[PARAM], errors="coerce"),
            }
        ).dropna()
        series = tidy.groupby("par")["value"].mean().sort_index()
        axes.plot(series.index, series.values, marker="o", markersize=3, label=str(name))

    axes.set_xlabel("PAR")
    axes.set_ylabel(PARAM)
    axes.set_title(TITLE)
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(frameon=False, fontsize=8)

    figure.tight_layout()
    figure.savefig(OUTPUT_PATH)
    print("已保存: " + str(OUTPUT_PATH))


if __name__ == "__main__":
    main()
'''


def _write_repro_script(
    image_path: str,
    title: str,
    csv_path: str,
    param: str,
    par_filter: float | None,
    *,
    kind: str,
) -> str:
    """为图表生成一份独立可跑的复现脚本，返回脚本路径。

    页面上的「下载 Python 脚本」直接读取该文件；脚本只依赖 pandas 与
    matplotlib，脱离本项目也能重算出同一张图。
    """
    target = Path(image_path).with_suffix(".py")
    body = _REPRO_BAR_BODY if kind == "bar" else _REPRO_CURVE_BODY
    header = _REPRO_HEADER.format(
        title=title,
        filename=target.name,
        csv_literal=repr(str(Path(csv_path).resolve())),
        output_literal=repr(str(Path(image_path).resolve())),
        param_literal=repr(param),
        par_literal=repr(None if par_filter is None else float(par_filter)),
        title_literal=repr(title),
    )
    target.write_text(header + body, encoding="utf-8")
    return str(target)



def _welch_p_value(groups: list[list[float]]) -> float | None:
    """两组比较时给出 Welch t 检验 p 值；缺少 scipy 或组数不符返回 None。"""
    if len(groups) != 2 or min(len(group) for group in groups) < 2:
        return None
    try:
        from scipy import stats  # 可选依赖
    except ImportError:
        return None
    try:
        result = stats.ttest_ind(groups[0], groups[1], equal_var=False)
    except ValueError:
        return None
    p_value = float(result.pvalue)
    return None if p_value != p_value else round(p_value, 4)  # 过滤 NaN


def _group_statistics(frame: pd.DataFrame, param: str) -> dict:
    """按样本组汇总均值、标准差与样本数。"""
    values = pd.to_numeric(frame[param], errors="coerce")
    tidy = pd.DataFrame({"group": frame["base_name"], "value": values}).dropna()
    if tidy.empty:
        return {"param": param, "groups": [], "stat_failed": True}

    groups: list[dict] = []
    raw_groups: list[list[float]] = []
    for name, chunk in tidy.groupby("group", sort=True):
        series = chunk["value"]
        groups.append(
            {
                "name": str(name),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std(ddof=1)), 4) if len(series) > 1 else 0.0,
                "n": int(len(series)),
            }
        )
        raw_groups.append(series.tolist())

    return {
        "param": param,
        "groups": groups,
        "p_value": _welch_p_value(raw_groups),
        "stat_failed": False,
    }


def _trend_hint(statistics: dict, unit: str = "") -> str:
    """把组间统计结果转成一句确定性的趋势事实，供报告引用。"""
    groups = [item for item in statistics.get("groups", []) if item["n"] > 0]
    if len(groups) < 2:
        return ""
    ordered = sorted(groups, key=lambda item: item["mean"])
    low, high = ordered[0], ordered[-1]
    if low["mean"] == 0:
        return f"{high['name']} 均值 {high['mean']}{unit}，其余组为 0，无法计算相对差异。"
    delta = (high["mean"] - low["mean"]) / abs(low["mean"]) * 100
    if abs(delta) < 0.1:
        # 均值差落在四舍五入精度内，不应硬报一个"高 0.0%"
        hint = (
            f"{high['name']} 与 {low['name']} 均值接近"
            f"（{high['mean']}{unit} 与 {low['mean']}{unit}）。"
        )
    else:
        direction = "高" if delta > 0 else "低"
        hint = (
            f"{high['name']} 均值最高（{high['mean']}{unit}），"
            f"{low['name']} 最低（{low['mean']}{unit}），"
            f"{high['name']} 比 {low['name']} {direction} {abs(delta):.1f}%。"
        )
    p_value = statistics.get("p_value")
    if p_value is not None:
        hint += f"两组 Welch t 检验 p = {p_value}。"
    return hint


def _curve_hint(frame: pd.DataFrame, param: str) -> str:
    """给出各组峰值及其对应 PAR，作为曲线类的确定性趋势事实。"""
    tidy = pd.DataFrame(
        {
            "group": frame["base_name"],
            "par": pd.to_numeric(frame["PAR"], errors="coerce"),
            "value": pd.to_numeric(frame[param], errors="coerce"),
        }
    ).dropna()
    if tidy.empty:
        return ""
    peaks = tidy.loc[tidy.groupby("group")["value"].idxmax()].sort_values(
        "value", ascending=False
    )
    parts = [
        f"{row.group} 在 PAR={row.par:g} 处达到最高 {param} {row.value:.3f}"
        for row in peaks.itertuples()
    ]
    return "；".join(parts) + "。"


def _render_bar(statistics: dict, title: str, ylabel: str,
                color: str = "#4C78A8") -> str:
    """绘制带误差棒的组间均值柱状图，返回图片路径。"""
    groups = statistics["groups"]
    labels = [item["name"] for item in groups]
    means = [item["mean"] for item in groups]
    errors = [item["std"] for item in groups]

    figure, axes = plt.subplots(figsize=(max(4.0, 1.1 * len(labels) + 2), 3.6))
    positions = range(len(labels))
    axes.bar(positions, means, yerr=errors, capsize=4, color=color, alpha=0.85)
    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels, rotation=0)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.spines[["top", "right"]].set_visible(False)

    path = _output_path(title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return str(path)


def _render_series(
    frame: pd.DataFrame,
    param: str,
    title: str,
    x_column: str,
    xlabel: str,
    ylabel: str,
) -> str:
    """绘制分组折线图（光响应曲线 / 动力学曲线共用）。"""
    figure, axes = plt.subplots(figsize=(6.4, 4.0))
    for name, chunk in frame.groupby("base_name", sort=True):
        values = pd.to_numeric(chunk[param], errors="coerce")
        tidy = pd.DataFrame({"x": pd.to_numeric(chunk[x_column], errors="coerce"),
                             "y": values}).dropna()
        if tidy.empty:
            continue
        series = tidy.groupby("x")["y"].mean().sort_index()
        axes.plot(series.index, series.values, marker="o", markersize=3, label=str(name))

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(frameon=False, fontsize=8)

    path = _output_path(title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return str(path)


def _parse_json(raw: str, fallback):
    """宽松解析 JSON 参数，非法输入回落到默认值。"""
    try:
        value = json.loads(raw) if raw else fallback
    except json.JSONDecodeError:
        return fallback
    return value if isinstance(value, type(fallback)) else fallback


def _first_param(params_json: str, default: str) -> str:
    params = _parse_json(params_json, [])
    return str(params[0]) if params else default


def _finish(chart_type: str, param: str, statistics: dict, image_path: str,
            title: str, trend_hint: str = "", code_path: str = "") -> str:
    """组装工具统一返回结构。"""
    return json.dumps(
        {
            "chart_type": chart_type,
            "param": param,
            "title": title,
            "image_path": image_path,
            "code_path": code_path,
            "statistics": statistics,
            "trend_hint": trend_hint,
        },
        ensure_ascii=False,
    )


def _error(chart_type: str, param: str, message: str) -> str:
    return json.dumps(
        {"chart_type": chart_type, "param": param, "error": message},
        ensure_ascii=False,
    )


def _bar_tool(
    chart_type: str,
    title_cn: str,
    csv_path: str,
    params_json: str,
    config_json: str,
    par_filter: float | None = None,
) -> str:
    """柱状图类工具的公共实现。"""
    param = _first_param(params_json, "Y(II)")
    config = _parse_json(config_json, {})
    title = str(config.get("title") or f"{title_cn} - {param}")
    color = str(config.get("color") or "#4C78A8")
    try:
        frame = _read_frame(csv_path, par_filter)
        if param not in frame.columns:
            return _error(chart_type, param, f"数据中不存在列 {param}")
        statistics = _group_statistics(frame, param)
        if statistics.get("stat_failed"):
            return _error(chart_type, param, f"{param} 无有效数值")
        statistics["par_filter"] = par_filter
        image_path = _render_bar(statistics, title, param, color=color)
        code_path = _write_repro_script(
            image_path, title, csv_path, param, par_filter, kind="bar"
        )
    except (KeyError, OSError, ValueError) as exc:
        return _error(chart_type, param, f"统计绘图失败: {exc}")
    return _finish(chart_type, param, statistics, image_path, title,
                   _trend_hint(statistics), code_path)


def _curve_tool(
    chart_type: str,
    title_cn: str,
    xlabel: str,
    csv_path: str,
    params_json: str,
    config_json: str,
    par_filter: float | None = None,
) -> str:
    """曲线类工具的公共实现（按 PAR 聚合）。"""
    param = _first_param(params_json, "rETR")
    config = _parse_json(config_json, {})
    title = str(config.get("title") or f"{title_cn} - {param}")
    try:
        frame = _read_frame(csv_path, par_filter)
        if param not in frame.columns:
            return _error(chart_type, param, f"数据中不存在列 {param}")
        if "PAR" not in frame.columns:
            return _error(chart_type, param, "数据缺少 PAR 列，无法绘制光响应曲线")
        par_levels = sorted(
            pd.to_numeric(frame["PAR"], errors="coerce").dropna().unique().tolist()
        )
        if len(par_levels) < 3:
            return _error(chart_type, param, "有效 PAR 梯度不足 3 个")
        image_path = _render_series(frame, param, title, "PAR", xlabel, param)
        statistics = {
            "param": param,
            "par_levels": [round(float(level), 2) for level in par_levels],
            "varieties": [str(name) for name in frame["base_name"].unique()],
            "stat_failed": False,
        }
        trend_hint = _curve_hint(frame, param)
        code_path = _write_repro_script(
            image_path, title, csv_path, param, par_filter, kind="curve"
        )
    except (KeyError, OSError, ValueError) as exc:
        return _error(chart_type, param, f"曲线绘制失败: {exc}")
    return _finish(chart_type, param, statistics, image_path, title,
                   trend_hint, code_path)


def generate_anova_bars(csv_path: str, params_json: str = "[]",
                        config_json: str = "{}",
                        par_filter: float | None = None) -> str:
    """多组（≥3）均值柱状图。"""
    return _bar_tool("anova_bars", "组间均值比较", csv_path, params_json,
                     config_json, par_filter)


def generate_ttest_bars(csv_path: str, params_json: str = "[]",
                        config_json: str = "{}",
                        par_filter: float | None = None) -> str:
    """两组均值柱状图（附 Welch t 检验 p 值）。"""
    return _bar_tool("ttest_bars", "两组均值比较", csv_path, params_json,
                     config_json, par_filter)


def generate_single_fvfm_bar(csv_path: str, params_json: str = "[]",
                             config_json: str = "{}",
                             par_filter: float | None = None) -> str:
    """单样品 Fv/Fm 柱状图。"""
    params_json = params_json if _parse_json(params_json, []) else '["Fv/Fm"]'
    return _bar_tool("single_fvfm", "暗适应最大光化学效率", csv_path, params_json,
                     config_json, par_filter)


def generate_aq_curves(csv_path: str, params_json: str = "[]",
                       config_json: str = "{}",
                       par_filter: float | None = None) -> str:
    """光响应曲线（横轴 PAR，纵轴 rETR）。"""
    return _curve_tool("aq_curves", "光响应曲线", "PAR", csv_path, params_json,
                       config_json, par_filter)


def generate_single_light_curve(csv_path: str, params_json: str = "[]",
                                config_json: str = "{}",
                                par_filter: float | None = None) -> str:
    """单样品光响应曲线。"""
    params_json = params_json if _parse_json(params_json, []) else '["rETR"]'
    return _curve_tool("single_curve", "单样品光响应曲线", "PAR", csv_path,
                       params_json, config_json, par_filter)


def generate_kinetics_curves(csv_path: str, params_json: str = "[]",
                             config_json: str = "{}",
                             par_filter: float | None = None) -> str:
    """荧光动力学时间序列曲线。

    Cycle 标注存在时按 Cycle 顺序作为横轴（相对测量点），否则按行序。
    """
    param = _first_param(params_json, "Y(II)")
    config = _parse_json(config_json, {})
    title = str(config.get("title") or f"荧光动力学时间序列 - {param}")
    try:
        frame = _read_frame(csv_path, par_filter)
        if param not in frame.columns:
            return _error("kinetics", param, f"数据中不存在列 {param}")
        # 行序即导出时的测量顺序；缺 Cycle 标注时只做描述性折线
        frame = frame.assign(_order=range(len(frame)))
        image_path = _render_series(frame, param, title, "_order", "测量点序号", param)
        code_path = _write_repro_script(
            image_path, title, csv_path, param, par_filter, kind="curve"
        )
        statistics = {
            "param": param,
            "par_filter": par_filter,
            "point_count": int(len(frame)),
            "has_cycle_labels": has_kinetics_cycles(frame),
            "stat_failed": False,
        }
    except (KeyError, OSError, ValueError) as exc:
        return _error("kinetics", param, f"动力学绘制失败: {exc}")
    return _finish("kinetics", param, statistics, image_path, title, code_path=code_path)


__all__ = [
    "CHART_OUTPUT_DIR",
    "generate_anova_bars",
    "generate_aq_curves",
    "generate_kinetics_curves",
    "generate_single_fvfm_bar",
    "generate_single_light_curve",
    "generate_ttest_bars",
    "to_csv",
]
