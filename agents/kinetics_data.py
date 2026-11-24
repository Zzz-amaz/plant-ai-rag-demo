"""荧光动力学序列的 Cycle 标记解析与时间轴处理。

仪器把光诱导—暗恢复过程导出为 ``Light_1..Light_n``、``Dark_1..Dark_n``
的 Cycle 标记。编号缺项会直接影响动力学解释，必须作为数据质量信息
暴露出来，而不能被当成连续数据静默处理。
"""

from __future__ import annotations

import re

import pandas as pd

_KINETICS_CYCLE_RE = re.compile(r"^(light|dark)[_\-\s]*(\d+)$", re.IGNORECASE)
_LIGHT_STEADY_STATE = "light_steady_state"
_STEADY_STATE_INDEX = 1_000_000


def parse_kinetics_cycle(value: object) -> tuple[str, int] | None:
    """解析 Light_1、Light_steady_state、Dark_2 等标记，无法识别返回 None。"""
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == _LIGHT_STEADY_STATE:
        # 稳态点排在所有编号 Light 点之后；绘图顺序仍以 DateTime 与行序为准
        return "light", _STEADY_STATE_INDEX
    match = _KINETICS_CYCLE_RE.match(str(value).strip())
    if not match:
        return None
    return match.group(1).lower(), int(match.group(2))


def kinetics_cycle_mask(frame: pd.DataFrame) -> pd.Series:
    """返回真正属于光/暗动力学序列的数据行掩码。"""
    if "Cycle" not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame["Cycle"].apply(lambda value: parse_kinetics_cycle(value) is not None)


def has_kinetics_cycles(frame: pd.DataFrame, minimum_points: int = 5) -> bool:
    """判断数据是否包含足够数量的 Cycle 标注点。"""
    return int(kinetics_cycle_mask(frame).sum()) >= minimum_points


def kinetics_cycle_gaps(frame: pd.DataFrame) -> dict[str, tuple[int, ...]]:
    """返回光/暗阶段缺失的编号区间。

    ``Light_steady_state`` 是合法的终点标记，不算缺项。
    """
    if "Cycle" not in frame.columns:
        return {}
    observed: dict[str, set[int]] = {"light": set(), "dark": set()}
    for value in frame["Cycle"]:
        parsed = parse_kinetics_cycle(value)
        if parsed is None:
            continue
        phase, index = parsed
        if index < _STEADY_STATE_INDEX:
            observed[phase].add(index)

    gaps: dict[str, tuple[int, ...]] = {}
    for phase, indexes in observed.items():
        if len(indexes) < 2:
            continue
        missing = tuple(
            index
            for index in range(min(indexes), max(indexes) + 1)
            if index not in indexes
        )
        if missing:
            gaps[phase] = missing
    return gaps
