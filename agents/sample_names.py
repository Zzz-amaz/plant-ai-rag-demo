"""样品名称结构识别：区分「名称自带数字」与「真正的重复样本」。

只有当至少两个基础品种都各自出现两个以上数字后缀时，才判定为
``品种_重复编号`` 设计。例如 ``A_1/A_2/B_1/B_2`` 是重复设计；
``Plant_1/Plant_2`` 则视为两个完整品种名。
"""

from __future__ import annotations

import re

import pandas as pd

_REPLICATE_PATTERN = re.compile(r"^(?P<base>.+)_(?P<index>\d+)$")


def classify_sample_names(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], bool]:
    """返回带 ``base_name`` 列的副本、品种列表与重复结构标记。

    Args:
        df: 至少包含 ``name`` 列的原始数据。

    Returns:
        (新增 base_name 列的数据, 品种名列表, 是否存在重复结构)
    """
    result = df.copy()
    names = result["name"].dropna().astype(str)
    unique_names = list(dict.fromkeys(names.tolist()))

    grouped_suffixes: dict[str, set[str]] = {}
    parsed: dict[str, str] = {}
    for name in unique_names:
        match = _REPLICATE_PATTERN.match(name)
        if not match:
            continue
        base = match.group("base")
        parsed[name] = base
        grouped_suffixes.setdefault(base, set()).add(match.group("index"))

    replicated_bases = [
        base for base, indexes in grouped_suffixes.items() if len(indexes) >= 2
    ]
    has_replicates = len(replicated_bases) >= 2

    if has_replicates:
        mapped = result["name"].astype(str).map(parsed)
        result["base_name"] = mapped.fillna(result["name"].astype(str))
    else:
        result["base_name"] = result["name"].astype(str)

    varieties = sorted(result["base_name"].dropna().astype(str).unique().tolist())
    return result, varieties, has_replicates


def has_biological_replicates(names: pd.Series) -> bool:
    """只判断名称集合是否构成至少两个品种的重复样本设计。"""
    frame = pd.DataFrame({"name": names.dropna().astype(str)})
    if frame.empty:
        return False
    _, _, has_replicates = classify_sample_names(frame)
    return has_replicates
