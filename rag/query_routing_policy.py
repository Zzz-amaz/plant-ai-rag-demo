"""Deterministic routing policy for the Device-A assistant.

This module classifies the *source of truth*, not answer quality.  Broad
cross-species literature syntheses are detected separately because they use
the paper route but must not be mistaken for a request to lock one paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Route = Literal["general", "manual", "paper", "mixed"]


@dataclass(frozen=True)
class QueryRouteDecision:
    route: Route
    reason: str


# Device anchors identify the Device-A product/manual as the source of truth.
# They deliberately exclude ambiguous scientific protocol words such as
# ``暗适应``: in "论文中的暗适应时间" the paper, not the manual, is authoritative.
DEVICE_ANCHORS = (
    "fdm", "fdm-n1", "仪器", "软件", "相机", "串口", "测量界面", "按钮", "点击",
    "操作手册", "使用手册", "设备手册", "报错", "故障", "连接", "校准", "去背景", "roi",
    "保存路径", "光强档位", "al档位", "ml档位", "button", "instrument", "software",
    "connect", "calibrat", "error", "setting", "manual",
)

# These requests normally concern a device workflow when no paper is named.
# A literature signal takes precedence unless a device anchor is also present.
MANUAL_OPERATION_SIGNALS = (
    "操作步骤", "测量步骤", "如何设置", "导出", "导入", "开始测量", "新建测量",
    "暗适应怎么", "怎么暗适应", "export", "import", "measure",
)

PAPER_SIGNALS = (
    "文献", "论文", "文章", "参考文献", "相关文章", "研究表明", "研究发现", "实验结果",
    "论文结论", "论文中", "文献中", "综述", "进展", "paper", "literature", "evidence",
    "study reports", "according to",
)

# Users do not reliably say “请查论文”.  A question that names a biological
# research object and asks about an experimentally variable process, a
# comparison, a mechanism or a genetic intervention is evidence-seeking by
# nature and should use the paper collection.  This intentionally does *not*
# include an ungrounded question such as “为什么低温下 ETR 先下降”: that
# remains a fast general explanation.
RESEARCH_OBJECT_SIGNALS = (
    "水稻", "小麦", "玉米", "番茄", "大豆", "烟草", "马铃薯", "辣椒", "棉花",
    "拟南芥", "本氏烟", "葡萄", "植物", "作物", "生态型", "品种", "基因组",
    "arabidopsis", "rice", "wheat", "maize", "tomato", "soybean", "tobacco",
    "potato", "pepper", "cotton", "grape", "genome", "ecotype",
)
RESEARCH_QUESTION_SIGNALS = (
    "胁迫", "低温", "冷害", "高温", "干旱", "盐", "高光", "弱光", "病害", "侵染",
    "变异", "差异", "对比", "比较", "主要参数", "影响", "表征", "程度", "响应",
    "成功案例", "改造", "育种", "基因", "位点", "qtl", "gwAS", "aba", "激素",
    "光合作用效率", "叶绿素荧光", "光合效率", "photosynthesis", "fluorescence",
    "stress", "chilling", "cold", "drought", "salinity", "high light", "gene",
    "locus", "variation", "response", "efficiency",
)


def decide_query_route(
    query: str,
    *,
    recommendation_intent: bool = False,
) -> QueryRouteDecision:
    """Choose the minimum authoritative source for a user question.

    - ``general``: stable concept/causal explanation; no source claimed.
    - ``manual``: Device-A operation or a manual-confirmable setting.
    - ``paper``: literature finding, evidence interpretation, or paper detail.
    - ``mixed``: the user explicitly combines device operation with literature.
    """

    text = str(query or "").casefold()
    has_device_anchor = any(signal in text for signal in DEVICE_ANCHORS)
    has_manual_operation = any(signal in text for signal in MANUAL_OPERATION_SIGNALS)
    explicit_paper = recommendation_intent or any(signal in text for signal in PAPER_SIGNALS)
    evidence_seeking_science = (
        any(signal in text for signal in RESEARCH_OBJECT_SIGNALS)
        and any(signal in text for signal in RESEARCH_QUESTION_SIGNALS)
    )
    has_paper = explicit_paper or evidence_seeking_science
    if has_paper and has_device_anchor:
        return QueryRouteDecision("mixed", "manual_and_literature")
    if has_paper:
        reason = "explicit_literature_request" if explicit_paper else "evidence_seeking_scientific_question"
        return QueryRouteDecision("paper", reason)
    if has_device_anchor or has_manual_operation:
        return QueryRouteDecision("manual", "manual_or_device_operation")
    return QueryRouteDecision("general", "general_scientific_question")
