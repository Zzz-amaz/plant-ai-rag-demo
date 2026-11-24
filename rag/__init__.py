"""RAG 检索链路核心组件。

- ``entity_alias_catalog``: 入库阶段固化的实体别名目录
- ``paper_rank_fusion``: RRF 排名融合与实体约束重排
- ``evidence``: 可溯源的证据片段对象
- ``citation_validator``: 证据绑定型答案的确定性校验
- ``paper_citations``: 参考文献渲染
"""

from .citation_validator import (
    REFUSAL_TEXT,
    ValidationResult,
    parse_answer_json,
    validate_and_render_answer,
)
from .entity_alias_catalog import AliasEntry, EntityAliasCatalog, build_alias_entries
from .evidence import Evidence, build_evidence_objects, evidence_prompt_block
from .paper_citations import PaperCitation, format_paper_reference, normalize_doi
from .paper_rank_fusion import (
    ConstraintAwarePaperRanker,
    PaperFusionResult,
    reciprocal_rank_fuse,
)

__all__ = [
    "AliasEntry",
    "ConstraintAwarePaperRanker",
    "EntityAliasCatalog",
    "Evidence",
    "PaperCitation",
    "PaperFusionResult",
    "REFUSAL_TEXT",
    "ValidationResult",
    "build_alias_entries",
    "build_evidence_objects",
    "evidence_prompt_block",
    "format_paper_reference",
    "normalize_doi",
    "parse_answer_json",
    "reciprocal_rank_fuse",
    "validate_and_render_answer",
]
