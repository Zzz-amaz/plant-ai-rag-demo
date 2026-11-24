"""Conservative, paper-level research tags for the complete local corpus.

These tags are a retrieval guardrail, not extracted scientific facts.  They
use locally available titles, abstracts and entity cards to identify what a
paper *appears* eligible to support.  Full-text snippets and, where present,
verified fact cards remain the authority for claims in an answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .query_understanding import organism_aliases


SCHEMA_VERSION = "paper-research-tags-v1"


def _text(row: Mapping[str, Any]) -> str:
    values = [
        row.get("title"), row.get("abstract"), row.get("organisms"),
        row.get("treatments"), row.get("study_types"), row.get("metrics"),
        row.get("keywords"), row.get("latin_names"),
    ]
    return " ".join(str(value or "") for value in values).casefold()


def _has(text: str, *patterns: str) -> bool:
    return any(re.search(pattern, text, flags=re.I) is not None for pattern in patterns)


@dataclass(frozen=True)
class PaperResearchTags:
    paper_id: str
    organisms: tuple[str, ...] = ()
    stresses: tuple[str, ...] = ()
    research_types: tuple[str, ...] = ()
    intervention_types: tuple[str, ...] = ()
    result_levels: tuple[str, ...] = ()
    evidence_roles: tuple[str, ...] = ()
    tag_confidence: str = "low"
    tag_sources: tuple[str, ...] = ()


def tag_profile(row: Mapping[str, Any]) -> PaperResearchTags:
    """Classify a paper conservatively from its local metadata.

    A review is deliberately never tagged as an intervention result.  A
    genetic/GWAS paper must show a matching result term before it can be used
    as an intervention or association case candidate.
    """

    text = _text(row)
    title = str(row.get("title") or "").casefold()
    organisms = list(dict.fromkeys(str(value).casefold() for value in row.get("organisms", ()) if value))
    organisms.extend(
        name for name, aliases in organism_aliases().items()
        if any(alias.casefold() in text for alias in aliases) and name not in organisms
    )
    stress_aliases = {
        "drought": ("drought", "water deficit", "water stress", "osmotic", "dehydration", "干旱", "水分胁迫"),
        "salt": ("salt stress", "salinity", "nacl", "盐胁迫", "盐害"),
        "heat": ("heat stress", "high temperature", "thermal stress", "热胁迫", "高温"),
        "cold": ("cold stress", "low temperature", "chilling", "freezing", "冷胁迫", "低温"),
        "high_light": ("high light", "high irradiance", "excess light", "photoinhibition", "强光", "高光"),
        "pathogen": ("pathogen", "fungal", "bacterial", "virus", "infection", "disease", "病原", "真菌", "细菌", "病毒"),
        "nutrition": ("nitrogen", "phosphorus", "nutrient", "fertilizer", "氮素", "磷素", "施肥"),
    }
    stresses = list(dict.fromkeys(str(value).casefold() for value in row.get("treatments", ()) if value))
    stresses.extend(
        name for name, aliases in stress_aliases.items()
        if any(alias in text for alias in aliases) and name not in stresses
    )
    research_types: list[str] = []
    interventions: list[str] = []
    results: list[str] = []
    roles: list[str] = []

    review = _has(title, r"\breview\b", r"\badvanc\w*\b", r"\bperspectives?\b", r"\boverview\b", r"\bmeta-analysis\b", r"综述")
    if review:
        research_types.append("review")
        roles.append("background_review")
    else:
        research_types.append("original_research")

    if _has(text, r"\bgenome[- ]wide association\b", r"\bgwas\b", r"\bqtl\b", r"\bquantitative trait loci?\b", r"\bgenetic mapping\b", r"\bmapping of .* loci\b"):
        research_types.append("genetic_association")
        interventions.append("natural_variation_or_mapping")

    # Genetic intervention tags are title-led by design.  Abstracts commonly
    # mention CRISPR or transformation as background, which otherwise creates
    # false "success cases" for inoculant or treatment studies.
    if _has(title, r"\bcrispr(?:[-/ ]cas\w*)?\b", r"\bgene edit(?:ing|ed)?\b", r"\bknockout\b", r"\bknock[- ]?out\b"):
        interventions.append("gene_editing")
    if _has(title, r"\boverexpress(?:ion|ed)?\b", r"\btransgenic\b", r"\bectopic expression\b"):
        interventions.append("transgenic_or_overexpression")
    if _has(title, r"\bmutant\b", r"\bmutagenesis\b", r"\bdeficient allele\b", r"\bloss[- ]of[- ]function\b"):
        interventions.append("mutagenesis_or_allelic_variant")
    if _has(text, r"\bspray(?:ed|ing)?\b", r"\bapplication\b", r"\btreatment\b", r"\bfertilizer\b", r"\birrigation\b", r"\bexogenous\b"):
        interventions.append("physiological_or_agronomic_treatment")

    if _has(text, r"\bfv/fm\b", r"\bnpq\b", r"\by\(ii\)\b", r"\bphi\s*psii\b", r"\bet\s*r\b", r"chlorophyll fluorescence", r"photosystem ii efficiency"):
        results.append("fluorescence_or_energy_partitioning")
    if _has(text, r"\bphotosynth(?:esis|etic)\b", r"\bco2 assimilation\b", r"\bcarbon fixation\b", r"\bgas exchange\b", r"\bnet photosynthetic rate\b"):
        results.append("photosynthetic_capacity_or_carbon_assimilation")
    if _has(text, r"\btolerance\b", r"\bresistance\b", r"\bsurvival\b", r"\bphotoinhibition\b", r"\bstress response\b"):
        results.append("stress_tolerance_or_protection")
    if _has(text, r"\byield\b", r"\bgrain\b", r"\bbiomass\b", r"\bproductivity\b", r"\bseed weight\b"):
        results.append("biomass_or_yield")

    if not review and "genetic_association" in research_types:
        roles.append("direct_genetic_association_candidate")
    if not review and interventions and results:
        roles.append("intervention_outcome_study_candidate")
    if not review and results and not interventions:
        roles.append("physiology_or_stress_response_study")
    if not results and not review:
        roles.append("method_or_background_candidate")

    # A successful modification case is a high bar.  It requires an original
    # intervention paper and an explicit positive outcome in the local record;
    # final use still needs a full-text evidence snippet.
    positive = _has(text, r"\bimprov(?:e|ed|ement|ing)\b", r"\bincreas(?:e|ed|ing)\b", r"\benhanc(?:e|ed|ement|ing)\b", r"提高", r"增强")
    if not review and interventions and results and positive:
        roles.append("positive_intervention_case_candidate")

    sources = [name for name, value in (("title", row.get("title")), ("abstract", row.get("abstract")), ("entity_profile", row.get("organisms") or row.get("treatments"))) if value]
    confidence = "high" if row.get("abstract") and (interventions or results) else "medium" if row.get("abstract") else "low"
    return PaperResearchTags(
        paper_id=str(row.get("paper_id") or "").strip(),
        organisms=tuple(dict.fromkeys(organisms)),
        stresses=tuple(dict.fromkeys(stresses)),
        research_types=tuple(dict.fromkeys(research_types)),
        intervention_types=tuple(dict.fromkeys(interventions)),
        result_levels=tuple(dict.fromkeys(results)),
        evidence_roles=tuple(dict.fromkeys(roles)),
        tag_confidence=confidence,
        tag_sources=tuple(sources),
    )


class PaperResearchTagIndex:
    def __init__(self, rows: Iterable[PaperResearchTags]):
        self.rows = tuple(row for row in rows if row.paper_id)
        self._by_id = {row.paper_id: row for row in self.rows}

    def get(self, paper_id: str) -> PaperResearchTags | None:
        return self._by_id.get(paper_id)

    def eligible(self, paper_id: str, *, required_role: str | None = None) -> bool:
        row = self.get(paper_id)
        return bool(row and (not required_role or required_role in row.evidence_roles))


def required_evidence_role(question: str) -> str | None:
    """Return a narrow evidence qualification for high-risk question forms."""

    text = str(question or "").casefold()
    if _has(text, r"成功案例", r"成功改造", r"成功实现", r"successful (?:case|example|engineering)", r"engineering success"):
        return "positive_intervention_case_candidate"
    if _has(text, r"哪些位点", r"哪些基因座", r"遗传位点", r"qtl", r"gwas", r"loci", r"locus"):
        return "direct_genetic_association_candidate"
    return None


def required_intervention_family(question: str) -> tuple[str, ...]:
    """Keep engineering questions separate from treatment-response papers."""

    text = str(question or "").casefold()
    if _has(text, r"高光合效率改造", r"基因改造", r"基因编辑", r"遗传改良", r"engineering", r"gene edit", r"genetic modification"):
        return (
            "gene_editing",
            "transgenic_or_overexpression",
            "mutagenesis_or_allelic_variant",
        )
    return ()


def write_research_tags(path: str | Path, rows: Iterable[PaperResearchTags]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_research_tags(path: str | Path) -> list[PaperResearchTags]:
    rows: list[PaperResearchTags] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            rows.append(PaperResearchTags(
                paper_id=str(value.get("paper_id") or ""),
                organisms=tuple(value.get("organisms") or ()),
                stresses=tuple(value.get("stresses") or ()),
                research_types=tuple(value.get("research_types") or ()),
                intervention_types=tuple(value.get("intervention_types") or ()),
                result_levels=tuple(value.get("result_levels") or ()),
                evidence_roles=tuple(value.get("evidence_roles") or ()),
                tag_confidence=str(value.get("tag_confidence") or "low"),
                tag_sources=tuple(value.get("tag_sources") or ()),
            ))
    return rows
