"""Strict, local-only matching for scientific entity-constrained evidence."""

from __future__ import annotations

from rag.query_understanding import EntityConstraints, organism_aliases


DIRECT_TREATMENT_TERMS = {
    "drought": ("drought", "water deficit", "water stress", "干旱", "水分胁迫"),
    "salt": ("salt", "salinity", "nacl", "盐胁迫", "盐害"),
    "heat": ("heat", "high temperature", "热胁迫", "高温"),
    "cold": ("cold", "chilling", "low temperature", "freezing", "冷胁迫", "低温"),
    "nitrogen": ("nitrogen", "nutrition", "fertilizer", "氮素", "施肥"),
    "light": ("light", "irradiance", "高光", "强光", "光照"),
}
DIRECT_METRIC_TERMS = {
    "fv/fm": ("fv/fm", "f v /f m"),
    "phi_psii": ("phi psii", "φpsii", "y(ii)", "yield of psii"),
    "npq": ("npq", "non-photochemical quenching"),
    "y_no": ("y(no)", "y no", "non-regulated energy dissipation"),
    "etr": ("etr", "electron transport rate"),
    "chlorophyll fluorescence": ("chlorophyll fluorescence", "chlorophyll-a fluorescence"),
}


def profile_matches_requested_entities(profile, constraints: EntityConstraints) -> bool:
    """Require every explicit species, treatment and metric in a source.

    Curated entity fields are preferred but cannot be the sole authority:
    legacy cards sometimes omit an organism or treatment that is unambiguous
    in the title or abstract.  This is still a strict match, not semantic
    similarity padding.
    """

    text = " ".join(
        str(value or "") for value in (
            getattr(profile, "title", ""), getattr(profile, "abstract", ""),
            *getattr(profile, "organisms", ()), *getattr(profile, "latin_names", ()),
            *getattr(profile, "treatments", ()), *getattr(profile, "metrics", ()),
        )
    ).casefold()
    if constraints.organisms and not any(
        any(alias.casefold() in text for alias in organism_aliases().get(organism, (organism,)))
        for organism in constraints.organisms
    ):
        return False
    if constraints.latin_names and not any(name.casefold() in text for name in constraints.latin_names):
        return False
    if constraints.treatments and not all(
        any(term.casefold() in text for term in DIRECT_TREATMENT_TERMS.get(treatment, (treatment,)))
        for treatment in constraints.treatments
    ):
        return False
    if constraints.metrics and not all(
        any(term.casefold() in text for term in DIRECT_METRIC_TERMS.get(metric, (metric,)))
        for metric in constraints.metrics
    ):
        return False
    return True


def direct_profile_fallback(paper_index, constraints: EntityConstraints, *, limit: int) -> list:
    """Recover direct papers missed by the vector candidate window.

    Vector retrieval is for semantic recall, but an exact title/abstract match
    must not disappear merely because its embedding did not land in the top
    24 chunks.  This scans only the small paper-card index (not full chunks),
    then leaves all answer evidence grounded in the original paper text.
    """

    from src.rag.paper_entities import RankedPaper

    rows = []
    for profile in getattr(paper_index, "profiles", ()):
        if not profile_matches_requested_entities(profile, constraints):
            continue
        constraint_score, _ = paper_index.score_constraints(profile, constraints)
        rows.append(RankedPaper(
            profile=profile,
            score=100.0 + float(constraint_score),
            lexical_score=0.0,
            constraint_score=float(constraint_score),
            hard_entity_match=True,
        ))
    rows.sort(key=lambda item: (-item.score, item.profile.paper_id))
    return rows[:max(0, int(limit))]
