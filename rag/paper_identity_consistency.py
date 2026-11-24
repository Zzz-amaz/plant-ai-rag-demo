"""Evidence-free paper identity consistency checks for routing."""

from __future__ import annotations

from dataclasses import dataclass

from .paper_entities import PaperEntityProfile
from .query_understanding import EntityConstraints


def _canonical(values: tuple[str, ...]) -> set[str]:
    return {
        value.casefold().replace("_", " ").replace("-", " ").replace("/", " ").strip()
        for value in values
        if value.strip()
    }


_DROUGHT_FAMILY = {"drought", "water deficit", "water stress", "osmotic", "dehydration", "soil drying"}
_FLUORESCENCE_FAMILY = {
    "chlorophyll fluorescence", "fv fm", "npq", "phi psii", "psii", "etr", "ojip",
    "jip test", "photochemical quenching",
}


def _matches_dimension(name: str, query_values: set[str], profile_values: set[str]) -> bool:
    if query_values.intersection(profile_values):
        return True
    if name == "treatment" and query_values.intersection(_DROUGHT_FAMILY):
        return bool(profile_values.intersection(_DROUGHT_FAMILY))
    if name == "metric" and "chlorophyll fluorescence" in query_values:
        return bool(profile_values.intersection(_FLUORESCENCE_FAMILY))
    return False


@dataclass(frozen=True)
class PaperIdentityConsistency:
    requested_dimensions: tuple[str, ...]
    matched_dimensions: tuple[str, ...]
    unknown_dimensions: tuple[str, ...]
    mismatched_dimensions: tuple[str, ...]

    @property
    def blocks_top1(self) -> bool:
        return bool(self.mismatched_dimensions)

    @property
    def three_way_verified(self) -> bool:
        required = {"research_object", "treatment", "metric"}
        return required.issubset(self.matched_dimensions)


def assess_paper_identity_consistency(
    profile: PaperEntityProfile, constraints: EntityConstraints
) -> PaperIdentityConsistency:
    """Check research object, treatment and metric without using answer labels."""

    requested: list[str] = []
    matched: list[str] = []
    unknown: list[str] = []
    mismatched: list[str] = []

    object_query = _canonical(constraints.organisms) | _canonical(constraints.latin_names)
    if object_query:
        requested.append("research_object")
        object_profile = _canonical(profile.organisms) | _canonical(profile.latin_names)
        if not object_profile:
            unknown.append("research_object")
        elif object_query.intersection(object_profile):
            matched.append("research_object")
        else:
            mismatched.append("research_object")

    for name, query_values, profile_values in (
        ("treatment", _canonical(constraints.treatments), _canonical(profile.treatments)),
        ("metric", _canonical(constraints.metrics), _canonical(profile.metrics)),
    ):
        if not query_values:
            continue
        requested.append(name)
        if not profile_values:
            unknown.append(name)
        elif _matches_dimension(name, query_values, profile_values):
            matched.append(name)
        else:
            mismatched.append(name)

    return PaperIdentityConsistency(
        tuple(requested), tuple(matched), tuple(unknown), tuple(mismatched)
    )
