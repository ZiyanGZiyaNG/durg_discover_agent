"""Typed output of every operation the compiler can run.

Each model lists the fields downstream steps rely on. Other fields coming from
the source are kept as they are (extra="allow"), so adding a field upstream does
not break a plan, but a missing or wrongly typed field is caught right here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, RootModel


class _Record(BaseModel):
    model_config = ConfigDict(extra="allow")


class ResolvedEntity(_Record):
    id: str
    name: str | None = ""
    symbol: str | None = ""


class ResolvedEntities(RootModel[dict[str, ResolvedEntity]]):
    """entity_ref -> resolved disease or target."""


class AssociationRow(_Record):
    target_id: str
    target_symbol: str = ""
    target_name: str = ""
    rank: int | None = None
    overall_association_score: float | None = None


class DiscoveredTargets(_Record):
    returned_target_count: int
    total_associated_target_count: int
    targets: list[AssociationRow]
    limitations: list[str] = []


class DatasourceEvidence(_Record):
    total_evidence_count: int
    fetched_evidence_count: int
    truncated: bool


class ExactPairEvidence(_Record):
    overall_association_score: float | None
    datasource_evidence: list[DatasourceEvidence]


class DrugRecord(_Record):
    exact_disease_match: bool


class TargetDrugs(_Record):
    total_drug_count: int
    fetched_drug_count: int
    drugs: list[DrugRecord]


class TargetDrugEvidence(_Record):
    by_target: dict[str, TargetDrugs]
    failed_targets: dict[str, str]


class TractabilityAssessment(_Record):
    modality: str
    value: bool
    label: str | None = ""


class TargetTractability(_Record):
    assessment_count: int
    assessments: list[TractabilityAssessment]


class TractabilityEvidence(_Record):
    by_target: dict[str, TargetTractability]
    failed_targets: dict[str, str]


class StratifiedRow(_Record):
    target_id: str
    association_rank: int | None
    overall_association_score: float | None
    drug_evidence: Literal["available", "missing"]
    known_drug_count: int | None
    exact_disease_drug_count: int | None
    tractability_evidence: Literal["available", "missing"]
    tractable_bucket_count: int | None
    tractable_modalities: list[str] | None
    target_translation_layer: Literal[
        "translational_support", "disease_association_only", "undetermined"
    ]


class StratifiedEvidence(_Record):
    rule_version: str
    targets: list[StratifiedRow]
    failed_targets: dict[str, dict[str, str]]
    limitations: list[str]


class RankedRow(_Record):
    target_id: str


class RankedTargets(_Record):
    targets: list[RankedRow]
    ranking_spec: dict[str, Any]


# operation -> the model its payload must satisfy (same keys as ADAPTERS)
OUTPUT_MODELS: dict[str, type[BaseModel]] = {
    "resolve_disease": ResolvedEntities,
    "resolve_target": ResolvedEntities,
    "retrieve_exact_pair": ExactPairEvidence,
    "discover_targets": DiscoveredTargets,
    "discover_targets_per_disease": DiscoveredTargets,
    "retrieve_target_drugs": TargetDrugEvidence,
    "retrieve_tractability": TractabilityEvidence,
    "stratify_target_evidence_layers": StratifiedEvidence,
    "rank_targets": RankedTargets,
}
