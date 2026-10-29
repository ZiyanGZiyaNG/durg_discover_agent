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


class FilteredTargets(_Record):
    targets: list[RankedRow]
    filter_spec: dict[str, Any]
    excluded_target_ids: list[str]
    unevaluable_target_ids: list[str]


class IndicationRow(_Record):
    disease_id: str
    disease_name: str
    max_clinical_stage: str | None = None


class DrugIndications(_Record):
    drug_id: str
    drug_name: str
    total_indication_count: int
    fetched_indication_count: int
    indications: list[IndicationRow]


class ApprovedIndications(_Record):
    by_drug: dict[str, DrugIndications]
    failed_drugs: dict[str, str]


class BroaderApproval(_Record):
    disease_id: str
    disease_name: str


class IndicationCheckRow(_Record):
    drug_id: str
    drug_name: str
    disease_id: str
    disease_name: str
    status: str
    exact_stage: str | None = None
    broader_approvals: list[BroaderApproval]
    indications_truncated: bool


class IndicationChecks(_Record):
    disease_id: str
    disease_name: str
    checks: list[IndicationCheckRow]
    unchecked_drugs: dict[str, str]


class DrugMechanism(_Record):
    action_type: str
    mechanism_of_action: str
    target_ids: list[str]


class DrugTarget(_Record):
    target_id: str
    target_symbol: str
    target_name: str


class DrugTargetRecord(_Record):
    drug_id: str
    drug_name: str
    mechanism_count: int
    mechanisms: list[DrugMechanism]
    targets: list[DrugTarget]


class DrugTargets(_Record):
    by_drug: dict[str, DrugTargetRecord]
    failed_drugs: dict[str, str]


class DiscoveredTargetsPerDisease(_Record):
    """One disease: the same fields as DiscoveredTargets. Several diseases:
    each disease's own result under by_disease, and `targets` holds the union
    of target IDs (no scores, since a score belongs to one disease)."""

    by_disease: dict[str, DiscoveredTargets]
    failed_diseases: dict[str, str]
    targets: list[AssociationRow]
    limitations: list[str] = []


class ExactPairs(_Record):
    pairs: list[ExactPairEvidence]
    failed_pairs: dict[str, str]


class DatasourceScoreRow(_Record):
    disease_id: str | None = None
    target_id: str
    datasource_id: str
    score: float | None = None


class DatasourceScores(_Record):
    rows: list[DatasourceScoreRow]
    pair_count: int
    limitations: list[str]


class RankedDatasource(_Record):
    datasource_id: str
    ranked_position: int


class DatasourceContributionRanking(_Record):
    by_pair: dict[str, list[RankedDatasource]]
    ranking_spec: dict[str, Any]


class SharedTarget(_Record):
    target_id: str
    per_disease: dict[str, dict[str, Any]]


class TargetIntersection(_Record):
    disease_ids: list[str]
    shared_targets: list[SharedTarget]
    not_shared_target_ids: dict[str, list[str]]
    set_sizes: dict[str, int]
    limitations: list[str]


class ComparisonRow(_Record):
    target_id: str
    overall_association_score: dict[str, float | None]
    datasource_scores: dict[str, dict[str, float | None]]


class AssociationComparison(_Record):
    disease_ids: list[str]
    datasource_ids: list[str]
    rows: list[ComparisonRow]
    limitations: list[str]


class DiseaseTargetDrugEdge(_Record):
    target_id: str
    drug_id: str


class JoinedDiseaseTargetDrug(_Record):
    edges: list[DiseaseTargetDrugEdge]
    targets_without_drug_records: list[str]
    targets_not_queried: list[str]
    failed_targets: dict[str, str]
    limitations: list[str]


class RankedCandidate(_Record):
    ranked_position: int


class RankedCandidates(_Record):
    candidates: list[RankedCandidate]
    source_capability: str
    ranking_spec: dict[str, Any]


class DrugModalityRecord(DrugTargetRecord):
    modality: str | None = None


class DrugModalities(_Record):
    by_drug: dict[str, DrugModalityRecord]
    failed_drugs: dict[str, str]


class ClinicalStageDrug(_Record):
    drug_id: str
    max_clinical_stage: str | None = None


class TargetClinicalStage(_Record):
    drug_record_count: int
    stages_present: list[str]
    drugs: list[ClinicalStageDrug]


class ClinicalStageEvidence(_Record):
    by_target: dict[str, TargetClinicalStage]
    failed_targets: dict[str, str]
    limitations: list[str]


class RankedDisease(_Record):
    disease_id: str
    ranked_position: int


class RankedDiseases(_Record):
    diseases: list[RankedDisease]
    ranking_spec: dict[str, Any]
    limitations: list[str]


class AssociatedDiseaseRow(_Record):
    disease_id: str
    disease_name: str = ""
    target_id: str
    rank: int | None = None
    overall_association_score: float | None = None


class TargetAssociatedDiseases(_Record):
    pairs: list[AssociatedDiseaseRow]
    by_target: dict[str, dict[str, Any]]
    failed_targets: dict[str, str]
    limitations: list[str] = []



class DiseaseRankComparison(_Record):
    disease_id: str
    found_in_ranking: bool
    ranked_position: int | None
    ranked_disease_count: int
    is_top: bool
    limitations: list[str]

# operation -> the model its payload must satisfy (same keys as ADAPTERS)
OUTPUT_MODELS: dict[str, type[BaseModel]] = {
    "resolve_disease": ResolvedEntities,
    "resolve_target": ResolvedEntities,
    "resolve_diseases": ResolvedEntities,
    "resolve_targets": ResolvedEntities,
    "retrieve_exact_pair": ExactPairEvidence,
    "discover_targets": DiscoveredTargets,
    "discover_targets_per_disease": DiscoveredTargetsPerDisease,
    "retrieve_target_drugs": TargetDrugEvidence,
    "retrieve_tractability": TractabilityEvidence,
    "stratify_target_evidence_layers": StratifiedEvidence,
    "rank_targets": RankedTargets,
    "filter_targets": FilteredTargets,
    "retrieve_approved_indications": ApprovedIndications,
    "check_indication": IndicationChecks,
    "resolve_drug": ResolvedEntities,
    "retrieve_drug_targets": DrugTargets,
    "resolve_disease_target": ResolvedEntities,
    "resolve_disease_targets": ResolvedEntities,
    "resolve_disease_and_drugs": ResolvedEntities,
    "retrieve_exact_pairs": ExactPairs,
    "retrieve_datasource_scores": DatasourceScores,
    "rank_datasource_contributions": DatasourceContributionRanking,
    "intersect_targets": TargetIntersection,
    "compare_association_evidence": AssociationComparison,
    "join_disease_target_drug": JoinedDiseaseTargetDrug,
    "rank_candidates": RankedCandidates,
    "retrieve_mechanisms": DrugTargets,
    "retrieve_mechanisms_and_modalities": DrugModalities,
    "retrieve_clinical_stage": ClinicalStageEvidence,
    "rank_diseases": RankedDiseases,
    "retrieve_associated_diseases": TargetAssociatedDiseases,
    "compare_disease_rank": DiseaseRankComparison,
}
