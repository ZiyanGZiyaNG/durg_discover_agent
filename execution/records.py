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



class GeneticEvidenceRow(_Record):
    target_id: str
    genetic_evidence_score: float | None


class GeneticEvidence(_Record):
    disease_id: str
    datatype_id: str
    targets: list[GeneticEvidenceRow]
    limitations: list[str]


class SupportedWithoutDrugRow(_Record):
    target_id: str
    genetic_evidence_score: float
    drug_record_count: int


class SupportedWithoutClinicalDrug(_Record):
    targets: list[SupportedWithoutDrugRow]
    excluded_target_ids: list[str]
    unevaluable_target_ids: list[str]
    filter_spec: dict[str, Any]
    evaluated_on: str
    limitations: list[str]

class GeneticSomaticEvidenceRow(_Record):
    target_id: str
    genetic_evidence_score: float | None
    somatic_evidence_score: float | None


class GeneticSomaticEvidence(_Record):
    disease_id: str
    datatype_ids: list[str]
    targets: list[GeneticSomaticEvidenceRow]
    limitations: list[str]


class FirstInClassRow(_Record):
    target_id: str
    max_genetic_somatic_score: float
    known_drug_count: int


class FirstInClassCandidates(_Record):
    targets: list[FirstInClassRow]
    excluded_target_ids: list[str]
    unevaluable_target_ids: list[str]
    filter_spec: dict[str, Any]
    evaluated_on: str
    limitations: list[str]


class RankedOpportunity(_Record):
    target_id: str
    ranked_position: int
    warnings: list[str]


class RankedOpportunities(_Record):
    opportunities: list[RankedOpportunity]
    ranking_spec: dict[str, Any]
    limitations: list[str]


class DatatypeEntry(_Record):
    datatype_id: str
    datatype_score: float | None
    datasources: list[dict[str, Any]]


class EvidencePair(_Record):
    disease_id: str
    target_id: str
    datatypes: list[DatatypeEntry]


class PairEvidenceByType(_Record):
    pairs: list[EvidencePair]
    evidence_type_filter: list[str] | None
    limitations: list[str]


class RankedEvidenceType(_Record):
    datatype_id: str
    ranked_position: int
    datatype_score: float | None


class EvidenceTypePair(_Record):
    disease_id: str
    target_id: str
    evidence_types: list[RankedEvidenceType]
    top_evidence_type: str | None
    top_is_tied: bool


class EvidenceTypeComparison(_Record):
    pairs: list[EvidenceTypePair]
    ranking_spec: dict[str, Any] | None
    limitations: list[str]


class NormalizedRow(_Record):
    disease_id: str
    target_id: str
    scores: dict[str, float | None]


class NormalizedComparison(_Record):
    dimensions: list[str]
    rows: list[NormalizedRow]
    limitations: list[str]


class DatatypeComparison(_Record):
    datatype_id: str
    scores: dict[str, float | None]
    leader_target_id: str | None
    complete: bool
    missing_target_ids: list[str]


class TargetComparison(_Record):
    disease_id: str
    comparisons: list[DatatypeComparison]
    limitations: list[str]


class DiseaseDrug(_Record):
    drug_id: str
    drug_name: str = ""
    max_clinical_stage: str | None = None


class DiseaseDrugs(_Record):
    disease_id: str
    disease_name: str = ""
    total_drug_count: int
    fetched_drug_count: int
    drugs: list[DiseaseDrug]
    named_drugs_not_found: list[dict[str, Any]]


class DiseaseDrugEvidence(_Record):
    by_disease: dict[str, DiseaseDrugs]
    failed_diseases: dict[str, str]
    limitations: list[str]


class TargetCancerRow(_Record):
    disease_id: str
    disease_name: str = ""
    overall_association_score: float | None = None
    is_cancer: bool | None = None


class TargetCancers(_Record):
    target_id: str
    target_symbol: str = ""
    total_associated_disease_count: int
    fetched_disease_count: int
    diseases: list[TargetCancerRow]
    non_cancer_count: int
    unclassified_count: int
    excluded_index_count: int


class TargetCancerEvidence(_Record):
    by_target: dict[str, TargetCancers]
    failed_targets: dict[str, str]
    cancer_only: bool
    truncated: bool
    limitations: list[str]


class CrossIndicationPath(_Record):
    drug_id: str
    drug_name: str = ""
    target_id: str
    target_symbol: str = ""
    disease_id: str
    disease_name: str = ""
    overall_association_score: float | None = None


class CrossIndicationGroup(_Record):
    drug_id: str
    target_id: str
    other_disease_count: int


class CrossIndicationPaths(_Record):
    drugs: list[dict[str, Any]]
    paths: list[CrossIndicationPath]
    groups: list[CrossIndicationGroup]
    incomplete: dict[str, str]
    limitations: list[str]


class TargetFamilyClass(_Record):
    target_id: str
    target_symbol: str = ""
    is_kinase: bool | None
    classes: list[str]


class KinaseClassification(_Record):
    targets: list[dict[str, Any]]
    by_disease: dict[str, dict[str, Any]]
    failed_diseases: dict[str, str]
    classification: dict[str, TargetFamilyClass]
    kinase_only: bool
    limitations: list[str]


class CoverageGroup(_Record):
    disease_id: str
    disease_name: str = ""
    denominator_target_ids: list[str]
    numerator_target_ids: list[str]
    unevaluable_target_ids: list[str]
    denominator_count: int
    numerator_count: int
    fetched_target_count: int
    total_associated_target_count: int


class CoverageDefinition(_Record):
    definition: dict[str, Any]
    by_disease: dict[str, CoverageGroup]
    aggregation_spec: dict[str, Any] | None
    failed_diseases: dict[str, str] = {}
    truncated: bool
    limitations: list[str]


class CoverageRate(_Record):
    disease_id: str
    disease_name: str = ""
    numerator: int
    denominator: int
    rate: float | None
    unevaluable_count: int


class CoverageRates(_Record):
    definition: dict[str, Any]
    rates: list[CoverageRate]
    failed_diseases: dict[str, str] = {}
    truncated: bool
    limitations: list[str]


class DiseaseRateComparison(_Record):
    definition: dict[str, Any]
    rates: list[CoverageRate]
    differences: list[dict[str, Any]]
    comparable: bool
    failed_diseases: dict[str, str] = {}
    limitations: list[str]


class SafetyEvent(_Record):
    event: str | None = None
    event_id: str | None = None
    datasource: str = ""
    literature: str | None = None
    url: str | None = None
    effects: list[dict[str, Any]]
    biosamples: list[dict[str, Any]]


class TargetSafety(_Record):
    target_id: str
    target_symbol: str = ""
    event_count: int
    events: list[SafetyEvent]


class SafetyEvidence(_Record):
    by_target: dict[str, TargetSafety]
    failed_targets: dict[str, str]
    limitations: list[str]


class TrialTermination(_Record):
    report_id: str
    overall_status: str | None = None
    why_stopped: str | None = None
    reason_recorded: bool
    stop_reason_categories: list[str]
    source: str | None = None
    provider: str | None = None
    url: str | None = None
    title: str | None = None
    year: int | None = None
    phase: str | None = None
    drugs: list[dict[str, Any]]
    diseases: list[str]


class TargetTerminations(_Record):
    target_id: str
    target_symbol: str = ""
    total_drug_row_count: int
    fetched_drug_row_count: int
    report_count_scanned: int
    terminated_count: int
    without_reason_count: int
    terminations: list[TrialTermination]


class TrialTerminationEvidence(_Record):
    by_target: dict[str, TargetTerminations]
    failed_targets: dict[str, str]
    status_filter: list[str]
    truncated: bool
    limitations: list[str]


class ExpressionGroup(_Record):
    datasource_id: str
    datatype_id: str
    unit: str
    level: str
    row_count: int
    top: list[dict[str, Any]]


class TargetExpression(_Record):
    target_id: str
    target_symbol: str = ""
    total_row_count: int
    fetched_row_count: int
    groups: dict[str, ExpressionGroup]
    tissue_index: dict[str, list[dict[str, Any]]]


class NormalTissueExpression(_Record):
    by_target: dict[str, TargetExpression]
    failed_targets: dict[str, str]
    ranking_spec: dict[str, Any] | None
    truncated: bool
    limitations: list[str]


class SafetyExpressionTarget(_Record):
    target_id: str
    target_symbol: str = ""
    risks: list[dict[str, Any]] | None
    terminations: list[dict[str, Any]] | None
    expression_groups: dict[str, Any] | None


class SafetyExpressionIntegration(_Record):
    targets: list[SafetyExpressionTarget]
    sources_present: list[str]
    sources_missing: list[str]
    failed_targets: dict[str, dict[str, str]]
    truncated: bool
    limitations: list[str]

class ClinicalTrialRecord(BaseModel):
    nct_id: str | None = None
    title: str | None = None
    status: str | None = None
    start_date: str | None = None
    completion_date: str | None = None
    why_stopped: str | None = None
    stop_reason_recorded: bool | None = None
    phases: list[str] = []
    study_type: str | None = None
    enrollment: int | None = None
    conditions: list[str] = []
    interventions: list[dict] = []
    sponsor: str | None = None
    url: str | None = None
    arm_groups: list[dict] = []


class ClinicalTrialQuery(BaseModel):
    source: str
    api_version: str
    query: dict
    retrieved_at: str
    total_count: int | None = None
    returned_count: int
    pages_fetched: int
    truncated: bool
    items: list[ClinicalTrialRecord]


class ClinicalTrialEvidence(BaseModel):
    queries: dict[str, ClinicalTrialQuery]
    failed_queries: dict[str, str]
    status_filter: list[str]
    truncated: bool
    limitations: list[str]


class LiteratureRecord(BaseModel):
    source_db: str | None = None
    id: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    title: str | None = None
    authors: str | None = None
    journal: str | None = None
    year: str | int | None = None
    pub_date: str | None = None
    cited_by: int | None = None
    open_access: str | bool | None = None
    abstract: str | None = None
    url: str | None = None


class LiteratureQuery(BaseModel):
    source: str
    query: dict
    retrieved_at: str
    total_count: int | None = None
    returned_count: int
    pages_fetched: int
    truncated: bool
    items: list[LiteratureRecord]


class LiteratureEvidence(BaseModel):
    queries: dict[str, LiteratureQuery]
    failed_queries: dict[str, str]
    truncated: bool
    limitations: list[str]

class InterventionItem(BaseModel):
    name: str | None = None
    type: str | None = None
    role: str
    arm_group_labels: list[str] = []


class InterventionTrial(BaseModel):
    nct_id: str
    title: str | None = None
    status: str | None = None
    phases: list[str] = []
    interventions: list[InterventionItem]
    combination_arms: list[str] = []
    is_combination: bool | None = None
    query_labels: list[str]


class InterventionEvidence(BaseModel):
    trial_count: int
    trials: list[InterventionTrial]
    truncated: bool
    limitations: list[str]


class CategoryCount(BaseModel):
    category: str
    category_count: int
    category_fraction: float


class CategoryDistribution(BaseModel):
    source_row_count: int
    metrics: list[str]
    distributions: dict[str, list[CategoryCount]]
    limitations: list[str]


class RankedCategory(CategoryCount):
    rank: int


class RankedCategories(BaseModel):
    ranked_by: str
    rankings: dict[str, list[RankedCategory]]
    limitations: list[str]

class MechanismClassificationRecord(BaseModel):
    drug_id: str
    drug_name: str | None = None
    modality: str
    mechanism_categories: list[str]
    action_types: list[str]
    mechanism_count: int
    multi_label: bool


class MechanismClassification(BaseModel):
    rule_version: str
    records: list[MechanismClassificationRecord]
    limitations: list[str]

class TerminationClassificationRecord(BaseModel):
    record_id: str | None = None
    origin: str
    status: str | None = None
    drug_names: list[str]
    why_stopped: str | None = None
    categories: list[str]
    matched_terms: list[str]
    source_stop_reason_categories: list[str] = []


class TerminationClassification(BaseModel):
    rule_version: str
    records: list[TerminationClassificationRecord]
    category_counts: dict[str, int]
    limitations: list[str]


class FailedProgram(BaseModel):
    program: str
    terminated_record_count: int
    category_counts: dict[str, int]
    record_ids: list[str]
    has_recorded_safety_or_efficacy_reason: bool


class FailedPrograms(BaseModel):
    program_count: int
    record_count: int
    programs: list[FailedProgram]
    limitations: list[str]


class PhaseFilterRecord(BaseModel):
    target_id: str
    drug_id: str
    drug_name: str | None = None
    max_clinical_stage: str | None = None
    stage_rank: int
    approval_checked: bool


class PhaseFilterExclusion(BaseModel):
    target_id: str
    drug_id: str | None = None
    max_clinical_stage: str | None = None
    reason: str


class PhaseFilterResult(BaseModel):
    disease_id: str | None = None
    disease_name: str | None = None
    min_stage_rank: int
    records: list[PhaseFilterRecord]
    kept_target_ids: list[str]
    excluded: list[PhaseFilterExclusion]
    failed: dict[str, dict[str, str]]
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
    "retrieve_genetic_evidence": GeneticEvidence,
    "filter_supported_without_clinical_drug": SupportedWithoutClinicalDrug,    
    "retrieve_genetic_and_somatic_evidence": GeneticSomaticEvidence,
    "filter_first_in_class_candidates": FirstInClassCandidates,
    "rank_opportunities": RankedOpportunities,
    "retrieve_datasource_evidence": PairEvidenceByType,
    "retrieve_evidence_by_type": PairEvidenceByType,
    "compare_evidence_types": EvidenceTypeComparison,
    "normalize_comparison_dimensions": NormalizedComparison,
    "compare_targets": TargetComparison,
    "retrieve_disease_drugs": DiseaseDrugEvidence,
    "retrieve_target_cancers": TargetCancerEvidence,
    "join_cross_indication_paths": CrossIndicationPaths,
    "classify_kinase_targets": KinaseClassification,
    "define_coverage_denominators": CoverageDefinition,
    "calculate_coverage_rates": CoverageRates,
    "compare_disease_rates": DiseaseRateComparison,
    "resolve_target_variants": ResolvedEntities,
    "retrieve_safety_evidence": SafetyEvidence,
    "retrieve_trial_terminations": TrialTerminationEvidence,
    "retrieve_normal_tissue_expression": NormalTissueExpression,
    "integrate_safety_and_expression": SafetyExpressionIntegration,
    "retrieve_clinical_trials": ClinicalTrialEvidence,
    "retrieve_trial_status_and_stop_reasons": ClinicalTrialEvidence, 
    "retrieve_negative_publications": LiteratureEvidence,
    "retrieve_interventions": InterventionEvidence,
    "aggregate_category_distribution": CategoryDistribution,
    "rank_mechanism_categories": RankedCategories,
    "classify_mechanism_categories": MechanismClassification,
    "classify_termination_reasons": TerminationClassification,
    "aggregate_failed_programs": FailedPrograms,
    "filter_phase2_not_pdac_approved": PhaseFilterResult,
}
