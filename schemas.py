"""Stable data contracts exchanged by the target-evidence agents."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


REQUIRED_RESEARCH_SOURCES = {
    "open_targets",
    "europe_pmc",
    "clinicaltrials_gov",
    "chembl",
}


class DiseaseTargetInput(BaseModel):
    disease: str = Field(min_length=1, description="Disease name")
    target: str = Field(min_length=1, description="Gene symbol or target name")


class DiseaseDiscoveryInput(BaseModel):
    disease: str = Field(min_length=1, description="Disease name")
    top_n: int = Field(default=10, ge=1, le=100)


class DataSourceEvidence(BaseModel):
    datasource_id: str
    score: float = Field(ge=0, le=1)


class CandidateTarget(BaseModel):
    rank: int = Field(ge=1)
    target_id: str
    target_symbol: str
    target_name: str
    overall_association_score: float = Field(ge=0, le=1)
    datasource_scores: list[DataSourceEvidence] = Field(default_factory=list)


class TargetDiscoveryResult(BaseModel):
    schema_version: str = "1.0"
    mode: Literal["disease_only_target_discovery"] = "disease_only_target_discovery"
    disease_query: str
    disease_id: str
    disease_name: str
    ranking_method: Literal["Open Targets overall association score"] = (
        "Open Targets overall association score"
    )
    requested_target_count: int = Field(ge=1, le=100)
    returned_target_count: int = Field(ge=0)
    total_associated_target_count: int = Field(ge=0)
    targets: list[CandidateTarget] = Field(default_factory=list)
    source: str = "Open Targets Platform GraphQL API"
    source_url: str = "https://platform.opentargets.org/"
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    score_warning: str = (
        "Open Targets overall association scores are initial ranking references, "
        "not causality, efficacy, clinical-success, or project-confidence probabilities."
    )
    limitations: list[str] = Field(default_factory=list)


class Agent1DiscoveryRunOutput(BaseModel):
    producer: Literal["agent1"]
    run_id: UUID
    generated_at: datetime
    target_discovery: TargetDiscoveryResult


class VerifiedEntity(BaseModel):
    query: str
    entity_id: str
    canonical_name: str
    symbol: str | None = None
    verified_synonyms: list[str] = Field(default_factory=list)
    source: Literal["Open Targets"] = "Open Targets"


class QueryManifest(BaseModel):
    """Machine-checkable contract created before any evidence retrieval."""

    schema_version: Literal["1.0"] = "1.0"
    manifest_id: UUID = Field(default_factory=uuid4)
    original_question: str = Field(min_length=1)
    query_type: Literal[
        "single_hop",
        "multi_hop",
        "evidence_directed",
        "filter_and_rank",
        "comparison_and_aggregation",
        "safety_and_negative_knowledge",
    ]
    disease: VerifiedEntity
    targets: list[VerifiedEntity] = Field(min_length=1, max_length=10)
    query_path: list[Literal["disease", "target", "drug", "evidence"]] = Field(
        min_length=2
    )
    target_expansion_strategy: Literal[
        "specified_target",
        "select_then_expand",
        "expand_all_top10",
    ]
    required_evidence: list[str] = Field(min_length=1)
    selected_sources: list[
        Literal["open_targets", "europe_pmc", "clinicaltrials_gov", "chembl"]
    ] = Field(min_length=1)
    filters: dict[str, Any] = Field(default_factory=dict)
    ranking: dict[str, Any] | None = None
    selection_reason: str = Field(min_length=1)
    created_before_retrieval: Literal[True] = True
    generated_at: datetime

    @model_validator(mode="after")
    def validate_manifest_strategy(self) -> "QueryManifest":
        if len(set(self.selected_sources)) != len(self.selected_sources):
            raise ValueError("selected_sources cannot contain duplicates")
        if self.target_expansion_strategy == "specified_target" and len(self.targets) != 1:
            raise ValueError("specified_target requires exactly one target")
        if self.target_expansion_strategy == "expand_all_top10" and len(self.targets) != 10:
            raise ValueError("expand_all_top10 requires exactly 10 targets")
        return self


class PlannedEntityV2(BaseModel):
    """Entity mentioned by the user and carried into a plan without invention."""

    entity_ref: str = Field(min_length=1)
    entity_type: Literal["disease", "target", "drug", "tissue", "mechanism"]
    mention: str = Field(min_length=1)
    entity_id: str | None = None
    canonical_name: str | None = None
    resolution_status: Literal["verified", "ambiguous", "unresolved"]


class RankingSpecV2(BaseModel):
    """Deterministic ranking contract with conservative legacy normalization."""

    model_config = ConfigDict(extra="forbid")

    primary_metric: str = Field(min_length=1)
    direction: Literal["ascending", "descending"]
    secondary_metrics: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_ranking(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if "criteria" in value:
            raise ValueError(
                "ranking.criteria is ambiguous; specify primary_metric, direction, "
                "and optional secondary_metrics"
            )

        normalized = dict(value)
        direction_aliases = {
            "asc": "ascending",
            "ascending": "ascending",
            "desc": "descending",
            "descending": "descending",
        }
        if "primary_metric" in normalized:
            direction = normalized.get("direction", normalized.get("order"))
            if isinstance(direction, str):
                normalized["direction"] = direction_aliases.get(
                    direction.casefold(), direction
                )
            normalized.pop("order", None)
            return normalized

        metric = next(
            (
                normalized[key]
                for key in ("sort_by", "rank_by", "order_by")
                if isinstance(normalized.get(key), str)
            ),
            None,
        )
        secondary: list[str] = []
        if metric is None and isinstance(normalized.get("sort"), list):
            metrics = [item for item in normalized["sort"] if isinstance(item, str)]
            if metrics:
                metric, secondary = metrics[0], metrics[1:]

        direction = normalized.get("direction", normalized.get("order"))
        if metric is not None:
            return {
                "primary_metric": metric,
                "direction": direction_aliases.get(
                    direction.casefold(), direction
                )
                if isinstance(direction, str)
                else "descending",
                "secondary_metrics": secondary,
            }

        # Backward compatibility for the former {metric: direction} form.
        if len(normalized) == 1:
            legacy_metric, legacy_direction = next(iter(normalized.items()))
            if isinstance(legacy_direction, str) and legacy_direction.casefold() in direction_aliases:
                return {
                    "primary_metric": legacy_metric,
                    "direction": direction_aliases[legacy_direction.casefold()],
                    "secondary_metrics": [],
                }
        return normalized

class StepInputRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str = Field(min_length=1)
    output: str = Field(min_length=1)



class ResearchPlanStepV2(BaseModel):
    """One executable or analytical operation in a research plan."""

    model_config = ConfigDict(validate_assignment=True)

    step_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    operation_status: Literal["approved", "proposed"] = "approved"
    proposed_operation_name: str | None = None
    proposal_reason: str | None = None
    description: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    entity_refs: list[str] = Field(default_factory=list)
    sources: list[
        Literal["open_targets", "europe_pmc", "clinicaltrials_gov", "chembl"]
    ] = Field(default_factory=list)
    outputs: list[str] = Field(min_length=1)
    filters: dict[str, Any] = Field(default_factory=dict)
    ranking: RankingSpecV2 | None = None
    aggregation: dict[str, Any] | None = None
    safeguards: list[str] = Field(default_factory=list)
    provides: list[str] = []
    inputs: dict[str, StepInputRef] = {}

    @model_validator(mode="after")
    def validate_operation_review_state(self) -> "ResearchPlanStepV2":
        if self.operation_status == "proposed":
            if self.operation != "proposed_new_operation":
                raise ValueError(
                    "proposed operations must use operation='proposed_new_operation'"
                )
            if not self.proposed_operation_name or not self.proposal_reason:
                raise ValueError(
                    "proposed operations require proposed_operation_name and proposal_reason"
                )
        elif self.proposed_operation_name is not None or self.proposal_reason is not None:
            raise ValueError(
                "approved operations cannot include proposal fields"
            )
        return self


class EntityResolution(BaseModel):
    status: Literal["verified", "ambiguous", "unresolved"]
    entity_id: str | None = None
    canonical_name: str | None = None
    candidates: list[dict[str, str]] = Field(default_factory=list)


class QueryCharacteristicsV2(BaseModel):
    """Descriptive labels for indexing; they do not determine the plan."""

    primary: Literal[
        "single_hop",
        "multi_hop",
        "evidence_directed",
        "filter_and_rank",
        "comparison_and_aggregation",
        "safety_and_negative_knowledge",
    ]
    secondary: list[
        Literal[
            "single_hop",
            "multi_hop",
            "evidence_directed",
            "filter_and_rank",
            "comparison_and_aggregation",
            "safety_and_negative_knowledge",
        ]
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def characteristics_must_be_unique(self) -> "QueryCharacteristicsV2":
        if self.primary in self.secondary or len(set(self.secondary)) != len(self.secondary):
            raise ValueError("query characteristics must be unique")
        return self


class QueryManifestV2(BaseModel):
    """Plan-first contract for complex natural-language research questions."""

    _reference_corrections: list[str] = PrivateAttr(default_factory=list)

    schema_version: Literal["2.0"] = "2.0"
    manifest_id: UUID = Field(default_factory=uuid4)
    original_question: str = Field(min_length=1)
    research_objective: str = Field(min_length=1)
    entities: list[PlannedEntityV2] = Field(min_length=1)
    plan_steps: list[ResearchPlanStepV2] = Field(min_length=1)
    evidence_requirements: list[str] = Field(min_length=1)
    completion_criteria: list[str] = Field(min_length=1)
    global_safeguards: list[str] = Field(min_length=1)
    characteristics: QueryCharacteristicsV2
    created_before_retrieval: Literal[True] = True
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def validate_plan_graph(self) -> "QueryManifestV2":
        entity_refs = [entity.entity_ref for entity in self.entities]
        if len(set(entity_refs)) != len(entity_refs):
            raise ValueError("entity_ref values must be unique")
        step_ids = [step.step_id for step in self.plan_steps]
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("step_id values must be unique")
        known_steps: set[str] = set()
        ancestors_by_step: dict[str, set[str]] = {}
        known_entities = set(entity_refs)
        output_producers: dict[str, list[str]] = {}
        for step in self.plan_steps:
            unknown_dependencies = set(step.depends_on) - known_steps
            if unknown_dependencies:
                raise ValueError(
                    f"step {step.step_id} has non-prior dependencies: "
                    f"{sorted(unknown_dependencies)}"
                )
            ancestors = set(step.depends_on)
            for dependency in step.depends_on:
                ancestors.update(ancestors_by_step[dependency])
            unknown_entities = set(step.entity_refs) - known_entities
            removable_output_refs: set[str] = set()
            for reference in unknown_entities:
                producers = output_producers.get(reference, [])
                if len(producers) == 1 and producers[0] in ancestors:
                    removable_output_refs.add(reference)
                    self._reference_corrections.append(
                        f"{step.step_id}: removed upstream-step output {reference!r} "
                        "from entity_refs; depends_on preserves the data flow"
                    )
            if removable_output_refs:
                step.entity_refs = [
                    reference
                    for reference in step.entity_refs
                    if reference not in removable_output_refs
                ]
                unknown_entities -= removable_output_refs
            if unknown_entities:
                raise ValueError(
                    f"step {step.step_id} references unknown entities: "
                    f"{sorted(unknown_entities)}"
                )
            known_steps.add(step.step_id)
            ancestors_by_step[step.step_id] = ancestors
            for output in step.outputs:
                output_producers.setdefault(output, []).append(step.step_id)
        return self

    @property
    def reference_corrections(self) -> list[str]:
        return list(self._reference_corrections)


class PlannedToolQuery(BaseModel):
    datasource: Literal[
        "open_targets", "europe_pmc", "clinicaltrials_gov", "chembl"
    ]
    tool_name: Literal[
        "open_targets_search",
        "europe_pmc_search",
        "clinicaltrials_gov_search",
        "chembl_search",
    ]
    query_purpose: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    verified_search_terms: list[str] = Field(default_factory=list)
    unvalidated_search_terms: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_question: str = Field(min_length=1)
    execution_order: int = Field(ge=1)
    collection_policy_status: Literal["configured", "not_configured"]

    @model_validator(mode="after")
    def datasource_must_match_tool(self) -> "PlannedToolQuery":
        expected = {
            "open_targets": "open_targets_search",
            "europe_pmc": "europe_pmc_search",
            "clinicaltrials_gov": "clinicaltrials_gov_search",
            "chembl": "chembl_search",
        }[self.datasource]
        if self.tool_name != expected:
            raise ValueError(f"{self.datasource} must use {expected}")
        if self.collection_policy_status != "not_configured":
            raise ValueError("collection policy is not configured in the current phase")
        return self


class TargetResearchSubplan(BaseModel):
    target: VerifiedEntity
    objective: str = Field(min_length=1)
    queries: list[PlannedToolQuery] = Field(min_length=1)
    anticipated_data_gaps: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def require_registered_unique_sources(self) -> "TargetResearchSubplan":
        sources = {query.datasource for query in self.queries}
        if not sources or not sources.issubset(REQUIRED_RESEARCH_SOURCES):
            raise ValueError("queries must use at least one registered source")
        if len(sources) != len(self.queries):
            raise ValueError("a target subplan cannot repeat a datasource")
        return self


class Agent1ResearchPlan(BaseModel):
    schema_version: Literal["1.1"] = "1.1"
    producer: Literal["agent1_research_planner"] = "agent1_research_planner"
    mode: Literal["disease_only", "disease_target"]
    original_question: str
    query_manifest: QueryManifest
    disease: VerifiedEntity
    overall_objective: str = Field(min_length=1)
    required_sources: list[
        Literal["open_targets", "europe_pmc", "clinicaltrials_gov", "chembl"]
    ]
    target_subplans: list[TargetResearchSubplan] = Field(min_length=1)
    first_round_completion_criteria: list[str] = Field(min_length=1)
    supplemental_round_policy: Literal["at_most_one_round"] = "at_most_one_round"
    generated_at: datetime
    model_provider: str
    model_name: str
    validation_repair_count: int = Field(default=0, ge=0, le=2)

    @model_validator(mode="after")
    def validate_plan_scope(self) -> "Agent1ResearchPlan":
        if not self.required_sources or not set(self.required_sources).issubset(
            REQUIRED_RESEARCH_SOURCES
        ):
            raise ValueError("required_sources must contain registered sources")
        queried_sources = {
            query.datasource
            for subplan in self.target_subplans
            for query in subplan.queries
        }
        if queried_sources != set(self.required_sources):
            raise ValueError("required_sources must match the planned tool queries")
        if self.mode == "disease_only" and len(self.target_subplans) > 10:
            raise ValueError("disease-only plan cannot contain more than 10 targets")
        if self.mode == "disease_target" and len(self.target_subplans) != 1:
            raise ValueError("disease-target plan must contain exactly one target")
        manifest_target_ids = {target.entity_id for target in self.query_manifest.targets}
        subplan_target_ids = {subplan.target.entity_id for subplan in self.target_subplans}
        if manifest_target_ids != subplan_target_ids:
            raise ValueError("query_manifest targets must match target_subplans")
        if self.query_manifest.disease.entity_id != self.disease.entity_id:
            raise ValueError("query_manifest disease must match plan disease")
        if set(self.query_manifest.selected_sources) != set(self.required_sources):
            raise ValueError("query_manifest selected_sources must match required_sources")
        return self


class PlannedExecutionOutput(BaseModel):
    producer: Literal["agent1"] = "agent1"
    run_id: UUID
    generated_at: datetime
    research_plan: Agent1ResearchPlan
    execution_status: Literal["collection_policy_not_configured"]
    executed_tools: list[str] = Field(default_factory=list)
    message: str


class DataSourceEvidenceDetails(BaseModel):
    """Underlying evidence rows contributing to one aggregated datasource score."""

    datasource_id: str
    aggregated_score: float | None = Field(default=None, ge=0, le=1)
    total_evidence_count: int = Field(ge=0)
    fetched_evidence_count: int = Field(ge=0)
    truncated: bool
    estimated_complete_size_bytes: int | None = Field(default=None, ge=0)
    evidence_fields: list[str] = Field(default_factory=list)
    collection_warnings: list[str] = Field(default_factory=list)
    records: list[dict[str, Any]] = Field(default_factory=list)


class FieldProfile(BaseModel):
    field_name: str
    plain_language_description: str
    populated_record_count: int = Field(ge=0)
    populated_percentage: float = Field(ge=0, le=100)
    value_type: str
    example_values: list[Any] = Field(default_factory=list)


class DrugEvidenceGroup(BaseModel):
    canonical_drug_name: str
    source_name_variants: list[str] = Field(default_factory=list)
    evidence_record_count: int = Field(ge=0)
    clinical_stage_counts: dict[str, int] = Field(default_factory=dict)
    highest_clinical_stage: str | None = None
    evidence_score_counts: dict[str, int] = Field(default_factory=dict)
    report_ids: list[str] = Field(default_factory=list)
    literature_ids: list[str] = Field(default_factory=list)
    direction_on_target_counts: dict[str, int] = Field(default_factory=dict)
    direction_on_trait_counts: dict[str, int] = Field(default_factory=dict)
    stopped_trial_record_count: int = Field(ge=0)
    negative_stop_record_count: int = Field(ge=0)
    safety_stop_record_count: int = Field(ge=0)
    stop_reason_categories: dict[str, int] = Field(default_factory=dict)
    stop_reason_examples: list[str] = Field(default_factory=list)


class DataSourceReviewView(BaseModel):
    datasource_id: str
    datasource_explanation: str
    aggregated_score: float | None = None
    score_interpretation_warning: str
    total_records: int = Field(ge=0)
    fetched_records: int = Field(ge=0)
    complete: bool
    raw_schema_field_count: int = Field(ge=0)
    populated_field_count: int = Field(ge=0)
    field_profiles: list[FieldProfile] = Field(default_factory=list)
    drug_groups: list[DrugEvidenceGroup] = Field(default_factory=list)
    review_questions: list[str] = Field(default_factory=list)


class ExpertReviewView(BaseModel):
    schema_version: str = "1.0"
    purpose: str = (
        "Human-readable inventory for domain experts to decide future database "
        "classification. It does not replace raw evidence."
    )
    disease: str
    disease_id: str
    target: str
    target_id: str
    datasource_reviews: list[DataSourceReviewView] = Field(default_factory=list)


class SynthesizerInterpretation(BaseModel):
    schema_version: str = "1.0"
    producer: Literal["agent1_evidence_synthesizer"] = "agent1_evidence_synthesizer"
    factual_inventory_summary: list[str] = Field(default_factory=list)
    potentially_supportive_signals: list[str] = Field(default_factory=list)
    potentially_conflicting_or_negative_signals: list[str] = Field(default_factory=list)
    missing_or_unresolved_information: list[str] = Field(default_factory=list)
    questions_for_domain_experts: list[str] = Field(default_factory=list)
    interpretation_warning: str = (
        "Model-generated interpretation of supplied evidence inventory; not a "
        "database fact, clinical conclusion, or success probability."
    )


class DiseaseTargetEvidence(BaseModel):
    schema_version: str = "1.1"
    disease_query: str
    target_query: str
    disease_id: str
    disease_name: str
    target_id: str
    target_symbol: str
    target_name: str
    overall_association_score: float | None = Field(default=None, ge=0, le=1)
    datasource_scores: list[DataSourceEvidence] = Field(default_factory=list)
    datasource_evidence: list[DataSourceEvidenceDetails] = Field(default_factory=list)
    rank_within_fetched_targets: int | None = None
    fetched_target_count: int
    source: str = "Open Targets Platform GraphQL API"
    source_url: str = "https://platform.opentargets.org/"
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    limitations: list[str] = Field(default_factory=list)


class Agent1RunOutput(BaseModel):
    """Strict handoff contract proving evidence came from an Agent 1 run."""

    producer: Literal["agent1"]
    run_id: UUID
    generated_at: datetime
    expert_review_view: ExpertReviewView | None = None
    model_interpretation: SynthesizerInterpretation | None = None
    agent_1_evidence: DiseaseTargetEvidence


class EvidenceInterpretation(BaseModel):
    schema_version: str = "1.0"
    disease: str
    target: str
    summary: str
    supporting_evidence: list[str]
    weak_or_missing_evidence: list[str]
    data_limitations: list[str]
    confidence_score_status: str = "pending_research_team_definition"
    confidence_score: float | None = None
    research_disclaimer: str = (
        "Research-use output only. This is not a clinical recommendation."
    )
