"""Hidden, deterministic rubrics for evaluating plan quality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas import QueryManifestV2


SourceName = Literal["open_targets", "europe_pmc", "clinicaltrials_gov", "chembl"]
EntityType = Literal["disease", "target", "drug", "tissue", "mechanism"]
# Path roles may also work on the association edge itself.
RowType = Literal["disease", "target", "drug", "tissue", "mechanism", "association"]


class ExpectedEntity(BaseModel):
    entity_type: Literal["disease", "target", "drug", "tissue", "mechanism"]
    mention: str = Field(min_length=1)


class PlannerAcceptanceCase(BaseModel):
    """Human-authored source specification retained for traceability."""

    case_id: str = Field(min_length=1)
    evaluation_set: Literal["development", "internal_holdout"]
    question: str = Field(min_length=1)
    research_intent: str = Field(min_length=1)
    required_entities: list[ExpectedEntity] = Field(min_length=1)
    required_operations: list[str] = Field(min_length=1)
    required_dependencies: list[tuple[str, str]] = Field(default_factory=list)
    required_sources: list[SourceName] = Field(min_length=1)
    required_safeguards: list[str] = Field(min_length=1)
    required_filters: list[str] = Field(default_factory=list)
    required_ranking: list[str] = Field(default_factory=list)
    required_aggregation: list[str] = Field(default_factory=list)
    allowed_characteristics: list[str] = Field(min_length=1)
    # The question set's own labels (entities, path, stated conditions) in the
    # slot planner's table shape; validated by slot_planner.QuerySlots.
    expected_slots: dict[str, Any] | None = None

    @model_validator(mode="after")
    def requirements_must_be_consistent(self) -> "PlannerAcceptanceCase":
        operations = set(self.required_operations)
        for before, after in self.required_dependencies:
            if before not in operations or after not in operations:
                raise ValueError(
                    f"dependency {before}->{after} must reference required_operations"
                )
        return self


class RequiredCapability(BaseModel):
    capability_id: str
    description: str
    equivalent_operations: list[str] = Field(min_length=1)


class ForbiddenCondition(BaseModel):
    condition_id: str
    description: str
    severity: Literal["blocking", "autocorrect"] = "blocking"


class OptionalCapability(BaseModel):
    capability_id: str
    description: str
    equivalent_operations: list[str] = Field(min_length=1)


class PlanAcceptanceRubric(BaseModel):
    """Hidden rubric: multiple equivalent plans may satisfy one question."""

    case_id: str
    question: str
    research_intent: str
    required_entities: list[ExpectedEntity]
    required_capabilities: list[RequiredCapability]
    required_dependencies: list[tuple[str, str]] = Field(default_factory=list)
    required_sources: list[SourceName]
    required_safeguards: list[str]
    required_filters: list[str] = Field(default_factory=list)
    required_ranking: list[str] = Field(default_factory=list)
    required_aggregation: list[str] = Field(default_factory=list)
    forbidden_conditions: list[ForbiddenCondition]
    optional_capabilities: list[OptionalCapability] = Field(default_factory=list)
    allowed_characteristics: list[str]


# Equivalences are conservative and expanded only after human review.
_REVIEWED_OPERATION_ALIASES: dict[str, list[str]] = {
    "resolve_disease": ["entity_resolution", "normalize_disease"],
    "resolve_target": ["normalize_target"],
    "resolve_drug": ["normalize_drug"],
    "resolve_drug": ["normalize_drug"],
    "compare_disease_rank": ["compare_pdac_rank"],
    "discover_targets": [
        "discover_targets_per_disease",
        "retrieve_disease_targets",
        "search_associated_targets",
        "target_retrieval",
    ],
    "rank_targets": ["prioritize_targets", "order_targets"],
    "filter_phase2_not_approved_for_disease": ["filter_phase2_not_pdac_approved"],
    "retrieve_target_drugs": [
        "retrieve_drugs_for_targets",
        "search_target_drugs",
    ],
    "retrieve_exact_pair": [
        "retrieve_disease_target_pair",
        "query_exact_association",
    ],
    "retrieve_exact_pairs": [
        "retrieve_disease_target_pairs",
        "query_exact_associations",
    ],
}

# This is capability documentation, not a per-question answer. The blind
# Planner may choose any subset and may propose a custom operation.
_CANONICAL_OPERATION_IDS = sorted(
    {
        "resolve_disease",
        "resolve_target",
        "resolve_drug",
        "resolve_diseases",
        "resolve_targets",
        "resolve_disease_target",
        "resolve_disease_targets",
        "resolve_disease_and_drugs",
        "resolve_target_variants",
        "discover_targets",
        "discover_targets_per_disease",
        "retrieve_exact_pair",
        "retrieve_exact_pairs",
        "retrieve_associated_diseases",
        "retrieve_target_drugs",
        "retrieve_drug_targets",
        "retrieve_disease_drugs",
        "retrieve_target_cancers",
        "retrieve_approved_indications",
        "retrieve_mechanisms",
        "retrieve_mechanisms_and_modalities",
        "retrieve_datasource_scores",
        "retrieve_datasource_evidence",
        "retrieve_evidence_by_type",
        "retrieve_genetic_evidence",
        "retrieve_genetic_and_somatic_evidence",
        "retrieve_tractability",
        "retrieve_clinical_trials",
        "retrieve_clinical_stage",
        "retrieve_interventions",
        "retrieve_trial_status_and_stop_reasons",
        "retrieve_trial_terminations",
        "retrieve_safety_evidence",
        "retrieve_normal_tissue_expression",
        "retrieve_negative_publications",
        "check_indication",
        "filter_targets",
        "filter_first_in_class_candidates",
        "filter_phase2_not_approved_for_disease",
        "filter_supported_without_clinical_drug",
        "rank_targets",
        "rank_diseases",
        "rank_candidates",
        "rank_opportunities",
        "rank_datasource_contributions",
        "rank_mechanism_categories",
        "compare_targets",
        "compare_association_evidence",
        "compare_evidence_types",
        "compare_disease_rank",
        "compare_disease_rates",
        "normalize_comparison_dimensions",
        "intersect_targets",
        "join_disease_target_drug",
        "join_cross_indication_paths",
        "integrate_safety_and_expression",
        "classify_kinase_targets",
        "classify_mechanism_categories",
        "classify_termination_reasons",
        "aggregate_category_distribution",
        "aggregate_failed_programs",
        "stratify_target_evidence_layers",
        "define_coverage_denominators",
        "calculate_coverage_rates",
        "compare_disease_rank",
    }
)

# Planner-facing semantics describe research intent, not executable tool APIs.
# Keep each definition short so the full catalog remains practical for a local LLM.
OPERATION_SEMANTICS: dict[str, str] = {
    "aggregate_category_distribution": "輸入已分類紀錄，計算各類別數量或比例；不用來建立證據層級。",
    "aggregate_failed_programs": "依藥物計畫與終止原因彙整失敗紀錄；缺少結果不等於失敗。",
    "calculate_coverage_rates": "使用已定義的分子與分母計算覆蓋率，不自行改變分母。",
    "check_indication": "對比已取得的核准適應症與 entity_refs 指定的疾病;在已查來源找不到只能標為『未確認』，不得視為『未核准』。",
    "classify_kinase_targets": "將已解析標的依家族註記判定為 kinase 或非 kinase。",
    "classify_mechanism_categories": "依預先定義規則將藥物機制或 modality 分類，組合療法須遵守多標籤政策。",
    "classify_termination_reasons": "將明確的試驗或計畫終止原因分類為安全性、療效或其他。",
    "compare_association_evidence": "以相同維度並列比較多個 disease-target 關聯證據，不建立未定義的綜合分數。",
    "compare_disease_rates": "比較各疾病已計算的比率，必須使用一致分母定義。",
    "compare_evidence_types": "比較不同證據類型的分布或貢獻，不把紀錄數直接當 datasource score。",
    "compare_disease_rank": "檢查 entity_refs 指定的疾病在已排序疾病中的相對位置，不重新查詢疾病清單；不在已取得範圍只能標為『未在已取得清單』，不得視為無關聯。",
    "compare_targets": "依一致比較維度並列多個標的的證據，不把缺失值視為零。",
    "define_coverage_denominators": "在計算覆蓋率前明確定義符合資格的總體、分子與分母。",
    "discover_targets": "輸入已解析疾病，取得 disease-associated targets；僅表示疾病關聯，不等同治療有效或可成藥。",
    "discover_targets_per_disease": "對多個已解析疾病分別取得 associated targets，保留各疾病獨立結果。",
    "filter_first_in_class_candidates": "依強證據與來源限定的無已知藥物條件篩選候選，不宣稱絕對不存在藥物。",
    "filter_phase2_not_approved_for_disease": "保留 Phase 2+ 且未在已查來源確認指定疾病核准的藥物或標的。",
    "filter_supported_without_clinical_drug": "保留有指定證據且在已查來源與日期未發現臨床藥物的標的。",
    "filter_targets": "依 Plan 明列的可執行條件篩選標的；不得以含糊的『足夠證據』代替門檻。",
    "integrate_safety_and_expression": "並列整合安全性、終止與正常組織表現，保持事實與風險推論分離。",
    "intersect_targets": "計算各疾病 target 集合交集，保留每個 target 在各疾病的原始證據。",
    "join_cross_indication_paths": "連接 disease-drug-target-other disease 路徑，不把關聯誤稱核准適應症。",
    "join_disease_target_drug": "連接 disease-target 與 target-drug 邊並保留來源；不推論藥物對疾病有效。",
    "normalize_comparison_dimensions": "在比較前統一證據類別、缺失值與分母定義。",
    "rank_candidates": "以明列的主要與次要指標排序候選，ranking 必須使用固定 schema。",
    "rank_datasource_contributions": "依 datasource score 或明定貢獻指標排序來源，不依紀錄數替代。",
    "rank_diseases": "依明列的 disease-target 關聯指標排序疾病。",
    "rank_mechanism_categories": "在完成分類與計數後依 category_count 排序機制類別。",
    "rank_opportunities": "依證據強度與明列警訊排序開發機會，不把排名解釋為成功機率。",
    "rank_targets": "對已取得的標的使用固定 ranking schema 排序；不負責擷取或分層證據。",
    "resolve_disease": "將一個疾病 mention 解析為候選標準實體；Plan 階段不得自行宣稱 ID 已驗證。",
    "resolve_disease_and_drugs": "分別解析一個疾病與多個藥物 mention，保留歧義。",
    "resolve_disease_target": "分別解析一個疾病與一個標的，供 exact-pair 查詢使用。",
    "resolve_disease_targets": "分別解析一個疾病與多個標的，供多個 exact-pair 查詢使用。",
    "resolve_diseases": "分別解析多個疾病 mention，不合併不同疾病範圍。",
    "resolve_drug": "將藥物 mention 解析為候選標準實體，保留鹽型、品牌或成分歧義。",
    "resolve_target": "將基因或標的 mention 解析為候選標準實體，保留同義詞歧義。",
    "resolve_target_variants": "解析 target 與特定變異型，變異證據不得回退為未分型 target 證據。",
    "resolve_targets": "分別解析多個 target mention，不合併不同蛋白或基因。",
    "retrieve_approved_indications": "取得資料庫記錄的核准適應症；正式核准結論仍需官方來源確認。",
    "retrieve_associated_diseases": "輸入已解析 target，取得 target-associated diseases 與關聯證據。",
    "retrieve_clinical_stage": "取得 target-drug 或 drug program 的臨床開發階段；phase 不等於核准。",
    "retrieve_clinical_trials": "依疾病、藥物或標的條件取得臨床試驗與狀態、日期及來源。",
    "retrieve_datasource_evidence": "取得 exact pair 的 datasource-level 原始證據與證據類型。",
    "retrieve_datasource_scores": "取得 exact pair 的 datasource-level scores；score 不等於紀錄數或機率。",
    "retrieve_disease_drugs": "取得與疾病用途相關的藥物紀錄，並區分核准、試驗與關聯。",
    "retrieve_drug_targets": "輸入已解析藥物，取得其已知作用標的與 mechanism evidence。",
    "retrieve_evidence_by_type": "依指定 evidence type 取得可比較的證據並保留 datasource。",
    "retrieve_exact_pair": "以已解析 disease ID 與 target ID 直接查詢單一關聯，不用排名窗口代替。",
    "retrieve_exact_pairs": "對多個已解析 disease-target 配對分別執行 exact-pair 查詢。",
    "retrieve_genetic_and_somatic_evidence": "取得遺傳與體細胞突變證據並保持兩類可區分。",
    "retrieve_genetic_evidence": "取得 disease-target genetic evidence；不包含未指定的其他證據類型。",
    "retrieve_interventions": "從已取得臨床試驗抽取介入措施、組合與角色。",
    "retrieve_mechanisms": "取得藥物-target 作用機制；不推論疾病療效或核准。",
    "retrieve_mechanisms_and_modalities": "取得介入措施的 mechanism 與 modality，供一致分類使用。",
    "retrieve_negative_publications": "查找明確報告負面、安全性或療效失敗的文獻；無文獻不等於成功。",
    "retrieve_normal_tissue_expression": "取得 target 在正常組織的表現資料；高表現不直接等於毒性。",
    "retrieve_safety_evidence": "取得 target 或藥物的已知安全性證據與來源，不以機制推測代替事實。",
    "retrieve_target_cancers": "輸入 target，取得其關聯癌別；關聯不代表治療適應症。",
    "retrieve_target_drugs": "輸入已解析 target，取得已知 target-drug 關係與證據；不負責 tractability、疾病療效或核准。",
    "retrieve_tractability": "輸入已解析 target，取得可成藥性與小分子 tractability 證據；不表示已有藥物或療效。",
    "retrieve_trial_status_and_stop_reasons": "取得試驗狀態及明確停止原因，缺少結果不得推定失敗。",
    "retrieve_trial_terminations": "取得已終止試驗及原始終止理由，不自行推測原因。",
    "stratify_target_evidence_layers": "輸入疾病關聯、target-drug 與 tractability 證據，分層輸出疾病相關標的及具轉譯支持標的；不做類別計數。",
}


class PathRole(BaseModel):
    """Where an operation sits on a question's path (read by the slot planner).

    kind="resolve":   resolves a named entity of subject_type ("variant" name
                      resolves a gene named with mutations).
    kind="hop":       walks from subject_type to object_type; name is the
                      relation, default marks the hop used when none is named.
    kind="attribute": extra evidence about rows of subject_type.
    kind="rank":      orders rows of subject_type.
    kind="check":     tests a named entity of subject_type against an
                      earlier result.
    kind="combine":   combines path results; "join_path" connects every edge
                      of a path whose types are path_types, "intersect" keeps
                      rows shared by several same-shape paths.
    kind="follow_up": added by the table, never named by the model: after a
                      combine (see then) or as an analysis prerequisite.
    kind="analysis":  a named analysis the question asks for; the slot planner
                      adds its prerequisites from requires_all/requires_any.
    kind="filter":    the step that applies stated conditions to rows of
                      subject_type (see FILTER_RULES).
    per_entity:       a hop or resolve that keeps several named entities apart
                      in one step; same-shape paths are merged into it.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "resolve", "hop", "attribute", "rank", "check", "combine", "follow_up",
        "analysis", "filter",
    ]
    subject_type: RowType
    object_type: EntityType | None = None
    # combine: the entity-type chain of the path(s) this operation combines.
    path_types: list[EntityType] = Field(default_factory=list)
    per_entity: bool = False
    # Operations the table always appends after this one.
    then: list[str] = Field(default_factory=list)
    name: str = "default"
    default: bool = False
    # Filters a hop always carries, e.g. approved indications only.
    fixed_filters: dict[str, Any] = Field(default_factory=dict)
    # resolve(variant): filter and grouping keys for every path leaving it.
    scope_filter: str | None = None
    scope_grouping: str | None = None
    # rank: metric used when a check needs a ranking the question did not name.
    default_metric: str | None = None
    # check: aggregation key the step must declare.
    aggregation_key: str | None = None
    # A filter the question must state before this operation can run.
    needs_stated_filter: str | None = None
    meaning: str = ""

    @model_validator(mode="after")
    def hop_needs_object_type(self) -> "PathRole":
        if (self.kind == "hop") != (self.object_type is not None):
            raise ValueError("object_type is required for hops and only for hops")
        if self.kind == "combine":
            if not self.path_types or self.path_types[-1] != self.subject_type:
                raise ValueError("combine needs path_types ending at subject_type")
            if self.name == "join_path" and len(self.path_types) < 3:
                raise ValueError("join_path needs a multi-hop path")
        return self


class OperationTableEntry(BaseModel):
    """One human-approved canonical operation and its reviewed equivalents."""

    canonical_operation: str
    equivalent_operations: list[str] = Field(default_factory=list)
    description: str
    operation_kind: Literal["resolution", "retrieval", "analysis"]
    allowed_entity_types: list[EntityType] = Field(default_factory=list)
    requires_all: list[str] = Field(default_factory=list)
    requires_any: list[str] = Field(default_factory=list)
    provides: list[str] = Field(default_factory=list)
    required_safeguards: list[str] = Field(default_factory=list)
    supported_filters: list[str] = Field(default_factory=list)
    requires_aggregation: bool = False
    reserved_outputs: list[str] = Field(default_factory=list)
    # Sources a Plan step declares for this operation. Empty for local analysis.
    default_sources: list[SourceName] = Field(default_factory=list)
    path_role: PathRole | None = None
    status: Literal["approved"] = "approved"


_OPERATION_CONTRACTS: dict[str, dict] = {
    "resolve_disease": {"allowed_entity_types": ["disease"], "provides": ["resolved_disease"]},
    "resolve_diseases": {"allowed_entity_types": ["disease"], "provides": ["resolved_disease"]},
    "resolve_target": {"allowed_entity_types": ["target"], "provides": ["resolved_target"]},
    "resolve_targets": {"allowed_entity_types": ["target"], "provides": ["resolved_target"]},
    "resolve_target_variants": {"allowed_entity_types": ["target"], "provides": ["resolved_target", "resolved_target_variants"], "required_safeguards": ["variant_specificity_required"]},
    "resolve_drug": {"allowed_entity_types": ["drug"], "provides": ["resolved_drug"]},
    "resolve_disease_target": {"allowed_entity_types": ["disease", "target"], "provides": ["resolved_disease", "resolved_target"]},
    "resolve_disease_targets": {"allowed_entity_types": ["disease", "target"], "provides": ["resolved_disease", "resolved_target"]},
    "resolve_disease_and_drugs": {"allowed_entity_types": ["disease", "drug"], "provides": ["resolved_disease", "resolved_drug"]},
    "discover_targets": {"requires_all": ["resolved_disease"], "provides": ["disease_target_association", "target_candidates"], "required_safeguards": ["association_score_not_probability", "association_does_not_imply_efficacy"]},
    "discover_targets_per_disease": {"requires_all": ["resolved_disease"], "provides": ["disease_target_association", "target_candidates"], "required_safeguards": ["association_score_not_probability", "association_does_not_imply_efficacy"]},
    "retrieve_exact_pair": {"requires_all": ["resolved_disease", "resolved_target"], "provides": ["disease_target_association"], "required_safeguards": ["association_score_not_probability", "association_does_not_imply_efficacy"]},
    "retrieve_exact_pairs": {"requires_all": ["resolved_disease", "resolved_target"], "provides": ["disease_target_association"], "required_safeguards": ["association_score_not_probability", "association_does_not_imply_efficacy"]},
    "retrieve_target_drugs": {"requires_any": ["resolved_target", "target_candidates"], "provides": ["target_drug_evidence"], "supported_filters": ["target_variant"]},
    "retrieve_drug_targets": {"requires_all": ["resolved_drug"], "provides": ["target_drug_evidence"]},
    "check_indication": {"requires_all": ["approval_evidence"], "provides": ["indication_check"], "required_safeguards": ["approval_requires_explicit_source", "database_approval_requires_official_confirmation", "missing_not_negative"]},
    "retrieve_tractability": {"requires_any": ["resolved_target", "target_candidates"], "provides": ["tractability_evidence"], "required_safeguards": ["tractability_not_efficacy"]},
    "retrieve_approved_indications": {"requires_any": ["resolved_drug", "target_drug_evidence"], "provides": ["approval_evidence"], "supported_filters": ["approval_status"]},
    "retrieve_clinical_trials": {"provides": ["clinical_trial_evidence"]},
    "retrieve_clinical_stage": {"requires_all": ["target_drug_evidence"], "provides": ["clinical_stage_evidence"]},
    "retrieve_safety_evidence": {"requires_any": ["resolved_target", "target_candidates"], "provides": ["safety_evidence"]},
    "retrieve_normal_tissue_expression": {"requires_any": ["resolved_target", "target_candidates"], "provides": ["normal_tissue_expression"], "required_safeguards": ["expression_not_toxicity", "specialized_expression_source_confirmation_required"]},
    "retrieve_trial_terminations": {"requires_any": ["resolved_target", "target_candidates"], "provides": ["trial_termination_evidence"], "required_safeguards": ["termination_reason_required"]},
    "integrate_safety_and_expression": {"requires_any": ["safety_evidence", "trial_termination_evidence", "normal_tissue_expression"], "provides": ["safety_expression_integration"], "required_safeguards": ["separate_fact_interpretation"]},
    "stratify_target_evidence_layers": {"requires_all": ["disease_target_association", "target_drug_evidence", "tractability_evidence"], "provides": ["stratified_target_evidence"], "requires_aggregation": True, "reserved_outputs": ["target_translation_layer"]},
    "aggregate_category_distribution": {"requires_aggregation": True, "reserved_outputs": ["category_distribution"]},
    "rank_targets": {"requires_any": ["target_candidates", "stratified_target_evidence"]},
    "retrieve_associated_diseases": {"requires_all": ["resolved_target"], "provides": ["disease_target_association"], "required_safeguards": ["association_score_not_probability", "association_does_not_imply_efficacy"]},
    "rank_diseases": {"requires_all": ["disease_target_association"], "provides": ["disease_ranking"]},
    "compare_disease_rank": {"requires_all": ["disease_ranking"], "provides": ["disease_rank_comparison"]},
    "retrieve_genetic_evidence": {"requires_any": ["target_candidates", "disease_target_association"], "provides": ["genetic_evidence"]},
    "filter_supported_without_clinical_drug": {"requires_all": ["genetic_evidence", "clinical_stage_evidence"], "provides": ["supported_without_clinical_drug"], "required_safeguards": ["no_drug_claim_source_bounded_and_dated"]},
    "retrieve_genetic_and_somatic_evidence": {"requires_any": ["target_candidates", "disease_target_association"], "provides": ["genetic_somatic_evidence"]},
    "filter_first_in_class_candidates": {"requires_all": ["genetic_somatic_evidence", "target_drug_evidence"], "provides": ["first_in_class_candidates"], "supported_filters": ["max_genetic_somatic_score"], "required_safeguards": ["no_drug_claim_source_bounded_and_dated", "missing_not_negative"]},
    "rank_opportunities": {"requires_all": ["first_in_class_candidates", "tractability_evidence"], "provides": ["opportunity_ranking"]},
    "filter_phase2_not_approved_for_disease": {"requires_all": ["clinical_stage_evidence", "approval_evidence"], "required_safeguards": ["trial_phase_not_approval", "approval_requires_explicit_source"]},
    "filter_targets": {"requires_any": ["target_candidates", "stratified_target_evidence"], "supported_filters": ["overall_association_score", "rank", "known_drug_count", "exact_disease_drug_count", "tractable_bucket_count", "tractable_modalities"]},
    "retrieve_datasource_evidence": {"requires_all": ["disease_target_association"], "provides": ["datasource_evidence"], "supported_filters": ["evidence_type"], "required_safeguards": ["datasource_score_not_record_count"]},
    "retrieve_evidence_by_type": {"requires_all": ["disease_target_association"], "provides": ["evidence_by_type"], "supported_filters": ["evidence_type"], "required_safeguards": ["datasource_score_not_record_count"]},
    "retrieve_datasource_scores": {"requires_all": ["disease_target_association"], "provides": ["datasource_scores"], "required_safeguards": ["datasource_score_not_record_count"]},
    "rank_datasource_contributions": {"requires_all": ["datasource_scores"]},
    "compare_evidence_types": {"requires_any": ["datasource_evidence", "evidence_by_type"], "provides": ["evidence_type_comparison"], "required_safeguards": ["datasource_score_not_record_count"]},
    "normalize_comparison_dimensions": {"requires_any": ["evidence_by_type", "datasource_evidence"], "provides": ["normalized_comparison"], "required_safeguards": ["same_comparison_denominator"]},
    "compare_targets": {"requires_all": ["normalized_comparison"], "provides": ["target_comparison"]},
    "retrieve_disease_drugs": {"requires_all": ["resolved_disease"], "provides": ["disease_drug_evidence"], "required_safeguards": ["approval_requires_explicit_source"]},
    "join_disease_target_drug": {"requires_all": ["disease_target_association", "target_drug_evidence"], "provides": ["disease_target_drug_edges"], "supported_filters": ["existing_drug"], "required_safeguards": ["association_does_not_imply_efficacy"]},
    "rank_candidates": {"requires_any": ["disease_target_drug_edges", "stratified_target_evidence", "target_candidates"]},
    "intersect_targets": {"requires_all": ["disease_target_association"], "provides": ["target_intersection"]},
    "compare_association_evidence": {"requires_all": ["disease_target_association"], "provides": ["association_comparison"], "required_safeguards": ["score_not_comparable_without_context"]},
    "retrieve_target_cancers": {"requires_any": ["target_drug_evidence", "resolved_target", "target_candidates"], "provides": ["target_cancer_evidence"], "supported_filters": ["cancer_only", "exclude_index_disease"], "required_safeguards": ["association_does_not_imply_indication"]},
    "join_cross_indication_paths": {"requires_all": ["disease_drug_evidence", "target_drug_evidence", "target_cancer_evidence"], "provides": ["cross_indication_paths"], "required_safeguards": ["association_does_not_imply_indication"]},
    "classify_kinase_targets": {"requires_any": ["target_candidates", "resolved_target"], "provides": ["target_family_classification", "target_candidates"]},
    "define_coverage_denominators": {"requires_all": ["target_family_classification", "target_drug_evidence"], "provides": ["coverage_definition"], "required_safeguards": ["define_drug_coverage", "same_comparison_denominator"]},
    "calculate_coverage_rates": {"requires_all": ["coverage_definition"], "provides": ["coverage_rates"]},
    "compare_disease_rates": {"requires_all": ["coverage_rates"], "provides": ["disease_rate_comparison"], "required_safeguards": ["same_comparison_denominator"]},
}


# Path index: how each operation sits on a question's path. The slot planner
# assembles Plans from these roles; nothing outside this table names them.
_OT = ["open_targets"]
# Open Targets drug, mechanism and indication records are ChEMBL-derived.
_OT_CHEMBL = ["open_targets", "chembl"]
_PATH_ROLES: dict[str, dict] = {
    "resolve_disease": {"kind": "resolve", "subject_type": "disease"},
    "resolve_target": {"kind": "resolve", "subject_type": "target"},
    "resolve_drug": {"kind": "resolve", "subject_type": "drug"},
    "resolve_diseases": {"kind": "resolve", "subject_type": "disease", "name": "many"},
    # Combined resolvers: the start entity plus named entities on its path.
    "resolve_disease_and_drugs": {"kind": "resolve", "subject_type": "disease", "name": "with_drugs"},
    "resolve_disease_target": {"kind": "resolve", "subject_type": "disease", "name": "pair"},
    "resolve_disease_targets": {"kind": "resolve", "subject_type": "disease", "name": "pairs"},
    "resolve_targets": {"kind": "resolve", "subject_type": "target", "name": "many"},
    "discover_targets_per_disease": {
        "kind": "hop", "subject_type": "disease", "object_type": "target",
        "name": "associated", "per_entity": True,
        "meaning": "每個疾病各自的相關聯標的，分開保留",
    },
    "resolve_target_variants": {
        "kind": "resolve", "subject_type": "target", "name": "variant",
        "scope_filter": "target_variant", "scope_grouping": "group_by_variant",
    },
    "discover_targets": {
        "kind": "hop", "subject_type": "disease", "object_type": "target",
        "name": "associated", "default": True, "meaning": "與疾病相關聯的標的",
    },
    "retrieve_associated_diseases": {
        "kind": "hop", "subject_type": "target", "object_type": "disease",
        "name": "associated", "default": True, "meaning": "與標的相關聯的疾病",
    },
    "retrieve_exact_pair": {
        "kind": "hop", "subject_type": "target", "object_type": "disease",
        "name": "exact", "meaning": "指名標的與指名疾病之間的那一條關聯",
    },
    "retrieve_exact_pairs": {
        "kind": "hop", "subject_type": "target", "object_type": "disease",
        "name": "exact", "per_entity": True,
        "meaning": "幾個指名標的各自與指名疾病之間的關聯，分開保留",
    },
    "retrieve_evidence_by_type": {
        "kind": "attribute", "subject_type": "association", "name": "evidence_by_type",
        # Without a stated type, compare every Open Targets evidence type.
        "fixed_filters": {"evidence_type": {"in": [
            "genetic_association", "somatic_mutation", "known_drug",
            "affected_pathway", "rna_expression", "literature", "animal_model",
        ]}},
        "meaning": "關聯依證據類型分開的分數",
    },
    "compare_evidence_types": {
        "kind": "analysis", "subject_type": "association", "name": "compare_evidence_types",
        "default_metric": "datasource_contribution",
        "aggregation_key": "group_by_evidence_type",
        "meaning": "一條關聯主要來自哪一類證據（依證據類型的貢獻排序）",
    },
    "rank_datasource_contributions": {
        "kind": "analysis", "subject_type": "association", "name": "rank_datasource_contributions",
        "default_metric": "datasource_score",
        "aggregation_key": "datasource_contribution_comparison",
        "meaning": "一條關聯中哪個資料來源貢獻最大（依 datasource score 排序）",
    },
    "compare_targets": {
        "kind": "analysis", "subject_type": "association", "name": "compare_targets",
        "aggregation_key": "group_by_target_and_evidence_type",
        "meaning": "在相同證據類型維度上並列比較幾個標的與同一疾病的關聯",
    },
    "filter_supported_without_clinical_drug": {
        "kind": "analysis", "subject_type": "target", "name": "supported_without_clinical_drug",
        "aggregation_key": "target_evidence_drug_join",
        "meaning": "有 genetic evidence、但在已查來源沒有藥物進入臨床的標的",
    },
    "filter_targets": {"kind": "filter", "subject_type": "target"},
    "define_coverage_denominators": {
        "kind": "follow_up", "subject_type": "target", "name": "coverage_denominators",
        "aggregation_key": "coverage_numerator_denominator",
        "meaning": "覆蓋率的分子與分母：家族內的相關標的，以及其中有藥物紀錄者",
    },
    "compare_disease_rates": {
        "kind": "analysis", "subject_type": "target", "name": "compare_kinase_drug_coverage",
        "aggregation_key": "compare_disease_rates",
        "meaning": "幾個疾病的相關標的中，kinase 家族被藥物覆蓋的比例並列比較",
    },
    "retrieve_safety_evidence": {
        "kind": "attribute", "subject_type": "target", "name": "safety_risk",
        # Known safety risk includes trials terminated with a recorded reason.
        "then": ["retrieve_trial_terminations"],
        "meaning": "標的已知的安全性風險（含已終止試驗及其理由）",
    },
    "retrieve_normal_tissue_expression": {
        "kind": "attribute", "subject_type": "target", "name": "normal_tissue_expression",
        "meaning": "標的在正常組織的表現量",
    },
    "integrate_safety_and_expression": {
        "kind": "analysis", "subject_type": "target", "name": "integrate_safety_and_expression",
        "aggregation_key": "group_by_target_risk_and_tissue",
        "meaning": "把標的的安全性風險、終止試驗與正常組織表現並列整合（事實與推論分開）",
    },
    "filter_phase2_not_approved_for_disease": {
        "kind": "analysis", "subject_type": "target", "name": "phase2_not_approved_for_disease",
        "aggregation_key": "target_drug_trial_indication_join",
        "meaning": "已有 Phase 2 以上藥物、但這些藥物在已查來源尚未核准用於起點疾病的標的",
    },
    "filter_first_in_class_candidates": {
        "kind": "follow_up", "subject_type": "target", "name": "first_in_class_candidates",
        "needs_stated_filter": "max_genetic_somatic_score",
        "aggregation_key": "target_evidence_drug_tractability_join",
        "meaning": "有遺傳或體細胞突變證據、但在已查來源沒有任何藥物的標的",
    },
    "rank_opportunities": {
        "kind": "analysis", "subject_type": "target", "name": "first_in_class_opportunities",
        "default_metric": "max_genetic_somatic_score",
        "meaning": "first-in-class 機會：上述標的依證據強度排序，並附 tractability 偏低警訊",
    },
    "retrieve_target_cancers": {
        "kind": "hop", "subject_type": "target", "object_type": "disease",
        "name": "cancer", "meaning": "標的關聯的癌別",
    },
    "retrieve_target_drugs": {
        "kind": "hop", "subject_type": "target", "object_type": "drug",
        "name": "known_drug", "default": True, "meaning": "作用在標的上的已知藥物",
    },
    "retrieve_drug_targets": {
        "kind": "hop", "subject_type": "drug", "object_type": "target",
        "name": "mechanism_target", "default": True, "meaning": "藥物的作用標的",
    },
    "retrieve_approved_indications": {
        "kind": "hop", "subject_type": "drug", "object_type": "disease",
        "name": "approved_indication", "default": True,
        "fixed_filters": {"approval_status": "approved"},
        "meaning": "藥物的核准適應症",
    },
    "retrieve_disease_drugs": {
        "kind": "hop", "subject_type": "disease", "object_type": "drug",
        "name": "known_drug", "default": True, "meaning": "用於該疾病的藥物",
    },
    "retrieve_mechanisms": {
        "kind": "attribute", "subject_type": "drug", "name": "mechanism_of_action",
        "meaning": "藥物的作用機制",
    },
    "retrieve_tractability": {
        "kind": "attribute", "subject_type": "target", "name": "tractability",
        "meaning": "標的的可成藥性（tractability）",
    },
    "rank_targets": {
        "kind": "rank", "subject_type": "target",
        "default_metric": "overall_association_score",
    },
    "rank_diseases": {
        "kind": "rank", "subject_type": "disease",
        "default_metric": "overall_association_score",
    },
    "rank_candidates": {
        "kind": "rank", "subject_type": "drug",
        "default_metric": "overall_association_score",
    },
    "intersect_targets": {
        "kind": "combine", "subject_type": "target", "name": "intersect",
        "path_types": ["disease", "target"], "per_entity": True,
        "aggregation_key": "target_set_intersection",
        "then": ["compare_association_evidence"],
        "meaning": "幾個疾病都有的相關聯標的（交集），各疾病的分數分開保留",
    },
    "compare_association_evidence": {
        "kind": "follow_up", "subject_type": "target",
        "aggregation_key": "report_scores_per_disease",
        "meaning": "並列各疾病的關聯分數，不合成單一分數",
    },
    "join_cross_indication_paths": {
        "kind": "combine", "subject_type": "disease", "name": "join_path",
        "path_types": ["disease", "drug", "target", "disease"],
        "aggregation_key": "group_by_drug_and_target",
        "meaning": "把起點疾病的藥物、藥物的標的、標的的其他疾病串成路徑，保留每條邊",
    },
    "join_disease_target_drug": {
        "kind": "combine", "subject_type": "drug", "name": "join_path",
        "path_types": ["disease", "target", "drug"],
        # The join keeps only drugs that have a record for the target.
        "fixed_filters": {"existing_drug": True},
        "aggregation_key": "disease_target_drug_join",
        "meaning": "把每個藥物經由它的標的連回起點疾病，保留每條邊與關聯分數",
    },
    "compare_disease_rank": {
        "kind": "check", "subject_type": "disease", "name": "rank_position",
        "aggregation_key": "disease_relative_rank",
        "meaning": "指名的疾病在排序結果中的位置（例如是否最強）",
    },
    "check_indication": {
        "kind": "check", "subject_type": "disease", "name": "approved_indication",
        "meaning": "指名的疾病是否在藥物的核准適應症中",
    },
}
_DEFAULT_SOURCES: dict[str, list[str]] = {
    "resolve_disease": _OT,
    "resolve_target": _OT,
    "resolve_drug": _OT,
    "resolve_diseases": _OT,
    "resolve_targets": _OT,
    "resolve_target_variants": _OT,
    "resolve_disease_and_drugs": _OT,
    "resolve_disease_target": _OT,
    "resolve_disease_targets": _OT,
    "retrieve_exact_pair": _OT,
    "retrieve_exact_pairs": _OT,
    "retrieve_datasource_evidence": _OT,
    "retrieve_evidence_by_type": _OT,
    "retrieve_genetic_evidence": _OT,
    "retrieve_genetic_and_somatic_evidence": _OT,
    "retrieve_safety_evidence": _OT,
    "retrieve_trial_terminations": _OT,
    "retrieve_normal_tissue_expression": _OT,
    "discover_targets_per_disease": _OT,
    "discover_targets": _OT,
    "retrieve_associated_diseases": _OT,
    "retrieve_target_cancers": _OT,
    "retrieve_tractability": _OT,
    "retrieve_target_drugs": _OT_CHEMBL,
    "retrieve_drug_targets": _OT_CHEMBL,
    "retrieve_approved_indications": _OT_CHEMBL,
    "retrieve_disease_drugs": _OT_CHEMBL,
    "retrieve_mechanisms": _OT_CHEMBL,
}


def _operation_kind(operation: str) -> Literal["resolution", "retrieval", "analysis"]:
    if operation.startswith("resolve_"):
        return "resolution"
    if operation.startswith(("retrieve_", "discover_")):
        return "retrieval"
    return "analysis"


def _operation_contract(operation: str) -> dict:
    contract = dict(_OPERATION_CONTRACTS.get(operation, {}))
    safeguards = set(contract.pop("required_safeguards", []))
    if _operation_kind(operation) == "retrieval":
        safeguards.update({"provenance_pagination_truncation", "missing_not_negative"})
    return {
        **contract,
        "required_safeguards": sorted(safeguards),
        "default_sources": _DEFAULT_SOURCES.get(operation, []),
        "path_role": _PATH_ROLES.get(operation),
    }


OPERATION_TABLE: dict[str, OperationTableEntry] = {
    operation: OperationTableEntry(
        canonical_operation=operation,
        equivalent_operations=_REVIEWED_OPERATION_ALIASES.get(operation, []),
        description=OPERATION_SEMANTICS[operation],
        operation_kind=_operation_kind(operation),
        **_operation_contract(operation),
    )
    for operation in _CANONICAL_OPERATION_IDS
}

# Compatibility views are derived from the authoritative table, never edited
# independently.
STANDARD_OPERATION_CATALOG = sorted(OPERATION_TABLE)
OPERATION_EQUIVALENCES: dict[str, list[str]] = {
    canonical: [canonical, *entry.equivalent_operations]
    for canonical, entry in OPERATION_TABLE.items()
    if entry.equivalent_operations
}

OPERATION_ALIAS_TO_CANONICAL = {
    alias: canonical
    for canonical, entry in OPERATION_TABLE.items()
    for alias in [canonical, *entry.equivalent_operations]
}


def _path_index(kind: str) -> dict[tuple, str]:
    index: dict[tuple, str] = {}
    for operation, entry in OPERATION_TABLE.items():
        role = entry.path_role
        if role is None or role.kind != kind:
            continue
        if kind == "hop" and role.per_entity:
            continue  # listed in PER_ENTITY_HOPS
        if kind == "hop":
            key = (role.subject_type, role.object_type, role.name)
        elif kind == "combine":
            key = (role.name, tuple(role.path_types))
        else:
            key = (role.subject_type, role.name)
        if key in index:
            raise ValueError(f"path index key {key} is claimed twice")
        index[key] = operation
    return index


# Path index views, derived from OPERATION_TABLE. Keys:
#   RESOLVE_INDEX[(entity_type, "default"|"variant")]
#   HOP_INDEX[(from_type, to_type, relation)]
#   ATTRIBUTE_INDEX / CHECK_INDEX[(entity_type, name)]
#   RANK_INDEX[(entity_type, "default")]
#   COMBINE_INDEX[(name, path_types)], e.g. ("join_path", ("disease", "target", "drug"))
RESOLVE_INDEX = _path_index("resolve")
HOP_INDEX = _path_index("hop")
ATTRIBUTE_INDEX = _path_index("attribute")
RANK_INDEX = _path_index("rank")
CHECK_INDEX = _path_index("check")
COMBINE_INDEX = _path_index("combine")
FILTER_STEPS: dict[str, str] = {
    entry.path_role.subject_type: operation
    for operation, entry in OPERATION_TABLE.items()
    if entry.path_role and entry.path_role.kind == "filter"
}
ANALYSIS_INDEX: dict[str, str] = {
    entry.path_role.name: operation
    for operation, entry in OPERATION_TABLE.items()
    if entry.path_role and entry.path_role.kind == "analysis"
}
# Filters the question can state, by name. Only filters an operation lists in
# supported_filters and its adapter applies belong here.
FILTER_MEANINGS: dict[str, str] = {
    "cancer_only": "只保留癌症類疾病",
    "exclude_index_disease": "排除起點疾病本身（「其他」疾病）",
    "evidence_type": "只看題目點名的證據類型；value 是清單",
    "overall_association_score": '與疾病的關聯分數門檻；value 例如 {"gt": 0.7}，數字要寫在題目裡',
    "small_molecule_tractable": "具已知小分子結合口袋（tractability 含 SM）",
    "max_genetic_somatic_score": '遺傳或體細胞突變證據分數的門檻；value 例如 {"gte": 0.5}，數字要寫在題目裡',
}
# Filters that become a filter step: executor field, fixed condition (if the
# name alone says it), and the evidence the field needs.
FILTER_RULES: dict[str, dict[str, Any]] = {
    "overall_association_score": {
        "field": "overall_association_score", "requires": "disease_target_association",
    },
    "small_molecule_tractable": {
        "field": "tractable_modalities", "condition": {"contains": "SM"},
        "requires": "tractability_evidence",
    },
}
# Allowed values for filters whose value is a list of names.
FILTER_VALUES: dict[str, list[str]] = {
    "evidence_type": [
        "genetic_association", "somatic_mutation", "known_drug",
        "affected_pathway", "rna_expression", "literature", "animal_model",
    ],
}
# Producer to add when a capability is missing and several operations make it.
DEFAULT_PRODUCERS: dict[str, str] = {
    "target_drug_evidence": "retrieve_target_drugs",
}

# One step that walks a hop for several named entities, keeping them apart.
PER_ENTITY_HOPS: dict[tuple[str, str, str], str] = {
    (entry.path_role.subject_type, entry.path_role.object_type, entry.path_role.name): operation
    for operation, entry in OPERATION_TABLE.items()
    if entry.path_role and entry.path_role.kind == "hop" and entry.path_role.per_entity
}
DEFAULT_HOPS: dict[tuple[str, str], str] = {}
for (_from, _to, _relation), _operation in HOP_INDEX.items():
    if OPERATION_TABLE[_operation].path_role.default:
        if (_from, _to) in DEFAULT_HOPS:
            raise ValueError(f"hop {_from}->{_to} has two default operations")
        DEFAULT_HOPS[(_from, _to)] = _operation


def normalize_plan_operations(plan: QueryManifestV2) -> list[str]:
    """Canonicalize reviewed aliases and reject undeclared approved operations."""

    corrections: list[str] = []
    for step in plan.plan_steps:
        if step.operation_status == "proposed":
            continue
        canonical = OPERATION_ALIAS_TO_CANONICAL.get(step.operation)
        if canonical is None:
            raise ValueError(
                f"operation {step.operation!r} is not in OPERATION_TABLE; use "
                "operation='proposed_new_operation', operation_status='proposed', "
                "proposed_operation_name, and proposal_reason for human review"
            )
        if canonical != step.operation:
            corrections.append(
                f"{step.step_id}: normalized operation {step.operation} -> {canonical}"
            )
            step.operation = canonical
    return corrections

RANKING_KEY_EQUIVALENCES: dict[str, list[str]] = {
    "overall_association_score": [
        "overall_association_score",
        "association_score",
        "target_association_score",
    ],
}

COMMON_FORBIDDEN_CONDITIONS = [
    ForbiddenCondition(
        condition_id="unverified_identity_claim",
        description="不得在未執行實體解析前宣稱 ID 或 canonical name 已驗證。",
        severity="autocorrect",
    ),
    ForbiddenCondition(
        condition_id="retrieval_claim_in_plan",
        description="Plan 階段不得宣稱證據已擷取或研究結論已成立。",
    ),
    ForbiddenCondition(
        condition_id="classification_substitutes_for_plan",
        description="六類描述標籤不得取代可執行的研究步驟。",
    ),
]

OPTIONAL_CAPABILITIES_BY_CASE: dict[str, list[OptionalCapability]] = {
    "pdac_drug_repurposing": [
        OptionalCapability(
            capability_id="check_trial_maturity",
            description="可用臨床試驗成熟度補充再利用候選排序。",
            equivalent_operations=["retrieve_clinical_trials", "retrieve_clinical_stage"],
        )
    ],
    "shp2_fak_safety_expression": [
        OptionalCapability(
            capability_id="retrieve_negative_publications",
            description="可補充文獻中的失敗或反向安全訊號。",
            equivalent_operations=["retrieve_negative_publications"],
        )
    ],
}


def build_acceptance_rubric(case: PlannerAcceptanceCase) -> PlanAcceptanceRubric:
    capabilities = [
        RequiredCapability(
            capability_id=operation,
            description=f"Plan 必須具備 {operation.replace('_', ' ')} 的研究能力。",
            equivalent_operations=OPERATION_EQUIVALENCES.get(operation, [operation]),
        )
        for operation in case.required_operations
    ]
    return PlanAcceptanceRubric(
        case_id=case.case_id,
        question=case.question,
        research_intent=case.research_intent,
        required_entities=case.required_entities,
        required_capabilities=capabilities,
        required_dependencies=case.required_dependencies,
        required_sources=case.required_sources,
        required_safeguards=case.required_safeguards,
        required_filters=case.required_filters,
        required_ranking=case.required_ranking,
        required_aggregation=case.required_aggregation,
        forbidden_conditions=COMMON_FORBIDDEN_CONDITIONS,
        optional_capabilities=OPTIONAL_CAPABILITIES_BY_CASE.get(case.case_id, []),
        allowed_characteristics=case.allowed_characteristics,
    )


class PlanAcceptanceResult(BaseModel):
    case_id: str
    passed: bool
    missing_entities: list[str] = Field(default_factory=list)
    missing_capabilities: list[str] = Field(default_factory=list)
    missing_operations: list[str] = Field(default_factory=list)
    missing_dependencies: list[str] = Field(default_factory=list)
    missing_sources: list[str] = Field(default_factory=list)
    missing_safeguards: list[str] = Field(default_factory=list)
    missing_filters: list[str] = Field(default_factory=list)
    missing_ranking: list[str] = Field(default_factory=list)
    missing_aggregation: list[str] = Field(default_factory=list)
    forbidden_findings: list[str] = Field(default_factory=list)
    optional_capabilities_present: list[str] = Field(default_factory=list)
    accepted_capabilities: list[str] = Field(default_factory=list)
    actual_operations: list[str] = Field(default_factory=list)
    unrecognized_operations: list[str] = Field(default_factory=list)
    proposed_operations: list[str] = Field(default_factory=list)


class RepairFeedback(BaseModel):
    """Actionable diagnostics only; never contains the complete hidden rubric."""

    missing: list[dict[str, str]] = Field(default_factory=list)
    incorrect: list[dict[str, str]] = Field(default_factory=list)
    preserve: list[dict[str, str]] = Field(default_factory=list)


def load_acceptance_cases(path: str | Path) -> list[PlannerAcceptanceCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [PlannerAcceptanceCase.model_validate(item) for item in payload]
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("acceptance case_id values must be unique")
    return cases


def load_acceptance_rubrics(path: str | Path) -> list[PlanAcceptanceRubric]:
    return [build_acceptance_rubric(case) for case in load_acceptance_cases(path)]


def _step_by_capability(
    rubric: PlanAcceptanceRubric, plan: QueryManifestV2
) -> dict[str, object]:
    result = {}
    for capability in rubric.required_capabilities:
        equivalent = set(capability.equivalent_operations)
        step = next(
            (step for step in plan.plan_steps if step.operation in equivalent), None
        )
        if step is not None:
            result[capability.capability_id] = step
    return result


def evaluate_plan(
    case_or_rubric: PlannerAcceptanceCase | PlanAcceptanceRubric,
    plan: QueryManifestV2,
) -> PlanAcceptanceResult:
    """Evaluate a hidden rubric without executing retrieval tools."""

    rubric = (
        build_acceptance_rubric(case_or_rubric)
        if isinstance(case_or_rubric, PlannerAcceptanceCase)
        else case_or_rubric
    )
    actual_entities = [
        (entity.entity_type, entity.mention.casefold()) for entity in plan.entities
    ]
    missing_entities = [
        f"{entity.entity_type}:{entity.mention}"
        for entity in rubric.required_entities
        if not any(
            actual_type == entity.entity_type
            and (
                entity.mention.casefold() in actual_mention
                or actual_mention in entity.mention.casefold()
            )
            for actual_type, actual_mention in actual_entities
        )
    ]
    capability_steps = _step_by_capability(rubric, plan)
    missing_capabilities = [
        item.capability_id
        for item in rubric.required_capabilities
        if item.capability_id not in capability_steps
    ]
    missing_dependencies = []
    for before, after in rubric.required_dependencies:
        before_step = capability_steps.get(before)
        after_step = capability_steps.get(after)
        if (
            before_step is None
            or after_step is None
            or before_step.step_id not in after_step.depends_on
        ):
            missing_dependencies.append(f"{before}->{after}")

    actual_sources = {source for step in plan.plan_steps for source in step.sources}
    actual_safeguards = set(plan.global_safeguards) | {
        safeguard for step in plan.plan_steps for safeguard in step.safeguards
    }
    filter_keys = {key for step in plan.plan_steps for key in step.filters}
    normalized_ranking_keys = {
        metric
        for step in plan.plan_steps
        if step.ranking is not None
        for metric in [step.ranking.primary_metric, *step.ranking.secondary_metrics]
    }
    for canonical, aliases in RANKING_KEY_EQUIVALENCES.items():
        if normalized_ranking_keys.intersection(aliases):
            normalized_ranking_keys.add(canonical)
    aggregation_keys = {
        key for step in plan.plan_steps for key in (step.aggregation or {})
    }
    text = " ".join(
        [plan.research_objective]
        + [step.description for step in plan.plan_steps]
        + plan.completion_criteria
    ).casefold()
    forbidden_findings = []
    if any(
        phrase in text
        for phrase in (
            "已擷取",
            "已證實關聯",
            "evidence was retrieved",
            "association is proven",
        )
    ):
        forbidden_findings.append("retrieval_claim_in_plan")
    if len(plan.plan_steps) < 2:
        forbidden_findings.append("classification_substitutes_for_plan")

    actual_operations = {step.operation for step in plan.plan_steps}
    proposed_operations = sorted(
        step.proposed_operation_name
        for step in plan.plan_steps
        if step.operation_status == "proposed" and step.proposed_operation_name
    )
    known_operations = set(STANDARD_OPERATION_CATALOG) | {
        alias for aliases in OPERATION_EQUIVALENCES.values() for alias in aliases
    } | {"proposed_new_operation"}
    optional_present = [
        item.capability_id
        for item in rubric.optional_capabilities
        if actual_operations.intersection(item.equivalent_operations)
    ]
    result = PlanAcceptanceResult(
        case_id=rubric.case_id,
        passed=False,
        missing_entities=missing_entities,
        missing_capabilities=missing_capabilities,
        missing_operations=missing_capabilities,
        missing_dependencies=missing_dependencies,
        missing_sources=sorted(set(rubric.required_sources) - actual_sources),
        missing_safeguards=sorted(
            set(rubric.required_safeguards) - actual_safeguards
        ),
        missing_filters=sorted(set(rubric.required_filters) - filter_keys),
        missing_ranking=sorted(
            set(rubric.required_ranking) - normalized_ranking_keys
        ),
        missing_aggregation=sorted(
            set(rubric.required_aggregation) - aggregation_keys
        ),
        forbidden_findings=forbidden_findings,
        optional_capabilities_present=optional_present,
        accepted_capabilities=sorted(capability_steps),
        actual_operations=sorted(actual_operations),
        unrecognized_operations=sorted(actual_operations - known_operations),
        proposed_operations=proposed_operations,
    )
    blocking_fields = (
        "missing_entities",
        "missing_capabilities",
        "missing_dependencies",
        "missing_sources",
        "missing_safeguards",
        "missing_filters",
        "missing_ranking",
        "missing_aggregation",
        "forbidden_findings",
        "proposed_operations",
    )
    result.passed = not any(getattr(result, field) for field in blocking_fields)
    return result


def build_repair_feedback(result: PlanAcceptanceResult) -> RepairFeedback:
    """Return specific gaps/errors without exposing the complete hidden rubric."""

    missing = []
    labels = {
        "missing_entities": "entity",
        "missing_capabilities": "research_capability",
        "missing_dependencies": "dependency",
        "missing_sources": "source_coverage",
        "missing_safeguards": "safeguard",
        "missing_filters": "filter",
        "missing_ranking": "ranking",
        "missing_aggregation": "aggregation",
    }
    for field, area in labels.items():
        for item in getattr(result, field):
            if field == "missing_filters":
                correction = (
                    f"在適當步驟的 filters 物件中加入頂層 key {item!r}，"
                    "並保留使用者要求的值。"
                )
            elif field == "missing_ranking":
                correction = (
                    f"在適當步驟的 ranking.primary_metric 或 "
                    f"ranking.secondary_metrics 中使用正式指標 {item!r}。"
                )
            elif field == "missing_aggregation":
                correction = (
                    f"在適當分析步驟的 aggregation 物件中加入頂層 key "
                    f"{item!r}；只放在 outputs、description 或巢狀文字不算完成。"
                )
            elif field == "missing_dependencies" and "->" in item:
                upstream_operation, downstream_operation = item.split("->", 1)
                correction = (
                    f"找到 operation={downstream_operation!r} 的下游步驟，"
                    f"把 operation={upstream_operation!r} 的上游 step_id 加入該"
                    "下游步驟的 depends_on。不可讓上游反過來依賴下游。"
                )
            else:
                correction = f"補上 {item} 的研究語意並保持整體 Plan 一致。"
            missing.append(
                {
                    "area": area,
                    "problem": f"缺少必要項目：{item}",
                    "required_correction": correction,
                }
            )
    incorrect = [
        {
            "area": "forbidden_condition",
            "problem": f"出現禁止錯誤：{item}",
            "required_correction": "移除錯誤主張，改成資料擷取前的中性規劃語句。",
        }
        for item in result.forbidden_findings
    ]
    incorrect.extend(
        {
            "area": "unrecognized_operation",
            "problem": f"operation 尚未被目錄或等價表辨識：{item}",
            "required_correction": (
                "先查 OPERATION_TABLE；若有等價能力，改用 canonical operation。"
                "若確實沒有，標記 proposed_new_operation 並等待人工審核。"
            ),
        }
        for item in result.unrecognized_operations
    )
    incorrect.extend(
        {
            "area": "proposed_operation",
            "problem": f"新 operation 等待人工審核：{item}",
            "required_correction": (
                "停止自動執行。由人工確認 OPERATION_TABLE 是否已有等價能力；"
                "若無，審核後加入 table，再重新產生 Plan。"
            ),
        }
        for item in result.proposed_operations
    )
    preserve = [
        {
            "area": "accepted_capability",
            "item": item,
            "instruction": "此能力已通過；修復其他缺項時不得刪除或改壞。",
        }
        for item in result.accepted_capabilities
    ]
    return RepairFeedback(missing=missing, incorrect=incorrect, preserve=preserve)
