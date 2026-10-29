"""Hidden, deterministic rubrics for evaluating plan quality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from schemas import QueryManifestV2


SourceName = Literal["open_targets", "europe_pmc", "clinicaltrials_gov", "chembl"]


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
        "filter_phase2_not_pdac_approved",
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
    "filter_phase2_not_pdac_approved": "保留 Phase 2+ 且未在已查來源確認 PDAC 核准的藥物或標的。",
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


class OperationTableEntry(BaseModel):
    """One human-approved canonical operation and its reviewed equivalents."""

    canonical_operation: str
    equivalent_operations: list[str] = Field(default_factory=list)
    description: str
    operation_kind: Literal["resolution", "retrieval", "analysis"]
    allowed_entity_types: list[
        Literal["disease", "target", "drug", "tissue", "mechanism"]
    ] = Field(default_factory=list)
    requires_all: list[str] = Field(default_factory=list)
    requires_any: list[str] = Field(default_factory=list)
    provides: list[str] = Field(default_factory=list)
    required_safeguards: list[str] = Field(default_factory=list)
    supported_filters: list[str] = Field(default_factory=list)
    requires_aggregation: bool = False
    reserved_outputs: list[str] = Field(default_factory=list)
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
    "retrieve_target_drugs": {"requires_any": ["resolved_target", "target_candidates"], "provides": ["target_drug_evidence"]},
    "retrieve_drug_targets": {"requires_all": ["resolved_drug"], "provides": ["target_drug_evidence"]},
    "check_indication": {"requires_all": ["approval_evidence"], "provides": ["indication_check"], "required_safeguards": ["approval_requires_explicit_source", "database_approval_requires_official_confirmation", "missing_not_negative"]},
    "retrieve_tractability": {"requires_any": ["resolved_target", "target_candidates"], "provides": ["tractability_evidence"], "required_safeguards": ["tractability_not_efficacy"]},
    "retrieve_approved_indications": {"provides": ["approval_evidence"]},
    "retrieve_clinical_trials": {"provides": ["clinical_trial_evidence"]},
    "retrieve_clinical_stage": {"provides": ["clinical_stage_evidence"]},
    "retrieve_safety_evidence": {"provides": ["safety_evidence"]},
    "retrieve_normal_tissue_expression": {"provides": ["normal_tissue_expression"]},
    "stratify_target_evidence_layers": {"requires_all": ["disease_target_association", "target_drug_evidence", "tractability_evidence"], "provides": ["stratified_target_evidence"], "requires_aggregation": True, "reserved_outputs": ["target_translation_layer"]},
    "aggregate_category_distribution": {"requires_aggregation": True, "reserved_outputs": ["category_distribution"]},
    "rank_targets": {"requires_any": ["target_candidates", "stratified_target_evidence"]},
    "retrieve_associated_diseases": {"requires_all": ["resolved_target"], "provides": ["disease_target_association"]},
    "rank_diseases": {"requires_all": ["disease_target_association"], "provides": ["disease_ranking"]},
    "compare_disease_rank": {"requires_all": ["disease_ranking"], "provides": ["disease_rank_comparison"]},
    "retrieve_genetic_evidence": {"requires_any": ["target_candidates", "disease_target_association"], "provides": ["genetic_evidence"]},
    "filter_supported_without_clinical_drug": {"requires_all": ["genetic_evidence", "clinical_stage_evidence"], "provides": ["supported_without_clinical_drug"]},
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
    return {**contract, "required_safeguards": sorted(safeguards)}


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
