"""General operation-level quality rules for QueryManifestV2 plans.

These rules validate research-plan structure. They do not encode the expected
answer to any individual acceptance question and never retrieve evidence.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, Field

from planner_acceptance import OPERATION_TABLE
from schemas import QueryManifestV2, ResearchPlanStepV2


RuleSeverity = Literal["hard_error", "repair_required", "warning"]


class OperationRuleDefinition(BaseModel):
    rule_id: str
    severity: RuleSeverity
    description_zh_tw: str


class OperationRuleFinding(BaseModel):
    rule_id: str
    severity: RuleSeverity
    step_id: str | None = None
    operation: str | None = None
    problem: str
    required_correction: str
    missing: list[str] = Field(default_factory=list)


class OperationRuleEvaluation(BaseModel):
    passed: bool
    findings: list[OperationRuleFinding] = Field(default_factory=list)

    @property
    def blocking_findings(self) -> list[OperationRuleFinding]:
        return [item for item in self.findings if item.severity != "warning"]

    @property
    def warnings(self) -> list[OperationRuleFinding]:
        return [item for item in self.findings if item.severity == "warning"]


RULE_TABLE: dict[str, OperationRuleDefinition] = {
    "required_inputs_must_exist": OperationRuleDefinition(
        rule_id="required_inputs_must_exist",
        severity="hard_error",
        description_zh_tw="需要上游資料的 operation，必須連到能產生該資料的先前步驟。",
    ),
    "operation_entity_type_compatibility": OperationRuleDefinition(
        rule_id="operation_entity_type_compatibility",
        severity="hard_error",
        description_zh_tw="operation 直接引用的實體類型必須符合該 operation 的用途。",
    ),
    "explicit_aliases_are_one_entity": OperationRuleDefinition(
        rule_id="explicit_aliases_are_one_entity",
        severity="warning",
        description_zh_tw="問題中明示的同義名稱不應被當成兩個研究對象。",
    ),
    "analysis_requires_upstream_data": OperationRuleDefinition(
        rule_id="analysis_requires_upstream_data",
        severity="hard_error",
        description_zh_tw="分析 operation 只能處理上游結果，不能以資料來源欄位代替擷取步驟。",
    ),
    "preserve_explicit_comparison_groups": OperationRuleDefinition(
        rule_id="preserve_explicit_comparison_groups",
        severity="warning",
        description_zh_tw="使用者明示要分別比較的群組，篩選與彙整時都應保持分開。",
    ),
    "constraints_require_authority": OperationRuleDefinition(
        rule_id="constraints_require_authority",
        severity="warning",
        description_zh_tw="Plan 新增的數值門檻必須說明來源，不能假裝是使用者要求。",
    ),
    "capability_requires_safeguards": OperationRuleDefinition(
        rule_id="capability_requires_safeguards",
        severity="repair_required",
        description_zh_tw="使用特定研究能力時，必須同時加入相應的解讀與缺失值保護。",
    ),
    "analysis_metric_requires_available_evidence": OperationRuleDefinition(
        rule_id="analysis_metric_requires_available_evidence",
        severity="hard_error",
        description_zh_tw=(
            "篩選條件須由 operation 契約支援；排名或彙整指標須已登記且由本步驟或上游提供。"
        ),
    ),
    "declared_entities_must_be_question_grounded": OperationRuleDefinition(
        rule_id="declared_entities_must_be_question_grounded",
        severity="hard_error",
        description_zh_tw=(
            "頂層實體必須來自原始問題；步驟輸出、集合與統計名稱不得偽裝成實體。"
        ),
    ),
    "operation_must_honor_capability_contract": OperationRuleDefinition(
        rule_id="operation_must_honor_capability_contract",
        severity="hard_error",
        description_zh_tw=(
            "已知 operation 必須填齊其必要結構，且不得聲稱其他 operation 專屬的輸出能力。"
        ),
    ),
    "inputs_must_reference_upstream_outputs": OperationRuleDefinition(
        rule_id="inputs_must_reference_upstream_outputs",
        severity="hard_error",
        description_zh_tw="step.inputs 只能引用上游步驟已宣告在 outputs 的名稱。",
    ),
}


# Only reviewed, unambiguous metrics belong here. Unknown/custom metrics are not
# guessed; they remain eligible for human review instead of being falsely blocked.
_METRIC_REQUIREMENTS: dict[str, str] = {
    "association_confidence": "disease_target_association",
    "association_score": "disease_target_association",
    "overall_association_score": "disease_target_association",
    "min_evidence_score": "disease_target_association",
    "datasource_score": "disease_target_association",
    "drug_count": "target_drug_evidence",
    "known_drug_count": "target_drug_evidence",
    "tractability_score": "tractability_evidence",
    "clinical_phase": "clinical_stage_evidence",
    "highest_clinical_stage": "clinical_stage_evidence",
    "safety_risk": "safety_evidence",
    "normal_tissue_expression": "normal_tissue_expression",
    "tissue_expression": "normal_tissue_expression",
    "drug_approval_status": "approval_evidence",
    "approval_status": "approval_evidence",
    "approved_indication_count": "approval_evidence",
}

_RESERVED_OUTPUT_CLAIMS = {
    output: operation
    for operation, contract in OPERATION_TABLE.items()
    for output in contract.reserved_outputs
}


def build_metric_catalog_prompt() -> str:
    """Expose the same reviewed metric names that validation enforces."""
    return "Reviewed metrics (metric -> required evidence capability):\n" + "\n".join(
        f"{metric} -> {capability}"
        for metric, capability in sorted(_METRIC_REQUIREMENTS.items())
    )


def _finding(rule_id: str, step: ResearchPlanStepV2 | None, problem: str,
             correction: str, missing: list[str] | None = None) -> OperationRuleFinding:
    rule = RULE_TABLE[rule_id]
    return OperationRuleFinding(
        rule_id=rule_id,
        severity=rule.severity,
        step_id=step.step_id if step else None,
        operation=step.operation if step else None,
        problem=problem,
        required_correction=correction,
        missing=missing or [],
    )


def _ancestor_steps(plan: QueryManifestV2, step: ResearchPlanStepV2) -> list[ResearchPlanStepV2]:
    by_id = {item.step_id: item for item in plan.plan_steps}
    seen: set[str] = set()
    stack = list(step.depends_on)
    while stack:
        step_id = stack.pop()
        if step_id in seen or step_id not in by_id:
            continue
        seen.add(step_id)
        stack.extend(by_id[step_id].depends_on)
    return [item for item in plan.plan_steps if item.step_id in seen]


def _upstream_capabilities(plan: QueryManifestV2, step: ResearchPlanStepV2) -> set[str]:
    return {
        capability
        for ancestor in _ancestor_steps(plan, step)
        for capability in (
            OPERATION_TABLE[ancestor.operation].provides
            if ancestor.operation in OPERATION_TABLE else []
        )
    }


def _all_safeguards(plan: QueryManifestV2) -> set[str]:
    return set(plan.global_safeguards) | {
        item for step in plan.plan_steps for item in step.safeguards
    }


def _numeric_values(value: Any) -> list[str]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _numeric_values(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _numeric_values(nested)]
    return []


def _grounding_text(value: str) -> str:
    """Normalize case, width, whitespace and punctuation for literal grounding."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _known_metrics(value: Any) -> set[str]:
    """Collect only reviewed metric identifiers from nested plan controls."""

    found: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in _METRIC_REQUIREMENTS:
                found.add(key)
            found.update(_known_metrics(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(_known_metrics(nested))
    elif isinstance(value, str) and value in _METRIC_REQUIREMENTS:
        found.add(value)
    return found


def _contains_aggregation_placeholder(value: Any) -> bool:
    """Detect a wholly undefined method, not optional empty subfields."""
    placeholders = {"reserved", "required", "placeholder", "tbd", "todo"}
    if isinstance(value, str):
        return not value.strip() or value.strip().casefold() in placeholders
    if isinstance(value, dict):
        if not value:
            return True
        return all(_contains_aggregation_placeholder(item) for item in value.values())
    if isinstance(value, list):
        if not value:
            return True
        return all(_contains_aggregation_placeholder(item) for item in value)
    return value is None


def evaluate_operation_rules(plan: QueryManifestV2) -> OperationRuleEvaluation:
    """Apply the reviewed general rules without using a case-specific rubric."""

    findings: list[OperationRuleFinding] = []
    entity_types = {item.entity_ref: item.entity_type for item in plan.entities}
    normalized_question = _grounding_text(plan.original_question)
    output_names = {
        _grounding_text(output)
        for step in plan.plan_steps
        for output in step.outputs
    }

    for entity in plan.entities:
        normalized_mention = _grounding_text(entity.mention)
        normalized_ref = _grounding_text(entity.entity_ref)
        not_in_question = (
            not normalized_mention or normalized_mention not in normalized_question
        )
        duplicates_output = (
            normalized_mention in output_names or normalized_ref in output_names
        )
        if not_in_question or duplicates_output:
            reasons = []
            if not_in_question:
                reasons.append("mention 無法在原始問題中找到")
            if duplicates_output:
                reasons.append("名稱同時被當成步驟 output")
            findings.append(_finding(
                "declared_entities_must_be_question_grounded", None,
                f"實體 {entity.entity_ref} 不具問題根據：{'；'.join(reasons)}。",
                "只保留問題中明示的實體；資料集合放在 outputs，後續僅以 depends_on 傳遞。",
                [entity.entity_ref],
            ))

    for step in plan.plan_steps:
        ancestor_outputs = {
            ancestor.step_id: set(ancestor.outputs)
            for ancestor in _ancestor_steps(plan, step)
        }
        for input_name, ref in step.inputs.items():
            if ref.output not in ancestor_outputs.get(ref.step_id, set()):
                findings.append(_finding(
                    "inputs_must_reference_upstream_outputs", step,
                    f"input {input_name!r} 引用了不存在的上游輸出 {ref.step_id}.{ref.output}。",
                    "改引用 depends_on 鏈上某個步驟已宣告在 outputs 的名稱。",
                    [f"{ref.step_id}.{ref.output}"],
                ))

        contract = OPERATION_TABLE.get(step.operation)
        if contract is None:
            continue
        allowed = set(contract.allowed_entity_types)
        incompatible = sorted(
            {entity_types[ref] for ref in step.entity_refs if ref in entity_types}
            - allowed
        ) if allowed else []
        if incompatible:
            findings.append(_finding(
                "operation_entity_type_compatibility", step,
                f"{step.operation} 引用了不相容的實體類型：{', '.join(incompatible)}。",
                f"改用 {', '.join(sorted(allowed))} 實體；若實體本身標錯，修正 entities 的 entity_type。",
                incompatible,
            ))

        upstream = _upstream_capabilities(plan, step)
        required_all = set(contract.requires_all)
        missing_all = sorted(required_all - upstream)
        if missing_all:
            findings.append(_finding(
                "required_inputs_must_exist", step,
                f"{step.operation} 缺少必要的全部上游輸入：{', '.join(missing_all)}。",
                "先加入能產生每項必要資料的 operation，並用 depends_on 連接。",
                missing_all,
            ))
        required_any = set(contract.requires_any)
        if required_any and not upstream.intersection(required_any):
            findings.append(_finding(
                "required_inputs_must_exist", step,
                f"{step.operation} 沒有連到可提供所需輸入的上游步驟。",
                "先加入產生所需資料的 operation，並用 depends_on 連接。",
                sorted(required_any),
            ))

        if contract.operation_kind == "analysis" and step.sources:
            findings.append(_finding(
                "analysis_requires_upstream_data", step,
                f"分析步驟 {step.operation} 直接填了資料來源，可能以分析代替資料擷取。",
                "把資料來源放到上游 retrieve/discover 步驟；本步只透過 depends_on 使用其輸出。",
            ))

        missing_aggregation = (
            contract.requires_aggregation
            and not step.aggregation
        )
        placeholder_aggregation = (
            contract.requires_aggregation
            and _contains_aggregation_placeholder(step.aggregation)
        )
        claimed_elsewhere = sorted(
            output
            for output in step.outputs
            if output in _RESERVED_OUTPUT_CLAIMS
            and _RESERVED_OUTPUT_CLAIMS[output] != step.operation
        )
        if missing_aggregation or placeholder_aggregation or claimed_elsewhere:
            details = []
            if missing_aggregation:
                details.append("缺少描述分層或彙整方法的 aggregation")
            if placeholder_aggregation:
                details.append("aggregation 使用 reserved/required/TBD 等空泛 placeholder")
            if claimed_elsewhere:
                details.append(
                    "錯誤聲稱其他 operation 的輸出：" + ", ".join(claimed_elsewhere)
                )
            findings.append(_finding(
                "operation_must_honor_capability_contract", step,
                f"{step.operation} 未遵守能力契約：{'；'.join(details)}。",
                "使用正確的 canonical operation，填入實際的分層或彙整方法；不要只改 output 名稱冒充能力。",
                (["aggregation"] if missing_aggregation or placeholder_aggregation else [])
                + claimed_elsewhere,
            ))

        available = upstream | set(contract.provides)
        unsupported_filters = sorted(
            set(step.filters) - set(contract.supported_filters)
        )
        controls: list[Any] = [step.filters, step.aggregation or {}]
        ranking_metrics: list[str] = []
        if step.ranking is not None:
            ranking_metrics = [
                step.ranking.primary_metric, *step.ranking.secondary_metrics
            ]
            controls.append(
                ranking_metrics
            )
        unregistered_metrics = sorted(
            metric for metric in ranking_metrics
            if metric not in _METRIC_REQUIREMENTS
        )
        used_metrics = set().union(*(_known_metrics(item) for item in controls))
        unavailable = sorted(
            metric
            for metric in used_metrics
            if _METRIC_REQUIREMENTS[metric] not in available
        )
        if unsupported_filters or unregistered_metrics or unavailable:
            requirements = sorted({_METRIC_REQUIREMENTS[item] for item in unavailable})
            details = []
            if unsupported_filters:
                details.append("未登記 filters：" + ", ".join(unsupported_filters))
            if unregistered_metrics:
                details.append("未登記 metrics：" + ", ".join(unregistered_metrics))
            if unavailable:
                details.append("尚未取得 metrics：" + ", ".join(unavailable))
            findings.append(_finding(
                "analysis_metric_requires_available_evidence", step,
                f"{step.operation} 使用未核准或尚不可用的分析控制：{'；'.join(details)}。",
                "只使用 operation table 支援的 filter 與 metric registry 已登記且有上游證據的指標；否則標記為待人工審核。",
                [*unsupported_filters, *unregistered_metrics, *requirements],
            ))

    # Conservative alias warning: text outside parentheses followed by an alias group.
    question = plan.original_question
    for outside, inside in re.findall(r"([^（）()，,。?？]{1,60})[（(]([^）)]{1,100})[）)]", question):
        terms = [outside.strip().split()[-1], *re.split(r"[,，/]", inside)]
        matching = [
            entity for entity in plan.entities
            if any(entity.mention.casefold() == term.strip().casefold() for term in terms)
        ]
        if len(matching) >= 2 and len({item.entity_type for item in matching}) == 1:
            findings.append(_finding(
                "explicit_aliases_are_one_entity", None,
                "括號內外的名稱可能是同一實體，Plan 卻建立了多個實體。",
                "保留一個 entity_ref，將其他名稱當別名；若確為不同實體，交由人工確認。",
                [item.entity_ref for item in matching],
            ))
            break

    # Explicit variant groups must remain visible in both filtering and grouping.
    variant_mentions = [
        item for item in plan.entities
        if item.entity_type == "target" and re.search(r"\b[A-Z]\d+[A-Z]\b", item.mention.upper())
    ]
    if len(variant_mentions) >= 2 and any(token in question.casefold() for token in ("各自", "分別", "each", "respectively")):
        filter_keys = {key for step in plan.plan_steps for key in step.filters}
        aggregation_keys = {key for step in plan.plan_steps for key in (step.aggregation or {})}
        missing = []
        if "target_variant" not in filter_keys:
            missing.append("target_variant")
        if "group_by_variant" not in aggregation_keys:
            missing.append("group_by_variant")
        if missing:
            findings.append(_finding(
                "preserve_explicit_comparison_groups", None,
                f"使用者要求分別處理多個變異，但 Plan 未完整保留分組：{', '.join(missing)}。",
                "在 filters 保留 target_variant，並在 aggregation 使用 group_by_variant。",
                missing,
            ))

    for step in plan.plan_steps:
        ungrounded = [value for value in _numeric_values(step.filters) if value not in question]
        if ungrounded:
            findings.append(_finding(
                "constraints_require_authority", step,
                f"Plan 新增了問題中沒有的數值門檻：{', '.join(ungrounded)}。",
                "說明門檻的外部依據，或移除該門檻；不要聲稱它是使用者指定。",
                ungrounded,
            ))

    safeguards = _all_safeguards(plan)
    operations = {step.operation for step in plan.plan_steps}
    required_safeguards = {
        safeguard
        for operation in operations
        for safeguard in (
            OPERATION_TABLE[operation].required_safeguards
            if operation in OPERATION_TABLE else []
        )
    }
    missing_safeguards = sorted(required_safeguards - safeguards)
    if missing_safeguards:
        findings.append(_finding(
            "capability_requires_safeguards", None,
            f"Plan 使用的研究能力缺少必要 safeguards：{', '.join(missing_safeguards)}。",
            "把列出的 safeguard 識別碼加入 global_safeguards 或相關步驟；不得用一般描述取代。",
            missing_safeguards,
        ))

    return OperationRuleEvaluation(
        passed=not any(item.severity != "warning" for item in findings),
        findings=findings,
    )
