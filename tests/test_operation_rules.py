from copy import deepcopy

from operation_rules import RULE_TABLE, evaluate_operation_rules
from schemas import QueryManifestV2
from test_natural_language_planner import QUESTION, _valid_payload


def _evaluate(payload=None):
    return evaluate_operation_rules(
        QueryManifestV2.model_validate(payload or _valid_payload())
    )


def _rule_ids(result):
    return {item.rule_id for item in result.findings}


def test_rule_table_has_seven_blocking_and_three_warning_rules():
    assert len(RULE_TABLE) == 10
    assert sum(rule.severity == "warning" for rule in RULE_TABLE.values()) == 3
    assert sum(rule.severity != "warning" for rule in RULE_TABLE.values()) == 7


def test_complete_therapeutic_target_plan_passes_general_rules():
    result = _evaluate()
    assert result.passed is True
    assert result.findings == []


def test_required_operation_input_must_come_from_an_upstream_step():
    payload = _valid_payload()
    payload["plan_steps"][1]["depends_on"] = []
    result = _evaluate(payload)
    assert "required_inputs_must_exist" in _rule_ids(result)
    assert result.passed is False


def test_resolution_operation_rejects_wrong_entity_type():
    payload = _valid_payload()
    payload["plan_steps"][0]["operation"] = "resolve_target_variants"
    payload["global_safeguards"].append("variant_specificity_required")
    result = _evaluate(payload)
    assert "operation_entity_type_compatibility" in _rule_ids(result)


def test_analysis_sources_do_not_replace_retrieval_steps():
    payload = _valid_payload()
    payload["plan_steps"][4]["sources"] = ["open_targets"]
    result = _evaluate(payload)
    assert "analysis_requires_upstream_data" in _rule_ids(result)


def test_target_stratification_requires_all_three_evidence_layers():
    payload = _valid_payload()
    payload["plan_steps"][4]["depends_on"] = ["step_4"]
    result = _evaluate(payload)
    finding = next(
        item
        for item in result.findings
        if item.rule_id == "required_inputs_must_exist" and item.step_id == "step_5"
    )
    assert finding.missing == ["target_drug_evidence"]


def test_capability_specific_safeguards_are_repair_required():
    payload = _valid_payload()
    payload["global_safeguards"].remove("missing_not_negative")
    result = _evaluate(payload)
    finding = next(
        item
        for item in result.findings
        if item.rule_id == "capability_requires_safeguards"
    )
    assert finding.severity == "repair_required"
    assert finding.missing == ["missing_not_negative"]
    assert result.passed is False


def test_explicit_parenthetical_aliases_warn_but_do_not_block():
    payload = _valid_payload(
        "胰臟癌（pancreatic ductal adenocarcinoma）有哪些治療標的？"
    )
    payload["entities"].append(
        {
            "entity_ref": "disease_2",
            "entity_type": "disease",
            "mention": "pancreatic ductal adenocarcinoma",
            "resolution_status": "unresolved",
        }
    )
    result = _evaluate(payload)
    assert "explicit_aliases_are_one_entity" in _rule_ids(result)
    assert result.passed is True


def test_explicit_variant_groups_warn_when_grouping_is_lost():
    question = "KRAS G12C 與 KRAS G12D 各自有哪些藥物？"
    payload = _valid_payload(question)
    payload["entities"] = [
        {
            "entity_ref": "target_1",
            "entity_type": "target",
            "mention": "KRAS G12C",
            "resolution_status": "unresolved",
        },
        {
            "entity_ref": "target_2",
            "entity_type": "target",
            "mention": "KRAS G12D",
            "resolution_status": "unresolved",
        },
    ]
    payload["plan_steps"] = [
        {
            "step_id": "step_1",
            "operation": "resolve_target_variants",
            "description": "解析兩個變異。",
            "entity_refs": ["target_1", "target_2"],
            "outputs": ["resolved_variants"],
        }
    ]
    payload["global_safeguards"] = ["variant_specificity_required"]
    result = _evaluate(payload)
    assert "preserve_explicit_comparison_groups" in _rule_ids(result)
    assert result.passed is True


def test_unrequested_unregistered_numeric_filter_warns_and_requires_review():
    payload = deepcopy(_valid_payload(QUESTION))
    payload["plan_steps"][1]["filters"] = {"minimum_score": 0.7}
    result = _evaluate(payload)
    assert "constraints_require_authority" in _rule_ids(result)
    assert "analysis_metric_requires_available_evidence" in _rule_ids(result)
    assert result.passed is False


def test_ranking_cannot_use_evidence_retrieved_only_by_a_later_step():
    payload = _valid_payload()
    payload["plan_steps"][1]["ranking"] = {
        "primary_metric": "overall_association_score",
        "direction": "descending",
        "secondary_metrics": ["tractability_score"],
    }
    result = _evaluate(payload)
    finding = next(
        item
        for item in result.findings
        if item.rule_id == "analysis_metric_requires_available_evidence"
    )
    assert finding.step_id == "step_2"
    assert finding.missing == ["tractability_evidence"]
    assert result.passed is False


def test_ranking_may_use_metrics_provided_by_ancestor_steps():
    payload = _valid_payload()
    payload["plan_steps"][5]["ranking"]["secondary_metrics"] = [
        "tractability_score",
        "drug_count",
    ]
    result = _evaluate(payload)
    assert "analysis_metric_requires_available_evidence" not in _rule_ids(result)
    assert result.passed is True


def test_unknown_custom_metric_requires_review_instead_of_being_ignored():
    payload = _valid_payload()
    payload["plan_steps"][5]["ranking"]["secondary_metrics"] = [
        "future_reviewed_metric"
    ]
    result = _evaluate(payload)
    finding = next(
        item
        for item in result.findings
        if item.rule_id == "analysis_metric_requires_available_evidence"
    )
    assert finding.missing == ["future_reviewed_metric"]
    assert result.passed is False


def test_exact_pair_requires_both_resolved_disease_and_target():
    payload = _valid_payload("胰臟癌與 KRAS 的關聯？")
    payload["entities"].append(
        {
            "entity_ref": "target_1",
            "entity_type": "target",
            "mention": "KRAS",
            "resolution_status": "unresolved",
        }
    )
    payload["plan_steps"] = [
        payload["plan_steps"][0],
        {
            "step_id": "step_pair",
            "operation": "retrieve_exact_pair",
            "description": "查詢精確配對。",
            "depends_on": ["step_1"],
            "entity_refs": ["disease_1", "target_1"],
            "sources": ["open_targets"],
            "outputs": ["pair_evidence"],
        },
    ]
    result = _evaluate(payload)
    finding = next(
        item for item in result.findings if item.rule_id == "required_inputs_must_exist"
    )
    assert finding.missing == ["resolved_target"]


def test_generated_collection_cannot_be_declared_as_a_target_entity():
    payload = _valid_payload()
    payload["entities"].append(
        {
            "entity_ref": "pdac_associated_targets",
            "entity_type": "target",
            "mention": "pdac_associated_targets",
            "resolution_status": "unresolved",
        }
    )
    payload["plan_steps"][1]["outputs"] = ["pdac_associated_targets"]
    result = _evaluate(payload)
    finding = next(
        item
        for item in result.findings
        if item.rule_id == "declared_entities_must_be_question_grounded"
    )
    assert finding.missing == ["pdac_associated_targets"]
    assert result.passed is False


def test_explicit_full_name_and_parenthetical_alias_are_grounded():
    question = "胰臟癌（pancreatic ductal adenocarcinoma, PDAC）有哪些治療標的？"
    payload = _valid_payload(question)
    payload["entities"] = [
        {
            "entity_ref": "disease_1",
            "entity_type": "disease",
            "mention": "胰臟癌",
            "resolution_status": "unresolved",
        },
        {
            "entity_ref": "disease_2",
            "entity_type": "disease",
            "mention": "pancreatic ductal adenocarcinoma (PDAC)",
            "resolution_status": "unresolved",
        },
    ]
    result = _evaluate(payload)
    assert "declared_entities_must_be_question_grounded" not in _rule_ids(result)


def test_entity_reference_name_may_differ_from_grounded_mention():
    payload = _valid_payload()
    payload["entities"][0]["entity_ref"] = "primary_disease"
    for step in payload["plan_steps"]:
        step["entity_refs"] = [
            "primary_disease" if ref == "disease_1" else ref
            for ref in step["entity_refs"]
        ]
    result = _evaluate(payload)
    assert "declared_entities_must_be_question_grounded" not in _rule_ids(result)


def test_category_aggregation_cannot_impersonate_target_stratification():
    payload = _valid_payload()
    payload["plan_steps"][4]["aggregation"] = None
    payload["plan_steps"].insert(
        5,
        {
            "step_id": "step_aggregate",
            "operation": "aggregate_category_distribution",
            "description": "彙整類別分布。",
            "depends_on": ["step_5"],
            "entity_refs": ["disease_1"],
            "outputs": ["target_translation_layer"],
            "aggregation": {"category_distribution": {"metrics": ["count"]}},
        },
    )
    payload["plan_steps"][6]["depends_on"] = ["step_aggregate"]
    result = _evaluate(payload)
    findings = [
        item
        for item in result.findings
        if item.rule_id == "operation_must_honor_capability_contract"
    ]
    assert {item.step_id for item in findings} == {"step_5", "step_aggregate"}
    assert result.passed is False


def test_category_aggregation_with_its_own_contract_is_valid():
    payload = _valid_payload()
    payload["plan_steps"].insert(
        5,
        {
            "step_id": "step_aggregate",
            "operation": "aggregate_category_distribution",
            "description": "彙整類別分布。",
            "depends_on": ["step_5"],
            "entity_refs": ["disease_1"],
            "outputs": ["category_counts"],
            "aggregation": {"category_distribution": {"metrics": ["count"]}},
        },
    )
    result = _evaluate(payload)
    assert "operation_must_honor_capability_contract" not in _rule_ids(result)


def test_stratification_contract_does_not_require_a_case_specific_key_name():
    payload = _valid_payload()
    payload["plan_steps"][4]["aggregation"] = {
        "evidence_tiers": {"dimensions": ["association", "drug", "tractability"]}
    }
    result = _evaluate(payload)
    assert "operation_must_honor_capability_contract" not in _rule_ids(result)


def test_required_aggregation_rejects_non_executable_placeholder():
    payload = _valid_payload()
    payload["plan_steps"][4]["aggregation"] = {"target_translation_layer": "reserved"}
    result = _evaluate(payload)

    finding = next(
        item
        for item in result.findings
        if item.rule_id == "operation_must_honor_capability_contract"
    )
    assert finding.step_id == "step_5"
    assert finding.missing == ["aggregation"]
    assert result.passed is False


def test_required_aggregation_rejects_empty_nested_method():
    payload = _valid_payload()
    payload["plan_steps"][4]["aggregation"] = {"target_translation_layer": {}}
    result = _evaluate(payload)

    assert "operation_must_honor_capability_contract" in _rule_ids(result)
    assert result.passed is False


def test_drug_approval_status_requires_approval_evidence():
    payload = _valid_payload()
    payload["plan_steps"][5]["ranking"]["secondary_metrics"] = ["drug_approval_status"]
    result = _evaluate(payload)
    finding = next(
        item
        for item in result.findings
        if item.rule_id == "analysis_metric_requires_available_evidence"
    )
    assert finding.missing == ["approval_evidence"]
    assert result.passed is False


def test_drug_approval_status_is_available_after_indication_retrieval():
    payload = _valid_payload()
    payload["plan_steps"].insert(
        5,
        {
            "step_id": "step_approval",
            "operation": "retrieve_approved_indications",
            "description": "取得核准適應症證據。",
            "depends_on": ["step_3"],
            "entity_refs": ["disease_1"],
            "sources": ["open_targets"],
            "outputs": ["approval_evidence"],
        },
    )
    payload["plan_steps"][6]["depends_on"] = ["step_5", "step_approval"]
    payload["plan_steps"][6]["ranking"]["secondary_metrics"] = ["drug_approval_status"]
    result = _evaluate(payload)
    assert "analysis_metric_requires_available_evidence" not in _rule_ids(result)
