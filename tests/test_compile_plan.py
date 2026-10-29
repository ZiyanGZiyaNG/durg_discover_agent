import json
from pathlib import Path

import pytest

from execution.__main__ import main
from execution.compiler import PlanRefusedError, compile_plan
from schemas import QueryManifestV2

PLANS = Path(__file__).resolve().parent.parent / "outputs" / "plans"
KNOWN_TARGETS = PLANS / "phase2_pdac_known_targets_batched_v4.json"


def _base():
    return json.loads(KNOWN_TARGETS.read_text())


def _step(step_id, operation, depends_on, entity_refs, outputs, **extra):
    return {
        "step_id": step_id,
        "operation": operation,
        "description": operation,
        "depends_on": depends_on,
        "entity_refs": entity_refs,
        "outputs": outputs,
        **extra,
    }


def _entity(ref, entity_type, mention):
    return {
        "entity_ref": ref,
        "entity_type": entity_type,
        "mention": mention,
        "entity_id": None,
        "canonical_name": None,
        "resolution_status": "unresolved",
    }


def test_known_targets_plan_compiles_to_ordered_tool_calls():
    calls = compile_plan(QueryManifestV2.model_validate(_base()))
    assert [call.operation for call in calls] == [
        "resolve_disease",
        "discover_targets",
        "retrieve_target_drugs",
        "retrieve_tractability",
        "stratify_target_evidence_layers",
        "rank_targets",
    ]
    by_step = {call.step_id: call for call in calls}
    assert by_step["discover_targets_1"].adapter == "DiscoverTargetsAdapter"
    assert by_step["discover_targets_1"].requires_all == ["resolved_disease"]
    assert by_step["rank_targets_1"].ranking["primary_metric"] == "overall_association_score"
    assert by_step["stratify_target_evidence_layers_1"].output_model == "StratifiedEvidence"


def test_filter_step_compiles_with_its_conditions():
    data = _base()
    data["plan_steps"].append(
        _step(
            "filter_1",
            "filter_targets",
            ["stratify_target_evidence_layers_1"],
            [],
            ["filtered_targets"],
            filters={"tractable_modalities": {"contains": "SM"}},
        )
    )
    calls = compile_plan(QueryManifestV2.model_validate(data))
    assert calls[-1].adapter == "FilterTargetsAdapter"
    assert calls[-1].filters == {"tractable_modalities": {"contains": "SM"}}
    assert calls[-1].output_model == "FilteredTargets"


def test_olaparib_manifest_compiles_through_the_drug_adapters():
    data = _base()
    data["original_question"] = "Olaparib 的標的是什麼？目前核准了哪些適應症？胰臟癌是其中之一嗎？"
    data["research_objective"] = "查明 Olaparib 的標靶、核准適應症並核對 PDAC。"
    data["entities"] = [
        _entity("entity_1", "drug", "Olaparib"),
        _entity("entity_2", "disease", "胰臟癌"),
    ]
    data["plan_steps"] = [
        _step("resolve_drug_1", "resolve_drug", [], ["entity_1"], ["resolved_drug_1"]),
        _step("resolve_disease_1", "resolve_disease", [], ["entity_2"], ["resolved_disease_1"]),
        _step("retrieve_drug_targets_1", "retrieve_drug_targets", ["resolve_drug_1"], ["entity_1"], ["target_drug_evidence_1"]),
        _step("retrieve_approved_indications_1", "retrieve_approved_indications", ["resolve_drug_1"], ["entity_1"], ["approval_evidence_1"]),
        _step("check_indication_1", "check_indication", ["retrieve_approved_indications_1", "resolve_disease_1"], ["entity_2"], ["indication_check_1"]),
    ]
    calls = compile_plan(QueryManifestV2.model_validate(data))
    assert [call.adapter for call in calls] == [
        "ResolveEntityAdapter",
        "ResolveEntityAdapter",
        "RetrieveDrugTargetsAdapter",
        "RetrieveApprovedIndicationsAdapter",
        "CheckIndicationAdapter",
    ]
    check = calls[-1]
    assert check.requires_all == ["approval_evidence"]
    assert check.provides == ["indication_check"]
    assert check.entities["entity_2"]["mention"] == "胰臟癌"
    assert check.output_model == "IndicationChecks"


def test_operation_without_an_adapter_is_refused():
    data = _base()
    data["plan_steps"] = [
        data["plan_steps"][0],
        _step("trials_1", "retrieve_clinical_trials", ["resolve_disease_1"], ["entity_1"], ["clinical_trial_evidence_1"]),
    ]
    with pytest.raises(PlanRefusedError) as error:
        compile_plan(QueryManifestV2.model_validate(data))
    assert any("no adapter" in problem for problem in error.value.problems)


def test_step_whose_input_is_never_provided_is_refused():
    data = _base()
    data["plan_steps"] = [
        data["plan_steps"][0],
        _step("tractability_1", "retrieve_tractability", ["resolve_disease_1"], ["entity_1"], ["tractability_evidence_1"]),
    ]
    with pytest.raises(PlanRefusedError) as error:
        compile_plan(QueryManifestV2.model_validate(data))
    assert any("no earlier step provides" in problem for problem in error.value.problems)


def test_dry_run_prints_the_tool_calls_without_running_anything(capsys):
    code = main([str(KNOWN_TARGETS), "--dry-run"])
    calls = json.loads(capsys.readouterr().out)
    assert code == 0
    assert calls[0]["operation"] == "resolve_disease"
    assert calls[-1]["operation"] == "rank_targets"