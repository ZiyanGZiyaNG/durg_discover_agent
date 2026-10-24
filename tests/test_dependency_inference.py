"""Offline tests for contract-derived dependency inference (no LLM, no retrieval)."""

from copy import deepcopy
import json

from langchain_core.messages import AIMessage

from dependency_inference import infer_step_dependencies
from natural_language_planner import (
    DirectJsonPlanAgent,
    QueryManifestPatch,
    apply_manifest_patch,
)
from operation_rules import evaluate_operation_rules
from schemas import QueryManifestV2
from test_natural_language_planner import _valid_payload


def _step(step_id, operation, depends_on=(), entity_refs=()):
    return {
        "step_id": step_id,
        "operation": operation,
        "depends_on": list(depends_on),
        "entity_refs": list(entity_refs),
    }


def _deps(steps):
    return {step["step_id"]: step["depends_on"] for step in steps}


def test_missing_edge_is_added_from_the_operation_contract():
    payload = _valid_payload()
    payload["plan_steps"][1]["depends_on"] = []
    steps, corrections, unresolved = infer_step_dependencies(payload["plan_steps"])
    assert _deps(steps)["step_2"] == ["step_1"]
    assert any("added dependency 'step_1'" in item for item in corrections)
    assert unresolved == []
    payload["plan_steps"] = steps
    assert evaluate_operation_rules(QueryManifestV2.model_validate(payload)).passed


def test_reversed_edge_is_removed_and_the_correct_edge_added():
    payload = _valid_payload()
    payload["plan_steps"][0]["depends_on"] = ["step_2"]  # resolve after discover
    payload["plan_steps"][1]["depends_on"] = []
    steps, corrections, _ = infer_step_dependencies(payload["plan_steps"])
    assert _deps(steps)["step_1"] == []
    assert _deps(steps)["step_2"] == ["step_1"]
    assert any("reversed" in item for item in corrections)
    payload["plan_steps"] = steps
    QueryManifestV2.model_validate(payload)  # no non-prior dependency error


def test_future_reference_is_compiled_into_a_valid_order():
    payload = _valid_payload()
    payload["plan_steps"].reverse()
    steps, corrections, _ = infer_step_dependencies(payload["plan_steps"])
    position = {item["step_id"]: index for index, item in enumerate(steps)}
    assert all(
        position[dependency] < position[item["step_id"]]
        for item in steps for dependency in item["depends_on"]
    )
    assert steps[0]["step_id"] == "step_1"
    assert any(item.startswith("reordered steps") for item in corrections)


def test_an_already_correct_plan_is_left_unchanged():
    payload = _valid_payload()
    steps, corrections, unresolved = infer_step_dependencies(payload["plan_steps"])
    assert [item["step_id"] for item in steps] == [
        item["step_id"] for item in payload["plan_steps"]
    ]
    assert _deps(steps) == {
        item["step_id"]: item.get("depends_on", []) for item in payload["plan_steps"]
    }
    assert corrections == [] and unresolved == []


def test_several_producers_are_not_guessed_without_entity_evidence():
    steps = [
        _step("a", "resolve_disease", entity_refs=["disease_a"]),
        _step("b", "resolve_disease", entity_refs=["disease_b"]),
        _step("c", "discover_targets"),
    ]
    result, corrections, unresolved = infer_step_dependencies(steps)
    assert _deps(result)["c"] == []
    assert corrections == []
    assert "not inferred" in unresolved[0]


def test_entity_overlap_selects_every_matching_producer():
    steps = [
        _step("a", "resolve_disease", entity_refs=["disease_a"]),
        _step("b", "resolve_disease", entity_refs=["disease_b"]),
        _step("c", "discover_targets_per_disease",
              entity_refs=["disease_a", "disease_b"]),
    ]
    result, _, unresolved = infer_step_dependencies(steps)
    assert _deps(result)["c"] == ["a", "b"]
    assert unresolved == []


def test_missing_producer_is_reported_and_never_invented():
    steps = [_step("only", "discover_targets")]
    result, corrections, unresolved = infer_step_dependencies(steps)
    assert [item["step_id"] for item in result] == ["only"]
    assert corrections == []
    assert "resolved_disease" in unresolved[0]


def test_declared_cycle_is_reported_and_order_kept():
    steps = [
        _step("x", "retrieve_clinical_trials", depends_on=["y"]),
        _step("y", "retrieve_safety_evidence", depends_on=["x"]),
    ]
    result, _, unresolved = infer_step_dependencies(steps)
    assert [item["step_id"] for item in result] == ["x", "y"]
    assert any("cycle" in item for item in unresolved)


def test_unknown_references_are_left_for_schema_validation():
    steps = [_step("a", "resolve_disease", depends_on=["ghost"])]
    result, corrections, _ = infer_step_dependencies(steps)
    assert _deps(result)["a"] == ["ghost"]
    assert corrections == []


def test_input_steps_are_not_mutated():
    payload = _valid_payload()
    payload["plan_steps"][1]["depends_on"] = []
    original = deepcopy(payload["plan_steps"])
    infer_step_dependencies(payload["plan_steps"])
    assert payload["plan_steps"] == original


class _FakeModel:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, messages, config=None):
        return AIMessage(content=json.dumps(self.payload, ensure_ascii=False))


def test_full_plan_agent_repairs_edges_and_order_without_another_model_call():
    payload = _valid_payload()
    payload["plan_steps"][1]["depends_on"] = []  # missing resolution edge
    payload["plan_steps"][4]["depends_on"] = ["step_3", "step_4"]
    payload["plan_steps"].reverse()  # every dependency now points forward
    result = DirectJsonPlanAgent(_FakeModel(payload)).invoke({"messages": []})
    plan = result["structured_response"]
    assert evaluate_operation_rules(plan).passed
    assert any("added dependency" in item for item in plan.reference_corrections)


def test_patch_merge_reinstates_contract_edges_and_records_them():
    base = QueryManifestV2.model_validate(_valid_payload())
    broken = base.plan_steps[3].model_copy(update={"depends_on": []})
    patch = QueryManifestPatch(upsert_steps=[broken])
    merged = apply_manifest_patch(base, patch)
    assert merged.plan_steps[3].depends_on == ["step_2"]
    assert any("added dependency 'step_2'" in item for item in merged.reference_corrections)
    assert evaluate_operation_rules(merged).passed
