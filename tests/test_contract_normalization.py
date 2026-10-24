"""Offline tests for program-authority contract normalization (no LLM)."""

from copy import deepcopy
import json

from langchain_core.messages import AIMessage

from contract_normalization import (
    drop_program_owned_fields,
    fill_null_step_defaults,
    normalize_contract_fields,
)
from natural_language_planner import (
    DirectJsonPatchAgent,
    DirectJsonPlanAgent,
    QueryManifestPatch,
    apply_manifest_patch,
)
from operation_rules import evaluate_operation_rules
from schemas import QueryManifestV2
from test_natural_language_planner import _valid_payload


def test_correct_payload_is_untouched():
    payload = _valid_payload()
    before = deepcopy(payload)
    assert normalize_contract_fields(payload) == []
    assert payload == before


def test_required_safeguards_are_added_from_the_operation_table():
    payload = _valid_payload()
    payload["global_safeguards"].remove("missing_not_negative")
    payload["plan_steps"][1]["safeguards"] = []  # drop the step-level one too
    notes = normalize_contract_fields(payload)
    assert "missing_not_negative" in payload["global_safeguards"]
    assert "association_score_not_probability" in payload["global_safeguards"]
    assert all("added required safeguard" in item for item in notes)
    assert evaluate_operation_rules(QueryManifestV2.model_validate(payload)).passed


def test_step_level_safeguards_count_as_present():
    payload = _valid_payload()
    # step_2 already carries association_score_not_probability at step level
    normalize_contract_fields(payload)
    assert "association_score_not_probability" not in payload["global_safeguards"]


def test_duplicated_analysis_sources_are_removed():
    payload = _valid_payload()
    payload["plan_steps"][4]["sources"] = ["open_targets"]
    notes = normalize_contract_fields(payload)
    assert payload["plan_steps"][4]["sources"] == []
    assert any("removed sources" in item for item in notes)
    assert evaluate_operation_rules(QueryManifestV2.model_validate(payload)).passed


def test_analysis_sources_not_declared_upstream_are_kept_for_the_model():
    payload = _valid_payload()
    payload["plan_steps"][4]["sources"] = ["europe_pmc"]
    assert normalize_contract_fields(payload) == []
    assert payload["plan_steps"][4]["sources"] == ["europe_pmc"]
    assert not evaluate_operation_rules(QueryManifestV2.model_validate(payload)).passed


def test_scientific_choices_are_never_normalized():
    payload = _valid_payload()
    payload["plan_steps"][1]["filters"] = {"association_type": "therapeutic"}
    payload["plan_steps"][5]["ranking"]["secondary_metrics"] = ["drug_evidence_count"]
    before = deepcopy(payload["plan_steps"])
    normalize_contract_fields(payload)
    assert payload["plan_steps"] == before


class _FakeModel:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, messages, config=None):
        return AIMessage(content=json.dumps(self.payload, ensure_ascii=False))


def test_full_plan_agent_applies_contract_fixes_without_another_model_call():
    payload = _valid_payload()
    payload["global_safeguards"].remove("missing_not_negative")
    payload["plan_steps"][4]["sources"] = ["open_targets"]
    plan = DirectJsonPlanAgent(_FakeModel(payload)).invoke({"messages": []})[
        "structured_response"
    ]
    assert evaluate_operation_rules(plan).passed
    notes = " ".join(plan.reference_corrections)
    assert "missing_not_negative" in notes and "removed sources" in notes


def test_patch_merge_applies_contract_fixes_and_records_them():
    base = QueryManifestV2.model_validate(_valid_payload())
    step = base.plan_steps[4].model_copy(update={"sources": ["open_targets"]})
    merged = apply_manifest_patch(base, QueryManifestPatch(upsert_steps=[step]))
    assert merged.plan_steps[4].sources == []
    assert any("removed sources" in item for item in merged.reference_corrections)
    assert evaluate_operation_rules(merged).passed


def test_invalid_runtime_fields_are_dropped_and_reported():
    payload = _valid_payload()
    payload["manifest_id"] = "a1b2c3d4-e5f6-4789-b012-3456789abcdef"  # 13-char last group
    payload["generated_at"] = "not a timestamp"
    payload["created_before_retrieval"] = False
    notes = drop_program_owned_fields(payload)
    assert not {"manifest_id", "generated_at", "created_before_retrieval"} & set(payload)
    assert len(notes) == 3 and all("program assigns" in item for item in notes)


def test_valid_runtime_fields_are_dropped_without_noise():
    payload = _valid_payload()
    before = set(payload)
    assert drop_program_owned_fields(payload) == []
    assert before - set(payload) == {
        "schema_version", "generated_at", "created_before_retrieval"
    }


def test_full_plan_agent_survives_a_malformed_model_manifest_id():
    payload = _valid_payload()
    payload["manifest_id"] = "a1b2c3d4-e5f6-4789-b012-3456789abcdef"
    plan = DirectJsonPlanAgent(_FakeModel(payload)).invoke({"messages": []})[
        "structured_response"
    ]
    assert str(plan.manifest_id) != payload["manifest_id"]
    assert any("manifest_id" in item for item in plan.reference_corrections)


def test_null_step_fields_become_empty_defaults():
    steps = [{"step_id": "a", "filters": None, "depends_on": None, "sources": ["x"]}]
    notes = fill_null_step_defaults(steps)
    assert steps[0]["filters"] == {} and steps[0]["depends_on"] == []
    assert steps[0]["sources"] == ["x"]
    assert len(notes) == 2


def test_full_plan_agent_survives_null_filters():
    payload = _valid_payload()
    for index in (2, 3, 4):
        payload["plan_steps"][index]["filters"] = None
    plan = DirectJsonPlanAgent(_FakeModel(payload)).invoke({"messages": []})[
        "structured_response"
    ]
    assert plan.plan_steps[2].filters == {}
    assert sum("null 'filters'" in item for item in plan.reference_corrections) == 3


def test_patch_agent_treats_null_as_omitted_so_the_base_step_is_preserved():
    base = QueryManifestV2.model_validate(_valid_payload())
    base.plan_steps[1].filters = {}
    update = {"step_id": "step_2", "filters": None, "description": "更新描述。"}
    model = _FakeModel({"upsert_steps": [update]})
    result = DirectJsonPatchAgent(model).invoke(
        {"messages": [], "base_plan": base}
    )
    patch = result["structured_response"]
    assert patch.upsert_steps[0].operation == "discover_targets"
    assert patch.upsert_steps[0].depends_on == ["step_1"]
    assert any("null 'filters'" in item for item in result["patch_normalizations"])
