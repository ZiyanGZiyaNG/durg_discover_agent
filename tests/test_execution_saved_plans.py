import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from execution.compiler import PlanRefusedError, run_plan
from schemas import QueryManifestV2
from test_execution_acceptance import FOUND, _drugs, _fake_resolve, _tractability
from execution.compiler import check_plan

PLANS = Path(__file__).resolve().parent.parent / "outputs" / "plans"


def _run_saved(name):
    plan = QueryManifestV2.model_validate(
        json.loads((PLANS / f"{name}.json").read_text())
    )
    with (
        patch("execution.adapters.resolve_entity", _fake_resolve),
        patch("execution.adapters.discover_disease_targets", lambda *a, **k: FOUND),
        patch("execution.more_adapters.discover_disease_targets", lambda *a, **k: FOUND),
        patch("execution.adapters.get_target_drug_evidence_by_ids", _drugs),
        patch("execution.adapters.get_target_tractability_by_id", _tractability),
    ):
        return run_plan(plan)


@pytest.mark.parametrize(
    "name",
    [
        "acceptance_pdac_known_targets_constrained",
        "phase2_pdac_known_targets_batched_v4",
        "pdac_known_targets_blind_v4",
        "pdac_known_targets_ranking_contract_v1",
    ],
)
def test_runs_to_the_end(name):
    trace = _run_saved(name)
    assert [r.status for r in trace] == ["success"] * len(trace)


@pytest.mark.parametrize(
    "name, missing_metric",
    [
        ("pdac_ctx8192_predict4096_v1", "tractability_score"),
        ("pdac_metric_dependency_rule_v1", "tractability_score"),
        ("pdac_direct_patch_transitive_refs_v1", "tractability_score"),
    ],
)
def test_undefined_metric_is_refused_before_retrieval(name, missing_metric):
    with pytest.raises(PlanRefusedError) as excinfo:
        _run_saved(name)
    assert any(missing_metric in problem for problem in excinfo.value.problems)

def test_old_operation_names_are_refused_before_retrieval():
    # Written before the operation table existed: refused up front, nothing is queried.
    with pytest.raises(PlanRefusedError):
        _run_saved("acceptance_pdac_known_targets")


def test_representative_manifest_produces_the_expected_tool_calls():
    plan = QueryManifestV2.model_validate(
        json.loads((PLANS / "phase2_pdac_known_targets_batched_v4.json").read_text())
    )
    tools = MagicMock()
    tools.resolve_entity.side_effect = _fake_resolve
    tools.discover_disease_targets.return_value = FOUND
    tools.get_target_drug_evidence_by_ids.side_effect = _drugs
    tools.get_target_tractability_by_id.side_effect = _tractability
    with patch.multiple(
        "execution.adapters",
        resolve_entity=tools.resolve_entity,
        discover_disease_targets=tools.discover_disease_targets,
        get_target_drug_evidence_by_ids=tools.get_target_drug_evidence_by_ids,
        get_target_tractability_by_id=tools.get_target_tractability_by_id,
    ):
        run_plan(plan)
    assert tools.mock_calls == [
        call.resolve_entity(
            "胰臟癌（pancreatic ductal adenocarcinoma, PDAC）", "disease", None
        ),
        call.discover_disease_targets(
            "pancreatic carcinoma", top_n=10, disease_id="EFO_1"
        ),
        call.get_target_drug_evidence_by_ids("T1", "EFO_1"),
        call.get_target_drug_evidence_by_ids("T2", "EFO_1"),
        call.get_target_drug_evidence_by_ids("T3", "EFO_1"),
        call.get_target_tractability_by_id("T1"),
        call.get_target_tractability_by_id("T2"),
        call.get_target_tractability_by_id("T3"),
    ]


def test_stratify_without_its_key_is_refused_before_retrieval():
    plan = QueryManifestV2.model_validate(
        json.loads((PLANS / "phase2_pdac_known_targets_batched_v4.json").read_text())
    )

    for step in plan.plan_steps:
        if step.operation == "stratify_target_evidence_layers":
            step.aggregation = {}
    assert any("target_translation_layer" in p for p in check_plan(plan))