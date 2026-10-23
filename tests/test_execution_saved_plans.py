import json
from pathlib import Path
from unittest.mock import patch
import pytest
from execution.compiler import run_plan
from schemas import QueryManifestV2
from test_execution_acceptance import FOUND, _drugs, _fake_resolve, _tractability

PLANS = Path(__file__).resolve().parent.parent / "outputs" / "plans"

def _run_saved(name):
    plan = QueryManifestV2.model_validate(json.loads((PLANS / f"{name}.json").read_text()))
    with patch("execution.adapters.resolve_entity", _fake_resolve), patch(
        "execution.adapters.discover_disease_targets", lambda *a, **k: FOUND
    ), patch("execution.adapters.get_target_drug_evidence_by_ids", _drugs), patch(
        "execution.adapters.get_target_tractability_by_id", _tractability
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
def test_upstream_plan_runs_to_the_end(name):
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
def test_upstream_plan_stops_only_at_the_undefined_ranking_metric(name, missing_metric):
    # Known gap: the metric's definition is waiting on a research-team decision.
    trace = _run_saved(name)
    assert [r.status for r in trace[:-1]] == ["success"] * (len(trace) - 1)
    assert trace[-1].status == "error"
    assert missing_metric in trace[-1].error