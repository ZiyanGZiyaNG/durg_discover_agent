from unittest.mock import patch

from execution.compiler import run_plan
from schemas import QueryManifestV2
from test_natural_language_planner import _valid_payload
from query_opentargets import OpenTargetsError


def _fake_resolve(name, kind, entity_id=None):
    return {"id": "EFO_1", "name": "pancreatic carcinoma", "symbol": ""}


def _row(target_id, rank, score):
    return {
        "target_id": target_id,
        "target_symbol": target_id,
        "target_name": target_id,
        "rank": rank,
        "overall_association_score": score,
    }


FOUND = {
    "returned_target_count": 3,
    "total_associated_target_count": 3,
    "limitations": [],
    "targets": [_row("T1", 1, 0.5), _row("T2", 2, 0.9), _row("T3", 3, 0.7)],
}


def _drugs(target_id, efo_id):
    count = 1 if target_id == "T1" else 0
    return {
        "ensembl_id": target_id,
        "efo_id": efo_id,
        "total_drug_count": count,
        "fetched_drug_count": count,
        "drugs": [{"exact_disease_match": True}] * count,
    }


def _tractability(target_id):
    return {
        "target_id": target_id,
        "assessment_count": 1,
        "assessments": [{"modality": "SM", "label": "x", "value": False}],
    }


def test_pdac_known_targets_runs_end_to_end():
    plan = QueryManifestV2.model_validate(_valid_payload())
    with (
        patch("execution.adapters.resolve_entity", side_effect=_fake_resolve),
        patch("execution.adapters.discover_disease_targets", return_value=FOUND),
        patch("execution.adapters.get_target_drug_evidence_by_ids", side_effect=_drugs),
        patch(
            "execution.adapters.get_target_tractability_by_id",
            side_effect=_tractability,
        ),
    ):
        trace = run_plan(plan)

    assert [r.step_id for r in trace] == [f"step_{i}" for i in range(1, 7)]
    assert [r.status for r in trace] == ["success"] * 6

    ranked = trace[-1].payload["targets"]
    assert [t["target_id"] for t in ranked] == ["T2", "T3", "T1"]
    layers = {t["target_id"]: t["target_translation_layer"] for t in ranked}
    assert layers == {
        "T1": "translational_support",
        "T2": "disease_association_only",
        "T3": "disease_association_only",
    }


def test_one_tractability_failure_keeps_the_plan_running():
    plan = QueryManifestV2.model_validate(_valid_payload())

    def flaky(target_id):
        if target_id == "T3":
            raise OpenTargetsError("boom")
        return _tractability(target_id)

    with (
        patch("execution.adapters.resolve_entity", side_effect=_fake_resolve),
        patch("execution.adapters.discover_disease_targets", return_value=FOUND),
        patch("execution.adapters.get_target_drug_evidence_by_ids", side_effect=_drugs),
        patch("execution.adapters.get_target_tractability_by_id", side_effect=flaky),
    ):
        trace = run_plan(plan)

    assert [r.status for r in trace] == [
        "success",
        "success",
        "success",
        "partial_error",
        "partial_error",
        "success",
    ]
    layers = {
        t["target_id"]: t["target_translation_layer"]
        for t in trace[-1].payload["targets"]
    }
    assert layers["T3"] == "undetermined"
    assert layers["T2"] == "disease_association_only"
    failed = trace[4].payload["failed_targets"]["tractability_evidence"]
    assert failed == {"T3": "boom"}
