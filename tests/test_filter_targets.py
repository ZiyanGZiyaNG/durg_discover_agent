import json
from pathlib import Path

import pytest

from execution.adapters import FilterTargetsAdapter
from execution.analysis import normalize_filters
from execution.compiler import check_plan
from execution.contracts import StepContext
from execution.records import OUTPUT_MODELS
from schemas import QueryManifestV2

PLANS = Path(__file__).resolve().parent.parent / "outputs" / "plans"


def _row(target_id, rank, score):
    return {
        "target_id": target_id,
        "target_symbol": target_id,
        "target_name": target_id,
        "rank": rank,
        "overall_association_score": score,
    }


CANDIDATES = {
    "returned_target_count": 3,
    "total_associated_target_count": 3,
    "limitations": [],
    "targets": [_row("T1", 1, 0.9), _row("T2", 2, 0.8), _row("T3", 3, 0.5)],
}


def _assessment(modality):
    return {
        "assessment_count": 1,
        "assessments": [{"modality": modality, "value": True, "label": ""}],
    }


TRACTABILITY = {
    "by_target": {"T1": _assessment("SM"), "T2": _assessment("AB")},
    "failed_targets": {},
}


def _run(filters, board=None):
    if board is None:
        board = {"target_candidates": CANDIDATES, "tractability_evidence": TRACTABILITY}
    ctx = StepContext(
        step_id="f", operation="filter_targets", entities={}, filters=filters, board=board
    )
    return FilterTargetsAdapter().run(ctx)


def test_generic_filter_keeps_matches_and_sets_aside_unjudgeable_rows():
    result = _run(
        {
            "overall_association_score": {"gt": 0.7},
            "tractable_modalities": {"contains": "SM"},
        }
    )
    assert result.status == "success"
    assert [row["target_id"] for row in result.payload["targets"]] == ["T1"]
    assert result.payload["excluded_target_ids"] == ["T2"]
    assert result.payload["unevaluable_target_ids"] == ["T3"]
    assert result.provides == ["target_candidates"]
    OUTPUT_MODELS["filter_targets"].model_validate(result.payload)


def test_min_and_max_are_accepted_for_gte_and_lte():
    assert normalize_filters({"x": {"min": 1, "max": 2}}) == [
        ("x", "gte", 1),
        ("x", "lte", 2),
    ]


def test_bare_value_means_equality():
    assert normalize_filters({"x": 3}) == [("x", "eq", 3)]


@pytest.mark.parametrize("filters", [{"x": {"near": 1}}, {"x": {"in": 3}}, {"x": {}}])
def test_malformed_conditions_are_rejected(filters):
    with pytest.raises(ValueError):
        normalize_filters(filters)


def test_unknown_field_is_an_error_not_an_empty_result():
    result = _run({"no_such_field": {"gt": 1}})
    assert result.status == "error"
    assert "no_such_field" in result.error


def test_filter_without_rows_is_an_error():
    assert _run({"overall_association_score": {"gt": 0.7}}, board={}).status == "error"


def _plan_with_filter(filters):
    data = json.loads(
        (PLANS / "phase2_pdac_known_targets_batched_v4.json").read_text()
    )
    data["plan_steps"].append(
        {
            "step_id": "filter_1",
            "operation": "filter_targets",
            "description": "filter targets",
            "depends_on": ["discover_targets_1"],
            "entity_refs": [],
            "outputs": ["filtered_targets"],
            "filters": filters,
        }
    )
    return QueryManifestV2.model_validate(data)


def test_filter_on_unavailable_field_is_refused_before_retrieval():
    problems = check_plan(_plan_with_filter({"no_such_field": {"gt": 1}}))
    assert any("no_such_field" in problem for problem in problems)


def test_filter_on_upstream_field_is_accepted():
    assert check_plan(_plan_with_filter({"overall_association_score": {"gt": 0.7}})) == []