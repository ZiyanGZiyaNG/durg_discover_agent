from unittest.mock import patch

from execution.compiler import run_plan
from schemas import QueryManifestV2
from unittest.mock import patch

import pytest

from execution.compiler import PlanRefusedError, run_plan
from query_opentargets import OpenTargetsError
from schemas import QueryManifestV2


def _step(step_id, operation, refs, deps=()):
    return {
        "step_id": step_id,
        "operation": operation,
        "description": "x",
        "entity_refs": refs,
        "depends_on": list(deps),
        "outputs": [step_id + "_out"],
    }


def _plan(steps):
    return QueryManifestV2.model_validate(
        {
            "original_question": "q",
            "research_objective": "o",
            "entities": [
                {
                    "entity_ref": "d",
                    "entity_type": "disease",
                    "mention": "pancreatic cancer",
                    "resolution_status": "unresolved",
                },
                {
                    "entity_ref": "t",
                    "entity_type": "target",
                    "mention": "KRAS",
                    "resolution_status": "unresolved",
                },
            ],
            "plan_steps": steps,
            "evidence_requirements": ["e"],
            "completion_criteria": ["c"],
            "global_safeguards": ["g"],
            "characteristics": {"primary": "single_hop"},
        }
    )


def _fake_resolve(name, kind, entity_id=None):
    return {"id": f"{kind}_ID", "name": name, "symbol": f"{kind}_SYM"}


def test_two_resolve_steps_succeed():
    plan = _plan(
        [
            _step("s1", "resolve_disease", ["d"]),
            _step("s2", "resolve_target", ["t"], ["s1"]),
        ]
    )
    with patch("execution.adapters.resolve_entity", side_effect=_fake_resolve):
        trace = run_plan(plan)

    assert [r.status for r in trace] == ["success", "success"]
    assert trace[1].provides == ["resolved_target"]


def test_unknown_operation_is_refused_before_any_retrieval():
    plan = _plan(
        [
            _step("s1", "resolve_disease", ["d"]),
            _step("s2", "retrieve_clinical_trials", ["d"], ["s1"]),
        ]
    )
    with patch("execution.adapters.resolve_entity") as resolve:
        with pytest.raises(PlanRefusedError) as excinfo:
            run_plan(plan)

    resolve.assert_not_called()
    assert "s2" in excinfo.value.problems[0]


def test_failed_step_blocks_its_dependents():
    plan = _plan(
        [
            _step("s1", "resolve_disease", ["d"]),
            _step("s2", "resolve_target", ["t"], ["s1"]),
        ]
    )
    with patch(
        "execution.adapters.resolve_entity", side_effect=OpenTargetsError("boom")
    ):
        trace = run_plan(plan)

    assert [r.status for r in trace] == ["error", "not_executed"]
    assert "s1" in trace[1].error


def test_exact_pair_uses_ids_resolved_upstream():
    plan = _plan(
        [
            _step("s1", "resolve_disease", ["d"]),
            _step("s2", "resolve_target", ["t"], ["s1"]),
            _step("s3", "retrieve_exact_pair", ["d", "t"], ["s1", "s2"]),
        ]
    )
    evidence = {"overall_association_score": 0.9, "datasource_evidence": []}
    with (
        patch("execution.adapters.resolve_entity", side_effect=_fake_resolve),
        patch(
            "execution.adapters.get_disease_target_evidence_by_ids",
            return_value=evidence,
        ) as pair,
    ):
        trace = run_plan(plan)

    assert [r.status for r in trace] == ["success", "success", "success"]
    pair.assert_called_once_with(
        "disease_ID",
        "target_ID",
        disease_query="pancreatic cancer",
        target_query="target_SYM",
    )
