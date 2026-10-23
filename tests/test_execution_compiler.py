from unittest.mock import patch

from execution.compiler import run_plan
from schemas import QueryManifestV2
from unittest.mock import patch
import pytest
from execution.compiler import PlanRefusedError, run_plan
from query_opentargets import OpenTargetsError
from schemas import QueryManifestV2
from execution.contracts import StepResult


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


def test_discover_targets_uses_disease_resolved_upstream():
    plan = _plan(
        [
            _step("s1", "resolve_disease", ["d"]),
            _step("s2", "discover_targets", ["d"], ["s1"]),
        ]
    )
    found = {
        "returned_target_count": 10,
        "total_associated_target_count": 250,
        "targets": [],
    }
    with (
        patch("execution.adapters.resolve_entity", side_effect=_fake_resolve),
        patch(
            "execution.adapters.discover_disease_targets", return_value=found
        ) as discover,
    ):
        trace = run_plan(plan)

    assert [r.status for r in trace] == ["success", "success"]
    assert trace[1].provides == ["disease_target_association", "target_candidates"]
    discover.assert_called_once_with(
        "pancreatic cancer", top_n=10, disease_id="disease_ID"
    )


def _drug_evidence(total=3, fetched=3):
    return {
        "total_drug_count": total,
        "fetched_drug_count": fetched,
        "drugs": [{}] * fetched,
    }


def test_target_drugs_uses_target_and_disease_resolved_upstream():
    plan = _plan(
        [
            _step("s1", "resolve_disease", ["d"]),
            _step("s2", "resolve_target", ["t"], ["s1"]),
            _step("s3", "retrieve_target_drugs", ["d", "t"], ["s1", "s2"]),
        ]
    )
    with (
        patch("execution.adapters.resolve_entity", side_effect=_fake_resolve),
        patch(
            "execution.adapters.get_target_drug_evidence_by_ids",
            return_value=_drug_evidence(),
        ) as drugs,
    ):
        trace = run_plan(plan)

    assert [r.status for r in trace] == ["success", "success", "success"]
    assert trace[2].provides == ["target_drug_evidence"]
    drugs.assert_called_once_with("target_ID", "disease_ID")


def test_target_drugs_without_disease_passes_empty_efo_id():
    plan = _plan(
        [
            _step("s1", "resolve_target", ["t"]),
            _step("s2", "retrieve_target_drugs", ["t"], ["s1"]),
        ]
    )
    with (
        patch("execution.adapters.resolve_entity", side_effect=_fake_resolve),
        patch(
            "execution.adapters.get_target_drug_evidence_by_ids",
            return_value=_drug_evidence(),
        ) as drugs,
    ):
        run_plan(plan)

    drugs.assert_called_once_with("target_ID", "")


def test_discover_then_target_drugs_queries_every_candidate():
    plan = _plan(
        [
            _step("s1", "resolve_disease", ["d"]),
            _step("s2", "discover_targets", ["d"], ["s1"]),
            _step("s3", "retrieve_target_drugs", ["d"], ["s2"]),
        ]
    )
    found = {
        "returned_target_count": 2,
        "total_associated_target_count": 2,
        "targets": [
            {"target_id": "ENSG1", "target_symbol": "KRAS", "target_name": "n1"},
            {"target_id": "ENSG2", "target_symbol": "TP53", "target_name": "n2"},
        ],
    }
    with (
        patch("execution.adapters.resolve_entity", side_effect=_fake_resolve),
        patch("execution.adapters.discover_disease_targets", return_value=found),
        patch(
            "execution.adapters.get_target_drug_evidence_by_ids",
            return_value=_drug_evidence(),
        ) as drugs,
    ):
        trace = run_plan(plan)

    assert [r.status for r in trace] == ["success", "success", "success"]
    assert [c.args for c in drugs.call_args_list] == [
        ("ENSG1", "disease_ID"),
        ("ENSG2", "disease_ID"),
    ]
    assert list(trace[2].payload["by_target"]) == ["ENSG1", "ENSG2"]


def _tractability(count=4):
    return {"assessment_count": count, "assessments": [{}] * count}


def test_discovered_targets_feed_both_drugs_and_tractability():
    plan = _plan(
        [
            _step("s1", "resolve_disease", ["d"]),
            _step("s2", "discover_targets", ["d"], ["s1"]),
            _step("s3", "retrieve_target_drugs", ["d"], ["s2"]),
            _step("s4", "retrieve_tractability", ["d"], ["s2"]),
        ]
    )
    found = {
        "returned_target_count": 2,
        "total_associated_target_count": 2,
        "targets": [
            {"target_id": "ENSG1", "target_symbol": "KRAS", "target_name": "n1"},
            {"target_id": "ENSG2", "target_symbol": "TP53", "target_name": "n2"},
        ],
    }

    with (
        patch("execution.adapters.resolve_entity", side_effect=_fake_resolve),
        patch("execution.adapters.discover_disease_targets", return_value=found),
        patch(
            "execution.adapters.get_target_drug_evidence_by_ids",
            return_value=_drug_evidence(),
        ) as drugs,
        patch(
            "execution.adapters.get_target_tractability_by_id",
            return_value=_tractability(),
        ) as tractability,
    ):
        trace = run_plan(plan)

    assert [r.status for r in trace] == ["success"] * 4
    assert [c.args for c in drugs.call_args_list] == [
        ("ENSG1", "disease_ID"),
        ("ENSG2", "disease_ID"),
    ]
    assert [c.args for c in tractability.call_args_list] == [("ENSG1",), ("ENSG2",)]
    assert list(trace[3].payload["by_target"]) == ["ENSG1", "ENSG2"]


class SpyAdapter:
    source = "spy"

    def __init__(self):
        self.ctx = None

    def run(self, ctx):
        self.ctx = ctx
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success",
            source=self.source,
        )

    def test_ranking_and_aggregation_reach_the_adapter():
        plan = _plan(
                [
                    _step("s1", "resolve_disease", ["d"]),
                    _step("s2", "discover_targets", ["d"], ["s1"]),
                    {
                        **_step("s3", "rank_targets", ["d"], ["s2"]),
                        "ranking": {
                            "primary_metric": "overall_association_score",
                            "direction": "descending",
                        },
                        "aggregation": {"target_translation_layer": "group"},
                    },
                ]
            )
        found = {
            "returned_target_count": 1,
            "total_associated_target_count": 1,
            "targets": [],
        }
        spy = SpyAdapter()
        with (
            patch("execution.adapters.resolve_entity", side_effect=_fake_resolve),
            patch("execution.adapters.discover_disease_targets", return_value=found),
            patch.dict("execution.compiler.ADAPTERS", {"rank_targets": spy}),
        ):
            run_plan(plan)
        assert spy.ctx.ranking["primary_metric"] == "overall_association_score"
        assert spy.ctx.ranking["direction"] == "descending"
        assert spy.ctx.aggregation == {"target_translation_layer": "group"}
