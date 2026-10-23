"""Small offline catalog of failures observed in real Planner runs.

These cases are deliberately about general structure and operation contracts.
They are not canonical scientific answers to the acceptance questions.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import pytest

from operation_rules import evaluate_operation_rules
from schemas import QueryManifestV2
from test_natural_language_planner import _valid_payload


PayloadMutation = Callable[[dict], None]


@dataclass(frozen=True)
class RegressionCase:
    case_id: str
    layer: str
    historical_problem: str
    mutate: PayloadMutation
    expected_rule: str


def _remove_discovery_dependency(payload):
    payload["plan_steps"][1]["depends_on"] = []


def _put_source_on_analysis(payload):
    payload["plan_steps"][4]["sources"] = ["open_targets"]


def _remove_missing_is_unknown_safeguard(payload):
    payload["global_safeguards"].remove("missing_not_negative")


def _rank_by_future_tractability(payload):
    payload["plan_steps"][1]["ranking"] = {
        "primary_metric": "overall_association_score",
        "direction": "descending",
        "secondary_metrics": ["tractability_score"],
    }


def _use_unsupported_therapeutic_filter(payload):
    payload["plan_steps"][1]["filters"] = {"association_type": "therapeutic"}


def _use_unregistered_drug_metric(payload):
    payload["plan_steps"][5]["ranking"]["secondary_metrics"] = ["drug_evidence_count"]


def _put_generated_collection_in_entities(payload):
    payload["entities"].append(
        {
            "entity_ref": "pdac_targets",
            "entity_type": "target",
            "mention": "pdac_targets",
            "resolution_status": "unresolved",
        }
    )


def _use_wrong_resolution_operation(payload):
    payload["plan_steps"][0]["operation"] = "resolve_target"


def _use_placeholder_aggregation(payload):
    payload["plan_steps"][4]["aggregation"] = {"target_translation_layer": "reserved"}


REGRESSION_CASES = (
    RegressionCase(
        "retrieval_without_resolution",
        "operation_contract",
        "A retrieval step did not depend on entity resolution.",
        _remove_discovery_dependency,
        "required_inputs_must_exist",
    ),
    RegressionCase(
        "analysis_source_substitutes_for_retrieval",
        "general_research_invariant",
        "An analysis step attached a datasource instead of consuming retrieval.",
        _put_source_on_analysis,
        "analysis_requires_upstream_data",
    ),
    RegressionCase(
        "missing_evidence_treated_as_negative",
        "operation_contract",
        "A retrieval plan omitted the missing-is-unknown safeguard.",
        _remove_missing_is_unknown_safeguard,
        "capability_requires_safeguards",
    ),
    RegressionCase(
        "ranking_uses_later_tractability",
        "general_research_invariant",
        "Discovery ranked by tractability before tractability was retrieved.",
        _rank_by_future_tractability,
        "analysis_metric_requires_available_evidence",
    ),
    RegressionCase(
        "unsupported_association_type_filter",
        "operation_contract",
        "Discovery invented association_type=therapeutic.",
        _use_unsupported_therapeutic_filter,
        "analysis_metric_requires_available_evidence",
    ),
    RegressionCase(
        "unregistered_drug_evidence_count",
        "operation_contract",
        "Ranking used an unreviewed drug_evidence_count metric.",
        _use_unregistered_drug_metric,
        "analysis_metric_requires_available_evidence",
    ),
    RegressionCase(
        "generated_collection_declared_as_entity",
        "general_research_invariant",
        "A generated target collection was promoted to a user-named entity.",
        _put_generated_collection_in_entities,
        "declared_entities_must_be_question_grounded",
    ),
    RegressionCase(
        "resolution_entity_type_mismatch",
        "operation_contract",
        "A disease entity was sent to target resolution.",
        _use_wrong_resolution_operation,
        "operation_entity_type_compatibility",
    ),
    RegressionCase(
        "aggregation_keyword_without_method",
        "operation_contract",
        "A stratification step used a placeholder instead of an executable method.",
        _use_placeholder_aggregation,
        "operation_must_honor_capability_contract",
    ),
)


@pytest.mark.parametrize("case", REGRESSION_CASES, ids=lambda case: case.case_id)
def test_historical_failure_catalog_remains_blocked(case):
    payload = deepcopy(_valid_payload())
    case.mutate(payload)
    plan = QueryManifestV2.model_validate(payload)
    result = evaluate_operation_rules(plan)

    assert result.passed is False
    assert case.expected_rule in {finding.rule_id for finding in result.findings}


def _without_optional_ranking(payload):
    payload["plan_steps"] = payload["plan_steps"][:-1]


def _alternate_stratification_shape(payload):
    payload["plan_steps"][4]["aggregation"] = {
        "evidence_tiers": {"dimensions": ["association", "drug", "tractability"]}
    }


def _rank_with_available_metrics(payload):
    payload["plan_steps"][5]["ranking"]["secondary_metrics"] = [
        "tractability_score",
        "drug_count",
    ]


ACCEPTED_ALTERNATIVES = (
    pytest.param(lambda payload: None, id="complete_reference_plan"),
    pytest.param(_without_optional_ranking, id="stratification_without_ranking"),
    pytest.param(_alternate_stratification_shape, id="equivalent_aggregation_shape"),
    pytest.param(_rank_with_available_metrics, id="ranking_with_upstream_evidence"),
)


@pytest.mark.parametrize("mutate", ACCEPTED_ALTERNATIVES)
def test_reasonable_alternatives_are_not_blocked(mutate):
    payload = deepcopy(_valid_payload())
    mutate(payload)
    plan = QueryManifestV2.model_validate(payload)
    result = evaluate_operation_rules(plan)

    assert result.passed is True, result.model_dump()


def test_catalog_cases_are_unique_and_classified():
    ids = [case.case_id for case in REGRESSION_CASES]
    assert len(ids) == len(set(ids))
    assert {case.layer for case in REGRESSION_CASES} <= {
        "schema_or_normalization",
        "operation_contract",
        "general_research_invariant",
        "case_specific_acceptance",
        "model_capability",
    }
