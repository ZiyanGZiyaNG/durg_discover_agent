from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from planner_acceptance import (
    OPERATION_TABLE,
    OPERATION_SEMANTICS,
    STANDARD_OPERATION_CATALOG,
    build_acceptance_rubric,
    build_repair_feedback,
    evaluate_plan,
    load_acceptance_cases,
    load_acceptance_rubrics,
    normalize_plan_operations,
)
from schemas import (
    PlannedEntityV2,
    QueryCharacteristicsV2,
    QueryManifestV2,
    ResearchPlanStepV2,
)


FIXTURE = Path(__file__).parent / "fixtures" / "planner_acceptance_cases.json"


def _compliant_plan(case):
    entity_refs = [
        f"entity_{index}" for index, _ in enumerate(case.required_entities, 1)
    ]
    entities = [
        PlannedEntityV2(
            entity_ref=entity_ref,
            entity_type=expected.entity_type,
            mention=expected.mention,
            resolution_status="unresolved",
        )
        for entity_ref, expected in zip(entity_refs, case.required_entities)
    ]
    step_ids = {
        operation: f"step_{index}"
        for index, operation in enumerate(case.required_operations, 1)
    }
    dependencies = {operation: [] for operation in case.required_operations}
    for before, after in case.required_dependencies:
        dependencies[after].append(step_ids[before])

    steps = []
    for index, operation in enumerate(case.required_operations):
        steps.append(
            ResearchPlanStepV2(
                step_id=step_ids[operation],
                operation=operation,
                description=f"Acceptance plan step: {operation}",
                depends_on=dependencies[operation],
                entity_refs=entity_refs,
                sources=case.required_sources if index == 0 else [],
                outputs=[f"{operation}_result"],
                filters={key: "required" for key in case.required_filters}
                if index == 0
                else {},
                ranking={
                    "primary_metric": case.required_ranking[0],
                    "direction": "descending",
                    "secondary_metrics": case.required_ranking[1:],
                }
                if index == 0 and case.required_ranking
                else None,
                aggregation={key: "required" for key in case.required_aggregation}
                if index == 0 and case.required_aggregation
                else None,
            )
        )

    return QueryManifestV2(
        original_question=case.question,
        research_objective=case.research_intent,
        entities=entities,
        plan_steps=steps,
        evidence_requirements=["Preserve evidence and provenance"],
        completion_criteria=["All planned outputs are accounted for"],
        global_safeguards=case.required_safeguards,
        characteristics=QueryCharacteristicsV2(
            primary=case.allowed_characteristics[0],
            secondary=case.allowed_characteristics[1:],
        ),
        generated_at=datetime.now(timezone.utc),
    )


def test_operation_table_contains_authoritative_contract_fields():
    exact = OPERATION_TABLE["retrieve_exact_pair"]
    assert exact.operation_kind == "retrieval"
    assert exact.requires_all == ["resolved_disease", "resolved_target"]
    assert exact.provides == ["disease_target_association"]
    assert "missing_not_negative" in exact.required_safeguards

    stratify = OPERATION_TABLE["stratify_target_evidence_layers"]
    assert stratify.operation_kind == "analysis"
    assert stratify.requires_aggregation is True
    assert set(stratify.requires_all) == {
        "disease_target_association",
        "target_drug_evidence",
        "tractability_evidence",
    }


def test_all_eighteen_questions_are_formal_plan_acceptance_cases():
    cases = load_acceptance_cases(FIXTURE)

    assert len(cases) == 18
    assert len({case.case_id for case in cases}) == 18
    assert all(case.question and case.research_intent for case in cases)
    assert sum(case.evaluation_set == "development" for case in cases) == 12
    assert sum(case.evaluation_set == "internal_holdout" for case in cases) == 6


def test_every_standard_operation_has_planner_facing_semantics():
    assert set(OPERATION_SEMANTICS) == set(STANDARD_OPERATION_CATALOG)
    assert set(OPERATION_TABLE) == set(STANDARD_OPERATION_CATALOG)
    assert all(len(description) >= 20 for description in OPERATION_SEMANTICS.values())


def test_reviewed_equivalent_is_normalized_to_canonical_operation():
    case = load_acceptance_cases(FIXTURE)[0]
    plan = _compliant_plan(case)
    step = next(item for item in plan.plan_steps if item.operation == "resolve_disease")
    step.operation = "normalize_disease"

    corrections = normalize_plan_operations(plan)

    assert step.operation == "resolve_disease"
    assert "normalize_disease -> resolve_disease" in corrections[0]


def test_proposed_operation_requires_human_review_and_blocks_acceptance():
    case = load_acceptance_cases(FIXTURE)[0]
    plan = _compliant_plan(case)
    plan.plan_steps.append(
        ResearchPlanStepV2(
            step_id="proposed_step",
            operation="proposed_new_operation",
            operation_status="proposed",
            proposed_operation_name="compare_single_cell_coexpression",
            proposal_reason="OPERATION_TABLE has no single-cell coexpression comparison.",
            description="Propose a new comparison capability.",
            depends_on=[plan.plan_steps[-1].step_id],
            entity_refs=[plan.entities[0].entity_ref],
            outputs=["proposal"],
        )
    )

    normalize_plan_operations(plan)
    result = evaluate_plan(case, plan)

    assert not result.passed
    assert result.proposed_operations == ["compare_single_cell_coexpression"]
    feedback = build_repair_feedback(result)
    assert any(item["area"] == "proposed_operation" for item in feedback.incorrect)


def test_unknown_approved_operation_must_be_marked_as_proposed():
    case = load_acceptance_cases(FIXTURE)[0]
    plan = _compliant_plan(case)
    plan.plan_steps[0].operation = "invented_operation"

    with pytest.raises(ValueError, match="not in OPERATION_TABLE"):
        normalize_plan_operations(plan)


def test_all_eighteen_cases_compile_to_required_forbidden_optional_equivalent_rubrics():
    rubrics = load_acceptance_rubrics(FIXTURE)

    assert len(rubrics) == 18
    assert all(rubric.required_capabilities for rubric in rubrics)
    assert all(rubric.forbidden_conditions for rubric in rubrics)
    assert all(
        capability.equivalent_operations
        for rubric in rubrics
        for capability in rubric.required_capabilities
    )
    assert all(rubric.optional_capabilities is not None for rubric in rubrics)


def test_acceptance_suite_covers_all_six_descriptive_characteristics():
    cases = load_acceptance_cases(FIXTURE)
    characteristics = {
        characteristic
        for case in cases
        for characteristic in case.allowed_characteristics
    }

    assert characteristics == {
        "single_hop",
        "multi_hop",
        "evidence_directed",
        "filter_and_rank",
        "comparison_and_aggregation",
        "safety_and_negative_knowledge",
    }


def test_all_cases_accept_a_semantically_complete_plan_without_retrieval():
    for case in load_acceptance_cases(FIXTURE):
        result = evaluate_plan(case, _compliant_plan(case))
        assert result.passed, (case.case_id, result.model_dump())


def test_evaluator_rejects_missing_source_and_safeguard():
    case = load_acceptance_cases(FIXTURE)[0]
    plan = _compliant_plan(case)
    plan = plan.model_copy(deep=True)
    plan.plan_steps[0].sources = []
    plan.global_safeguards = []

    result = evaluate_plan(case, plan)

    assert not result.passed
    assert result.missing_sources == sorted(case.required_sources)
    assert result.missing_safeguards == sorted(case.required_safeguards)


def test_query_characteristics_do_not_substitute_for_plan_semantics():
    case = load_acceptance_cases(FIXTURE)[0]
    plan = _compliant_plan(case)
    incomplete = deepcopy(plan)
    incomplete.plan_steps[0].ranking = None

    result = evaluate_plan(case, incomplete)

    assert not result.passed
    assert result.missing_ranking == case.required_ranking


def test_equivalent_operation_names_satisfy_capabilities():
    case = load_acceptance_cases(FIXTURE)[0]
    plan = _compliant_plan(case)
    aliases = {
        "resolve_disease": "entity_resolution",
        "discover_targets": "retrieve_disease_targets",
        "rank_targets": "prioritize_targets",
    }
    for step in plan.plan_steps:
        step.operation = aliases.get(step.operation, step.operation)

    result = evaluate_plan(build_acceptance_rubric(case), plan)

    assert result.passed


def test_repair_feedback_contains_only_actual_gaps_and_errors():
    case = load_acceptance_cases(FIXTURE)[0]
    plan = _compliant_plan(case)
    plan.plan_steps = plan.plan_steps[:1]
    result = evaluate_plan(case, plan)

    feedback = build_repair_feedback(result)

    assert feedback.missing
    assert any("discover_targets" in item["problem"] for item in feedback.missing)
    assert "research_intent" not in feedback.model_dump_json()


def test_dependency_feedback_explains_arrow_direction_as_depends_on_edit():
    case = load_acceptance_cases(FIXTURE)[0]
    plan = _compliant_plan(case)
    rank_step = next(
        step for step in plan.plan_steps if step.operation == "rank_targets"
    )
    rank_step.depends_on = []
    result = evaluate_plan(case, plan)

    feedback = build_repair_feedback(result)
    item = next(entry for entry in feedback.missing if entry["area"] == "dependency")

    assert "下游步驟的 depends_on" in item["required_correction"]
    assert "不可讓上游反過來依賴下游" in item["required_correction"]


def test_ranking_alias_is_accepted_and_actual_operations_are_reported():
    case = load_acceptance_cases(FIXTURE)[0]
    plan = _compliant_plan(case)
    for step in plan.plan_steps:
        step.ranking = None
    rank_step = next(
        step for step in plan.plan_steps if step.operation == "rank_targets"
    )
    rank_step.ranking = {"association_score": "descending"}

    result = evaluate_plan(case, plan)

    assert result.passed
    assert result.actual_operations == [
        "discover_targets",
        "rank_targets",
        "resolve_disease",
        "retrieve_target_drugs",
        "retrieve_tractability",
        "stratify_target_evidence_layers",
    ]


def test_ranking_criteria_is_rejected_as_ambiguous():
    case = load_acceptance_cases(FIXTURE)[0]

    with pytest.raises(ValueError, match="ranking.criteria is ambiguous"):
        ResearchPlanStepV2(
            step_id="ambiguous_ranking",
            operation="rank_targets",
            description="Ambiguous ranking criteria.",
            outputs=["ranked_targets"],
            ranking={"criteria": ["overall_association_score", "tractability"]},
        )


def test_discover_per_disease_and_sort_by_value_are_valid_equivalents():
    case = load_acceptance_cases(FIXTURE)[0]
    plan = _compliant_plan(case)
    plan.plan_steps[1].operation = "discover_targets_per_disease"
    plan.plan_steps[0].ranking = None
    plan.plan_steps[2].ranking = {
        "sort_by": "overall_association_score",
        "order": "descending",
    }

    result = evaluate_plan(case, plan)

    assert result.passed
    assert result.missing_capabilities == []
    assert result.missing_ranking == []


def test_sort_list_is_normalized_to_primary_and_secondary_metrics():
    step = ResearchPlanStepV2(
        step_id="rank",
        operation="rank_targets",
        description="Rank targets.",
        outputs=["ranked_targets"],
        ranking={
            "sort": ["overall_association_score", "tractability_score"],
            "order": "desc",
        },
    )

    assert step.ranking.primary_metric == "overall_association_score"
    assert step.ranking.direction == "descending"
    assert step.ranking.secondary_metrics == ["tractability_score"]


def test_unknown_operation_is_reported_without_erasing_valid_capabilities():
    case = load_acceptance_cases(FIXTURE)[0]
    plan = _compliant_plan(case)
    plan.plan_steps.append(
        ResearchPlanStepV2(
            step_id="step_custom",
            operation="compare_single_cell_coexpression",
            description="Compare single-cell coexpression.",
            depends_on=[plan.plan_steps[-1].step_id],
            entity_refs=[plan.entities[0].entity_ref],
            outputs=["coexpression_result"],
        )
    )

    result = evaluate_plan(case, plan)

    assert result.passed
    assert result.unrecognized_operations == ["compare_single_cell_coexpression"]
