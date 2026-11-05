import json

from planner_acceptance import load_acceptance_cases
from staged_planner import (
    StagedNaturalLanguagePlanner,
    operation_candidates,
)


class FakeAgent:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def invoke(self, request):
        self.calls.append(request)
        return {"structured_response": self.payload}


def _case():
    cases = load_acceptance_cases("tests/fixtures/planner_acceptance_cases.json")
    return next(case for case in cases if case.case_id == "pdac_known_targets")


def _strategy():
    shared = {
        "entity_refs": ["disease_1"],
        "filters": {},
        "ranking": None,
        "aggregation": None,
        "safeguards": [],
    }
    return {
        "research_objective": "分層辨識並排序 PDAC 治療標的。",
        "entities": [
            {
                "entity_ref": "disease_1",
                "entity_type": "disease",
                "mention": "胰臟癌",
                "resolution_status": "unresolved",
            }
        ],
        "needs": [
            {
                **shared,
                "need_id": "resolve",
                "purpose": "解析疾病",
                "operation_search_query": "resolve disease entity",
                "depends_on": [],
                "required_inputs": ["disease mention"],
                "outputs": ["resolved_disease"],
                "sources": ["open_targets"],
            },
            {
                **shared,
                "need_id": "discover",
                "purpose": "取得疾病相關標的",
                "operation_search_query": "discover disease associated targets",
                "depends_on": ["resolve"],
                "required_inputs": ["resolved_disease"],
                "outputs": ["candidate_targets"],
                "sources": ["open_targets"],
                "safeguards": ["association_score_not_probability"],
            },
            {
                **shared,
                "need_id": "drugs",
                "purpose": "取得標的藥物證據",
                "operation_search_query": "retrieve target drug evidence",
                "depends_on": ["discover"],
                "required_inputs": ["candidate_targets"],
                "outputs": ["target_drug_evidence"],
                "sources": ["open_targets", "chembl"],
            },
            {
                **shared,
                "need_id": "tractability",
                "purpose": "取得可成藥性證據",
                "operation_search_query": "retrieve target tractability evidence",
                "depends_on": ["discover"],
                "required_inputs": ["candidate_targets"],
                "outputs": ["target_tractability_evidence"],
                "sources": ["open_targets"],
            },
            {
                **shared,
                "need_id": "stratify",
                "purpose": "分層轉譯證據",
                "operation_search_query": "stratify target evidence layers",
                "depends_on": ["drugs", "tractability"],
                "required_inputs": [
                    "target_drug_evidence",
                    "target_tractability_evidence",
                ],
                "outputs": ["target_evidence_layers"],
                "sources": [],
                "aggregation": {"target_translation_layer": "group"},
            },
            {
                **shared,
                "need_id": "rank",
                "purpose": "排序標的",
                "operation_search_query": "rank targets",
                "depends_on": ["stratify"],
                "required_inputs": ["target_evidence_layers"],
                "outputs": ["ranked_targets"],
                "sources": [],
                "ranking": {
                    "primary_metric": "overall_association_score",
                    "direction": "descending",
                    "secondary_metrics": [],
                },
            },
        ],
        "evidence_requirements": ["疾病關聯、藥物與可成藥性證據"],
        "completion_criteria": ["完成標的分層與排序"],
        "global_safeguards": [
            "association_does_not_imply_efficacy",
            "tractability_not_efficacy",
            "provenance_pagination_truncation",
        ],
        "characteristics": {"primary": "single_hop", "secondary": ["filter_and_rank"]},
    }


def _mapping(overrides=None):
    operations = {
        "resolve": "resolve_disease",
        "discover": "discover_targets",
        "drugs": "retrieve_target_drugs",
        "tractability": "retrieve_tractability",
        "stratify": "stratify_target_evidence_layers",
        "rank": "rank_targets",
    }
    operations.update(overrides or {})
    return {
        "matches": [
            {
                "need_id": need_id,
                "match_status": "matched",
                "canonical_operation": operation,
            }
            for need_id, operation in operations.items()
        ]
    }


def test_candidate_shortlist_finds_key_operation_boundaries():
    assert "retrieve_target_drugs" in operation_candidates(
        "retrieve target drug evidence"
    )
    assert "retrieve_tractability" in operation_candidates(
        "target tractability evidence"
    )
    assert "stratify_target_evidence_layers" in operation_candidates(
        "stratify target evidence layers"
    )


def test_staged_planner_compiles_and_passes_hidden_case(tmp_path):
    case = _case()
    strategy_agent = FakeAgent(_strategy())
    mapping_agent = FakeAgent(_mapping())
    checkpoint = tmp_path / "staged.report.json"

    report = StagedNaturalLanguagePlanner(
        strategy_agent, mapping_agent
    ).run_with_report(case.question, acceptance_case=case, checkpoint_path=checkpoint)

    assert report.status == "passed"
    assert report.semantic_acceptance.passed is True
    assert report.final_plan.original_question == case.question
    assert report.executed_retrieval_tools == []
    request = json.loads(mapping_agent.calls[0]["messages"][0]["content"])
    assert len(request["atomic_needs_with_candidates"][0]["candidates"]) == 5
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["status"] == "passed"


def test_mapping_cannot_select_operation_outside_shortlist():
    case = _case()
    report = StagedNaturalLanguagePlanner(
        FakeAgent(_strategy()), FakeAgent(_mapping({"resolve": "rank_drugs"}))
    ).run_with_report(case.question, acceptance_case=case)

    assert report.status == "failed"
    assert "outside its candidates" in report.errors[0]
    assert report.final_plan is None


def test_no_equivalent_operation_is_held_for_human_review():
    strategy = _strategy()
    mapping = _mapping()
    mapping["matches"][0] = {
        "need_id": "resolve",
        "match_status": "no_equivalent",
        "proposed_operation_name": "resolve_rare_disease_context",
        "proposal_reason": "候選項不涵蓋罕病語境解析。",
    }
    report = StagedNaturalLanguagePlanner(
        FakeAgent(strategy), FakeAgent(mapping)
    ).run_with_report(_case().question)

    assert report.status == "pending_human_review"
    assert report.final_plan is None
    assert report.executed_retrieval_tools == []


def test_compiled_steps_reference_upstream_outputs_as_inputs():
    case = _case()
    report = StagedNaturalLanguagePlanner(
        FakeAgent(_strategy()), FakeAgent(_mapping())
    ).run_with_report(case.question)

    steps = {step.step_id: step for step in report.final_plan.plan_steps}
    assert steps["step_resolve"].inputs == {}

    # One reference per dependency, pointing at that step's first output.
    stratify = steps["step_stratify"]
    assert set(stratify.inputs) == {"step_drugs", "step_tractability"}
    assert stratify.inputs["step_drugs"].step_id == "step_drugs"
    assert stratify.inputs["step_drugs"].output == "target_drug_evidence"
    assert (
        stratify.inputs["step_tractability"].output
        == "target_tractability_evidence"
    )

    # Every reference must resolve to an existing output of a prior step.
    for step in steps.values():
        for ref in step.inputs.values():
            assert ref.step_id in step.depends_on
            assert ref.output in steps[ref.step_id].outputs


def test_shortlist_keeps_reviewed_operations_for_kras_needs():
    """Replay of real staged queries whose reviewed operation once missed the top 5."""
    cases = [
        (
            "resolve_kras_entity",
            "KRAS human gene target canonical identifier Open Targets ChEMBL",
            "resolve_target",
        ),
        (
            "retrieve_kras_disease_associations",
            "KRAS disease association Open Targets evidence score data types disease ID",
            "retrieve_associated_diseases",
        ),
        (
            "rank_kras_disease_associations",
            "rank KRAS disease associations by association score evidence count "
            "evidence type diversity",
            "rank_diseases",
        ),
        (
            "kras_disease_evidence",
            "Open Targets KRAS target disease associations evidence strength score",
            "retrieve_associated_diseases",
        ),
    ]
    for need_id, query, expected in cases:
        assert expected in operation_candidates(query, need_id=need_id), need_id
