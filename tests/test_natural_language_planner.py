from datetime import datetime, timezone
import json
import time

import pytest
import natural_language_planner as planner_module
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from natural_language_planner import (
    _cleanup_ollama_after_timeout,
    apply_manifest_patch,
    build_dependency_repair_prompt,
    build_dependency_repair_task,
    build_repair_scope,
    build_repair_operation_catalog,
    build_operation_catalog_prompt,
    COMPACT_BOUNDARY_OPERATIONS,
    DirectJsonPlanAgent,
    DirectJsonPatchAgent,
    NaturalLanguagePlannerError,
    NaturalLanguageResearchPlanner,
    ModelCallTraceHandler,
    OPERATION_CATALOG_PROMPT,
    OPERATION_GROUPS,
    PLAN_ONLY_PROMPT,
    QueryManifestPatch,
    RepairScope,
    select_repair_batch,
    validate_patch_scope,
)
from planner_acceptance import PlanAcceptanceResult, RepairFeedback
from planner_acceptance import (
    OPERATION_SEMANTICS,
    STANDARD_OPERATION_CATALOG,
    load_acceptance_cases,
)
from schemas import QueryManifestV2
from operation_rules import evaluate_operation_rules


QUESTION = "胰臟癌目前有哪些已知的治療標的？"


def _valid_payload(question=QUESTION):
    return {
        "schema_version": "2.0",
        "original_question": question,
        "research_objective": "規劃辨識並排序胰臟癌相關治療標的。",
        "entities": [
            {
                "entity_ref": "disease_1",
                "entity_type": "disease",
                "mention": "胰臟癌",
                "resolution_status": "unresolved",
            }
        ],
        "plan_steps": [
            {
                "step_id": "step_1",
                "operation": "resolve_disease",
                "description": "解析疾病實體。",
                "entity_refs": ["disease_1"],
                "sources": ["open_targets"],
                "outputs": ["resolved_disease"],
            },
            {
                "step_id": "step_2",
                "operation": "discover_targets",
                "description": "取得疾病相關標的。",
                "depends_on": ["step_1"],
                "entity_refs": ["disease_1"],
                "sources": ["open_targets"],
                "outputs": ["candidate_targets"],
                "safeguards": ["association_score_not_probability"],
            },
            {
                "step_id": "step_3",
                "operation": "retrieve_target_drugs",
                "description": "取得候選標的的已知藥物證據。",
                "depends_on": ["step_2"],
                "entity_refs": ["disease_1"],
                "sources": ["open_targets", "chembl"],
                "outputs": ["target_drug_evidence"],
            },
            {
                "step_id": "step_4",
                "operation": "retrieve_tractability",
                "description": "取得候選標的的可成藥性證據。",
                "depends_on": ["step_2"],
                "entity_refs": ["disease_1"],
                "sources": ["open_targets"],
                "outputs": ["target_tractability_evidence"],
            },
            {
                "step_id": "step_5",
                "operation": "stratify_target_evidence_layers",
                "description": "區分疾病相關標的與具有藥物或可成藥性證據的標的。",
                "depends_on": ["step_3", "step_4"],
                "entity_refs": ["disease_1"],
                "sources": [],
                "outputs": ["target_evidence_layers"],
                "aggregation": {"target_translation_layer": "group"},
            },
            {
                "step_id": "step_6",
                "operation": "rank_targets",
                "description": "依關聯分數排序候選標的。",
                "depends_on": ["step_5"],
                "entity_refs": ["disease_1"],
                "sources": [],
                "outputs": ["ranked_targets"],
                "ranking": {
                    "primary_metric": "overall_association_score",
                    "direction": "descending",
                    "secondary_metrics": [],
                },
            },
        ],
        "evidence_requirements": ["疾病與標的關聯證據及來源"],
        "completion_criteria": ["每個步驟均定義輸出"],
        "global_safeguards": [
            "provenance_pagination_truncation",
            "missing_not_negative",
            "association_does_not_imply_efficacy",
            "tractability_not_efficacy",
        ],
        "characteristics": {"primary": "single_hop", "secondary": []},
        "created_before_retrieval": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


class FakeAgent:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def invoke(self, request):
        self.calls.append(request)
        return next(self.responses)


class CallbackTraceAgent:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, request, config=None):
        callback = config["callbacks"][0]
        for index in (1, 2):
            run_id = uuid4()
            callback.on_chat_model_start(
                {}, [[HumanMessage(content=f"底層實際提示{index}")]], run_id=run_id
            )
            callback.on_llm_end(
                LLMResult(
                    generations=[
                        [ChatGeneration(message=AIMessage(content=f"原始回覆{index}"))]
                    ]
                ),
                run_id=run_id,
            )
        return {"structured_response": self.payload}


class CheckpointAwareAgent(FakeAgent):
    def __init__(self, responses, checkpoint_path):
        super().__init__(responses)
        self.checkpoint_path = checkpoint_path

    def invoke(self, request):
        if self.calls:
            checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            assert checkpoint["status"] == "running"
            assert len(checkpoint["attempts"]) == 1
            assert checkpoint["attempts"][0]["candidate_plan"] is not None
        return super().invoke(request)


def test_planner_returns_v2_plan_without_any_tool_runner():
    agent = FakeAgent([{"structured_response": _valid_payload()}])

    plan = NaturalLanguageResearchPlanner(agent=agent).run(QUESTION)

    assert plan.schema_version == "2.0"
    assert plan.original_question == QUESTION
    assert plan.created_before_retrieval is True
    assert len(agent.calls) == 1


def test_each_underlying_model_call_is_captured_in_report():
    report = NaturalLanguageResearchPlanner(
        agent=CallbackTraceAgent(_valid_payload()), capture_model_calls=True
    ).run_with_report(QUESTION)

    calls = report.attempts[0].model_calls
    assert len(calls) == 2
    assert [call.call_index for call in calls] == [1, 2]
    assert all(call.status == "completed" for call in calls)
    assert calls[0].input_messages[0]["content"] == "底層實際提示1"
    assert "原始回覆2" in json.dumps(calls[1].raw_response, ensure_ascii=False)


def test_running_low_level_call_can_be_marked_timed_out():
    trace = ModelCallTraceHandler()
    trace.on_chat_model_start({}, [[HumanMessage(content="尚未完成")]], run_id=uuid4())
    trace.mark_running_calls_timed_out()
    call = trace.snapshot()[0]
    assert call.status == "timed_out"
    assert call.error_type == "OuterAttemptTimeout"


def test_prompt_defines_therapeutic_targets_without_leaking_a_case_answer():
    prompt = " ".join(PLAN_ONLY_PROMPT.casefold().split())
    assert "do not equate every disease-associated target" in prompt
    assert "known drug evidence or tractability" in prompt
    assert "never put prose sentences" in prompt
    assert "PDAC" not in PLAN_ONLY_PROMPT
    assert "required_operations" not in PLAN_ONLY_PROMPT


def test_prompt_explains_metadata_and_reference_contracts():
    assert "`characteristics` is metadata with exactly this shape" in PLAN_ONLY_PROMPT
    assert "Never put a prior step's output name in entity_refs" in PLAN_ONLY_PROMPT
    assert "Do not generate `generated_at`" in PLAN_ONLY_PROMPT


def test_query_manifest_generated_at_is_runtime_default_not_model_requirement():
    payload = _valid_payload()
    payload.pop("generated_at")
    plan = QueryManifestV2.model_validate(payload)
    assert plan.generated_at.tzinfo is not None
    assert "generated_at" not in QueryManifestV2.model_json_schema()["required"]


def test_patch_schema_does_not_request_parent_plan_metadata():
    properties = QueryManifestPatch.model_json_schema()["properties"]
    assert "schema_version" not in properties
    assert "original_question" not in properties
    assert "manifest_id" not in properties
    assert "generated_at" not in properties


def test_default_full_and_patch_paths_use_direct_models(monkeypatch):
    models = []

    def fake_model():
        model = object()
        models.append(model)
        return model

    monkeypatch.setattr(planner_module, "get_chat_model", fake_model)
    planner = NaturalLanguageResearchPlanner(timeout_cleanup=lambda model: {})

    assert len(models) == 2
    assert isinstance(planner._agent, DirectJsonPlanAgent)
    assert planner._agent._model is models[0]
    assert isinstance(planner._patch_agent, DirectJsonPatchAgent)
    assert planner._patch_agent._model is models[1]


def test_direct_full_plan_agent_makes_one_call_and_normalizes_output_dependency():
    payload = _valid_payload()
    payload["plan_steps"][1]["depends_on"] = ["resolved_disease"]

    class DirectModel:
        def __init__(self):
            self.calls = []

        def invoke(self, messages, config=None):
            self.calls.append((messages, config))
            return AIMessage(
                content="</think>\n" + json.dumps(payload, ensure_ascii=False)
            )

    model = DirectModel()
    result = DirectJsonPlanAgent(model).invoke(
        {"messages": [{"role": "user", "content": "建立 Plan"}]}
    )

    plan = result["structured_response"]
    assert len(model.calls) == 1
    assert plan.plan_steps[1].depends_on == ["step_1"]
    assert "replaced output dependency" in plan.reference_corrections[0]


def test_dependency_normalization_does_not_guess_unknown_or_future_output():
    payload = _valid_payload()
    payload["plan_steps"][1]["depends_on"] = ["target_drug_evidence"]

    class DirectModel:
        def invoke(self, messages, config=None):
            return AIMessage(content=json.dumps(payload, ensure_ascii=False))

    with pytest.raises(Exception, match="non-prior dependencies"):
        DirectJsonPlanAgent(DirectModel()).invoke({"messages": []})


def test_direct_patch_agent_makes_one_call_and_parses_json_after_thinking():
    class DirectModel:
        def __init__(self):
            self.calls = []

        def invoke(self, messages, config=None):
            self.calls.append((messages, config))
            return AIMessage(
                content=(
                    "內部思考不屬於 JSON。\n</think>\n"
                    '{"upsert_steps": [], "global_safeguards": '
                    '["missing_not_negative"]}'
                )
            )

    model = DirectModel()
    result = DirectJsonPatchAgent(model).invoke(
        {"messages": [{"role": "user", "content": "修正 Plan"}]}
    )

    assert len(model.calls) == 1
    assert result["structured_response"].global_safeguards == ["missing_not_negative"]


def test_direct_patch_preserves_omitted_fields_for_existing_step():
    base = QueryManifestV2.model_validate(_valid_payload())

    class DirectModel:
        def invoke(self, messages, config=None):
            return AIMessage(
                content=json.dumps(
                    {
                        "upsert_steps": [
                            {
                                "step_id": "step_6",
                                "ranking": {
                                    "primary_metric": "overall_association_score",
                                    "direction": "descending",
                                    "secondary_metrics": [],
                                },
                            }
                        ]
                    }
                )
            )

    result = DirectJsonPatchAgent(DirectModel()).invoke(
        {
            "messages": [],
            "base_plan": base,
        }
    )

    step = result["structured_response"].upsert_steps[0]
    assert step.outputs == ["ranked_targets"]
    assert step.operation == "rank_targets"
    assert "preserved omitted existing-step fields" in result["patch_normalizations"][0]


def test_direct_patch_does_not_complete_an_incomplete_new_step():
    base = QueryManifestV2.model_validate(_valid_payload())

    class DirectModel:
        def invoke(self, messages, config=None):
            return AIMessage(
                content=json.dumps(
                    {
                        "upsert_steps": [
                            {
                                "step_id": "new_step",
                                "operation": "retrieve_target_drugs",
                            }
                        ]
                    }
                )
            )

    with pytest.raises(Exception, match="Field required"):
        DirectJsonPatchAgent(DirectModel()).invoke(
            {
                "messages": [],
                "base_plan": base,
            }
        )


def test_direct_dependency_output_is_removed_from_entity_refs_and_recorded():
    payload = _valid_payload()
    payload["plan_steps"][2]["entity_refs"] = ["candidate_targets"]
    plan = QueryManifestV2.model_validate(payload)
    assert plan.plan_steps[2].entity_refs == []
    assert "candidate_targets" in plan.reference_corrections[0]


def test_output_ref_from_non_dependency_is_still_rejected():
    payload = _valid_payload()
    payload["plan_steps"][2]["depends_on"] = []
    payload["plan_steps"][2]["entity_refs"] = ["candidate_targets"]
    with pytest.raises(Exception, match="unknown entities"):
        QueryManifestV2.model_validate(payload)


def test_transitive_upstream_output_ref_is_removed_and_recorded():
    payload = _valid_payload()
    payload["plan_steps"][4]["entity_refs"] = ["candidate_targets"]

    plan = QueryManifestV2.model_validate(payload)

    assert plan.plan_steps[4].entity_refs == []
    assert any(
        "upstream-step output 'candidate_targets'" in item
        for item in plan.reference_corrections
    )


def test_arbitrary_unknown_entity_ref_is_still_rejected():
    payload = _valid_payload()
    payload["plan_steps"][2]["entity_refs"] = ["invented_reference"]
    with pytest.raises(Exception, match="unknown entities"):
        QueryManifestV2.model_validate(payload)


def test_planner_report_records_safe_output_ref_normalization():
    payload = _valid_payload()
    payload["plan_steps"][2]["entity_refs"] = ["candidate_targets"]
    report = NaturalLanguageResearchPlanner(
        agent=FakeAgent([{"structured_response": payload}])
    ).run_with_report(QUESTION)
    assert report.status == "passed"
    assert any(
        "candidate_targets" in correction
        for correction in report.attempts[0].corrected_model_claims
    )


def test_operation_catalog_explains_boundaries_in_chinese():
    assert "canonical=retrieve_target_drugs" in OPERATION_CATALOG_PROMPT
    assert "不負責 tractability" in OPERATION_CATALOG_PROMPT
    assert "canonical=stratify_target_evidence_layers" in OPERATION_CATALOG_PROMPT
    assert "不做類別計數" in OPERATION_CATALOG_PROMPT
    assert "canonical=aggregate_category_distribution" in OPERATION_CATALOG_PROMPT
    assert "不用來建立證據層級" in OPERATION_CATALOG_PROMPT


def test_grouped_compact_catalog_is_complete_shorter_and_reversible():
    grouped_operations = {
        item for operations in OPERATION_GROUPS.values() for item in operations
    }
    compact = build_operation_catalog_prompt("grouped_compact")
    full = build_operation_catalog_prompt("full")

    assert grouped_operations == set(STANDARD_OPERATION_CATALOG)
    assert len(compact) < len(full)
    assert all(f"canonical={item}" in compact for item in STANDARD_OPERATION_CATALOG)
    assert all(
        OPERATION_SEMANTICS[item] in compact for item in COMPACT_BOUNDARY_OPERATIONS
    )
    assert full == OPERATION_CATALOG_PROMPT


def test_report_records_selected_operation_catalog_mode(monkeypatch):
    monkeypatch.setenv("PLANNER_OPERATION_CATALOG_MODE", "grouped_compact")
    report = NaturalLanguageResearchPlanner(
        agent=FakeAgent([{"structured_response": _valid_payload()}])
    ).run_with_report(QUESTION)

    assert report.operation_catalog_mode == "grouped_compact"


def test_unknown_operation_catalog_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("PLANNER_OPERATION_CATALOG_MODE", "unknown")

    with pytest.raises(NaturalLanguagePlannerError, match="grouped_compact"):
        NaturalLanguageResearchPlanner(
            agent=FakeAgent([{"structured_response": _valid_payload()}])
        ).run_with_report(QUESTION)


def test_planner_repairs_a_changed_original_question():
    agent = FakeAgent(
        [
            {"structured_response": _valid_payload("被模型改寫的問題")},
            {"structured_response": _valid_payload()},
        ]
    )

    plan = NaturalLanguageResearchPlanner(agent=agent).run(QUESTION)

    assert plan.original_question == QUESTION
    assert len(agent.calls) == 2
    assert "original_question" in agent.calls[1]["messages"][0]["content"]


def test_planner_rejects_empty_question_before_model_call():
    agent = FakeAgent([])

    with pytest.raises(NaturalLanguagePlannerError, match="cannot be empty"):
        NaturalLanguageResearchPlanner(agent=agent).run("  ")

    assert agent.calls == []


def test_planner_deterministically_clears_invented_verified_entity_identity():
    invented = _valid_payload()
    invented["entities"][0].update(
        {
            "entity_id": "invented_id",
            "canonical_name": "invented name",
            "resolution_status": "verified",
        }
    )
    agent = FakeAgent([{"structured_response": invented}])

    plan = NaturalLanguageResearchPlanner(agent=agent).run(QUESTION)

    assert plan.entities[0].resolution_status == "unresolved"
    assert plan.entities[0].entity_id is None
    assert plan.entities[0].canonical_name is None
    assert len(agent.calls) == 1


def test_hidden_acceptance_is_not_leaked_and_only_diagnostics_are_repaired():
    case = load_acceptance_cases("tests/fixtures/planner_acceptance_cases.json")[0]
    incomplete = _valid_payload(case.question)
    incomplete["plan_steps"] = incomplete["plan_steps"][:1]
    agent = FakeAgent(
        [
            {"structured_response": incomplete},
            {"structured_response": _valid_payload(case.question)},
        ]
    )

    report = NaturalLanguageResearchPlanner(agent=agent).run_with_report(
        case.question,
        acceptance_case=case,
    )

    assert [step.operation for step in report.final_plan.plan_steps] == [
        "resolve_disease",
        "discover_targets",
        "retrieve_target_drugs",
        "retrieve_tractability",
        "stratify_target_evidence_layers",
        "rank_targets",
    ]
    assert len(agent.calls) == 2
    first_prompt = agent.calls[0]["messages"][0]["content"]
    assert "required_operations" not in first_prompt
    assert case.research_intent not in first_prompt
    assert build_operation_catalog_prompt() in first_prompt
    repair_prompt = agent.calls[1]["messages"][0]["content"]
    assert "rank_targets" in repair_prompt
    assert "research_intent" not in repair_prompt
    assert "上一輪候選 Plan" in repair_prompt
    assert '"operation": "resolve_disease"' in repair_prompt
    assert report.first_attempt_passed is False
    assert report.repair_count == 1
    assert report.evidence_collection_started is False
    assert report.attempts[0].semantic_acceptance.passed is False
    assert report.attempts[1].semantic_acceptance.passed is True
    assert report.attempts[0].candidate_plan is not None
    assert report.attempts[0].semantic_acceptance.actual_operations == [
        "resolve_disease"
    ]


def test_schema_valid_failure_is_repaired_with_bounded_patch_when_available():
    broken = _valid_payload()
    broken["plan_steps"][4]["sources"] = ["open_targets"]
    broken["plan_steps"][4]["aggregation"] = None
    repaired_step = _valid_payload()["plan_steps"][4]
    patch = {
        "schema_version": "1.0",
        "original_question": QUESTION,
        "upsert_steps": [repaired_step],
    }
    full_agent = FakeAgent([{"structured_response": broken}])
    patch_agent = FakeAgent([{"structured_response": patch}])

    report = NaturalLanguageResearchPlanner(
        agent=full_agent,
        patch_agent=patch_agent,
    ).run_with_report(QUESTION)

    assert report.status == "passed"
    assert report.repair_mode == "local_patch"
    assert len(full_agent.calls) == 1
    assert len(patch_agent.calls) == 1
    assert report.attempts[1].generation_mode == "local_patch"
    assert report.attempts[1].applied_patch.upsert_steps[0].step_id == "step_5"
    assert (
        report.final_plan.plan_steps[0]
        == QueryManifestV2.model_validate(_valid_payload()).plan_steps[0]
    )
    patch_prompt = patch_agent.calls[0]["messages"][0]["content"]
    assert "只回傳 QueryManifestPatch" in patch_prompt


def test_patch_step_order_must_cover_every_final_step():
    base = QueryManifestV2.model_validate(_valid_payload())
    patch = QueryManifestPatch.model_validate(
        {
            "original_question": QUESTION,
            "step_order": ["step_1"],
        }
    )
    with pytest.raises(ValueError, match="every final step ID"):
        apply_manifest_patch(base, patch)


def test_stale_step_order_is_ignored_when_patch_only_updates_existing_steps():
    base = QueryManifestV2.model_validate(_valid_payload())
    changed = _valid_payload()["plan_steps"][5]
    changed["description"] = "只修改既有排序步驟。"
    patch = QueryManifestPatch.model_validate(
        {
            "upsert_steps": [changed],
            "step_order": ["step_1"],
        }
    )

    repaired = apply_manifest_patch(base, patch)

    assert [step.step_id for step in repaired.plan_steps] == [
        step.step_id for step in base.plan_steps
    ]
    assert repaired.plan_steps[5].description == "只修改既有排序步驟。"


def test_repair_feedback_batches_structure_before_ranking():
    feedback = RepairFeedback(
        missing=[
            {
                "area": "research_capability",
                "problem": "缺少必要項目：retrieve_target_drugs",
                "required_correction": "補上查詢。",
            },
            {
                "area": "ranking",
                "problem": "缺少必要項目：overall_association_score",
                "required_correction": "補上排序。",
            },
        ],
        incorrect=[
            {
                "area": "analysis_metric_requires_available_evidence",
                "problem": "step_id=rank_targets_1; metric 未登記",
                "required_correction": "改用正式指標。",
            }
        ],
        preserve=[
            {
                "area": "accepted_capability",
                "item": "discover_targets",
                "instruction": "保留。",
            }
        ],
    )

    batch = select_repair_batch(feedback)

    assert [item["area"] for item in batch.missing] == ["research_capability"]
    assert batch.incorrect == []
    assert batch.preserve == feedback.preserve


def test_repair_feedback_sends_ranking_after_structure_is_clear():
    feedback = RepairFeedback(
        incorrect=[
            {
                "area": "analysis_metric_requires_available_evidence",
                "problem": "step_id=rank_targets_1; metric 未登記",
                "required_correction": "改用正式指標。",
            }
        ]
    )

    batch = select_repair_batch(feedback)

    assert batch.incorrect == feedback.incorrect


def test_four_attempt_run_can_be_recorded_without_report_schema_failure():
    broken = _valid_payload()
    broken["plan_steps"][4]["aggregation"] = None
    unchanged_step = broken["plan_steps"][4]
    report = NaturalLanguageResearchPlanner(
        agent=FakeAgent([{"structured_response": broken}]),
        patch_agent=FakeAgent(
            [
                {"structured_response": {"upsert_steps": [unchanged_step]}},
                {"structured_response": {"upsert_steps": [unchanged_step]}},
                {"structured_response": {"upsert_steps": [unchanged_step]}},
            ]
        ),
    ).run_with_report(QUESTION, max_attempts=4)

    assert report.status == "failed"
    assert len(report.attempts) == 4
    assert report.repair_count == 3


def test_resume_starts_with_patch_from_supplied_best_plan():
    broken = _valid_payload()
    broken["plan_steps"][4]["aggregation"] = {"target_translation_layer": "reserved"}
    repaired_step = _valid_payload()["plan_steps"][4]

    report = NaturalLanguageResearchPlanner(
        agent=FakeAgent([]),
        patch_agent=FakeAgent(
            [{"structured_response": {"upsert_steps": [repaired_step]}}]
        ),
    ).run_with_report(
        QUESTION,
        max_attempts=1,
        initial_plan=QueryManifestV2.model_validate(broken),
    )

    assert report.status == "passed"
    assert report.resumed_from_plan is True
    assert report.attempts[0].generation_mode == "local_patch"


def test_preserved_operations_do_not_expand_repair_scope():
    plan = QueryManifestV2.model_validate(_valid_payload())
    rules = evaluate_operation_rules(plan)
    feedback = RepairFeedback(
        missing=[
            {
                "area": "aggregation",
                "problem": "缺少必要項目：target_translation_layer",
                "required_correction": "加入 aggregation 頂層 key。",
            }
        ],
        preserve=[
            {
                "area": "accepted_capability",
                "item": "rank_targets",
                "instruction": "保留。",
            }
        ],
    )
    acceptance = PlanAcceptanceResult(
        case_id="scope_test",
        passed=False,
        missing_aggregation=["target_translation_layer"],
    )

    scope = build_repair_scope(plan, acceptance, rules, feedback)

    assert scope.step_ids == ["step_5"]
    assert "step_6" not in scope.step_ids


def _dependency_feedback(*, rejected_step_id=None):
    incorrect = []
    if rejected_step_id is not None:
        incorrect.append(
            {
                "area": "schema_or_integrity",
                "problem": (
                    "patch attempted edits outside the authorized repair scope: "
                    f"upsert_steps:{rejected_step_id}"
                ),
                "required_correction": "只修改授權步驟。",
            }
        )
    return RepairFeedback(
        missing=[
            {
                "area": "dependency",
                "problem": (
                    "缺少必要項目：stratify_target_evidence_layers->rank_targets"
                ),
                "required_correction": "加入正確方向的依賴。",
            }
        ],
        incorrect=incorrect,
    )


def _plan_missing_final_ranking_dependency():
    payload = _valid_payload()
    payload["plan_steps"][5]["depends_on"] = ["step_2", "step_3", "step_4"]
    return QueryManifestV2.model_validate(payload)


def test_dependency_repair_task_resolves_exact_direction_and_step_ids():
    plan = _plan_missing_final_ranking_dependency()
    task = build_dependency_repair_task(
        plan,
        _dependency_feedback(),
        RepairScope(step_ids=["step_6"]),
    )

    assert task is not None
    assert task.editable_step_id == "step_6"
    assert task.required_dependency_step_id == "step_5"
    assert task.required_final_dependencies == ["step_2", "step_3", "step_4", "step_5"]
    assert task.forbidden_reverse_edge == "step_5 depends_on step_6"


def test_dependency_only_prompt_omits_unrelated_catalogs():
    plan = _plan_missing_final_ranking_dependency()
    task = build_dependency_repair_task(
        plan,
        _dependency_feedback(),
        RepairScope(step_ids=["step_6"]),
    )

    prompt = build_dependency_repair_prompt(QUESTION, plan, task)

    assert "唯一可修改步驟" in prompt
    assert '"editable_step_id": "step_6"' in prompt
    assert "OPERATION_TABLE" not in prompt
    assert "Reviewed metrics" not in prompt


def test_dependency_retry_names_the_previously_rejected_step():
    plan = _plan_missing_final_ranking_dependency()
    task = build_dependency_repair_task(
        plan,
        _dependency_feedback(rejected_step_id="step_5"),
        RepairScope(step_ids=["step_6"]),
    )

    prompt = build_dependency_repair_prompt(QUESTION, plan, task)

    assert task.previously_rejected_step_ids == ["step_5"]
    assert "上一輪被拒絕" in prompt
    assert "唯一允許出現的 step_id 是：step_6" in prompt


def test_repeated_identical_out_of_scope_patch_stops_early():
    case = next(
        item
        for item in load_acceptance_cases(
            "tests/fixtures/planner_acceptance_cases.json"
        )
        if item.case_id == "pdac_known_targets"
    )
    plan = _plan_missing_final_ranking_dependency()
    wrong_step = plan.plan_steps[4].model_copy(deep=True)
    wrong_step.depends_on.append("step_6")
    wrong_patch = {"upsert_steps": [wrong_step.model_dump()]}
    patch_agent = FakeAgent(
        [
            {"structured_response": wrong_patch},
            {"structured_response": wrong_patch},
        ]
    )

    report = NaturalLanguageResearchPlanner(
        agent=FakeAgent([]), patch_agent=patch_agent
    ).run_with_report(
        QUESTION,
        acceptance_case=case,
        initial_plan=plan,
        max_attempts=5,
    )

    assert report.status == "failed"
    assert len(report.attempts) == 2
    assert report.attempts[1].repeated_repair_failure is True
    assert len(patch_agent.calls) == 2


def test_full_rewrite_repair_mode_remains_reversible(monkeypatch):
    monkeypatch.setenv("PLANNER_REPAIR_MODE", "full_rewrite")
    broken = _valid_payload()
    broken["plan_steps"][4]["aggregation"] = None
    full_agent = FakeAgent(
        [
            {"structured_response": broken},
            {"structured_response": _valid_payload()},
        ]
    )
    patch_agent = FakeAgent([])

    report = NaturalLanguageResearchPlanner(
        agent=full_agent,
        patch_agent=patch_agent,
    ).run_with_report(QUESTION)

    assert report.status == "passed"
    assert report.repair_mode == "full_rewrite"
    assert len(full_agent.calls) == 2
    assert patch_agent.calls == []
    assert report.attempts[1].generation_mode == "full_plan"


def test_patch_cannot_modify_an_unflagged_step():
    base = QueryManifestV2.model_validate(_valid_payload())
    changed = _valid_payload()["plan_steps"][0]
    changed["description"] = "越權改寫正確步驟。"
    patch = QueryManifestPatch.model_validate(
        {
            "original_question": QUESTION,
            "upsert_steps": [changed],
        }
    )
    scope = RepairScope(step_ids=["step_5"])

    with pytest.raises(ValueError, match="outside the authorized repair scope"):
        validate_patch_scope(base, patch, scope)


def test_invalid_patch_keeps_original_feedback_for_next_repair():
    broken = _valid_payload()
    broken["plan_steps"][4]["aggregation"] = None
    unauthorized = _valid_payload()["plan_steps"][0]
    unauthorized["description"] = "不應修改。"
    valid_repair = _valid_payload()["plan_steps"][4]
    full_agent = FakeAgent([{"structured_response": broken}])
    patch_agent = FakeAgent(
        [
            {
                "structured_response": {
                    "original_question": QUESTION,
                    "upsert_steps": [unauthorized],
                }
            },
            {
                "structured_response": {
                    "original_question": QUESTION,
                    "upsert_steps": [valid_repair],
                }
            },
        ]
    )

    report = NaturalLanguageResearchPlanner(
        agent=full_agent, patch_agent=patch_agent
    ).run_with_report(QUESTION)

    assert report.status == "passed"
    assert report.attempts[1].schema_valid is False
    third_prompt = patch_agent.calls[1]["messages"][0]["content"]
    assert "缺少描述分層或彙整方法" in third_prompt
    assert "outside the authorized repair scope" in third_prompt


def test_repair_catalog_contains_relevant_operations_not_the_full_table():
    plan = QueryManifestV2.model_validate(_valid_payload())
    feedback = RepairFeedback(
        missing=[
            {
                "area": "research_capability",
                "problem": "缺少必要項目：retrieve_approved_indications",
                "required_correction": "補上核准資料。",
            }
        ]
    )

    catalog = build_repair_operation_catalog(plan, feedback)

    assert "canonical=retrieve_approved_indications" in catalog
    assert "canonical=stratify_target_evidence_layers" in catalog
    assert "canonical=retrieve_safety_evidence" not in catalog


def test_equal_quality_patch_does_not_replace_the_better_repair_base():
    broken = _valid_payload()
    broken["plan_steps"][4]["sources"] = ["open_targets"]
    broken["plan_steps"][4]["aggregation"] = None
    fixes_sources_only = _valid_payload()["plan_steps"][4]
    fixes_sources_only["aggregation"] = None
    trades_one_error_for_another = _valid_payload()["plan_steps"][4]
    trades_one_error_for_another["sources"] = ["open_targets"]

    report = NaturalLanguageResearchPlanner(
        agent=FakeAgent([{"structured_response": broken}]),
        patch_agent=FakeAgent(
            [
                {
                    "structured_response": {
                        "original_question": QUESTION,
                        "upsert_steps": [fixes_sources_only],
                    }
                },
                {
                    "structured_response": {
                        "original_question": QUESTION,
                        "upsert_steps": [trades_one_error_for_another],
                    }
                },
            ]
        ),
    ).run_with_report(QUESTION)

    assert report.status == "failed"
    assert [item.promoted_to_repair_base for item in report.attempts] == [
        True,
        True,
        False,
    ]
    assert "did not improve" in report.attempts[2].errors[-1]


def test_local_patch_can_add_hidden_acceptance_missing_operations():
    case = load_acceptance_cases("tests/fixtures/planner_acceptance_cases.json")[0]
    complete = _valid_payload(case.question)
    incomplete = _valid_payload(case.question)
    incomplete["plan_steps"] = incomplete["plan_steps"][:1]
    structural_patch = {
        "original_question": case.question,
        "upsert_steps": complete["plan_steps"][1:5],
        "step_order": [item["step_id"] for item in complete["plan_steps"][:5]],
        "global_safeguards": complete["global_safeguards"],
    }
    ranking_patch = {
        "original_question": case.question,
        "upsert_steps": [complete["plan_steps"][5]],
        "step_order": [item["step_id"] for item in complete["plan_steps"]],
    }

    report = NaturalLanguageResearchPlanner(
        agent=FakeAgent([{"structured_response": incomplete}]),
        patch_agent=FakeAgent(
            [
                {"structured_response": structural_patch},
                {"structured_response": ranking_patch},
            ]
        ),
    ).run_with_report(case.question, acceptance_case=case)

    assert report.status == "passed"
    assert report.attempts[1].generation_mode == "local_patch"
    assert set(report.attempts[1].repair_scope.new_operations) >= {
        "discover_targets",
        "retrieve_target_drugs",
        "retrieve_tractability",
        "stratify_target_evidence_layers",
    }
    assert "rank_targets" in report.attempts[2].repair_scope.new_operations


def test_run_report_records_corrected_claims_and_zero_retrieval():
    invented = _valid_payload()
    invented["entities"][0].update(
        {
            "entity_id": "invented_id",
            "canonical_name": "invented name",
            "resolution_status": "verified",
        }
    )
    report = NaturalLanguageResearchPlanner(
        agent=FakeAgent([{"structured_response": invented}])
    ).run_with_report(QUESTION)

    assert report.status == "passed"
    assert report.attempts[0].corrected_model_claims
    assert report.attempts[0].executed_retrieval_tools == []
    assert report.attempts[0].duration_seconds >= 0


def test_runtime_overwrites_model_generated_timestamp():
    payload = _valid_payload()
    payload["generated_at"] = "2023-10-05T12:00:00Z"
    report = NaturalLanguageResearchPlanner(
        agent=FakeAgent([{"structured_response": payload}])
    ).run_with_report(QUESTION)

    assert report.final_plan.generated_at.year == 2026
    assert report.attempts[0].candidate_plan.generated_at.year == 2026


def test_repair_report_detects_regression_between_attempts():
    case = load_acceptance_cases("tests/fixtures/planner_acceptance_cases.json")[0]
    first = _valid_payload(case.question)
    first["plan_steps"] = first["plan_steps"][:1]
    second = _valid_payload(case.question)
    second["plan_steps"] = second["plan_steps"][1:]
    second["plan_steps"][0]["depends_on"] = []
    third = _valid_payload(case.question)
    report = NaturalLanguageResearchPlanner(
        agent=FakeAgent(
            [
                {"structured_response": first},
                {"structured_response": second},
                {"structured_response": third},
            ]
        )
    ).run_with_report(case.question, acceptance_case=case)

    assert report.status == "passed"
    assert report.attempts[1].regressed_components == ["resolve_disease"]
    assert any(
        item["area"] == "regression"
        for item in report.attempts[1].repair_feedback_sent_next.incorrect
    )


def test_checkpoint_is_written_after_each_completed_attempt(tmp_path):
    case = load_acceptance_cases("tests/fixtures/planner_acceptance_cases.json")[0]
    first = _valid_payload(case.question)
    first["plan_steps"] = first["plan_steps"][:1]
    checkpoint_path = tmp_path / "planner.report.json"
    agent = CheckpointAwareAgent(
        [
            {"structured_response": first},
            {"structured_response": _valid_payload(case.question)},
        ],
        checkpoint_path,
    )

    report = NaturalLanguageResearchPlanner(agent=agent).run_with_report(
        case.question,
        acceptance_case=case,
        checkpoint_path=checkpoint_path,
    )

    saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert report.status == "passed"
    assert saved["status"] == "passed"
    assert len(saved["attempts"]) == 2


def test_attempt_timeout_saves_failure_and_does_not_start_repair(tmp_path, monkeypatch):
    class SlowAgent:
        calls = 0

        def invoke(self, request):
            self.calls += 1
            time.sleep(0.1)
            return {"structured_response": _valid_payload()}

    monkeypatch.setenv("PLANNER_ATTEMPT_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("PLANNER_TOTAL_TIMEOUT_SECONDS", "0.05")
    checkpoint_path = tmp_path / "timeout.report.json"
    agent = SlowAgent()

    report = NaturalLanguageResearchPlanner(agent=agent).run_with_report(
        QUESTION,
        checkpoint_path=checkpoint_path,
    )

    assert report.status == "failed"
    assert len(report.attempts) == 1
    assert report.attempts[0].errors[0].startswith("ModelAttemptTimeout:")
    assert report.attempts[0].model_generation_timeout is True
    assert agent.calls == 1
    assert checkpoint_path.exists()


def test_attempt_timeout_records_ollama_cleanup_outcome(tmp_path, monkeypatch):
    class SlowAgent:
        def invoke(self, request):
            time.sleep(0.1)
            return {"structured_response": _valid_payload()}

    cleanup_calls = []

    def cleanup():
        cleanup_calls.append(True)
        return {
            "ollama_stop_requested": True,
            "ollama_stopping_detected": True,
            "ollama_stopping_timeout": False,
            "ollama_cleanup_succeeded": True,
            "ollama_cleanup_error": None,
        }

    monkeypatch.setenv("PLANNER_ATTEMPT_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("PLANNER_TOTAL_TIMEOUT_SECONDS", "0.05")
    report = NaturalLanguageResearchPlanner(
        agent=SlowAgent(), timeout_cleanup=cleanup
    ).run_with_report(
        QUESTION,
        checkpoint_path=tmp_path / "cleanup.report.json",
    )

    attempt = report.attempts[0]
    assert cleanup_calls == [True]
    assert attempt.model_generation_timeout is True
    assert attempt.ollama_stop_requested is True
    assert attempt.ollama_stopping_detected is True
    assert attempt.ollama_stopping_timeout is False
    assert attempt.ollama_cleanup_succeeded is True


def test_ollama_cleanup_detects_stopping_then_confirms_model_removed(monkeypatch):
    states = iter([(True, True), (False, False)])
    stop_calls = []

    monkeypatch.setattr(
        "natural_language_planner._ollama_executable", lambda: "ollama-test"
    )
    monkeypatch.setattr(
        "natural_language_planner._ollama_process_state",
        lambda executable, model: next(states),
    )
    monkeypatch.setattr(
        "natural_language_planner.subprocess.run",
        lambda *args, **kwargs: stop_calls.append(args[0]),
    )
    monkeypatch.setenv("PLANNER_OLLAMA_STOP_GRACE_SECONDS", "1")
    monkeypatch.setenv("PLANNER_OLLAMA_STOP_POLL_SECONDS", "0.001")

    result = _cleanup_ollama_after_timeout()

    assert stop_calls == [["ollama-test", "stop", "qwen3:14b"]]
    assert result["ollama_stop_requested"] is True
    assert result["ollama_stopping_detected"] is True
    assert result["ollama_stopping_timeout"] is False
    assert result["ollama_cleanup_succeeded"] is True
