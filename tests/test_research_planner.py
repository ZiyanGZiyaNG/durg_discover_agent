from datetime import datetime, timezone

import pytest

from research_planner import _validate_against_verified_context
from schemas import Agent1ResearchPlan, QueryManifest


def _valid_plan():
    target = {
        "query": "KRAS",
        "entity_id": "ENSG00000133703",
        "canonical_name": "KRAS proto-oncogene, GTPase",
        "symbol": "KRAS",
        "verified_synonyms": [],
        "source": "Open Targets",
    }
    queries = []
    tools = {
        "open_targets": "open_targets_search",
        "europe_pmc": "europe_pmc_search",
        "clinicaltrials_gov": "clinicaltrials_gov_search",
        "chembl": "chembl_search",
    }
    for order, (source, tool) in enumerate(tools.items(), 1):
        queries.append(
            {
                "datasource": source,
                "tool_name": tool,
                "query_purpose": "查詢證據",
                "rationale": "必要來源",
                "verified_search_terms": ["KRAS", "pancreatic cancer"],
                "unvalidated_search_terms": [],
                "parameters": {},
                "expected_question": "是否有相關證據？",
                "execution_order": order,
                "collection_policy_status": "not_configured",
            }
        )
    plan = Agent1ResearchPlan(
        mode="disease_target",
        original_question="test",
        query_manifest={
            "original_question": "test",
            "query_type": "evidence_directed",
            "disease": {
                "query": "pancreatic cancer",
                "entity_id": "MONDO_0005192",
                "canonical_name": "pancreatic cancer",
            },
            "targets": [target],
            "query_path": ["disease", "target", "evidence"],
            "target_expansion_strategy": "specified_target",
            "required_evidence": ["disease-target association evidence"],
            "selected_sources": list(tools),
            "selection_reason": "指定 disease-target 證據查詢",
            "generated_at": datetime.now(timezone.utc),
        },
        disease={
            "query": "pancreatic cancer",
            "entity_id": "MONDO_0005192",
            "canonical_name": "pancreatic cancer",
        },
        overall_objective="規劃查詢",
        required_sources=list(tools),
        target_subplans=[
            {
                "target": target,
                "objective": "查詢",
                "queries": queries,
                "anticipated_data_gaps": ["臨床結果可能尚未發布"],
            }
        ],
        first_round_completion_criteria=["四個來源皆完成或記錄錯誤"],
        generated_at=datetime.now(timezone.utc),
        model_provider="ollama",
        model_name="qwen3:14b",
    )
    context = {
        "mode": "disease_target",
        "disease": plan.disease.model_dump(),
        "targets": [target],
    }
    return plan, context


def test_verified_context_accepts_exact_entities():
    plan, context = _valid_plan()
    _validate_against_verified_context(plan, context)


def test_verified_context_rejects_llm_id_change():
    plan, context = _valid_plan()
    plan.disease.entity_id = "FAKE_ID"
    try:
        _validate_against_verified_context(plan, context)
    except ValueError as exc:
        assert "verified disease" in str(exc)
    else:
        raise AssertionError("LLM ID change was not rejected")


def test_verified_context_restores_immutable_target_symbol():
    plan, context = _valid_plan()
    plan.target_subplans[0].target.symbol = None
    _validate_against_verified_context(plan, context)
    assert plan.target_subplans[0].target.symbol == "KRAS"


def test_verified_context_rejects_manifest_target_change():
    plan, context = _valid_plan()
    plan.query_manifest.targets[0].entity_id = "FAKE_ID"
    try:
        _validate_against_verified_context(plan, context)
    except ValueError as exc:
        assert "manifest targets" in str(exc)
    else:
        raise AssertionError("LLM manifest target change was not rejected")


def test_query_manifest_accepts_all_six_query_categories():
    plan, _ = _valid_plan()
    manifest = plan.query_manifest.model_dump()
    query_types = {
        "single_hop",
        "multi_hop",
        "evidence_directed",
        "filter_and_rank",
        "comparison_and_aggregation",
        "safety_and_negative_knowledge",
    }

    for query_type in query_types:
        candidate = {**manifest, "query_type": query_type}
        assert QueryManifest.model_validate(candidate).query_type == query_type


def test_verified_context_rejects_incompatible_query_type():
    plan, context = _valid_plan()
    plan.query_manifest.query_type = "single_hop"

    with pytest.raises(ValueError, match="requires evidence_directed query type"):
        _validate_against_verified_context(plan, context)


def test_verified_context_rejects_incompatible_query_path():
    plan, context = _valid_plan()
    plan.query_manifest.query_path = ["disease", "target"]

    with pytest.raises(ValueError, match="requires disease-target-evidence path"):
        _validate_against_verified_context(plan, context)


def test_query_manifest_rejects_duplicate_selected_sources():
    plan, _ = _valid_plan()
    manifest = plan.query_manifest.model_dump()
    manifest["selected_sources"] = ["open_targets", "open_targets"]

    with pytest.raises(ValueError, match="selected_sources cannot contain duplicates"):
        QueryManifest.model_validate(manifest)
