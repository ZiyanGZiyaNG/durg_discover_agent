import json

import pytest

from planner_examples import build_few_shot_prompt, examples
from planner_acceptance import OPERATION_TABLE, load_acceptance_cases
from schemas import QueryManifestV2
from natural_language_planner import NaturalLanguageResearchPlanner
from test_natural_language_planner import FakeAgent, _valid_payload, QUESTION


def test_demonstrations_are_valid_and_not_acceptance_answers():
    questions = {
        x.question
        for x in load_acceptance_cases("tests/fixtures/planner_acceptance_cases.json")
    }
    for item in examples():
        plan = QueryManifestV2.model_validate(item["plan"])
        assert plan.original_question not in questions
        assert all(s.operation in OPERATION_TABLE for s in plan.plan_steps)
        assert all(e.resolution_status == "unresolved" for e in plan.entities)
        assert plan.created_before_retrieval


def test_few_shot_is_reversible_and_rejects_unknown_mode():
    assert build_few_shot_prompt("off") == ""
    assert "synthetic_pair_sources" in build_few_shot_prompt("synthetic_v1")
    with pytest.raises(ValueError):
        build_few_shot_prompt("unknown")


def test_examples_are_sent_and_reported_without_hidden_rubric(monkeypatch):
    monkeypatch.setenv("PLANNER_FEW_SHOT_MODE", "synthetic_v1")
    agent = FakeAgent([{"structured_response": _valid_payload()}])
    report = NaturalLanguageResearchPlanner(agent=agent).run_with_report(QUESTION)
    request = json.dumps(agent.calls, ensure_ascii=False)
    assert "synthetic_pair_sources" in request
    assert "required_operations" not in request
    assert report.few_shot_mode == "synthetic_v1"
    assert report.executed_retrieval_tools == []


def test_first_attempt_comparison_never_repairs(monkeypatch):
    monkeypatch.setenv("PLANNER_FEW_SHOT_MODE", "off")
    agent = FakeAgent([{"structured_response": {}}])
    report = NaturalLanguageResearchPlanner(agent=agent).run_with_report(
        QUESTION, max_attempts=1
    )
    assert report.status == "failed"
    assert len(agent.calls) == 1
    assert report.repair_count == 0


def test_comparison_rejects_holdout_before_model_call(monkeypatch, tmp_path):
    import compare_few_shot

    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_few_shot.py",
            "--cases",
            "kras_associated_diseases",
            "--output",
            str(tmp_path / "no_run"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        compare_few_shot.main()
    assert exc.value.code == 2
    assert not (tmp_path / "no_run").exists()


def test_comparison_rejects_inadequate_context(monkeypatch, tmp_path):
    import compare_few_shot

    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_few_shot.py",
            "--num-ctx",
            "4096",
            "--output",
            str(tmp_path / "no_run"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        compare_few_shot.main()
    assert exc.value.code == 2
