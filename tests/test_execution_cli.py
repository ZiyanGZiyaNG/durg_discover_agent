from unittest.mock import patch

from execution.__main__ import main
from test_execution_acceptance import FOUND, _drugs, _fake_resolve, _tractability

PLAN = "outputs/plans/phase2_pdac_known_targets_batched_v4.json"


def test_cli_prints_every_step_and_the_ranking(capsys):
    with (
        patch("execution.adapters.resolve_entity", _fake_resolve),
        patch("execution.adapters.discover_disease_targets", lambda *a, **k: FOUND),
        patch("execution.adapters.get_target_drug_evidence_by_ids", _drugs),
        patch("execution.adapters.get_target_tractability_by_id", _tractability),
    ):
        code = main([PLAN, "--disease-id", "EFO_1"])
    out = capsys.readouterr().out
    assert code == 0
    assert out.count("success") == 6
    assert out.index("T2") < out.index("T3") < out.index("T1")


def test_cli_reports_a_refused_plan_without_running_anything(capsys):
    code = main(["outputs/plans/acceptance_pdac_known_targets.json"])
    assert code == 2
    assert "refused" in capsys.readouterr().out
