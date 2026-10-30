from types import SimpleNamespace

from execution.compiler import _inherit_truncation
from execution.contracts import StepResult


def _result(step_id, status="success", source="local_analysis", truncated=False, payload=None):
    return StepResult(
        step_id=step_id,
        operation="x",
        status=status,
        source=source,
        truncated=truncated,
        payload=payload,
    )


def _step(*deps):
    return SimpleNamespace(depends_on=list(deps))


def test_local_analysis_inherits_truncation():
    done = {"a": _result("a", "partial_truncated", "clinicaltrials_gov", True)}
    out = _inherit_truncation(_step("a"), _result("b", payload={"limitations": []}), done)
    assert out.status == "partial_truncated"
    assert out.truncated
    assert "a" in out.payload["limitations"][-1]


def test_complete_upstream_stays_success():
    done = {"a": _result("a", "success", "clinicaltrials_gov")}
    out = _inherit_truncation(_step("a"), _result("b"), done)
    assert out.status == "success"
    assert not out.truncated


def test_no_records_is_not_overwritten():
    done = {"a": _result("a", "partial_truncated", "clinicaltrials_gov", True)}
    out = _inherit_truncation(_step("a"), _result("b", "no_records"), done)
    assert out.status == "no_records"


def test_retrieval_step_is_left_alone():
    done = {"a": _result("a", "partial_truncated", "open_targets", True)}
    out = _inherit_truncation(_step("a"), _result("b", source="chembl"), done)
    assert out.status == "success"
