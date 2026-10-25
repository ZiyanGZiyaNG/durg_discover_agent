# tests/test_step_inputs.py
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict

from execution import compiler
from execution.contracts import StepResult
from schemas import QueryManifestV2


class AnyPayload(BaseModel):
    model_config = ConfigDict(extra="allow")


class Fake:
    def __init__(self, fn):
        self.fn = fn

    def run(self, ctx):
        return self.fn(ctx)


def _result(ctx, payload, provides):
    return StepResult(  # 欄位依你的 StepResult 調整
        step_id=ctx.step_id, operation=ctx.operation, status="success",
        source="fake", payload=payload, provides=provides,
    )


def _plan():
    def step(step_id, op, deps, outputs, inputs=None):
        return {
            "step_id": step_id, "operation": op, "description": "x",
            "depends_on": deps, "outputs": outputs, "inputs": inputs or {},
            "entity_refs": ["d"] if not deps else [],
        }

    return QueryManifestV2.model_validate({
        "original_question": "q", "research_objective": "o",
        "entities": [{"entity_ref": "d", "entity_type": "disease",
                      "mention": "d", "resolution_status": "verified"}],
        "plan_steps": [
            step("s1", "fake_get", [], ["targets"]),
            step("s2", "fake_filter", ["s1"], ["kept"],
                 {"src": {"step_id": "s1", "output": "targets"}}),
            step("s3", "fake_rank", ["s2"], ["ranked"],
                 {"src": {"step_id": "s2", "output": "kept"}}),
        ],
        "evidence_requirements": ["e"], "completion_criteria": ["c"],
        "global_safeguards": ["g"],
        "characteristics": {"primary": "filter_and_rank"},
    })


def test_rank_only_sees_filtered_targets(monkeypatch):
    seen = {}

    def get_(ctx):
        return _result(ctx, {"A": 1, "B": 2, "C": 3}, ["cap_a"])

    def filter_(ctx):
        kept = {k: v for k, v in ctx.inputs["src"].items() if k != "C"}
        return _result(ctx, kept, ["cap_a"])

    def rank_(ctx):
        seen["keys"] = sorted(ctx.inputs["src"])
        return _result(ctx, {"ranked": seen["keys"]}, [])

    ops = ("fake_get", "fake_filter", "fake_rank")
    monkeypatch.setattr(compiler, "ADAPTERS", dict(zip(ops, map(Fake, (get_, filter_, rank_)))))
    monkeypatch.setattr(compiler, "OPERATION_TABLE",
                        {op: SimpleNamespace(requires_all=[], requires_any=[]) for op in ops})
    monkeypatch.setattr(compiler, "OUTPUT_MODELS", {op: AnyPayload for op in ops})

    trace = compiler.run_plan(_plan())

    assert [r.status for r in trace] == ["success"] * 3
    assert seen["keys"] == ["A", "B"]  # 舊做法會讀到 A、B、C