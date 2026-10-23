from __future__ import annotations

from execution.adapters import ADAPTERS
from execution.contracts import StepContext, StepResult
from planner_acceptance import OPERATION_TABLE
from schemas import QueryManifestV2

FAILED = {"error", "not_executed"}


class PlanRefusedError(Exception):
    """The plan cannot be run; raised before any retrieval starts."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


def check_plan(plan: QueryManifestV2) -> list[str]:
    problems = []
    for step in plan.plan_steps:
        if step.operation_status != "approved":
            problems.append(f"{step.step_id}: operation not approved")
        elif step.operation not in OPERATION_TABLE:
            problems.append(f"{step.step_id}: unknown operation {step.operation!r}")
        elif step.operation not in ADAPTERS:
            problems.append(f"{step.step_id}: no adapter for {step.operation!r}")
    return problems


def _missing(operation: str, board: dict) -> list[str]:
    entry = OPERATION_TABLE[operation]
    missing = [cap for cap in entry.requires_all if cap not in board]
    if entry.requires_any and not any(cap in board for cap in entry.requires_any):
        missing.append("one of " + "/".join(entry.requires_any))
    return missing


def _post_to_board(board: dict, result: StepResult) -> None:
    for capability in result.provides:
        old = board.get(capability)
        if isinstance(old, dict) and isinstance(result.payload, dict):
            board[capability] = {**old, **result.payload}
        else:
            board[capability] = result.payload


def run_plan(plan: QueryManifestV2) -> list[StepResult]:
    problems = check_plan(plan)
    if problems:
        raise PlanRefusedError(problems)

    entities = {entity.entity_ref: entity.model_dump() for entity in plan.entities}
    board: dict = {}
    done: dict[str, StepResult] = {}
    trace: list[StepResult] = []

    for step in plan.plan_steps:
        failed_deps = [d for d in step.depends_on if done[d].status in FAILED]
        missing = _missing(step.operation, board)

        if failed_deps or missing:
            reason = (
                f"upstream failed: {failed_deps}"
                if failed_deps
                else f"missing: {missing}"
            )
            result = StepResult(
                step_id=step.step_id,
                operation=step.operation,
                status="not_executed",
                source="-",
                error=reason,
            )
        else:
            ctx = StepContext(
                step_id=step.step_id,
                operation=step.operation,
                entities={ref: entities[ref] for ref in step.entity_refs},
                board=board,
                filters=step.filters,
            )
            result = ADAPTERS[step.operation].run(ctx)
            _post_to_board(board, result)

        done[step.step_id] = result
        trace.append(result)

    return trace
