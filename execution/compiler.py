from __future__ import annotations

from execution.adapters import ADAPTERS
from execution.contracts import StepContext, StepResult
from execution.records import OUTPUT_MODELS, AssociationRow, StratifiedRow
from execution.board import Board, MissingInput
from planner_acceptance import OPERATION_TABLE
from pydantic import ValidationError
from schemas import QueryManifestV2
from execution.analysis import normalize_filters
from execution.contracts import StepContext, StepResult, ToolCall


FAILED = {"error", "not_executed"}
HAS_PAYLOAD = {
    "success",
    "partial_truncated",
    "partial_timeout",
    "partial_error",
    "no_records",
}


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
        else:
            problems.extend(_check_step_contract(plan, step))
    return problems


def _missing(operation: str, board: dict) -> list[str]:
    entry = OPERATION_TABLE[operation]
    missing = [cap for cap in entry.requires_all if cap not in board]
    if entry.requires_any and not any(cap in board for cap in entry.requires_any):
        missing.append("one of " + "/".join(entry.requires_any))
    return missing

REQUDRED_AGGREGATION_KEYS = {
    "stratify_target_evidence_layers": ["target_translation_layer"],
}
def _ancestor_operation(plan: QueryManifestV2, step) -> set[str]:
    by_id = {item.step_id: item for item in plan.plan_steps}
    seen: set[str] = set()
    stack = list(step.depends_on)
    while stack:
        step_id = stack.pop()
        if step_id in seen or step_id not in by_id:
            continue
        seen.add(step_id)
        stack.extend(by_id[step_id].depends_on)
    return {by_id[step_id].operation for step_id in seen}

DRUG_FIELDS = {"known_drug_count", "exact_disease_drug_count"}
TRACTABILITY_FIELDS = {"tractable_bucket_count", "tractable_modalities"}

def _filterable_fields(upstream: set[str]) -> set[str] | None:
    if "stratify_target_evidence_layers" in upstream:
        return set(StratifiedRow.model_fields)
    if upstream & {"discover_targets", "discover_targets_per_disease"}:
        fields = set(AssociationRow.model_fields)
        if "retrieve_target_drugs" in upstream:
            fields |= DRUG_FIELDS
        if "retrieve_tractability" in upstream:
            fields |= TRACTABILITY_FIELDS 
        return fields
    return None



def _check_step_contract(plan: QueryManifestV2, step) -> list[str]:
    problems = []
    missing_keys = [
        key
        for key in REQUDRED_AGGREGATION_KEYS.get(step.operation, [])
        if key not in (step.aggregation or {})
    ]
    if missing_keys:
        problems.append(
            f"{step.step_id}: {step.operation} needs aggregation keys {missing_keys}"
        )


    if step.operation == "filter_targets":
        if not step.filters:
            problems.append(f"{step.step_id}: filter_targets needs filters")
            return problems
        try:
            fields = {field for field, _, _ in normalize_filters(step.filters)}
        except ValueError as exc:
            problems.append(f"{step.step_id}: {exc}")
            return problems
        available = _filterable_fields(_ancestor_operation(plan, step))
        if available is not None:
            unknown = sorted(fields - available)
            if unknown:
                problems.append(
                    f"{step.step_id}: filter field(s) {unknown} are not available "
                    f"upstream; available: {sorted(available)}"
                )

    if step.operation == "rank_targets":
        if step.ranking is None:
            problems.append(f"{step.step_id}: rank_targets needs a ranking spec")
            return problems
        upstream = _ancestor_operation(plan, step)
        if "stratify_target_evidence_layers" in upstream:
            row_model = StratifiedRow
        elif upstream & {"discover_targets", "discover_targets_per_disease"}:
            row_model = AssociationRow
        else:
            return problems

        available = set(row_model.model_fields)
        metrics = [step.ranking.primary_metric, *step.ranking.secondary_metrics]
        unknown = [metric for metric in metrics if metric not in available]
        if unknown:
            problems.append(
                f"{step.step_id}: ranking metric(s) {unknown} are not produced"
                f"upstream; availbale: {sorted(available)}"
            )
    return problems



def _check_output(result: StepResult) -> StepResult:
    """An adapter whose payload breaks its operation's model is an error step."""
    if result.status not in HAS_PAYLOAD:
        return result
    model = OUTPUT_MODELS[result.operation]
    try:
        model.model_validate(result.payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        return result.model_copy(
            update={
                "status": "error",
                "payload": None,
                "provides": [],
                "error": (
                    f"output does not match {model.__name__}: {where}: {first['msg']}"
                ),
            }
        )
    return result


def _post_to_board(board: dict, result: StepResult) -> None:
    for capability in result.provides:
        old = board.get(capability)
        if isinstance(old, dict) and isinstance(result.payload, dict):
            board[capability] = {**old, **result.payload}
        else:
            board[capability] = result.payload




def _check_dataflow(plan: QueryManifestV2) -> list[str]:
    """Static version of the runtime check: every capability a step needs must
    be provided by a step listed before it."""
    problems = []
    available: set[str] = set()
    for step in plan.plan_steps:
        entry = OPERATION_TABLE.get(step.operation)
        if entry is None:
            continue
        missing = [cap for cap in entry.requires_all if cap not in available]
        if entry.requires_any and not any(cap in available for cap in entry.requires_any):
            missing.append("one of " + "/".join(entry.requires_any))
        if missing:
            problems.append(
                f"{step.step_id}: {step.operation} needs {missing}, "
                "which no earlier step provides"
            )
        available.update(entry.provides)
    return problems


def compile_plan(plan: QueryManifestV2) -> list[ToolCall]:
    """Turn an approved plan into an ordered list of tool calls; runs nothing."""
    problems = check_plan(plan) + _check_dataflow(plan)
    if problems:
        raise PlanRefusedError(problems)
    entities = {entity.entity_ref: entity.model_dump() for entity in plan.entities}
    calls = []
    for step in plan.plan_steps:
        entry = OPERATION_TABLE[step.operation]
        adapter = ADAPTERS[step.operation]
        calls.append(
            ToolCall(
                step_id=step.step_id,
                operation=step.operation,
                adapter=type(adapter).__name__,
                source=adapter.source,
                declared_sources=list(step.sources),
                entities={ref: entities[ref] for ref in step.entity_refs},
                filters=step.filters,
                ranking=step.ranking.model_dump() if step.ranking else None,
                aggregation=step.aggregation,
                inputs={
                    name: f"{ref.step_id}.{ref.output}"
                    for name, ref in step.inputs.items()
                },
                depends_on=list(step.depends_on),
                requires_all=list(entry.requires_all),
                requires_any=list(entry.requires_any),
                provides=list(entry.provides),
                output_model=OUTPUT_MODELS[step.operation].__name__,
            )
        )
    return calls


def _inherit_truncation(step, result: StepResult, done: dict[str, StepResult]) -> StepResult:
    """A local analysis run on cut-off upstream data cannot claim to be complete."""
    if result.status != "success" or result.source != "local_analysis":
        return result
    cut_off = [
        d
        for d in step.depends_on
        if done[d].truncated or done[d].status == "partial_truncated"
    ]
    if not cut_off:
        return result
    note = (
        f"Computed from truncated upstream data ({', '.join(cut_off)}); "
        "counts cover only the records that were fetched."
    )
    if isinstance(result.payload, dict):
        limits = result.payload.setdefault("limitations", [])
        if isinstance(limits, list):
            limits.append(note)
    return result.model_copy(update={"status": "partial_truncated", "truncated": True})


def run_plan(plan: QueryManifestV2) -> list[StepResult]:
    problems = check_plan(plan)
    if problems:
        raise PlanRefusedError(problems)

    step_board = Board()
    entities = {entity.entity_ref: entity.model_dump() for entity in plan.entities}
    board: dict = {}
    done: dict[str, StepResult] = {}
    trace: list[StepResult] = []

    for step in plan.plan_steps:
        failed_deps = [d for d in step.depends_on if done[d].status in FAILED]
        inputs: dict = {}
        missing_inputs: list[str] = []
        for name, ref in step.inputs.items():
            try:
                inputs[name] = step_board.get(ref)
            except MissingInput:
                missing_inputs.append(f"input {ref.step_id}.{ref.output}")
        missing = _missing(step.operation, board) + missing_inputs

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
                ranking=step.ranking.model_dump() if step.ranking else None,
                aggregation=step.aggregation,
                inputs=inputs,
            )
            result = _inherit_truncation(
                step, _check_output(ADAPTERS[step.operation].run(ctx)), done
            )
            _post_to_board(board, result)
            if result.status in HAS_PAYLOAD:
                for output in step.outputs:
                    step_board.put(step.step_id, output, result.payload)

        done[step.step_id] = result
        trace.append(result)

    return trace
