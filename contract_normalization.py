"""Program-authority normalization of fields the operation table already decides.

Two defect classes were repeatedly sent back to the model although the answer
is fixed by OPERATION_TABLE, not by scientific judgement:

* a required safeguard is missing from the plan;
* an analysis step repeats data sources that its upstream retrieval steps
  already declare.

Both are corrected here and recorded in the attempt diagnostics. Metrics,
filters, thresholds, operation choice and any biomedical content stay with the
model and the human reviewer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from planner_acceptance import OPERATION_ALIAS_TO_CANONICAL, OPERATION_TABLE


def _canonical(step: dict[str, Any]) -> str | None:
    if step.get("operation_status") == "proposed":
        return None
    operation = step.get("operation")
    return OPERATION_ALIAS_TO_CANONICAL.get(operation) if isinstance(operation, str) else None


def _add_required_safeguards(payload: dict[str, Any], steps: list[dict[str, Any]]) -> list[str]:
    declared = set(payload.get("global_safeguards") or [])
    for step in steps:
        declared.update(step.get("safeguards") or [])
    notes: list[str] = []
    global_safeguards = list(payload.get("global_safeguards") or [])
    for step in steps:
        operation = _canonical(step)
        if operation is None:
            continue
        for safeguard in OPERATION_TABLE[operation].required_safeguards:
            if safeguard not in declared:
                declared.add(safeguard)
                global_safeguards.append(safeguard)
                notes.append(
                    f"added required safeguard {safeguard!r} to global_safeguards "
                    f"(required by {operation} in {step.get('step_id')})"
                )
    if notes:
        payload["global_safeguards"] = global_safeguards
    return notes


def _upstream_steps(step: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    stack = list(step.get("depends_on") or [])
    while stack:
        step_id = stack.pop()
        if step_id in seen or step_id not in by_id:
            continue
        seen.add(step_id)
        stack.extend(by_id[step_id].get("depends_on") or [])
    return [by_id[item] for item in seen]


def _drop_duplicated_analysis_sources(steps: list[dict[str, Any]]) -> list[str]:
    by_id = {step["step_id"]: step for step in steps if isinstance(step.get("step_id"), str)}
    notes: list[str] = []
    for step in steps:
        operation = _canonical(step)
        if operation is None or OPERATION_TABLE[operation].operation_kind != "analysis":
            continue
        sources = list(step.get("sources") or [])
        if not sources:
            continue
        upstream_sources = {
            source
            for upstream in _upstream_steps(step, by_id)
            if (_canonical(upstream) is None
                or OPERATION_TABLE[_canonical(upstream)].operation_kind != "analysis")
            for source in upstream.get("sources") or []
        }
        removable = [source for source in sources if source in upstream_sources]
        if removable:
            step["sources"] = [source for source in sources if source not in removable]
            notes.append(
                f"{step.get('step_id')}: removed sources {removable} from analysis step "
                f"{operation}; upstream retrieval steps already declare them"
            )
    return notes


# Runtime metadata is program authority. The prompt already tells the model not to
# write these; a model-invented value (for example a malformed UUID) must never
# cost a whole generation attempt. Valid values are dropped silently; only an
# invalid one is reported.
def _is_valid_runtime_value(key: str, value: Any) -> bool:
    try:
        if key == "manifest_id":
            UUID(str(value))
        elif key == "generated_at":
            datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        elif key == "schema_version":
            return value == "2.0"
        elif key == "created_before_retrieval":
            return value is True
    except ValueError:
        return False
    return True


def drop_program_owned_fields(payload: dict[str, Any]) -> list[str]:
    """Remove model-supplied runtime metadata so schema defaults apply."""

    notes: list[str] = []
    for key in ("schema_version", "manifest_id", "generated_at", "created_before_retrieval"):
        if key not in payload:
            continue
        value = payload.pop(key)
        if not _is_valid_runtime_value(key, value):
            notes.append(
                f"removed invalid model-supplied runtime field {key!r}; the program assigns it"
            )
    return notes


# A JSON null where the schema has an empty default (``"filters": null``) is
# "nothing here", not a scientific claim. It once cost a whole 184-second attempt.
_STEP_EMPTY_DEFAULTS: dict[str, Any] = {
    "depends_on": list,
    "entity_refs": list,
    "sources": list,
    "safeguards": list,
    "filters": dict,
}


def fill_null_step_defaults(steps: Any, *, omit: bool = False) -> list[str]:
    """Replace null step fields with their empty defaults, in place.

    With ``omit=True`` (partial patch updates) the key is removed instead, so
    an existing step keeps its previous value.
    """

    notes: list[str] = []
    if not isinstance(steps, list):
        return notes
    for step in steps:
        if not isinstance(step, dict):
            continue
        for key, factory in _STEP_EMPTY_DEFAULTS.items():
            if key in step and step[key] is None:
                if omit:
                    step.pop(key)
                else:
                    step[key] = factory()
                notes.append(
                    f"{step.get('step_id')}: null {key!r} treated as "
                    + ("omitted" if omit else "empty")
                )
        if "operation_status" in step and step["operation_status"] is None:
            step.pop("operation_status")
    return notes


def normalize_contract_fields(payload: dict[str, Any]) -> list[str]:
    """Normalize ``payload`` in place and return one note per correction."""

    steps = payload.get("plan_steps")
    if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
        return []
    return _add_required_safeguards(payload, steps) + _drop_duplicated_analysis_sources(steps)
