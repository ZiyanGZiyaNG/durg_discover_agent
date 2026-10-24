"""Deterministic dependency inference from the authoritative operation table.

The Planner model chooses operations; Python derives the data-flow edges those
operations already imply. For every step, ``requires_all`` / ``requires_any``
in OPERATION_TABLE name the capabilities it consumes and ``provides`` names the
capabilities each step produces. This module therefore:

* adds a missing ``depends_on`` edge only when the producer is unambiguous;
* removes a declared edge only when it runs against the table contract
  (the upstream step consumes what the downstream step provides, and not the
  reverse);
* reorders steps so every declared edge points to an earlier step.

It never guesses between several equally valid producers, never invents a
step, and never uses an acceptance rubric. Anything it cannot decide is
returned as ``unresolved`` and left to validation and the repair loop.
"""

from __future__ import annotations

from typing import Any

from planner_acceptance import OPERATION_ALIAS_TO_CANONICAL, OPERATION_TABLE


def _canonical_operation(step: dict[str, Any]) -> str | None:
    if step.get("operation_status") == "proposed":
        return None
    operation = step.get("operation")
    return OPERATION_ALIAS_TO_CANONICAL.get(operation) if isinstance(operation, str) else None


def _provides(operation: str | None) -> set[str]:
    return set(OPERATION_TABLE[operation].provides) if operation else set()


def _requires(operation: str | None) -> set[str]:
    if not operation:
        return set()
    contract = OPERATION_TABLE[operation]
    return set(contract.requires_all) | set(contract.requires_any)


def _ancestors(step_id: str, graph: dict[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    stack = list(graph[step_id])
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(graph.get(current, []))
    return seen


def _stable_order(steps: list[dict[str, Any]], known: set[str]) -> list[dict[str, Any]] | None:
    """Stable topological order over declared edges; None if a cycle exists."""
    pending = list(steps)
    ordered: list[dict[str, Any]] = []
    emitted: set[str] = set()
    while pending:
        ready = next(
            (
                step for step in pending
                if {d for d in step["depends_on"] if d in known} <= emitted
            ),
            None,
        )
        if ready is None:
            return None
        pending.remove(ready)
        ordered.append(ready)
        emitted.add(ready["step_id"])
    return ordered


def infer_step_dependencies(
    steps: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Return ``(steps, corrections, unresolved)`` without mutating the input."""

    work = [{**step, "depends_on": list(step.get("depends_on") or [])} for step in steps]
    ids = [step.get("step_id") for step in work]
    if not all(isinstance(item, str) and item for item in ids) or len(set(ids)) != len(ids):
        return work, [], []  # malformed or duplicate IDs: schema validation reports it
    by_id = {step["step_id"]: step for step in work}
    known = set(by_id)
    operations = {step_id: _canonical_operation(step) for step_id, step in by_id.items()}
    corrections: list[str] = []
    unresolved: list[str] = []

    def graph() -> dict[str, list[str]]:
        return {
            step_id: [d for d in step["depends_on"] if d in known]
            for step_id, step in by_id.items()
        }

    # 1. Edges that contradict the contract: B (upstream) consumes what A
    #    (downstream) provides, and A consumes nothing from B.
    for step in work:
        step_id = step["step_id"]
        for dependency in list(step["depends_on"]):
            if dependency not in known or dependency == step_id:
                continue
            a_op, b_op = operations[step_id], operations[dependency]
            if (_requires(b_op) & _provides(a_op)) and not (_requires(a_op) & _provides(b_op)):
                step["depends_on"].remove(dependency)
                corrections.append(
                    f"{step_id}: removed dependency {dependency!r}; the contract of "
                    f"{b_op} consumes {step_id}'s output, so the edge was reversed"
                )

    # 2. Missing edges implied by requires_all / requires_any.
    for step in work:
        step_id = step["step_id"]
        contract = OPERATION_TABLE.get(operations[step_id]) if operations[step_id] else None
        if contract is None:
            continue
        groups: list[tuple[str, list[str]]] = []
        current_graph = graph()
        upstream = _ancestors(step_id, current_graph)
        upstream_caps = {c for item in upstream for c in _provides(operations[item])}
        groups.extend((c, [c]) for c in contract.requires_all if c not in upstream_caps)
        if contract.requires_any and not (upstream_caps & set(contract.requires_any)):
            groups.append(("|".join(contract.requires_any), list(contract.requires_any)))
        for label, caps in groups:
            current_graph = graph()
            upstream = _ancestors(step_id, current_graph)
            if set(caps) & {c for item in upstream for c in _provides(operations[item])}:
                continue  # an edge added for an earlier group already covers it
            candidates = [
                other for other in by_id
                if other != step_id
                and other not in upstream
                and _provides(operations[other]) & set(caps)
                and step_id not in _ancestors(other, current_graph)
            ]
            if not candidates:
                unresolved.append(
                    f"{step_id}: no earlier-compatible step provides {label}; "
                    "a producing step must be added"
                )
                continue
            if len(candidates) > 1:
                entities = set(step.get("entity_refs") or [])
                overlapping = [
                    other for other in candidates
                    if entities & set(by_id[other].get("entity_refs") or [])
                ]
                if len(caps) == 1 and overlapping:
                    candidates = overlapping
                elif len(overlapping) == 1:
                    candidates = overlapping
                else:
                    unresolved.append(
                        f"{step_id}: several steps provide {label} "
                        f"({', '.join(candidates)}); not inferred"
                    )
                    continue
            for producer in candidates:
                step["depends_on"].append(producer)
                corrections.append(
                    f"{step_id}: added dependency {producer!r}; "
                    f"{operations[step_id]} requires {label}"
                )

    ordered = _stable_order(work, known)
    if ordered is None:
        unresolved.append("declared dependencies contain a cycle; step order unchanged")
        return work, corrections, unresolved
    if [s["step_id"] for s in ordered] != ids:
        corrections.append(
            "reordered steps so dependencies precede dependents: "
            + ", ".join(s["step_id"] for s in ordered)
        )
    return ordered, corrections, unresolved


def apply_to_payload(payload: dict[str, Any]) -> list[str]:
    """Rewrite ``payload['plan_steps']`` in place; return diagnostics notes."""

    steps = payload.get("plan_steps")
    if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
        return []
    ordered, corrections, unresolved = infer_step_dependencies(steps)
    payload["plan_steps"] = ordered
    return corrections + [f"not inferred: {item}" for item in unresolved]
