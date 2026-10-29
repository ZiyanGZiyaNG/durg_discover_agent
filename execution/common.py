"""Helpers shared by the adapter modules."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from execution.contracts import StepContext, StepResult
from query_opentargets import OpenTargetsError


def _resolved_for_step(
    ctx: StepContext, entity_type: str, capability: str
) -> list[dict]:
    on_board = ctx.board.get(capability, {})
    named = [
        ref
        for ref, entity in ctx.entities.items()
        if entity.get("entity_type") == entity_type
    ]
    if not named:
        return [
            entity
            for entity in on_board.values()
            if entity.get("entity_type", entity_type) == entity_type
        ]
    return [on_board[ref] for ref in named if ref in on_board]


def _targets_for_step(ctx: StepContext) -> list[dict]:
    explicit = _resolved_for_step(ctx, "target", "resolved_target")
    if explicit:
        return explicit
    candidates = ctx.board.get("target_candidates") or {}
    return [
        {
            "id": target["target_id"],
            "symbol": target["target_symbol"],
            "name": target["target_name"],
        }
        for target in candidates.get("targets", [])
    ]


def _collect_per_target(
    targets: list[dict], fetch: Callable[[dict], dict]
) -> tuple[dict[str, dict], dict[str, str]]:
    by_target: dict[str, dict] = {}
    failed_targets: dict[str, str] = {}
    for target in targets:
        try:
            by_target[target["id"]] = fetch(target)
        except OpenTargetsError as exc:
            failed_targets[target["id"]] = str(exc)
    return by_target, failed_targets


def error_result(
    ctx: StepContext, source: str, start: float, message: str
) -> StepResult:
    return StepResult(
        step_id=ctx.step_id,
        operation=ctx.operation,
        status="error",
        source=source,
        error=message,
        elapsed_seconds=time.monotonic() - start,
    )


def association_rows(board: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the disease-target association data on the board into one list
    of rows (one per disease-target pair), whichever step produced it."""

    assoc = board.get("disease_target_association") or {}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(row: dict[str, Any]) -> None:
        key = (row.get("disease_id") or "", row.get("target_id") or "")
        if not key[1] or key in seen:
            return
        seen.add(key)
        rows.append(row)

    for pair in assoc.get("pairs") or []:
        add(dict(pair))
    if assoc.get("target_id") and assoc.get("disease_id") and "datasource_scores" in assoc:
        add(
            {
                key: assoc.get(key)
                for key in (
                    "disease_id",
                    "disease_name",
                    "target_id",
                    "target_symbol",
                    "target_name",
                    "overall_association_score",
                    "datasource_scores",
                )
            }
        )
    by_disease = assoc.get("by_disease")
    if by_disease:
        sources = [(found.get("disease_id"), found) for found in by_disease.values()]
    elif assoc.get("targets"):
        sources = [(assoc.get("disease_id"), assoc)]
    else:
        sources = []
    for disease_id, found in sources:
        for target in found.get("targets") or []:
            add(
                {
                    "disease_id": disease_id,
                    "disease_name": found.get("disease_name"),
                    "target_id": target.get("target_id"),
                    "target_symbol": target.get("target_symbol"),
                    "target_name": target.get("target_name"),
                    "rank": target.get("rank"),
                    "overall_association_score": target.get("overall_association_score"),
                    "datasource_scores": target.get("datasource_scores") or [],
                }
            )
    return rows
