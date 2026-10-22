from __future__ import annotations

import time

from execution.contracts import Adapter, StepContext, StepResult
from query_opentargets import (
    OpenTargetsError,
    discover_disease_targets,
    get_disease_target_evidence_by_ids,
    resolve_entity,
)


class ResolveEntityAdapter:
    source = "open_targets"

    def __init__(self, entity_type: str, capability: str) -> None:
        self.entity_type = entity_type
        self.capability = capability

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()

        mentions = {
            ref: entity
            for ref, entity in ctx.entities.items()
            if entity.get("entity_type") == self.entity_type
        }
        if not mentions:
            return StepResult(
                step_id=ctx.step_id,
                operation=ctx.operation,
                status="error",
                source=self.source,
                error=f"no {self.entity_type} entity in this step",
                elapsed_seconds=time.monotonic() - start,
            )
        resolved: dict[str, dict] = {}
        try:
            for ref, entity in mentions.items():
                resolved[ref] = resolve_entity(
                    entity["mention"], self.entity_type, entity.get("entity_id")
                )
        except OpenTargetsError as exc:
            return StepResult(
                step_id=ctx.step_id,
                operation=ctx.operation,
                status="error",
                source=self.source,
                error=str(exc),
                elapsed_seconds=time.monotonic() - start,
            )
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success",
            source=self.source,
            payload=resolved,
            provides=[self.capability],
            total_count=len(resolved),
            fetched_count=len(resolved),
            elapsed_seconds=time.monotonic() - start,
        )


def _resolved_for_step(
    ctx: StepContext, entity_type: str, capability: str
) -> list[dict]:
    on_board = ctx.board.get(capability, {})
    return [
        on_board[ref]
        for ref, entity in ctx.entities.items()
        if entity.get("entity_type") == entity_type and ref in on_board
    ]


class RetrieveExactPairAdapter:
    source = "open_targets"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        diseases = _resolved_for_step(ctx, "disease", "resolved_disease")
        targets = _resolved_for_step(ctx, "target", "resolved_target")

        if len(diseases) != 1 or len(targets) != 1:
            return StepResult(
                step_id=ctx.step_id,
                operation=ctx.operation,
                status="error",
                source=self.source,
                error=(
                    "exact pair needs exactly one resolved disease "
                    "and one resolved target"
                ),
                elapsed_seconds=time.monotonic() - start,
            )
        disease, target = diseases[0], targets[0]
        try:
            evidence = get_disease_target_evidence_by_ids(
                disease["id"],
                target["id"],
                disease_query=disease.get("name", ""),
                target_query=target.get("symbol") or target.get("name", ""),
            )
        except OpenTargetsError as exc:
            return StepResult(
                step_id=ctx.step_id,
                operation=ctx.operation,
                status="error",
                source=self.source,
                error=str(exc),
                elapsed_seconds=time.monotonic() - start,
            )

        records = evidence["datasource_evidence"]
        total = sum(r["total_evidence_count"] for r in records) if records else None
        fetched = sum(r["fetched_evidence_count"] for r in records) if records else None
        truncated = any(r["truncated"] for r in records)

        if evidence["overall_association_score"] is None:
            status = "no_records"
        elif truncated:
            status = "partial_truncated"
        else:
            status = "success"

        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status=status,
            source=self.source,
            payload=evidence,
            provides=["disease_target_association"],
            total_count=total,
            fetched_count=fetched,
            truncated=truncated,
            elapsed_seconds=time.monotonic() - start,
        )


class DiscoverTargetsAdapter:
    source = "open_targets"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        diseases = _resolved_for_step(ctx, "disease", "resolved_disease")
        if len(diseases) != 1:
            return StepResult(
                step_id=ctx.step_id,
                operation=ctx.operation,
                status="error",
                source=self.source,
                error="discover_targets needs exactly one resolved disease",
                elapsed_seconds=time.monotonic() - start,
            )
        disease = diseases[0]
        top_n = ctx.filters.get("top_n", 10)
        try:
            found = discover_disease_targets(
                disease.get("name", ""), top_n=top_n, disease_id=disease["id"]
            )
        except (OpenTargetsError, ValueError) as exc:
            return StepResult(
                step_id=ctx.step_id,
                operation=ctx.operation,
                status="error",
                source=self.source,
                error=str(exc),
                elapsed_seconds=time.monotonic() - start,
            )

        returned = found["returned_target_count"]
        total = found["total_associated_target_count"]
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success" if returned else "no_records",
            source=self.source,
            payload=found,
            provides=["disease_target_association", "target_candidates"],
            total_count=total,
            fetched_count=returned,
            truncated=returned < total,
            elapsed_seconds=time.monotonic() - start,
        )


ADAPTERS: dict[str, Adapter] = {
    "resolve_disease": ResolveEntityAdapter("disease", "resolved_disease"),
    "resolve_target": ResolveEntityAdapter("target", "resolved_target"),
    "retrieve_exact_pair": RetrieveExactPairAdapter(),
    "discover_targets": DiscoverTargetsAdapter(),
}
