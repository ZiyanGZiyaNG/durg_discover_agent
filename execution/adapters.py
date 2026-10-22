from __future__ import annotations
import time
from execution.contracts import Adapter, StepContext, StepResult
from query_opentargets import OpenTargetsError, resolve_entity


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

ADAPTERS: dict[str, Adapter] = {
    "resolve_disease": ResolveEntityAdapter("disease", "resolved_disease"),
    "resolve_target": ResolveEntityAdapter("target", "resolved_target"),
}