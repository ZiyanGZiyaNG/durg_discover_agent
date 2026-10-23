from __future__ import annotations
import time
from collections.abc import Callable
from execution.contracts import Adapter, StepContext, StepResult
from query_opentargets import (
    OpenTargetsError,
    discover_disease_targets,
    get_disease_target_evidence_by_ids,
    get_target_drug_evidence_by_ids,
    get_target_tractability_by_id,
    resolve_entity,
)
from execution.analysis import RULE_NOTE, RULE_VERSION, rank_rows, stratify_targets


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
    named = [
        ref
        for ref, entity in ctx.entities.items()
        if entity.get("entity_type") == entity_type
    ]
    if not named:
        return list(on_board.values())
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


class RetrieveTargetDrugsAdapter:
    source = "open_targets"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        targets = _targets_for_step(ctx)
        diseases = _resolved_for_step(ctx, "disease", "resolved_disease")

        if not targets:
            return StepResult(
                step_id=ctx.step_id,
                operation=ctx.operation,
                status="error",
                source=self.source,
                error="retrieve_target_drugs needs at least one resolved target",
                elapsed_seconds=time.monotonic() - start,
            )

        efo_id = diseases[0]["id"] if len(diseases) == 1 else ""
        by_target, failed_targets = _collect_per_target(
            targets,
            lambda target: get_target_drug_evidence_by_ids(target["id"], efo_id),
        )

        errors = "; ".join(f"{tid}: {msg}" for tid, msg in failed_targets.items())
        if not by_target:
            return StepResult(
                step_id=ctx.step_id,
                operation=ctx.operation,
                status="error",
                source=self.source,
                error=errors,
                elapsed_seconds=time.monotonic() - start,
            )

        total = sum(e["total_drug_count"] for e in by_target.values())
        fetched = sum(e["fetched_drug_count"] for e in by_target.values())
        truncated = fetched < total

        if failed_targets:
            status = "partial_error"
        elif total == 0:
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
            payload={"by_target": by_target, "failed_targets": failed_targets},
            provides=["target_drug_evidence"],
            total_count=total,
            fetched_count=fetched,
            truncated=truncated,
            error=errors or None,
            elapsed_seconds=time.monotonic() - start,
        )


class RetrieveTractabilityAdapter:
    source = "open_targets"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        targets = _targets_for_step(ctx)

        if not targets:
            return StepResult(
                step_id=ctx.step_id,
                operation=ctx.operation,
                status="error",
                source=self.source,
                error="retrieve_tractability needs at least one resolved target",
                elapsed_seconds=time.monotonic() - start,
            )
        by_target, failed_targets = _collect_per_target(
            targets, lambda target: get_target_tractability_by_id(target["id"])
        )

        errors = "; ".join(f"{tid}: {msg}" for tid, msg in failed_targets.items())
        if not by_target:
            return StepResult(
                step_id=ctx.step_id,
                operation=ctx.operation,
                status="error",
                source=self.source,
                error=errors,
                elapsed_seconds=time.monotonic() - start,
            )
        total = sum(e["assessment_count"] for e in by_target.values())
        if failed_targets:
            status = "partial_error"
        elif total == 0:
            status = "no_records"
        else:
            status = "success"

        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status=status,
            source=self.source,
            payload={"by_target": by_target, "failed_targets": failed_targets},
            provides=["tractability_evidence"],
            total_count=total,
            fetched_count=total,
            error=errors or None,
            elapsed_seconds=time.monotonic() - start,
        )


class RankTargetsAdapter:
    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        evidence = ctx.board.get("stratified_target_evidence") or ctx.board.get(
            "target_candidates"
        )
        problem = None
        if ctx.ranking is None:
            problem = "rank_targets needs a ranking spec"
        elif not evidence:
            problem = (
                "rank_targets needs stratified_target_evidence or target_candidates"
            )
        ranked: list[dict] = []
        if problem is None:
            try:
                ranked = rank_rows(
                    evidence.get("targets", []),
                    ctx.ranking["primary_metric"],
                    ctx.ranking["direction"],
                    ctx.ranking.get("secondary_metrics"),
                )
            except ValueError as exc:
                problem = str(exc)
        if problem is not None:
            return StepResult(
                step_id=ctx.step_id,
                operation=ctx.operation,
                status="error",
                source=self.source,
                error=problem,
                elapsed_seconds=time.monotonic() - start,
            )
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success" if ranked else "no_records",
            source=self.source,
            payload={**evidence, "targets": ranked, "ranking_spec": ctx.ranking},
            elapsed_seconds=time.monotonic() - start,
        )


class StratifyTargetEvidenceAdapter:
    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        association = ctx.board.get("disease_target_association") or {}
        drug_evidence = ctx.board.get("target_drug_evidence") or {}
        tractability_evidence = ctx.board.get("tractability_evidence") or {}
        problem = None
        if "target_translation_layer" not in (ctx.aggregation or {}):
            problem = "stratify needs aggregation.target_translation_layer"
        elif "targets" not in association:
            problem = "stratify needs a target list from discover_targets"

        if problem is not None:
            return StepResult(
                step_id=ctx.step_id,
                operation=ctx.operation,
                status="error",
                source=self.source,
                error=problem,
                elapsed_seconds=time.monotonic() - start,
            )
        stratified = stratify_targets(
            association["targets"], drug_evidence, tractability_evidence
        )
        failed_targets = {
            "target_drug_evidence": drug_evidence.get("failed_targets", {}),
            "tractability_evidence": tractability_evidence.get("failed_targets", {}),
        }

        if not stratified:
            status = "no_records"
        elif any(failed_targets.values()):
            status = "partial_error"
        else:
            status = "success"

        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status=status,
            source=self.source,
            payload={
                "rule_version": RULE_VERSION,
                "targets": stratified,
                "failed_targets": failed_targets,
                "limitations": [*association.get("limitations", []), RULE_NOTE],
            },
            provides=["stratified_target_evidence"],
            elapsed_seconds=time.monotonic() - start,
        )


ADAPTERS: dict[str, Adapter] = {
    "resolve_disease": ResolveEntityAdapter("disease", "resolved_disease"),
    "resolve_target": ResolveEntityAdapter("target", "resolved_target"),
    "retrieve_exact_pair": RetrieveExactPairAdapter(),
    "discover_targets": DiscoverTargetsAdapter(),
    "retrieve_target_drugs": RetrieveTargetDrugsAdapter(),
    "retrieve_tractability": RetrieveTractabilityAdapter(),
    "rank_targets": RankTargetsAdapter(),
    "stratify_target_evidence_layers": StratifyTargetEvidenceAdapter(),
    "discover_targets_per_disease": DiscoverTargetsAdapter(),
}
