from __future__ import annotations
import time
from collections.abc import Callable
from execution.common import _collect_per_target, _resolved_for_step, _targets_for_step
from execution.contracts import Adapter, StepContext, StepResult
from execution.more_adapters import (
    CompareAssociationEvidenceAdapter,
    DiscoverTargetsPerDiseaseAdapter,
    IntersectTargetsAdapter,
    JoinDiseaseTargetDrugAdapter,
    RankCandidatesAdapter,
    RankDatasourceContributionsAdapter,
    ResolveEntitiesAdapter,
    RetrieveClinicalStageAdapter,
    RetrieveDatasourceScoresAdapter,
    RetrieveExactPairsAdapter,
    RetrieveMechanismsAdapter,
    RetrieveAssociatedDiseasesAdapter,
    FilterSupportedWithoutClinicalDrugAdapter,
    RetrieveGeneticEvidenceAdapter,
    FilterFirstInClassCandidatesAdapter,
    RankOpportunitiesAdapter,
    RetrieveGeneticAndSomaticEvidenceAdapter,
    RetrieveDatasourceEvidenceAdapter,
    RetrieveEvidenceByTypeAdapter,
    CompareEvidenceTypesAdapter,
    NormalizeComparisonDimensionsAdapter,
    CompareTargetsAdapter,
    RetrieveDiseaseDrugsAdapter,
    RetrieveTargetCancersAdapter,
    JoinCrossIndicationPathsAdapter,
    ClassifyKinaseTargetsAdapter,
    DefineCoverageDenominatorsAdapter,
    CalculateCoverageRatesAdapter,
    CompareDiseaseRatesAdapter,
    ResolveTargetVariantsAdapter,
    RetrieveSafetyEvidenceAdapter,
    RetrieveTrialTerminationsAdapter,
    RetrieveNormalTissueExpressionAdapter,
    IntegrateSafetyAndExpressionAdapter,
)
from query_opentargets import (
    OpenTargetsError,
    discover_disease_targets,
    get_disease_ancestors,
    get_disease_target_evidence_by_ids,
    get_drug_indications_by_id,
    get_drug_targets_by_id,
    get_target_drug_evidence_by_ids,
    get_target_tractability_by_id,
    resolve_entity,
    resolve_mention,
)
from execution.analysis import (
    RULE_NOTE,
    RULE_VERSION,
    classify_indication,
    filter_rows,
    rank_rows,
    stratify_targets,
)

from execution.more_adapters import RankDiseasesAdapter
from execution.common import (
    _collect_per_target,
    _resolved_for_step,
    _targets_for_step,
    error_result,
    variant_payload,
)

from execution.external_source_adapters import (
    RetrieveClinicalTrialsAdapter,
    RetrieveInterventionsAdapter,
    RetrieveNegativePublicationsAdapter,
    RetrieveTrialStatusAndStopReasonsAdapter,
)

from execution.trial_analysis_adapters import (
    AggregateCategoryDistributionAdapter,
    AggregateFailedProgramsAdapter,
    ClassifyMechanismCategoriesAdapter,
    ClassifyTerminationReasonsAdapter,
    FilterPhase2NotApprovedAdapter,
    RankMechanismCategoriesAdapter,
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
                resolved[ref] = resolve_mention(
                    entity["mention"],
                    self.entity_type,
                    entity.get("entity_id"),
                    resolver=resolve_entity,
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
            payload={
                "by_target": by_target,
                "failed_targets": failed_targets,
                **variant_payload(targets),
            },
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



class FilterTargetsAdapter:
    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        stratified = ctx.board.get("stratified_target_evidence")
        base = stratified or ctx.board.get("target_candidates")
        capability = "stratified_target_evidence" if stratified else "target_candidates"
        problem = None
        if not ctx.filters:
            problem = "filter_targets needs filters"
        elif not base:
            problem = "filter_targets needs stratified_target_evidence or target_candidates"
        rows: list[dict] = []
        kept_rows: list[dict] = []
        rejected: list[str] = []
        unevaluable: list[str] = []

        if problem is None:
            rows = base.get("targets", [])
            view = rows if stratified else [
                {k: v for k, v in row.items() if k != "target_translation_layer"}
                for row in stratify_targets(                    rows, 
                    ctx.board.get("target_drug_evidence") or {},
                    ctx.board.get("tractability_evidence") or {},
                )
            ]

            try:
                kept, rejected, unevaluable = filter_rows(view, ctx.filters)
            except ValueError as exc:
                problem = str(exc)
            else:
                kept_ids = set(kept)
                kept_rows = [row for row in rows if row["target_id"] in kept_ids]
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
            status="success" if kept_rows else "no_records",
            source=self.source,
            payload={
                **base,
                "targets": kept_rows,
                "filter_spec": ctx.filters,
                "excluded_target_ids": rejected,
                "unevaluable_target_ids": unevaluable,
            },
            provides=[capability],
            total_count=len(rows),
            fetched_count=len(kept_rows),
            elapsed_seconds=time.monotonic() - start,
        )


MAX_APPROVAL_DRUGS = 50


def _drugs_for_step(ctx: StepContext) -> list[dict]:
    """Resolved drug entities, else the drugs found for the targets in an earlier step."""
    explicit = _resolved_for_step(ctx, "drug", "resolved_drug")
    if explicit:
        return explicit
    by_target = (ctx.board.get("target_drug_evidence") or {}).get("by_target") or {}
    drugs: dict[str, dict] = {}
    for record in by_target.values():
        for drug in record.get("drugs") or []:
            drugs.setdefault(
                drug["drug_id"], {"id": drug["drug_id"], "name": drug.get("drug_name", "")}
            )
    return list(drugs.values())


class RetrieveApprovedIndicationsAdapter:
    source = "open_targets"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        all_drugs = _drugs_for_step(ctx)
        if not all_drugs:
            return StepResult(
                step_id=ctx.step_id,
                operation=ctx.operation,
                status="error",
                source=self.source,
                error="retrieve_approved_indications needs at least one resolved drug",
                elapsed_seconds=time.monotonic() - start,
            )
        drugs = all_drugs[:MAX_APPROVAL_DRUGS]
        not_queried = len(all_drugs) - len(drugs)
        by_drug, failed_drugs = _collect_per_target(
            drugs, lambda drug: get_drug_indications_by_id(drug["id"])
        )
        notes = [f"{did}: {msg}" for did, msg in failed_drugs.items()]
        if not_queried:
            notes.append(f"{not_queried} drug(s) not queried (limit {MAX_APPROVAL_DRUGS})")
        errors = "; ".join(notes)
        if not by_drug:
            return StepResult(
                step_id=ctx.step_id,
                operation=ctx.operation,
                status="error",
                source=self.source,
                error=errors,
                elapsed_seconds=time.monotonic() - start,
            )
        total = sum(e["total_indication_count"] for e in by_drug.values())
        fetched = sum(e["fetched_indication_count"] for e in by_drug.values())
        truncated = fetched < total or not_queried > 0
        if failed_drugs:
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
            payload={"by_drug": by_drug, "failed_drugs": failed_drugs},
            provides=["approval_evidence"],
            total_count=total,
            fetched_count=fetched,
            truncated=truncated,
            error=errors or None,
            elapsed_seconds=time.monotonic() - start,
        )


def _disease_for_check(ctx: StepContext) -> dict | None:
    resolved = _resolved_for_step(ctx, "disease", "resolved_disease")
    if len(resolved) == 1:
        return resolved[0]
    named = [e for e in ctx.entities.values() if e.get("entity_type") == "disease"]
    if resolved or len(named) != 1:
        return None
    entity = named[0]
    return resolve_mention(
        entity["mention"], "disease", entity.get("entity_id"), resolver=resolve_entity
    )


class CheckIndicationAdapter:
    source = "open_targets"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()

        def fail(message: str) -> StepResult:
            return StepResult(
                step_id=ctx.step_id,
                operation=ctx.operation,
                status="error",
                source=self.source,
                error=message,
                elapsed_seconds=time.monotonic() - start,
            )

        evidence = ctx.board.get("approval_evidence") or {}
        by_drug = evidence.get("by_drug") or {}
        if not by_drug:
            return fail("check_indication needs approval_evidence for at least one drug")
        try:
            disease = _disease_for_check(ctx)
            if disease is None:
                return fail("check_indication needs exactly one disease")
            ancestors = get_disease_ancestors(disease["id"])
        except OpenTargetsError as exc:
            return fail(str(exc))

        checks = []
        for drug_id, entry in by_drug.items():
            verdict = classify_indication(
                entry["indications"], disease["id"], ancestors["ancestor_ids"]
            )
            checks.append(
                {
                    "drug_id": drug_id,
                    "drug_name": entry["drug_name"],
                    "disease_id": disease["id"],
                    "disease_name": disease.get("name", ""),
                    **verdict,
                    "indications_truncated": (
                        entry["fetched_indication_count"]
                        < entry["total_indication_count"]
                    ),
                }
            )
        unchecked = evidence.get("failed_drugs") or {}
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="partial_error" if unchecked else "success",
            source=self.source,
            payload={
                "disease_id": disease["id"],
                "disease_name": disease.get("name", ""),
                "checks": checks,
                "unchecked_drugs": unchecked,
                "limitations": [
                    "Stages are Open Targets database records, not an official approval conclusion.",
                    "approved_on_broader_term means approval is recorded on a broader disease term; it is not approval for the specific disease.",
                    "not_found_in_source means unconfirmed in this source, not 'not approved'.",
                ],
            },
            provides=["indication_check"],
            total_count=len(by_drug) + len(unchecked),
            fetched_count=len(checks),
            error="; ".join(f"{d}: {m}" for d, m in unchecked.items()) or None,
            elapsed_seconds=time.monotonic() - start,
        )


class CompareDiseaseRankAdapter:
    """Where the named disease sits in the ranking an earlier step produced."""

    source = "open_targets"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        ranked = (ctx.board.get("disease_ranking") or {}).get("diseases") or []
        if not ranked:
            return error_result(
                ctx,
                self.source,
                start,
                "compare_disease_rank needs disease_ranking from rank_diseases",
            )
        try:
            disease = _disease_for_check(ctx)
        except OpenTargetsError as exc:
            return error_result(ctx, self.source, start, str(exc))
        if disease is None:
            return error_result(
                ctx,
                self.source,
                start,
                "compare_disease_rank needs exactly one disease in entity_refs",
            )
        match = next(
            (row for row in ranked if row.get("disease_id") == disease["id"]), None
        )
        top = ranked[0]
        limitations = [
            f"The position is within the {len(ranked)} diseases fetched, "
            "not among all associated diseases.",
            "Only an exact disease ID match counts; related or broader disease "
            "terms are not merged.",
        ]
        if match is None:
            limitations.append(
                "The disease is not in the fetched ranking; this is not evidence "
                "of no association."
            )
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success",
            source=self.source,
            payload={
                "disease_id": disease["id"],
                "disease_name": disease.get("name", ""),
                "found_in_ranking": match is not None,
                "ranked_position": match["ranked_position"] if match else None,
                "overall_association_score": (
                    match.get("overall_association_score") if match else None
                ),
                "ranked_disease_count": len(ranked),
                "top_disease_id": top.get("disease_id"),
                "top_disease_name": top.get("disease_name"),
                "is_top": match is not None and match["ranked_position"] == 1,
                "limitations": limitations,
            },
            provides=["disease_rank_comparison"],
            elapsed_seconds=time.monotonic() - start,
        )

class RetrieveDrugTargetsAdapter:
    source = "open_targets"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        drugs = _resolved_for_step(ctx, "drug", "resolved_drug")
        if not drugs:
            return StepResult(
                step_id=ctx.step_id,
                operation=ctx.operation,
                status="error",
                source=self.source,
                error="retrieve_drug_targets needs at least one resolved drug",
                elapsed_seconds=time.monotonic() - start,
            )
        by_drug, failed_drugs = _collect_per_target(
            drugs, lambda drug: get_drug_targets_by_id(drug["id"])
        )
        errors = "; ".join(f"{did}: {msg}" for did, msg in failed_drugs.items())
        if not by_drug:
            return StepResult(
                step_id=ctx.step_id,
                operation=ctx.operation,
                status="error",
                source=self.source,
                error=errors,
                elapsed_seconds=time.monotonic() - start,
            )
        total = sum(len(entry["targets"]) for entry in by_drug.values())
        if failed_drugs:
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
            payload={"by_drug": by_drug, "failed_drugs": failed_drugs},
            provides=["target_drug_evidence"],
            total_count=total,
            fetched_count=total,
            error=errors or None,
            elapsed_seconds=time.monotonic() - start,
        )


ADAPTERS: dict[str, Adapter] = {
    "resolve_disease": ResolveEntityAdapter("disease", "resolved_disease"),
    "resolve_diseases": ResolveEntityAdapter("disease", "resolved_disease"),
    "resolve_targets": ResolveEntityAdapter("target", "resolved_target"),
    "resolve_target": ResolveEntityAdapter("target", "resolved_target"),
    "retrieve_exact_pair": RetrieveExactPairAdapter(),
    "discover_targets": DiscoverTargetsAdapter(),
    "retrieve_target_drugs": RetrieveTargetDrugsAdapter(),
    "retrieve_tractability": RetrieveTractabilityAdapter(),
    "rank_targets": RankTargetsAdapter(),
    "stratify_target_evidence_layers": StratifyTargetEvidenceAdapter(),
    "discover_targets_per_disease": DiscoverTargetsPerDiseaseAdapter(),
    "filter_targets": FilterTargetsAdapter(),
    "retrieve_approved_indications": RetrieveApprovedIndicationsAdapter(),
    "check_indication": CheckIndicationAdapter(),
    "resolve_drug": ResolveEntityAdapter("drug", "resolved_drug"),
    "retrieve_drug_targets": RetrieveDrugTargetsAdapter(),
    "resolve_disease_target": ResolveEntitiesAdapter(
        {"disease": "resolved_disease", "target": "resolved_target"}
    ),
    "resolve_disease_targets": ResolveEntitiesAdapter(
        {"disease": "resolved_disease", "target": "resolved_target"}
    ),
    "resolve_disease_and_drugs": ResolveEntitiesAdapter(
        {"disease": "resolved_disease", "drug": "resolved_drug"}
    ),
    "retrieve_exact_pairs": RetrieveExactPairsAdapter(),
    "retrieve_datasource_scores": RetrieveDatasourceScoresAdapter(),
    "rank_datasource_contributions": RankDatasourceContributionsAdapter(),
    "intersect_targets": IntersectTargetsAdapter(),
    "compare_association_evidence": CompareAssociationEvidenceAdapter(),
    "join_disease_target_drug": JoinDiseaseTargetDrugAdapter(),
    "rank_candidates": RankCandidatesAdapter(),
    "retrieve_mechanisms": RetrieveMechanismsAdapter(),
    "retrieve_mechanisms_and_modalities": RetrieveMechanismsAdapter(with_modality=True),
    "retrieve_clinical_stage": RetrieveClinicalStageAdapter(),
    "rank_diseases": RankDiseasesAdapter(),
    "retrieve_associated_diseases": RetrieveAssociatedDiseasesAdapter(),
    "compare_disease_rank": CompareDiseaseRankAdapter(),
    "retrieve_genetic_evidence": RetrieveGeneticEvidenceAdapter(),
    "filter_supported_without_clinical_drug": FilterSupportedWithoutClinicalDrugAdapter(),
    "retrieve_genetic_and_somatic_evidence": RetrieveGeneticAndSomaticEvidenceAdapter(),
    "filter_first_in_class_candidates": FilterFirstInClassCandidatesAdapter(),
    "rank_opportunities": RankOpportunitiesAdapter(),
    "retrieve_datasource_evidence": RetrieveDatasourceEvidenceAdapter(),
    "retrieve_evidence_by_type": RetrieveEvidenceByTypeAdapter(),
    "compare_evidence_types": CompareEvidenceTypesAdapter(),
    "normalize_comparison_dimensions": NormalizeComparisonDimensionsAdapter(),
    "compare_targets": CompareTargetsAdapter(),
    "retrieve_disease_drugs": RetrieveDiseaseDrugsAdapter(),
    "retrieve_target_cancers": RetrieveTargetCancersAdapter(),
    "join_cross_indication_paths": JoinCrossIndicationPathsAdapter(),
    "classify_kinase_targets": ClassifyKinaseTargetsAdapter(),
    "define_coverage_denominators": DefineCoverageDenominatorsAdapter(),
    "calculate_coverage_rates": CalculateCoverageRatesAdapter(),
    "compare_disease_rates": CompareDiseaseRatesAdapter(),
    "resolve_target_variants": ResolveTargetVariantsAdapter(),
    "retrieve_safety_evidence": RetrieveSafetyEvidenceAdapter(),
    "retrieve_trial_terminations": RetrieveTrialTerminationsAdapter(),
    "retrieve_normal_tissue_expression": RetrieveNormalTissueExpressionAdapter(),
    "integrate_safety_and_expression": IntegrateSafetyAndExpressionAdapter(),
    "retrieve_clinical_trials": RetrieveClinicalTrialsAdapter(),
    "retrieve_trial_status_and_stop_reasons": RetrieveTrialStatusAndStopReasonsAdapter(),
    "retrieve_negative_publications": RetrieveNegativePublicationsAdapter(),
    "retrieve_interventions": RetrieveInterventionsAdapter(),
    "aggregate_category_distribution": AggregateCategoryDistributionAdapter(),
    "rank_mechanism_categories": RankMechanismCategoriesAdapter(),
    "classify_mechanism_categories": ClassifyMechanismCategoriesAdapter(),
    "classify_termination_reasons": ClassifyTerminationReasonsAdapter(),
    "aggregate_failed_programs": AggregateFailedProgramsAdapter(),
    "filter_phase2_not_pdac_approved": FilterPhase2NotApprovedAdapter(),
}
