"""Adapters that gather for several entities at once, or combine data that
earlier steps already collected.

Nothing here makes up a score: values come from Open Targets as given, a
missing value stays missing (never zero), and each combined table says what it
covers.
"""

from __future__ import annotations

import time
from typing import Any

from execution.analysis import rank_rows, filter_rows
from execution.common import (
    _collect_per_target,
    _resolved_for_step,
    _targets_for_step,
    association_rows,
    error_result,
)
from execution.contracts import StepContext, StepResult
from query_opentargets import (
    OpenTargetsError,
    discover_disease_targets,
    get_disease_target_evidence_by_ids,
    get_drug_targets_by_id,
    resolve_entity,
    resolve_mention,
    get_target_associated_diseases,
    get_disease_targets_datatype_scores,
    get_association_datasource_types,
    get_disease_drug_records,
    get_disease_lineage,
)
from datetime import datetime, timezone


def _pair_key(disease_id: str | None, target_id: str) -> str:
    return f"{disease_id}|{target_id}"


class ResolveEntitiesAdapter:
    """Resolve every named entity of several types in one step."""

    source = "open_targets"

    def __init__(self, capabilities: dict[str, str]) -> None:
        self.capabilities = capabilities  # entity type -> capability it provides

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        named = {
            ref: entity
            for ref, entity in ctx.entities.items()
            if entity.get("entity_type") in self.capabilities
        }
        present = {entity["entity_type"] for entity in named.values()}
        absent = [kind for kind in self.capabilities if kind not in present]
        if absent:
            return error_result(
                ctx, self.source, start, f"needs a {' and a '.join(absent)} entity in this step"
            )
        resolved: dict[str, dict] = {}
        try:
            for ref, entity in named.items():
                resolved[ref] = {
                    **resolve_mention(
                        entity["mention"],
                        entity["entity_type"],
                        entity.get("entity_id"),
                        resolver=resolve_entity,
                    ),
                    "entity_type": entity["entity_type"],
                }
        except OpenTargetsError as exc:
            return error_result(ctx, self.source, start, str(exc))
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success",
            source=self.source,
            payload=resolved,
            provides=list(self.capabilities.values()),
            total_count=len(resolved),
            fetched_count=len(resolved),
            elapsed_seconds=time.monotonic() - start,
        )


class DiscoverTargetsPerDiseaseAdapter:
    """Top associated targets for each resolved disease, kept separate."""

    source = "open_targets"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        diseases = _resolved_for_step(ctx, "disease", "resolved_disease")
        if not diseases:
            return error_result(
                ctx, self.source, start, "discover_targets_per_disease needs at least one resolved disease"
            )
        top_n = ctx.filters.get("top_n", 10)
        by_disease: dict[str, dict] = {}
        failed: dict[str, str] = {}
        for disease in diseases:
            try:
                by_disease[disease["id"]] = discover_disease_targets(
                    disease.get("name", ""), top_n=top_n, disease_id=disease["id"]
                )
            except (OpenTargetsError, ValueError) as exc:
                failed[disease["id"]] = str(exc)
        if not by_disease:
            return error_result(
                ctx, self.source, start, "; ".join(f"{did}: {msg}" for did, msg in failed.items())
            )

        if len(by_disease) == 1:
            only = next(iter(by_disease.values()))
            payload = {**only}
        else:
            union: dict[str, dict] = {}
            for disease_id, found in by_disease.items():
                for row in found["targets"]:
                    entry = union.setdefault(
                        row["target_id"],
                        {
                            "target_id": row["target_id"],
                            "target_symbol": row["target_symbol"],
                            "target_name": row["target_name"],
                            "disease_ids": [],
                        },
                    )
                    entry["disease_ids"].append(disease_id)
            payload = {
                "targets": list(union.values()),
                "limitations": [
                    "Each disease keeps its own scores under by_disease; the merged "
                    "target list has IDs only, because a score belongs to one disease.",
                    f"Only the top {top_n} targets of each disease were fetched.",
                ],
            }
        payload["by_disease"] = by_disease
        payload["failed_diseases"] = failed

        returned = sum(found["returned_target_count"] for found in by_disease.values())
        total = sum(found["total_associated_target_count"] for found in by_disease.values())
        if failed:
            status = "partial_error"
        else:
            status = "success" if returned else "no_records"
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status=status,
            source=self.source,
            payload=payload,
            provides=["disease_target_association", "target_candidates"],
            total_count=total,
            fetched_count=returned,
            truncated=returned < total,
            error="; ".join(f"{did}: {msg}" for did, msg in failed.items()) or None,
            elapsed_seconds=time.monotonic() - start,
        )


class RetrieveExactPairsAdapter:
    """Exact-pair evidence for every resolved disease x target combination."""

    source = "open_targets"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        diseases = _resolved_for_step(ctx, "disease", "resolved_disease")
        targets = _resolved_for_step(ctx, "target", "resolved_target")
        if not diseases or not targets:
            return error_result(
                ctx,
                self.source,
                start,
                "exact pairs need at least one resolved disease and one resolved target",
            )
        pairs: list[dict] = []
        failed: dict[str, str] = {}
        for disease in diseases:
            for target in targets:
                try:
                    pairs.append(
                        get_disease_target_evidence_by_ids(
                            disease["id"],
                            target["id"],
                            disease_query=disease.get("name", ""),
                            target_query=target.get("symbol") or target.get("name", ""),
                        )
                    )
                except OpenTargetsError as exc:
                    failed[_pair_key(disease["id"], target["id"])] = str(exc)
        if not pairs:
            return error_result(
                ctx, self.source, start, "; ".join(f"{key}: {msg}" for key, msg in failed.items())
            )
        records = [r for pair in pairs for r in pair["datasource_evidence"]]
        total = sum(r["total_evidence_count"] for r in records) if records else None
        fetched = sum(r["fetched_evidence_count"] for r in records) if records else None
        truncated = any(r["truncated"] for r in records)
        if failed:
            status = "partial_error"
        elif all(pair["overall_association_score"] is None for pair in pairs):
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
            payload={"pairs": pairs, "failed_pairs": failed},
            provides=["disease_target_association"],
            total_count=total,
            fetched_count=fetched,
            truncated=truncated,
            error="; ".join(f"{key}: {msg}" for key, msg in failed.items()) or None,
            elapsed_seconds=time.monotonic() - start,
        )


class RetrieveDatasourceScoresAdapter:
    """Datasource-level scores of the disease-target pairs already retrieved."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        pair_rows = association_rows(ctx.board)
        if not pair_rows:
            return error_result(
                ctx, self.source, start, "needs disease_target_association data from an earlier step"
            )
        rows = [
            {
                "disease_id": pair.get("disease_id"),
                "disease_name": pair.get("disease_name"),
                "target_id": pair["target_id"],
                "target_symbol": pair.get("target_symbol"),
                "datasource_id": item["datasource_id"],
                "score": item.get("score"),
            }
            for pair in pair_rows
            for item in pair.get("datasource_scores") or []
        ]
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success" if rows else "no_records",
            source=self.source,
            payload={
                "rows": rows,
                "pair_count": len(pair_rows),
                "limitations": [
                    "Datasource scores are Open Targets association scores; they are "
                    "not record counts, probabilities or clinical confidence.",
                    "A pair with no datasource listed has no score there; that is "
                    "missing evidence, not a score of zero.",
                ],
            },
            provides=["datasource_scores"],
            total_count=len(rows),
            fetched_count=len(rows),
            elapsed_seconds=time.monotonic() - start,
        )


class RankDatasourceContributionsAdapter:
    """Order the datasources of each pair by their own score."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        scores = ctx.board.get("datasource_scores")
        if not scores or not scores.get("rows"):
            return error_result(
                ctx, self.source, start, "rank_datasource_contributions needs datasource_scores"
            )
        spec = ctx.ranking or {"primary_metric": "score", "direction": "descending"}
        grouped: dict[str, list[dict]] = {}
        for row in scores["rows"]:
            grouped.setdefault(_pair_key(row.get("disease_id"), row["target_id"]), []).append(row)
        try:
            by_pair = {
                key: rank_rows(
                    rows,
                    spec["primary_metric"],
                    spec["direction"],
                    spec.get("secondary_metrics"),
                )
                for key, rows in grouped.items()
            }
        except ValueError as exc:
            return error_result(ctx, self.source, start, str(exc))
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success",
            source=self.source,
            payload={"by_pair": by_pair, "ranking_spec": spec},
            elapsed_seconds=time.monotonic() - start,
        )


class IntersectTargetsAdapter:
    """Targets found for every disease, each with its own per-disease evidence."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        assoc = ctx.board.get("disease_target_association") or {}
        by_disease = assoc.get("by_disease") or {}
        if len(by_disease) < 2:
            return error_result(
                ctx,
                self.source,
                start,
                "intersect_targets needs targets of at least two diseases "
                "(from discover_targets_per_disease)",
            )
        disease_ids = list(by_disease)
        rows = {
            disease_id: {row["target_id"]: row for row in found["targets"]}
            for disease_id, found in by_disease.items()
        }
        shared_ids = [
            target_id
            for target_id in rows[disease_ids[0]]
            if all(target_id in rows[other] for other in disease_ids[1:])
        ]
        shared_set = set(shared_ids)
        shared = []
        for target_id in shared_ids:
            first = rows[disease_ids[0]][target_id]
            shared.append(
                {
                    "target_id": target_id,
                    "target_symbol": first.get("target_symbol"),
                    "target_name": first.get("target_name"),
                    "per_disease": {
                        disease_id: {
                            "rank": rows[disease_id][target_id].get("rank"),
                            "overall_association_score": rows[disease_id][target_id].get(
                                "overall_association_score"
                            ),
                            "datasource_scores": rows[disease_id][target_id].get(
                                "datasource_scores"
                            )
                            or [],
                        }
                        for disease_id in disease_ids
                    },
                }
            )
        truncated = any(
            found["returned_target_count"] < found["total_associated_target_count"]
            for found in by_disease.values()
        )
        limitations = [
            "Only the targets fetched for each disease were compared; a target "
            "outside a disease's fetched list may still be associated with it."
        ]
        if assoc.get("failed_diseases"):
            limitations.append(
                f"Diseases that failed earlier are not included: {sorted(assoc['failed_diseases'])}."
            )
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success" if shared else "no_records",
            source=self.source,
            payload={
                "disease_ids": disease_ids,
                "shared_targets": shared,
                "not_shared_target_ids": {
                    disease_id: [t for t in rows[disease_id] if t not in shared_set]
                    for disease_id in disease_ids
                },
                "set_sizes": {disease_id: len(rows[disease_id]) for disease_id in disease_ids},
                "limitations": limitations,
            },
            provides=["target_intersection"],
            total_count=len(shared),
            fetched_count=len(shared),
            truncated=truncated,
            elapsed_seconds=time.monotonic() - start,
        )


class CompareAssociationEvidenceAdapter:
    """Side-by-side scores per disease on the same dimensions; no combined score."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        pair_rows = association_rows(ctx.board)
        if not pair_rows:
            return error_result(
                ctx, self.source, start, "needs disease_target_association data from an earlier step"
            )
        intersection = ctx.board.get("target_intersection")
        if intersection:
            keep = {row["target_id"] for row in intersection.get("shared_targets", [])}
            pair_rows = [row for row in pair_rows if row["target_id"] in keep]
        disease_ids = list(dict.fromkeys(row["disease_id"] for row in pair_rows))
        datasource_ids = sorted(
            {
                item["datasource_id"]
                for row in pair_rows
                for item in row.get("datasource_scores") or []
            }
        )
        by_target: dict[str, dict[str, Any]] = {}
        for row in pair_rows:
            entry = by_target.setdefault(
                row["target_id"],
                {
                    "target_id": row["target_id"],
                    "target_symbol": row.get("target_symbol"),
                    "overall_association_score": {d: None for d in disease_ids},
                    "datasource_scores": {
                        ds: {d: None for d in disease_ids} for ds in datasource_ids
                    },
                },
            )
            entry["overall_association_score"][row["disease_id"]] = row.get(
                "overall_association_score"
            )
            for item in row.get("datasource_scores") or []:
                entry["datasource_scores"][item["datasource_id"]][row["disease_id"]] = item.get(
                    "score"
                )
        rows = list(by_target.values())
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success" if rows else "no_records",
            source=self.source,
            payload={
                "disease_ids": disease_ids,
                "datasource_ids": datasource_ids,
                "rows": rows,
                "limitations": [
                    "Missing values are null, not zero.",
                    "No combined score is computed; scores are Open Targets "
                    "association scores, not probabilities of success.",
                ],
            },
            provides=["association_comparison"],
            total_count=len(rows),
            fetched_count=len(rows),
            elapsed_seconds=time.monotonic() - start,
        )


class JoinDiseaseTargetDrugAdapter:
    """Join disease-target associations with target-drug records."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        pair_rows = association_rows(ctx.board)
        drug_evidence = ctx.board.get("target_drug_evidence") or {}
        by_target = drug_evidence.get("by_target")
        if not pair_rows or by_target is None:
            return error_result(
                ctx,
                self.source,
                start,
                "join_disease_target_drug needs disease_target_association and target_drug_evidence",
            )
        failed_targets = drug_evidence.get("failed_targets", {})
        edges: list[dict] = []
        without_drugs: list[str] = []
        not_queried: list[str] = []
        for pair in pair_rows:
            target_id = pair["target_id"]
            record = by_target.get(target_id)
            if record is None:
                if target_id not in failed_targets:
                    not_queried.append(target_id)
                continue
            if not record.get("drugs"):
                without_drugs.append(target_id)
            for drug in record.get("drugs") or []:
                edges.append(
                    {
                        "disease_id": pair.get("disease_id"),
                        "disease_name": pair.get("disease_name"),
                        "target_id": target_id,
                        "target_symbol": pair.get("target_symbol"),
                        "association_rank": pair.get("rank"),
                        "overall_association_score": pair.get("overall_association_score"),
                        "drug_id": drug["drug_id"],
                        "drug_name": drug.get("drug_name"),
                        "drug_type": drug.get("drug_type"),
                        "max_clinical_stage": drug.get("max_clinical_stage"),
                        "mechanisms_of_action": drug.get("mechanisms_of_action") or [],
                        "exact_disease_match": drug.get("exact_disease_match"),
                    }
                )
        if failed_targets:
            status = "partial_error"
        else:
            status = "success" if edges else "no_records"
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status=status,
            source=self.source,
            payload={
                "edges": edges,
                "targets_without_drug_records": without_drugs,
                "targets_not_queried": not_queried,
                "failed_targets": failed_targets,
                "limitations": [
                    "An edge shows that Open Targets has a drug record for the target; "
                    "it does not show that the drug works for the disease.",
                    "A target with no drug record is not shown to be undruggable.",
                ],
            },
            provides=["disease_target_drug_edges"],
            total_count=len(edges),
            fetched_count=len(edges),
            elapsed_seconds=time.monotonic() - start,
        )


class RankCandidatesAdapter:
    """Rank the most refined candidate table on the board by a numeric metric."""

    source = "local_analysis"
    _SOURCES = (
        ("disease_target_drug_edges", "edges"),
        ("stratified_target_evidence", "targets"),
        ("target_candidates", "targets"),
    )

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        if ctx.ranking is None:
            return error_result(ctx, self.source, start, "rank_candidates needs a ranking spec")
        capability, rows = None, []
        for name, key in self._SOURCES:
            data = ctx.board.get(name)
            if data and data.get(key):
                capability, rows = name, data[key]
                break
        if capability is None:
            return error_result(
                ctx,
                self.source,
                start,
                "rank_candidates needs disease_target_drug_edges, "
                "stratified_target_evidence or target_candidates",
            )
        metrics = [ctx.ranking["primary_metric"], *(ctx.ranking.get("secondary_metrics") or [])]
        for metric in metrics:
            values = [row[metric] for row in rows if row.get(metric) is not None]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
                return error_result(
                    ctx, self.source, start, f"ranking metric {metric!r} is not numeric"
                )
        try:
            ranked = rank_rows(
                rows,
                ctx.ranking["primary_metric"],
                ctx.ranking["direction"],
                ctx.ranking.get("secondary_metrics"),
            )
        except ValueError as exc:
            return error_result(ctx, self.source, start, str(exc))
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success",
            source=self.source,
            payload={
                "candidates": ranked,
                "source_capability": capability,
                "ranking_spec": ctx.ranking,
            },
            elapsed_seconds=time.monotonic() - start,
        )


class RetrieveMechanismsAdapter:
    """Mechanisms of action for the resolved drugs, or for the drugs found
    for the targets in an earlier step."""

    source = "open_targets"
    default_max_drugs = 50

    def __init__(self, with_modality: bool = False) -> None:
        self.with_modality = with_modality

    def _drugs(self, ctx: StepContext) -> list[dict]:
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

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        drugs = self._drugs(ctx)
        if not drugs:
            return error_result(
                ctx,
                self.source,
                start,
                f"{ctx.operation} needs a resolved drug or target_drug_evidence from an earlier step",
            )
        limit = ctx.filters.get("max_drugs", self.default_max_drugs)
        skipped = drugs[limit:]
        drugs = drugs[:limit]
        by_drug, failed = _collect_per_target(drugs, lambda drug: get_drug_targets_by_id(drug["id"]))
        if self.with_modality:
            by_drug = {
                drug_id: {**record, "modality": record.get("drug_type")}
                for drug_id, record in by_drug.items()
            }
        errors = "; ".join(f"{did}: {msg}" for did, msg in failed.items())
        if not by_drug:
            return error_result(ctx, self.source, start, errors)
        if failed:
            status = "partial_error"
        elif skipped:
            status = "partial_truncated"
        else:
            status = "success"
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status=status,
            source=self.source,
            payload={"by_drug": by_drug, "failed_drugs": failed},
            provides=["mechanism_evidence"],
            total_count=len(drugs) + len(skipped),
            fetched_count=len(by_drug),
            truncated=bool(skipped),
            error=errors or None,
            elapsed_seconds=time.monotonic() - start,
        )


class RetrieveClinicalStageAdapter:
    """Development stage of the drugs already found for each target."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        drug_evidence = ctx.board.get("target_drug_evidence") or {}
        by_target = drug_evidence.get("by_target")
        if by_target is None:
            return error_result(
                ctx, self.source, start, "retrieve_clinical_stage needs target_drug_evidence"
            )
        result = {}
        for target_id, record in by_target.items():
            drugs = [
                {
                    "drug_id": drug["drug_id"],
                    "drug_name": drug.get("drug_name"),
                    "max_clinical_stage": drug.get("max_clinical_stage"),
                }
                for drug in record.get("drugs") or []
            ]
            result[target_id] = {
                "drug_record_count": len(drugs),
                "stages_present": sorted(
                    {d["max_clinical_stage"] for d in drugs if d["max_clinical_stage"]}
                ),
                "drugs": drugs,
            }
        failed = drug_evidence.get("failed_targets", {})
        total = sum(entry["drug_record_count"] for entry in result.values())
        if failed:
            status = "partial_error"
        else:
            status = "success" if total else "no_records"
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status=status,
            source=self.source,
            payload={
                "by_target": result,
                "failed_targets": failed,
                "limitations": [
                    "A clinical stage is a development stage, not an approval.",
                    "No drug record is missing evidence, not proof that no drug exists.",
                ],
            },
            provides=["clinical_stage_evidence"],
            total_count=total,
            fetched_count=total,
            elapsed_seconds=time.monotonic() - start,
        )



class RetrieveAssociatedDiseasesAdapter:
    """Top associated diseases for each resolved target, kept as separate pairs."""

    source = "open_targets"
    default_top_n = 10

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        targets = _resolved_for_step(ctx, "target", "resolved_target")
        if not targets:
            return error_result(
                ctx,
                self.source,
                start,
                "retrieve_associated_diseases needs at least one resolved target",
            )
        top_n = ctx.filters.get("top_n", self.default_top_n)
        if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 100:
            return error_result(
                ctx, self.source, start, "top_n must be an integer between 1 and 100"
            )
        by_target, failed = _collect_per_target(
            targets,
            lambda target: get_target_associated_diseases(target["id"], top_n=top_n),
        )
        errors = "; ".join(f"{tid}: {msg}" for tid, msg in failed.items())
        if not by_target:
            return error_result(ctx, self.source, start, errors)
        pairs = [row for found in by_target.values() for row in found["diseases"]]
        total = sum(found["total_associated_disease_count"] for found in by_target.values())
        if failed:
            status = "partial_error"
        else:
            status = "success" if pairs else "no_records"
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status=status,
            source=self.source,
            payload={
                "pairs": pairs,
                "by_target": {
                    tid: {key: value for key, value in found.items() if key != "diseases"}
                    for tid, found in by_target.items()
                },
                "failed_targets": failed,
                "limitations": [
                    f"Only the top {top_n} diseases of each target were fetched.",
                    "An association score is not a probability and does not imply efficacy.",
                ],
            },
            provides=["disease_target_association"],
            total_count=total,
            fetched_count=len(pairs),
            truncated=len(pairs) < total,
            error=errors or None,
            elapsed_seconds=time.monotonic() - start,
        )

class RankDiseasesAdapter:
    """Rank diseases by a numeric disease-target association metric."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        if ctx.ranking is None:
            return error_result(ctx, self.source, start, "rank_diseases needs a ranking spec")
        rows = association_rows(ctx.board)
        if not rows:
            return error_result(
                ctx, self.source, start, "rank_diseases needs disease_target_association"
            )
        metrics = [ctx.ranking["primary_metric"], *(ctx.ranking.get("secondary_metrics") or [])]
        for metric in metrics:
            values = [row[metric] for row in rows if row.get(metric) is not None]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
                return error_result(
                    ctx, self.source, start, f"ranking metric {metric!r} is not numeric"
                )
        try:
            ranked = rank_rows(
                rows,
                ctx.ranking["primary_metric"],
                ctx.ranking["direction"],
                ctx.ranking.get("secondary_metrics"),
            )
        except ValueError as exc:
            return error_result(ctx, self.source, start, str(exc))
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success",
            source=self.source,
            payload={
                "diseases": ranked,
                "ranking_spec": ctx.ranking,
                "limitations": [
                    "Only the disease-target pairs retrieved by earlier steps are ranked.",
                    "An association score is not a probability and does not imply efficacy.",
                ],
            },
            provides=["disease_ranking"],
            total_count=len(ranked),
            fetched_count=len(ranked),
            elapsed_seconds=time.monotonic() - start,
        )


GENETIC_DATATYPE = "genetic_association"
SOMATIC_DATATYPE = "somatic_mutation"


class _DatatypeEvidenceAdapter:
    """Datatype-level association scores of each candidate target for one disease."""

    source = "open_targets"
    capability = ""
    fields: dict[str, str] = {}  # output field -> Open Targets datatype id

    def extra_payload(self) -> dict[str, Any]:
        return {}

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        diseases = _resolved_for_step(ctx, "disease", "resolved_disease")
        if len(diseases) == 1:
            disease_id = diseases[0]["id"]
        else:
            disease_id = (ctx.board.get("disease_target_association") or {}).get("disease_id")
        if not disease_id:
            return error_result(
                ctx, self.source, start, f"{ctx.operation} needs one resolved disease"
            )
        targets = _targets_for_step(ctx)
        if not targets:
            return error_result(
                ctx,
                self.source,
                start,
                f"{ctx.operation} needs target candidates or resolved targets",
            )
        try:
            found = get_disease_targets_datatype_scores(
                disease_id, [target["id"] for target in targets]
            )
        except OpenTargetsError as exc:
            return error_result(ctx, self.source, start, str(exc))
        rows = []
        for target in targets:
            record = found["by_target"].get(target["id"]) or {}
            scores = {
                item["datatype_id"]: item["score"]
                for item in record.get("datatype_scores") or []
            }
            rows.append(
                {
                    "target_id": target["id"],
                    "target_symbol": target.get("symbol", ""),
                    "target_name": target.get("name", ""),
                    **{field: scores.get(datatype) for field, datatype in self.fields.items()},
                }
            )
        with_score = sum(
            any(row[field] is not None for field in self.fields) for row in rows
        )
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success" if with_score else "no_records",
            source=self.source,
            payload={
                "disease_id": found["disease_id"],
                **self.extra_payload(),
                "targets": rows,
                "limitations": [
                    "Scores are Open Targets datatype association scores for this "
                    f"disease ({', '.join(self.fields.values())}); they are not "
                    "probabilities or evidence of efficacy.",
                    "A target with no score for a datatype has none listed in this "
                    "source; the score is missing, not zero.",
                    "Only the candidate targets from earlier steps were queried.",
                ],
            },
            provides=[self.capability],
            total_count=len(rows),
            fetched_count=with_score,
            elapsed_seconds=time.monotonic() - start,
        )


class RetrieveGeneticEvidenceAdapter(_DatatypeEvidenceAdapter):
    capability = "genetic_evidence"
    fields = {"genetic_evidence_score": GENETIC_DATATYPE}

    def extra_payload(self) -> dict[str, Any]:
        return {"datatype_id": GENETIC_DATATYPE}


class RetrieveGeneticAndSomaticEvidenceAdapter(_DatatypeEvidenceAdapter):
    capability = "genetic_somatic_evidence"
    fields = {
        "genetic_evidence_score": GENETIC_DATATYPE,
        "somatic_evidence_score": SOMATIC_DATATYPE,
    }

    def extra_payload(self) -> dict[str, Any]:
        return {"datatype_ids": [GENETIC_DATATYPE, SOMATIC_DATATYPE]}


class FilterSupportedWithoutClinicalDrugAdapter:
    """Targets with genetic evidence and no drug record in the queried source."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        genetic = ctx.board.get("genetic_evidence") or {}
        stage = ctx.board.get("clinical_stage_evidence") or {}
        if not genetic.get("targets") or stage.get("by_target") is None:
            return error_result(
                ctx,
                self.source,
                start,
                "filter_supported_without_clinical_drug needs genetic_evidence "
                "and clinical_stage_evidence",
            )
        by_stage = stage["by_target"]
        failed = stage.get("failed_targets") or {}
        candidates: list[dict] = []
        rejected: list[str] = []
        unevaluable: list[str] = []
        for row in genetic["targets"]:
            record = by_stage.get(row["target_id"])
            view = {
                **row,
                "drug_record_count": record["drug_record_count"] if record else None,
                "stages_present": record["stages_present"] if record else None,
            }
            if view["genetic_evidence_score"] is None:
                rejected.append(row["target_id"])
            elif view["drug_record_count"] is None:
                unevaluable.append(row["target_id"])
            elif view["drug_record_count"] > 0:
                rejected.append(row["target_id"])
            else:
                candidates.append(view)
        kept_rows = candidates
        if ctx.filters and candidates:
            try:
                kept_ids, extra_rejected, extra_unevaluable = filter_rows(
                    candidates, ctx.filters
                )
            except ValueError as exc:
                return error_result(ctx, self.source, start, str(exc))
            kept_set = set(kept_ids)
            kept_rows = [row for row in candidates if row["target_id"] in kept_set]
            rejected += extra_rejected
            unevaluable += extra_unevaluable
        if failed:
            status = "partial_error"
        else:
            status = "success" if kept_rows else "no_records"
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status=status,
            source=self.source,
            payload={
                "targets": kept_rows,
                "excluded_target_ids": rejected,
                "unevaluable_target_ids": unevaluable,
                "filter_spec": ctx.filters,
                "evaluated_on": datetime.now(timezone.utc).date().isoformat(),
                "limitations": [
                    "No drug record means none was found in the Open Targets "
                    "known-drug data on the evaluated_on date; it does not prove that "
                    "no drug exists.",
                    "Any drug record counts as a drug here; stages_present is "
                    "listed on excluded rows' source data so the reader can judge stages.",
                    "A target whose drug retrieval failed is unevaluable, not 'no drug'.",
                    "Genetic evidence is an Open Targets association score, not a "
                    "probability or evidence of efficacy.",
                ],
            },
            provides=["supported_without_clinical_drug"],
            total_count=len(genetic["targets"]),
            fetched_count=len(kept_rows),
            error="; ".join(f"{tid}: {msg}" for tid, msg in failed.items()) or None,
            elapsed_seconds=time.monotonic() - start,
        )


class FilterFirstInClassCandidatesAdapter:
    """Targets with strong genetic or somatic evidence and no drug record found."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        evidence = ctx.board.get("genetic_somatic_evidence") or {}
        drugs = ctx.board.get("target_drug_evidence") or {}
        if not evidence.get("targets") or drugs.get("by_target") is None:
            return error_result(
                ctx,
                self.source,
                start,
                "filter_first_in_class_candidates needs genetic_somatic_evidence "
                "and target_drug_evidence",
            )
        if not ctx.filters:
            return error_result(
                ctx,
                self.source,
                start,
                "filter_first_in_class_candidates needs an explicit evidence-strength "
                "filter (for example max_genetic_somatic_score); the project has not "
                "defined 'strong'",
            )
        by_drug = drugs["by_target"]
        failed = drugs.get("failed_targets") or {}
        candidates: list[dict] = []
        rejected: list[str] = []
        unevaluable: list[str] = []
        for row in evidence["targets"]:
            scores = [
                row[field]
                for field in ("genetic_evidence_score", "somatic_evidence_score")
                if row.get(field) is not None
            ]
            if not scores:
                rejected.append(row["target_id"])
                continue
            record = by_drug.get(row["target_id"])
            if record is None:
                unevaluable.append(row["target_id"])
                continue
            if record["total_drug_count"] > 0:
                rejected.append(row["target_id"])
                continue
            candidates.append(
                {
                    **row,
                    "max_genetic_somatic_score": max(scores),
                    "known_drug_count": record["total_drug_count"],
                }
            )
        try:
            kept_ids, extra_rejected, extra_unevaluable = filter_rows(
                candidates, ctx.filters
            )
        except ValueError as exc:
            return error_result(ctx, self.source, start, str(exc))
        kept_set = set(kept_ids)
        kept_rows = [row for row in candidates if row["target_id"] in kept_set]
        rejected += extra_rejected
        unevaluable += extra_unevaluable
        if failed:
            status = "partial_error"
        else:
            status = "success" if kept_rows else "no_records"
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status=status,
            source=self.source,
            payload={
                "targets": kept_rows,
                "excluded_target_ids": rejected,
                "unevaluable_target_ids": unevaluable,
                "filter_spec": ctx.filters,
                "evaluated_on": datetime.now(timezone.utc).date().isoformat(),
                "limitations": [
                    "No known drug means no drug record was found for the target in "
                    "the Open Targets known-drug data on the evaluated_on date; it "
                    "does not prove that no drug exists.",
                    "A target whose drug retrieval failed is unevaluable, not 'no drug'.",
                    "Evidence scores are Open Targets association scores, not "
                    "probabilities or evidence of efficacy.",
                ],
            },
            provides=["first_in_class_candidates"],
            total_count=len(evidence["targets"]),
            fetched_count=len(kept_rows),
            error="; ".join(f"{tid}: {msg}" for tid, msg in failed.items()) or None,
            elapsed_seconds=time.monotonic() - start,
        )


class RankOpportunitiesAdapter:
    """Order candidates by an explicit metric; tractability only adds warnings."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        if ctx.ranking is None:
            return error_result(
                ctx, self.source, start, "rank_opportunities needs a ranking spec"
            )
        candidates = ctx.board.get("first_in_class_candidates")
        if candidates is None or candidates.get("targets") is None:
            return error_result(
                ctx, self.source, start, "rank_opportunities needs first_in_class_candidates"
            )
        tractability = (ctx.board.get("tractability_evidence") or {}).get("by_target") or {}
        rows = []
        for row in candidates["targets"]:
            record = tractability.get(row["target_id"])
            warnings = []
            if record is None:
                bucket_count = modalities = None
                warnings.append(
                    "Tractability was not assessed for this target; missing is not "
                    "'not tractable'."
                )
            else:
                positives = [a for a in record["assessments"] if a["value"]]
                bucket_count = len(positives)
                modalities = sorted({a["modality"] for a in positives})
                if not positives:
                    warnings.append(
                        "No tractability bucket is assessed as true for this target."
                    )
            rows.append(
                {
                    **row,
                    "tractable_bucket_count": bucket_count,
                    "tractable_modalities": modalities,
                    "warnings": warnings,
                }
            )
        metrics = [ctx.ranking["primary_metric"], *(ctx.ranking.get("secondary_metrics") or [])]
        for metric in metrics:
            values = [row[metric] for row in rows if row.get(metric) is not None]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
                return error_result(
                    ctx, self.source, start, f"ranking metric {metric!r} is not numeric"
                )
        try:
            ranked = rank_rows(
                rows,
                ctx.ranking["primary_metric"],
                ctx.ranking["direction"],
                ctx.ranking.get("secondary_metrics"),
            )
        except ValueError as exc:
            return error_result(ctx, self.source, start, str(exc))
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success" if ranked else "no_records",
            source=self.source,
            payload={
                "opportunities": ranked,
                "ranking_spec": ctx.ranking,
                "limitations": [
                    "The order reflects the chosen metric only; it is not a probability "
                    "of success.",
                    "Tractability is shown as warnings and does not change the order.",
                ],
            },
            provides=["opportunity_ranking"],
            total_count=len(ranked),
            fetched_count=len(ranked),
            elapsed_seconds=time.monotonic() - start,
        )


def _datatype_key(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _evidence_types_from_filters(filters: dict[str, Any]) -> set[str] | None:
    """The datatype ids a plan asks for, or None when it names none."""
    if "evidence_type" not in filters:
        return None
    condition = filters["evidence_type"]
    if isinstance(condition, dict):
        if set(condition) - {"in", "eq"}:
            raise ValueError("evidence_type filter supports only 'in' and 'eq'")
        condition = condition.get("in", condition.get("eq"))
    if condition is None:
        raise ValueError("evidence_type filter has no value")
    values = condition if isinstance(condition, list) else [condition]
    return {_datatype_key(value) for value in values}


class _PairEvidenceByTypeAdapter:
    """Datasource scores of disease-target pairs, grouped by evidence datatype."""

    source = "open_targets"
    capability = ""
    needs_type_filter = False

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        pair_rows = association_rows(ctx.board)
        if not pair_rows:
            return error_result(
                ctx,
                self.source,
                start,
                f"{ctx.operation} needs disease_target_association data from an earlier step",
            )
        unsupported = sorted(set(ctx.filters) - {"evidence_type"})
        if unsupported:
            return error_result(
                ctx, self.source, start, f"unsupported filter(s): {unsupported}"
            )
        try:
            wanted = _evidence_types_from_filters(ctx.filters)
        except ValueError as exc:
            return error_result(ctx, self.source, start, str(exc))
        if self.needs_type_filter and not wanted:
            return error_result(
                ctx, self.source, start, f"{ctx.operation} needs an evidence_type filter"
            )
        try:
            types_by_datasource = {
                datasource: _datatype_key(datatype)
                for datasource, datatype in get_association_datasource_types().items()
            }
            targets_by_disease: dict[str, set[str]] = {}
            for row in pair_rows:
                targets_by_disease.setdefault(row["disease_id"], set()).add(row["target_id"])
            datatype_scores = {
                disease_id: get_disease_targets_datatype_scores(
                    disease_id, sorted(target_ids)
                )["by_target"]
                for disease_id, target_ids in targets_by_disease.items()
            }
        except OpenTargetsError as exc:
            return error_result(ctx, self.source, start, str(exc))

        pairs = []
        unknown_datasources: set[str] = set()
        for row in pair_rows:
            record = datatype_scores[row["disease_id"]].get(row["target_id"]) or {}
            scores = {
                _datatype_key(item["datatype_id"]): item["score"]
                for item in record.get("datatype_scores") or []
            }
            grouped: dict[str, list[dict]] = {}
            for item in row.get("datasource_scores") or []:
                datatype = types_by_datasource.get(item["datasource_id"])
                if datatype is None:
                    datatype = "unknown"
                    unknown_datasources.add(item["datasource_id"])
                grouped.setdefault(datatype, []).append(
                    {"datasource_id": item["datasource_id"], "score": item.get("score")}
                )
            names = sorted(wanted) if wanted else sorted(set(scores) | set(grouped))
            pairs.append(
                {
                    "disease_id": row["disease_id"],
                    "disease_name": row.get("disease_name"),
                    "target_id": row["target_id"],
                    "target_symbol": row.get("target_symbol"),
                    "overall_association_score": row.get("overall_association_score"),
                    "datatypes": [
                        {
                            "datatype_id": name,
                            "datatype_score": scores.get(name),
                            "datasources": grouped.get(name, []),
                        }
                        for name in names
                    ],
                }
            )
        limitations = [
            "Scores are Open Targets association scores per datatype and datasource; "
            "they are not probabilities, and evidence record counts are not scores.",
            "Raw evidence records are not downloaded in this step; only the "
            "clinical_precedence records fetched by the exact-pair step exist.",
            "A datatype with no score is missing in this source, not zero.",
        ]
        if unknown_datasources:
            limitations.append(
                "Datasources with no datatype in the metadata are grouped as "
                f"'unknown': {sorted(unknown_datasources)}."
            )
        has_score = any(
            entry["datatype_score"] is not None
            for pair in pairs
            for entry in pair["datatypes"]
        )
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success" if has_score else "no_records",
            source=self.source,
            payload={
                "pairs": pairs,
                "evidence_type_filter": sorted(wanted) if wanted else None,
                "limitations": limitations,
            },
            provides=[self.capability],
            total_count=len(pairs),
            fetched_count=len(pairs),
            elapsed_seconds=time.monotonic() - start,
        )


class RetrieveDatasourceEvidenceAdapter(_PairEvidenceByTypeAdapter):
    capability = "datasource_evidence"


class RetrieveEvidenceByTypeAdapter(_PairEvidenceByTypeAdapter):
    capability = "evidence_by_type"
    needs_type_filter = True


def _typed_evidence(ctx: StepContext) -> dict[str, Any] | None:
    data = ctx.board.get("evidence_by_type") or ctx.board.get("datasource_evidence")
    return data if data and data.get("pairs") else None


class CompareEvidenceTypesAdapter:
    """Rank the evidence types of each pair by their own datatype score."""

    source = "local_analysis"
    _METRICS = {"datatype_score", "datasource_contribution", "score"}

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        data = _typed_evidence(ctx)
        if data is None:
            return error_result(
                ctx,
                self.source,
                start,
                "compare_evidence_types needs datasource_evidence or evidence_by_type",
            )
        direction = "descending"
        if ctx.ranking is not None:
            if ctx.ranking["primary_metric"] not in self._METRICS:
                return error_result(
                    ctx,
                    self.source,
                    start,
                    "compare_evidence_types ranks by datatype_score only, "
                    f"not {ctx.ranking['primary_metric']!r}",
                )
            direction = ctx.ranking["direction"]
        results = []
        for pair in data["pairs"]:
            rows = [
                {
                    "datatype_id": entry["datatype_id"],
                    "datatype_score": entry["datatype_score"],
                    "datasource_count": len(entry["datasources"]),
                }
                for entry in pair["datatypes"]
            ]
            ranked = rank_rows(rows, "datatype_score", direction) if rows else []
            scored = [row for row in ranked if row["datatype_score"] is not None]
            results.append(
                {
                    "disease_id": pair["disease_id"],
                    "target_id": pair["target_id"],
                    "target_symbol": pair.get("target_symbol"),
                    "evidence_types": ranked,
                    "top_evidence_type": scored[0]["datatype_id"] if scored else None,
                    "top_is_tied": len(scored) > 1
                    and scored[0]["datatype_score"] == scored[1]["datatype_score"],
                }
            )
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success",
            source=self.source,
            payload={
                "pairs": results,
                "ranking_spec": ctx.ranking,
                "limitations": [
                    "Datatype scores are not shares that add up to a whole; the top "
                    "type has the strongest score, not a percentage of the evidence.",
                    "An evidence type with no score is missing, not zero.",
                ],
            },
            provides=["evidence_type_comparison"],
            total_count=len(results),
            fetched_count=len(results),
            elapsed_seconds=time.monotonic() - start,
        )


class NormalizeComparisonDimensionsAdapter:
    """Put every target on the same evidence-type dimensions; gaps stay null."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        data = _typed_evidence(ctx)
        if data is None:
            return error_result(
                ctx,
                self.source,
                start,
                "normalize_comparison_dimensions needs evidence_by_type or datasource_evidence",
            )
        dimensions = sorted(
            {entry["datatype_id"] for pair in data["pairs"] for entry in pair["datatypes"]}
        )
        rows = []
        for pair in data["pairs"]:
            by_type = {entry["datatype_id"]: entry["datatype_score"] for entry in pair["datatypes"]}
            rows.append(
                {
                    "disease_id": pair["disease_id"],
                    "target_id": pair["target_id"],
                    "target_symbol": pair.get("target_symbol"),
                    "scores": {name: by_type.get(name) for name in dimensions},
                }
            )
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success" if rows else "no_records",
            source=self.source,
            payload={
                "dimensions": dimensions,
                "rows": rows,
                "limitations": [
                    "All scores are on the same 0-1 Open Targets scale, so no "
                    "denominator is applied.",
                    "A missing score stays null; it is never turned into zero.",
                ],
            },
            provides=["normalized_comparison"],
            total_count=len(rows),
            fetched_count=len(rows),
            elapsed_seconds=time.monotonic() - start,
        )


class CompareTargetsAdapter:
    """Side-by-side scores per evidence type; no combined score, no winner overall."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        data = ctx.board.get("normalized_comparison")
        if not data or not data.get("rows"):
            return error_result(
                ctx, self.source, start, "compare_targets needs normalized_comparison"
            )
        rows = data["rows"]
        disease_ids = {row["disease_id"] for row in rows}
        if len(disease_ids) != 1:
            return error_result(
                ctx,
                self.source,
                start,
                "compare_targets compares targets inside one disease, "
                f"got {sorted(str(d) for d in disease_ids)}",
            )
        if len({row["target_id"] for row in rows}) < 2:
            return error_result(
                ctx, self.source, start, "compare_targets needs at least two targets"
            )
        comparisons = []
        for name in data["dimensions"]:
            scores = {row["target_id"]: row["scores"].get(name) for row in rows}
            missing = [target_id for target_id, score in scores.items() if score is None]
            complete = not missing
            leader = None
            tied = False
            if complete:
                top = max(scores.values())
                leaders = [target_id for target_id, score in scores.items() if score == top]
                tied = len(leaders) > 1
                leader = None if tied else leaders[0]
            comparisons.append(
                {
                    "datatype_id": name,
                    "scores": scores,
                    "leader_target_id": leader,
                    "tied": tied,
                    "complete": complete,
                    "missing_target_ids": missing,
                }
            )
        has_score = any(
            score is not None for item in comparisons for score in item["scores"].values()
        )
        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success" if has_score else "no_records",
            source=self.source,
            payload={
                "disease_id": next(iter(disease_ids)),
                "target_symbols": {row["target_id"]: row.get("target_symbol") for row in rows},
                "comparisons": comparisons,
                "limitations": [
                    "Scores are Open Targets datatype scores on one 0-1 scale; no "
                    "combined score is computed and no overall winner is named.",
                    "A leader is named only when every compared target has a score "
                    "for that evidence type; a missing score is not zero.",
                    "Equal scores are reported as tied, with no leader.",
                ],
            },
            provides=["target_comparison"],
            total_count=len(comparisons),
            fetched_count=len(comparisons),
            elapsed_seconds=time.monotonic() - start,
        )


CANCER_AREA_IDS = {"MONDO_0045024", "EFO_0000616"}
CANCER_AREA_NAMES = {"cancer or benign tumor", "neoplasm"}


def _is_cancer(areas: list[dict[str, Any]]) -> bool | None:
    """True/False from Open Targets therapeutic areas; None when none are listed."""
    if not areas:
        return None
    return any(
        area.get("id") in CANCER_AREA_IDS
        or str(area.get("name") or "").strip().lower() in CANCER_AREA_NAMES
        for area in areas
    )


def _flag(filters: dict[str, Any], name: str, default: bool) -> bool:
    if name not in filters:
        return default
    value = filters[name]
    if isinstance(value, dict):
        if set(value) - {"eq"}:
            raise ValueError(f"{name} filter supports only 'eq'")
        value = value.get("eq")
    if not isinstance(value, bool):
        raise ValueError(f"{name} filter must be true or false")
    return value


class RetrieveDiseaseDrugsAdapter:
    """Drug records Open Targets holds for the resolved disease(s)."""

    source = "open_targets"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        if ctx.filters:
            return error_result(
                ctx, self.source, start, f"unsupported filter(s): {sorted(ctx.filters)}"
            )
        diseases = _resolved_for_step(ctx, "disease", "resolved_disease")
        if not diseases:
            return error_result(
                ctx,
                self.source,
                start,
                "retrieve_disease_drugs needs at least one resolved disease",
            )
        named = _resolved_for_step(ctx, "drug", "resolved_drug")
        by_disease, failed = _collect_per_target(
            diseases, lambda disease: get_disease_drug_records(disease["id"])
        )
        errors = "; ".join(f"{did}: {msg}" for did, msg in failed.items())
        if not by_disease:
            return error_result(ctx, self.source, start, errors)
        total = sum(found["total_drug_count"] for found in by_disease.values())
        fetched = sum(found["fetched_drug_count"] for found in by_disease.values())
        named_ids = {drug["id"] for drug in named}
        for found in by_disease.values():
            if named:
                kept = [d for d in found["drugs"] if d["drug_id"] in named_ids]
                present = {d["drug_id"] for d in kept}
                found["drugs"] = kept
                found["named_drugs_not_found"] = [
                    {"drug_id": d["id"], "drug_name": d.get("name") or ""}
                    for d in named
                    if d["id"] not in present
                ]
            else:
                found["named_drugs_not_found"] = []
        kept_count = sum(len(found["drugs"]) for found in by_disease.values())
        truncated = fetched < total
        if failed:
            status = "partial_error"
        elif kept_count == 0:
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
                "by_disease": by_disease,
                "failed_diseases": failed,
                "limitations": [
                    "Only records filed under the exact disease id are listed; a drug "
                    "recorded under a broader or narrower disease term does not appear.",
                    "A named drug with no record here is not found in this source; that "
                    "is not evidence that it is unused or unapproved.",
                    "A stage is the Open Targets record for this drug and disease, not an "
                    "approval conclusion; approval needs an explicit source.",
                ],
            },
            provides=["disease_drug_evidence"],
            total_count=total,
            fetched_count=fetched,
            truncated=truncated,
            error=errors or None,
            elapsed_seconds=time.monotonic() - start,
        )


def _targets_from_drugs(board: dict[str, Any]) -> list[dict[str, str]]:
    evidence = board.get("target_drug_evidence") or {}
    found: dict[str, dict[str, str]] = {}
    for record in (evidence.get("by_drug") or {}).values():
        for target in record.get("targets") or []:
            found.setdefault(
                target["target_id"],
                {
                    "id": target["target_id"],
                    "symbol": target.get("target_symbol") or "",
                    "name": target.get("target_name") or "",
                },
            )
    return list(found.values())


class RetrieveTargetCancersAdapter:
    """Cancers associated with each target; an association is not an indication."""

    source = "open_targets"
    default_top_n = 100
    _FILTERS = {"cancer_only", "exclude_index_disease", "top_n"}

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        unsupported = sorted(set(ctx.filters) - self._FILTERS)
        if unsupported:
            return error_result(
                ctx, self.source, start, f"unsupported filter(s): {unsupported}"
            )
        try:
            cancer_only = _flag(ctx.filters, "cancer_only", True)
            exclude_index = _flag(ctx.filters, "exclude_index_disease", False)
        except ValueError as exc:
            return error_result(ctx, self.source, start, str(exc))
        top_n = ctx.filters.get("top_n", self.default_top_n)
        if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 100:
            return error_result(
                ctx, self.source, start, "top_n must be an integer between 1 and 100"
            )
        targets = (
            _resolved_for_step(ctx, "target", "resolved_target")
            or _targets_from_drugs(ctx.board)
            or _targets_for_step(ctx)
        )
        if not targets:
            return error_result(
                ctx,
                self.source,
                start,
                "retrieve_target_cancers needs resolved targets or drug targets",
            )
        index_ids: set[str] = set()
        index_disease_ids: list[str] = []
        if exclude_index:
            indexes = _resolved_for_step(ctx, "disease", "resolved_disease")
            if not indexes:
                return error_result(
                    ctx,
                    self.source,
                    start,
                    "exclude_index_disease needs a resolved disease",
                )
            try:
                for disease in indexes:
                    lineage = get_disease_lineage(disease["id"])
                    index_disease_ids.append(lineage["disease_id"])
                    index_ids |= {
                        lineage["disease_id"],
                        *lineage["ancestor_ids"],
                        *lineage["descendant_ids"],
                    }
            except OpenTargetsError as exc:
                return error_result(ctx, self.source, start, str(exc))
        fetched_by_target, failed = _collect_per_target(
            targets,
            lambda target: get_target_associated_diseases(target["id"], top_n=top_n),
        )
        errors = "; ".join(f"{tid}: {msg}" for tid, msg in failed.items())
        if not fetched_by_target:
            return error_result(ctx, self.source, start, errors)

        by_target = {}
        for target_id, found in fetched_by_target.items():
            kept, excluded, non_cancer, unclassified = [], 0, 0, 0
            for row in found["diseases"]:
                if row["disease_id"] in index_ids:
                    excluded += 1
                    continue
                flag = _is_cancer(row.get("therapeutic_areas") or [])
                if cancer_only:
                    if flag is None:
                        unclassified += 1
                        continue
                    if not flag:
                        non_cancer += 1
                        continue
                kept.append(
                    {
                        "disease_id": row["disease_id"],
                        "disease_name": row["disease_name"],
                        "overall_association_score": row["overall_association_score"],
                        "is_cancer": flag,
                    }
                )
            by_target[target_id] = {
                "target_id": target_id,
                "target_symbol": found["target_symbol"],
                "total_associated_disease_count": found["total_associated_disease_count"],
                "fetched_disease_count": found["returned_disease_count"],
                "diseases": kept,
                "non_cancer_count": non_cancer,
                "unclassified_count": unclassified,
                "excluded_index_count": excluded,
            }
        total = sum(e["total_associated_disease_count"] for e in by_target.values())
        fetched = sum(e["fetched_disease_count"] for e in by_target.values())
        kept_count = sum(len(e["diseases"]) for e in by_target.values())
        truncated = fetched < total
        limitations = [
            f"Only the top {top_n} associated diseases of each target were fetched, "
            "so lower-ranked diseases are not listed.",
            "An association is not a treatment indication; it does not show that a drug "
            "works or is approved in that disease.",
        ]
        if cancer_only:
            limitations.append(
                "Cancer follows the Open Targets therapeutic areas 'cancer or benign "
                "tumor' / 'neoplasm' (benign tumours included). Diseases with no "
                "therapeutic area are counted as unclassified and left out, not treated "
                "as non-cancer."
            )
        if exclude_index:
            limitations.append(
                "Index disease exclusion removes the disease together with its ancestors "
                "and descendants in the ontology."
            )
        if failed:
            status = "partial_error"
        elif kept_count == 0:
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
                "failed_targets": failed,
                "cancer_only": cancer_only,
                "index_disease_ids": index_disease_ids,
                "index_exclusion_term_count": len(index_ids),
                "truncated": truncated,
                "limitations": limitations,
            },
            provides=["target_cancer_evidence"],
            total_count=total,
            fetched_count=fetched,
            truncated=truncated,
            error=errors or None,
            elapsed_seconds=time.monotonic() - start,
        )


class JoinCrossIndicationPathsAdapter:
    """Chain index-disease drug -> target -> other disease. Paths are associations."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        disease_drugs = ctx.board.get("disease_drug_evidence")
        drug_targets = ctx.board.get("target_drug_evidence")
        target_cancers = ctx.board.get("target_cancer_evidence")
        missing = [
            name
            for name, data in (
                ("disease_drug_evidence", disease_drugs),
                ("target_drug_evidence", drug_targets),
                ("target_cancer_evidence", target_cancers),
            )
            if not data
        ]
        if missing:
            return error_result(
                ctx,
                self.source,
                start,
                f"join_cross_indication_paths needs {', '.join(missing)}",
            )
        by_drug = drug_targets.get("by_drug")
        if by_drug is None:
            return error_result(
                ctx,
                self.source,
                start,
                "join_cross_indication_paths needs drug-to-target data from retrieve_drug_targets",
            )
        cancers = target_cancers.get("by_target") or {}
        failed_targets = target_cancers.get("failed_targets") or {}
        incomplete = {
            f"drug:{drug_id}": message
            for drug_id, message in (drug_targets.get("failed_drugs") or {}).items()
        }
        records: dict[str, list[dict[str, Any]]] = {drug_id: [] for drug_id in by_drug}
        for disease_id, found in (disease_drugs.get("by_disease") or {}).items():
            listed = {d["drug_id"]: d for d in found.get("drugs") or []}
            for drug_id in by_drug:
                record = listed.get(drug_id)
                records[drug_id].append(
                    {
                        "disease_id": disease_id,
                        "disease_name": found.get("disease_name"),
                        "found_in_source": record is not None,
                        "max_clinical_stage": record["max_clinical_stage"] if record else None,
                    }
                )
        drugs, paths, groups = [], [], []
        for drug_id, drug in by_drug.items():
            drugs.append(
                {
                    "drug_id": drug_id,
                    "drug_name": drug.get("drug_name"),
                    "index_disease_records": records[drug_id],
                }
            )
            for target in drug.get("targets") or []:
                target_id = target["target_id"]
                found = cancers.get(target_id)
                if found is None:
                    incomplete.setdefault(
                        f"target:{target_id}",
                        failed_targets.get(target_id, "no disease data for this target"),
                    )
                    continue
                for disease in found["diseases"]:
                    paths.append(
                        {
                            "drug_id": drug_id,
                            "drug_name": drug.get("drug_name"),
                            "target_id": target_id,
                            "target_symbol": target.get("target_symbol"),
                            "disease_id": disease["disease_id"],
                            "disease_name": disease["disease_name"],
                            "overall_association_score": disease["overall_association_score"],
                        }
                    )
                groups.append(
                    {
                        "drug_id": drug_id,
                        "drug_name": drug.get("drug_name"),
                        "target_id": target_id,
                        "target_symbol": target.get("target_symbol"),
                        "other_disease_count": len(found["diseases"]),
                    }
                )
        truncated = bool(target_cancers.get("truncated"))
        if incomplete:
            status = "partial_error"
        elif not paths:
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
                "drugs": drugs,
                "paths": paths,
                "groups": groups,
                "incomplete": incomplete,
                "limitations": [
                    "A path shows that a drug's target is associated with another "
                    "disease; it is not a treatment indication and not evidence that "
                    "the drug works or is approved there.",
                    "Paths are grouped by drug and target; the same disease reached "
                    "through different targets or drugs appears once per path.",
                    "Stages in index_disease_records are Open Targets records, not "
                    "approval conclusions; approval needs an explicit source.",
                ]
                + (
                    ["Each target's disease list is truncated to its top associations."]
                    if truncated
                    else []
                ),
            },
            provides=["cross_indication_paths"],
            total_count=len(paths),
            fetched_count=len(paths),
            truncated=truncated,
            error="; ".join(f"{k}: {v}" for k, v in incomplete.items()) or None,
            elapsed_seconds=time.monotonic() - start,
        )