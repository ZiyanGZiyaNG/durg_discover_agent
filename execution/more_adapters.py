"""Adapters that gather for several entities at once, or combine data that
earlier steps already collected.

Nothing here makes up a score: values come from Open Targets as given, a
missing value stays missing (never zero), and each combined table says what it
covers.
"""

from __future__ import annotations

import time
from typing import Any

from execution.analysis import rank_rows
from execution.common import (
    _collect_per_target,
    _resolved_for_step,
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
)


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
