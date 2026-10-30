"""Local analysis adapters over trial, literature and evidence records."""

from __future__ import annotations

import time

from execution.common import error_result
from execution.contracts import StepContext, StepResult

ROW_KEYS = ("records", "rows", "targets", "items", "trials")


def _rows_of(payload) -> list[dict] | None:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = next(
            (payload[key] for key in ROW_KEYS if isinstance(payload.get(key), list)),
            None,
        )
    else:
        return None
    if rows and all(isinstance(row, dict) for row in rows):
        return rows
    return None


class AggregateCategoryDistributionAdapter:
    """Count records per category for the requested fields."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        spec = (ctx.aggregation or {}).get("category_distribution") or {}
        metrics = spec.get("metrics")
        if (
            not isinstance(metrics, list)
            or not metrics
            or not all(isinstance(m, str) for m in metrics)
        ):
            return error_result(
                ctx,
                self.source,
                start,
                "aggregate_category_distribution needs "
                "aggregation.category_distribution.metrics (a list of field names)",
            )

        rows = None
        latest_fields: list[str] = []
        for candidate in list(ctx.inputs.values()) + list(reversed(list(ctx.board.values()))):
            found = _rows_of(candidate)
            if not found:
                continue
            if not latest_fields:
                latest_fields = sorted(found[0])
            if all(metric in found[0] for metric in metrics):
                rows = found
                break
        if rows is None:
            return error_result(
                ctx,
                self.source,
                start,
                f"no upstream records contain all of the fields {metrics}; "
                f"fields of the latest upstream records: {latest_fields}",
            )

        total = len(rows)
        distributions: dict[str, list[dict]] = {}
        for metric in metrics:
            counts: dict[str, int] = {}
            for row in rows:
                value = row.get(metric)
                values = value if isinstance(value, list) else [value]
                categories = {
                    "(missing)" if v is None else str(v) for v in values
                } or {"(missing)"}
                for category in categories:
                    counts[category] = counts.get(category, 0) + 1
            distributions[metric] = [
                {
                    "category": category,
                    "category_count": n,
                    "category_fraction": round(n / total, 4),
                }
                for category, n in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            ]

        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success",
            source=self.source,
            payload={
                "source_row_count": total,
                "metrics": metrics,
                "distributions": distributions,
                "limitations": [
                    "Counts cover the records of the nearest upstream result that contains "
                    "all requested fields; list-valued fields count each element once, so "
                    "fractions can add up to more than 1.",
                    "A category count describes the retrieved records only; it is not an "
                    "evidence level.",
                ],
            },
            provides=["category_distribution"],
            total_count=total,
            fetched_count=total,
            elapsed_seconds=time.monotonic() - start,
        )


class RankMechanismCategoriesAdapter:
    """Rank the categories of a finished distribution by category_count."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        distribution = next(
            (
                p
                for p in ctx.inputs.values()
                if isinstance(p, dict) and "distributions" in p
            ),
            None,
        )
        if distribution is None:
            candidate = ctx.board.get("category_distribution")
            if isinstance(candidate, dict) and "distributions" in candidate:
                distribution = candidate
        if distribution is None:
            return error_result(
                ctx,
                self.source,
                start,
                "rank_mechanism_categories needs a category distribution "
                "from aggregate_category_distribution",
            )

        rankings: dict[str, list[dict]] = {}
        for metric, rows in distribution["distributions"].items():
            ordered = sorted(rows, key=lambda r: (-r["category_count"], r["category"]))
            rankings[metric] = [
                {"rank": position, **row} for position, row in enumerate(ordered, start=1)
            ]
        total = sum(len(rows) for rows in rankings.values())

        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success" if total else "no_records",
            source=self.source,
            payload={
                "ranked_by": "category_count",
                "rankings": rankings,
                "limitations": [
                    "Categories with equal counts are ordered alphabetically; the order "
                    "between them carries no meaning.",
                ],
            },
            total_count=total,
            fetched_count=total,
            elapsed_seconds=time.monotonic() - start,
        )

MECHANISM_RULE_VERSION = "mechanism_category_rules_v1"


def _mechanism_category(action_type: str | None) -> str:
    text = (action_type or "").upper()
    if not text:
        return "unspecified"
    if "INHIBITOR" in text:
        return "inhibition"
    if "DEGRADER" in text:
        return "degradation"
    if "ANTAGONIST" in text or "BLOCKER" in text or "INVERSE AGONIST" in text:
        return "antagonism_or_blockade"
    if "AGONIST" in text or "ACTIVATOR" in text or "OPENER" in text:
        return "activation"
    if "MODULATOR" in text:
        return "modulation"
    return "other"


class ClassifyMechanismCategoriesAdapter:
    """Rule-based mechanism categories per drug; one drug can carry several categories."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        by_drug = None
        for candidate in list(ctx.inputs.values()) + list(reversed(list(ctx.board.values()))):
            if not isinstance(candidate, dict):
                continue
            drugs = candidate.get("by_drug")
            if (
                isinstance(drugs, dict)
                and drugs
                and all(isinstance(d, dict) and "mechanisms" in d for d in drugs.values())
            ):
                by_drug = drugs
                break
        if by_drug is None:
            return error_result(
                ctx,
                self.source,
                start,
                "classify_mechanism_categories needs drug mechanism records "
                "(from retrieve_mechanisms or retrieve_drug_targets)",
            )

        records = []
        for drug_id, drug in by_drug.items():
            mechanisms = drug.get("mechanisms") or []
            categories = sorted(
                {_mechanism_category(m.get("action_type")) for m in mechanisms}
            )
            records.append(
                {
                    "drug_id": drug.get("drug_id") or drug_id,
                    "drug_name": drug.get("drug_name"),
                    "modality": drug.get("modality") or "unknown",
                    "mechanism_categories": categories or ["unspecified"],
                    "action_types": sorted(
                        {m.get("action_type") for m in mechanisms if m.get("action_type")}
                    ),
                    "mechanism_count": len(mechanisms),
                    "multi_label": len(categories) > 1,
                }
            )

        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success",
            source=self.source,
            payload={
                "rule_version": MECHANISM_RULE_VERSION,
                "records": records,
                "limitations": [
                    "Categories come from a fixed keyword rule on the Open Targets "
                    "action_type (see rule_version); they describe the recorded "
                    "mechanism, not efficacy.",
                    "A drug with several recorded mechanisms keeps every category "
                    "(multi-label), so category counts can exceed the number of drugs.",
                    "modality is 'unknown' unless modality evidence was retrieved.",
                ],
            },
            total_count=len(records),
            fetched_count=len(records),
            elapsed_seconds=time.monotonic() - start,
        )