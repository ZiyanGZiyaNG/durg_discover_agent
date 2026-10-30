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