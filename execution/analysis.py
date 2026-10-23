from __future__ import annotations
from operator import itemgetter
from typing import Any


def rank_rows(
    rows: list[dict[str, Any]],
    primary_metric: str,
    direction: str,
    secondary_metrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    if direction not in {"ascending", "descending"}:
        raise ValueError(f"unknown direction: {direction!r}")
    metrics = [primary_metric, *(secondary_metrics or [])]
    for metric in metrics:
        if rows and not any(metric in row for row in rows):
            raise ValueError(f"metric not available in data: {metric}")
    ordered = list(rows)
    for metric in reversed(metrics):
        present = [row for row in ordered if row.get(metric) is not None]
        missing = [row for row in ordered if row.get(metric) is None]
        present.sort(key=itemgetter(metric), reverse=direction == "descending")
        ordered = present + missing

    return [
        {**row, "ranked_position": position}
        for position, row in enumerate(ordered, start=1)
    ]
