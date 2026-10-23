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


RULE_VERSION = "provisional_v0"


def translation_layer(
    known_drug_count: int | None, tractable_bucket_count: int | None
) -> str:
    if (known_drug_count or 0) > 0 or (tractable_bucket_count or 0) > 0:
        return "translational_support"
    if known_drug_count is None or tractable_bucket_count is None:
        return "undetermined"
    return "disease_association_only"


def stratify_targets(
    association_rows: list[dict[str, Any]],
    drug_evidence: dict[str, Any],
    tractability_evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    drugs_by_target = drug_evidence.get("by_target", {})
    tractability_by_target = tractability_evidence.get("by_target", {})
    stratified = []
    for row in association_rows:
        target_id = row["target_id"]
        drug = drugs_by_target.get(target_id)
        tractability = tractability_by_target.get(target_id)

        if drug is None:
            known_drug_count = exact_drug_count = drug_truncated = None
        else:
            known_drug_count = drug["total_drug_count"]
            exact_drug_count = sum(1 for d in drug["drugs"] if d["exact_disease_match"])
            drug_truncated = drug["fetched_drug_count"] < drug["total_drug_count"]

        if tractability is None:
            bucket_count = modalities = None
        else:
            positives = [a for a in tractability["assessments"] if a["value"]]
            bucket_count = len(positives)
            modalities = sorted({a["modality"] for a in positives})

        stratified.append(
            {
                "target_id": target_id,
                "target_symbol": row.get("target_symbol", ""),
                "target_name": row.get("target_name", ""),
                "association_rank": row.get("rank"),
                "overall_association_score": row.get("overall_association_score"),
                "drug_evidence": "missing" if drug is None else "available",
                "known_drug_count": known_drug_count,
                "exact_disease_drug_count": exact_drug_count,
                "drug_records_truncated": drug_truncated,
                "tractability_evidence": (
                    "missing" if tractability is None else "available"
                ),
                "tractable_bucket_count": bucket_count,
                "tractable_modalities": modalities,
                "target_translation_layer": translation_layer(
                    known_drug_count, bucket_count
                ),
            }
        )
    return stratified
