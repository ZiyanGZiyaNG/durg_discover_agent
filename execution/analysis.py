from __future__ import annotations
from operator import itemgetter
from typing import Any


APPROVED_STAGE = "APPROVAL"


def classify_indication(
    indications: list[dict[str, Any]],
    disease_id: str,
    ancestor_ids: list[str],
) -> dict[str, Any]:
    """Compare one drug's indication rows with one disease.

    Rows look like {"disease_id", "disease_name", "max_clinical_stage"}.
    An approval recorded on a broader (ancestor) term is reported on its own:
    it does not mean the specific disease is approved.
    """


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
RULE_NOTE = (
    "Layer definitions are provisional and pending research-team approval "
    "(docs/DECISIONS_REQUIRED.md). translational_support means at least one "
    "known-drug record or at least one tractability assessment with value true; "
    "no thresholds or scores are applied. undetermined means a required evidence "
    "source is missing for the target, so absence is not established."
)


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



FILTER_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "contains"}
_OPERATOR_ALIASES = {"min": "gte", "max": "lte"}


def normalize_filters(filters: dict[str, Any]) -> list[tuple[str, str, Any]]:
    """{field: {operator: value}} -> [(field, operator, value)].

    A bare value means equality: "min"/"max" are accepted for gte/lte.
    """

    conditions = []
    for field, condition in filters.items():
        if not isinstance(condition, dict):
            condition = {"eq": condition}
        if not condition:
            raise ValueError(f"filter on {field!r} has no operator")
        for operator, expected in condition.items():
            operator = _OPERATOR_ALIASES.get(operator, operator)
            if operator not in FILTER_OPERATORS:
                raise ValueError(f"unknown filter operator {operator!r} on {field!r}")
            if operator == "in" and not isinstance(expected, list):
                raise ValueError(f"'in' on {field!r} needs a list")
            conditions.append((field, operator, expected))
    return conditions


def _holds(value: Any, operator: str, expected: Any) -> bool:
    if operator == "eq":
        return value == expected
    if operator == "ne":
        return value != expected
    if operator == "in":
        return value in expected
    if operator == "contains":
        return expected in value
    if operator == "gt":
        return value > expected
    if operator == "gte":
        return value >= expected
    if operator == "lt":
        return value < expected
    return value <= expected


def filter_rows(
        rows: list[dict[str, Any]], filters: dict[str, Any]
) -> tuple[list[str], list[str], list[str]]:
    """Return (kept, rejected, unevaluable) target ids.
    
    A row missing any filtered value cannot be judged, so it is set aside
    instead of being counted as a failed condition.
    """
    conditions = normalize_filters(filters)
    fields = {field for field, _, _ in conditions}
    for field in sorted(fields):
        if rows and not any(field in row for row in rows):
            raise ValueError(f"field not available in data: {field}")
    kept, rejected, unevaluable = [], [], []
    for row in rows:
        if any(row.get(field) is None for field in fields):
            unevaluable.append(row["target_id"])
            continue
        try:
            passed = all(
                _holds(row[field], operator, expected)
                for field, operator, expected in conditions
            )
        except TypeError as exc:
            raise ValueError(f"cannot apply filter to {row['target_id']}: {exc}") from None
        (kept if passed else rejected).append(row["target_id"])
    return kept, rejected, unevaluable

def classify_indication(
        indications: list[dict[str, Any]],
        disease_id: str,
        ancestor_ids: list[str],
) -> dict[str, Any]:
    """Compare one drug's indication rows with one disease.
    Rows look like {"disease_id", "disease_name", "max_clinical_stage"}.
    An approval recorded on a broader (ancestor) term is reported on its own:
    it does not mean the specific disease is approved. Ancestors can be very
    broad, so the matched terms are always listed for the reader to judge.
    """
    ancestors = set(ancestor_ids)
    exact = next((row for row in indications if row["disease_id"] == disease_id), None)
    broader = [
        {"disease_id": row["disease_id"], "disease_name": row["disease_name"]}
        for row in indications
        if row["disease_id"] in ancestors and row["max_clinical_stage"] == APPROVED_STAGE
    ]

    if exact is not None and exact["max_clinical_stage"] == APPROVED_STAGE:
        status = "approved_exact"
    elif broader: 
        status = "approved_on_broader_term"
    elif exact is not None:
        status = "studied_not_approved"
    else:
        status = "not_found_in_source"
    return {
        "status": status,
        "exact_stage": exact["max_clinical_stage"] if exact is not None else None,
        "broader_approvals": broader,
    }
