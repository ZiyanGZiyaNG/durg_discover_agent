"""Local analysis adapters over trial, literature and evidence records."""

from __future__ import annotations

import time
import re

from execution.common import _resolved_for_step, error_result
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
            provides=["mechanism_classification"],
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

STOP_STATUSES = {"TERMINATED", "WITHDRAWN", "SUSPENDED"}
COMBINATION_DRUG_TYPES = {"DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT"}
TERMINATION_RULE_VERSION = "termination_reason_rules_v1"
SAFETY_TERMS = (
    "safety",
    "toxic",
    "adverse",
    "side effect",
    "tolerab",
    "dose-limiting",
    "dose limiting",
)
EFFICACY_TERMS = (
    "efficacy",
    "futility",
    "futile",
    "limited anti",
    "limited activity",
    "lack of activity",
    "lack of response",
    "no clinical benefit",
    "no benefit",
    "ineffective",
    "insufficient activity",
    "insufficient clinical",
    "primary endpoint not met",
    "did not meet the primary",
    "failed to meet the primary",
)
_CONCERN = r"\b(?:safety|efficacy|toxicity|adverse)\w*"
_NEGATED_CONCERN = re.compile(
    r"\b(?:not|nor)\s+(?:be\s+)?(?:based\s+on|due\s+to|related\s+to|because\s+of"
    r"|a\s+result\s+of|driven\s+by|caused\s+by)\b[^.;]{0,60}" + _CONCERN
    + r"|\b(?:unrelated\s+to|independent\s+of)\b[^.;]{0,60}" + _CONCERN
    + r"|\bno\b(?:\s+\w+){0,2}\s+(?:safety|toxicity)\s+(?:concern|issue|signal|problem)s?",
    re.IGNORECASE,
)


def _find_payload(ctx: StepContext, predicate):
    for candidate in list(ctx.inputs.values()) + list(reversed(list(ctx.board.values()))):
        if isinstance(candidate, dict) and predicate(candidate):
            return candidate
    return None


def _records_with_key(node, key):
    if isinstance(node, dict):
        if key in node and not isinstance(node[key], (dict, list)):
            yield node
        else:
            for value in node.values():
                yield from _records_with_key(value, key)
    elif isinstance(node, list):
        for item in node:
            yield from _records_with_key(item, key)


def _classify_reason(text) -> tuple[list[str], list[str]]:
    reason = str(text).strip() if text is not None else ""
    if not reason:
        return ["no_reason_recorded"], []
    cleaned = _NEGATED_CONCERN.sub(" ", reason).lower()
    safety = [term for term in SAFETY_TERMS if term in cleaned]
    efficacy = [term for term in EFFICACY_TERMS if term in cleaned]
    categories = []
    if safety:
        categories.append("safety")
    if efficacy:
        categories.append("efficacy")
    return (categories or ["other"]), safety + efficacy


def _drug_names(record: dict) -> list[str]:
    names = []
    for drug in record.get("drugs") or []:
        if not isinstance(drug, dict):
            continue
        name = (
            drug.get("drug_name")
            or drug.get("name")
            or drug.get("drug_id")
            or drug.get("id")
            or next((v for v in drug.values() if isinstance(v, str)), None)
        )
        if name:
            names.append(str(name))
    for item in record.get("interventions") or []:
        if item.get("type") in COMBINATION_DRUG_TYPES and item.get("name"):
            names.append(str(item["name"]))
    return sorted(set(names))


class ClassifyTerminationReasonsAdapter:
    """Rule-based safety / efficacy / other classification of recorded stop reasons."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        payloads = list(ctx.inputs.values()) or [
            p
            for p in (
                ctx.board.get("trial_termination_evidence"),
                ctx.board.get("clinical_trial_evidence"),
            )
            if p
        ]
        if not payloads:
            return error_result(
                ctx,
                self.source,
                start,
                "classify_termination_reasons needs trial termination or trial status records",
            )

        seen: set[tuple[str, str]] = set()
        records = []
        for payload in payloads:
            for rec in _records_with_key(payload, "why_stopped"):
                record_id = rec.get("report_id") or rec.get("nct_id")
                origin = "open_targets" if "report_id" in rec else "clinicaltrials_gov"
                status = rec.get("overall_status") or rec.get("status")
                reason = rec.get("why_stopped")
                stopped = str(status or "").upper() in STOP_STATUSES
                if not (stopped or (reason and str(reason).strip())):
                    continue
                if record_id:
                    if (origin, record_id) in seen:
                        continue
                    seen.add((origin, record_id))
                categories, matched = _classify_reason(reason)
                records.append(
                    {
                        "record_id": record_id,
                        "origin": origin,
                        "status": status,
                        "drug_names": _drug_names(rec),
                        "why_stopped": reason,
                        "categories": categories,
                        "matched_terms": matched,
                        "source_stop_reason_categories": list(
                            rec.get("stop_reason_categories") or []
                        ),
                    }
                )

        counts: dict[str, int] = {}
        for rec in records:
            for category in rec["categories"]:
                counts[category] = counts.get(category, 0) + 1

        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success" if records else "no_records",
            source=self.source,
            payload={
                "rule_version": TERMINATION_RULE_VERSION,
                "records": records,
                "category_counts": counts,
                "limitations": [
                    "Categories come from a fixed keyword rule on the recorded stop "
                    "reason text (see rule_version); matched_terms shows what triggered "
                    "each label. A reason that mentions both safety and efficacy keeps both.",
                    "A record without a stop reason is labelled no_reason_recorded; no "
                    "reason is inferred, and a missing reason is not evidence of failure.",
                    "Only records that are terminated, withdrawn or suspended, or that "
                    "carry a stop reason, are included.",
                ],
            },
            provides=["termination_classification"],
            total_count=len(records),
            fetched_count=len(records),
            elapsed_seconds=time.monotonic() - start,
        )


class AggregateFailedProgramsAdapter:
    """Group classified termination records by drug program."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        classification = _find_payload(
            ctx,
            lambda p: p.get("rule_version") == TERMINATION_RULE_VERSION
            and isinstance(p.get("records"), list),
        )
        if classification is None:
            return error_result(
                ctx,
                self.source,
                start,
                "aggregate_failed_programs needs the output of classify_termination_reasons",
            )

        programs: dict[str, dict] = {}
        for rec in classification["records"]:
            for name in rec.get("drug_names") or ["(no drug recorded)"]:
                program = programs.setdefault(
                    name,
                    {
                        "program": name,
                        "terminated_record_count": 0,
                        "category_counts": {},
                        "record_ids": [],
                    },
                )
                program["terminated_record_count"] += 1
                for category in rec["categories"]:
                    program["category_counts"][category] = (
                        program["category_counts"].get(category, 0) + 1
                    )
                if rec.get("record_id"):
                    program["record_ids"].append(rec["record_id"])

        rows = sorted(
            programs.values(),
            key=lambda p: (-p["terminated_record_count"], p["program"]),
        )
        for row in rows:
            counts = row["category_counts"]
            row["has_recorded_safety_or_efficacy_reason"] = bool(
                counts.get("safety") or counts.get("efficacy")
            )

        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status="success" if rows else "no_records",
            source=self.source,
            payload={
                "program_count": len(rows),
                "record_count": len(classification["records"]),
                "programs": rows,
                "limitations": [
                    "Only programs with at least one recorded termination are listed; a "
                    "termination record is not proof that the program failed, and a "
                    "program without records is not evidence of success.",
                    "A record naming several drugs counts toward each of them, so program "
                    "counts can exceed the number of records.",
                ],
            },
            total_count=len(rows),
            fetched_count=len(rows),
            elapsed_seconds=time.monotonic() - start,
        )

MIN_STAGE_RANK = 2


def _stage_rank(stage) -> int | None:
    text = str(stage or "").upper()
    if not text:
        return None
    if text.startswith("APPROV"):
        return 5
    if text.startswith("PREAPPROV"):
        return 4
    numbers = [int(n) for n in re.findall(r"\d+", text)]
    if "PHASE" in text and numbers:
        return max(numbers)
    return None


def _is_approved(stage) -> bool:
    return str(stage or "").upper().startswith("APPROV")


def _is_clinical_stage_payload(payload: dict) -> bool:
    by_target = payload.get("by_target")
    return (
        isinstance(by_target, dict)
        and bool(by_target)
        and all(isinstance(v, dict) and "stages_present" in v for v in by_target.values())
    )


def _is_approval_payload(payload: dict) -> bool:
    by_drug = payload.get("by_drug")
    return (
        isinstance(by_drug, dict)
        and bool(by_drug)
        and all(isinstance(v, dict) and "indications" in v for v in by_drug.values())
    )


class FilterPhase2NotApprovedAdapter:
    """Keep drug-target pairs at Phase 2 or later that are not confirmed approved
    for the step's disease."""

    source = "local_analysis"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        diseases = _resolved_for_step(ctx, "disease", "resolved_disease")
        if len(diseases) != 1:
            return error_result(
                ctx,
                self.source,
                start,
                "this filter needs exactly one resolved disease",
            )
        disease = diseases[0]
        stage_evidence = _find_payload(ctx, _is_clinical_stage_payload)
        approval = _find_payload(ctx, _is_approval_payload)
        if stage_evidence is None:
            return error_result(
                ctx, self.source, start, "this filter needs clinical stage evidence "
                "(retrieve_clinical_stage)",
            )
        if approval is None:
            return error_result(
                ctx, self.source, start, "this filter needs approved-indication evidence "
                "(retrieve_approved_indications)",
            )

        approved: set[str] = set()
        names: dict[str, str | None] = {}
        for drug_id, drug in approval["by_drug"].items():
            names[drug_id] = drug.get("drug_name")
            for row in drug.get("indications") or []:
                if row.get("disease_id") == disease.get("id") and _is_approved(
                    row.get("max_clinical_stage")
                ):
                    approved.add(drug_id)

        records: list[dict] = []
        excluded: list[dict] = []
        for target_id, target in stage_evidence["by_target"].items():
            for drug in target.get("drugs") or []:
                drug_id = drug.get("drug_id")
                stage = drug.get("max_clinical_stage")
                rank = _stage_rank(stage)
                if rank is None:
                    reason = "stage_unparsed"
                elif rank < MIN_STAGE_RANK:
                    reason = "below_phase_2"
                elif drug_id in approved:
                    reason = "approved_for_disease"
                else:
                    reason = None
                if reason:
                    excluded.append(
                        {
                            "target_id": target_id,
                            "drug_id": drug_id,
                            "max_clinical_stage": stage,
                            "reason": reason,
                        }
                    )
                    continue
                records.append(
                    {
                        "target_id": target_id,
                        "drug_id": drug_id,
                        "drug_name": names.get(drug_id),
                        "max_clinical_stage": stage,
                        "stage_rank": rank,
                        "approval_checked": drug_id in approval["by_drug"],
                    }
                )

        failed = {
            "clinical_stage": dict(stage_evidence.get("failed_targets") or {}),
            "approval": dict(approval.get("failed_drugs") or {}),
        }
        if any(failed.values()):
            status = "partial_error"
        elif not records:
            status = "no_records"
        else:
            status = "success"

        return StepResult(
            step_id=ctx.step_id,
            operation=ctx.operation,
            status=status,
            source=self.source,
            payload={
                "disease_id": disease.get("id"),
                "disease_name": disease.get("name"),
                "min_stage_rank": MIN_STAGE_RANK,
                "records": records,
                "kept_target_ids": sorted({r["target_id"] for r in records}),
                "excluded": excluded,
                "failed": failed,
                "limitations": [
                    "Phase 2 or later is judged from each drug's recorded maximum "
                    "clinical stage in the clinical stage evidence.",
                    "Approval is checked by exact disease ID in the retrieved approved "
                    "indications; approvals recorded under a broader or narrower disease "
                    "term are not detected (see check_indication).",
                    "approval_checked is false when the drug has no approved-indication "
                    "record; such pairs are kept because approval was not confirmed, not "
                    "because it was ruled out.",
                    "Pairs whose stage cannot be read as a phase are excluded as "
                    "stage_unparsed.",
                ],
            },
            total_count=len(records),
            fetched_count=len(records),
            elapsed_seconds=time.monotonic() - start,
        )