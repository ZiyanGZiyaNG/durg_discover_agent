"""Adapters for ClinicalTrials.gov and Europe PMC."""

from __future__ import annotations

import time

from execution.common import _resolved_for_step, error_result
from execution.contracts import StepContext, StepResult
from execution.more_adapters import _filter_values
from query_clinicaltrials import search_trials
from query_europepmc import search_negative_literature
from source_http import SourceError

MAX_QUERIES = 20
MAX_TRIALS_PER_QUERY = 50


class RetrieveClinicalTrialsAdapter:
    """Trials from ClinicalTrials.gov for the step's disease, target and drug entities."""

    source = "clinicaltrials_gov"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        unsupported = sorted(set(ctx.filters) - {"trial_status"})
        if unsupported:
            return error_result(
                ctx, self.source, start, f"unsupported filter(s): {unsupported}"
            )
        try:
            statuses = _filter_values(ctx.filters, "trial_status", upper=True)
        except ValueError as exc:
            return error_result(ctx, self.source, start, str(exc))
        status_list = sorted(statuses) if statuses else None

        diseases = _disease_term_groups(ctx) or [None]
        targets = [
            t.get("symbol") or t.get("name")
            for t in _resolved_for_step(ctx, "target", "resolved_target")
        ]
        drugs = [
            d.get("name") for d in _resolved_for_step(ctx, "drug", "resolved_drug")
        ]

        queries: dict[str, dict] = {}
        for terms in diseases:
            label_disease = terms[0] if terms else "any disease"
            condition = _or_join(terms)
            for symbol in targets:
                queries[f"{symbol} | {label_disease}"] = {
                    "condition": condition,
                    "term": symbol,
                }
            for drug in drugs:
                queries[f"{drug} | {label_disease}"] = {
                    "condition": condition,
                    "intervention": drug,
                }
            if not targets and not drugs and condition:
                queries[label_disease] = {"condition": condition}
        if not queries:
            return error_result(
                ctx,
                self.source,
                start,
                "retrieve_clinical_trials needs at least one resolved disease, target or drug",
            )
        if len(queries) > MAX_QUERIES:
            return error_result(
                ctx, self.source, start, f"too many queries ({len(queries)} > {MAX_QUERIES})"
            )

        results: dict[str, dict] = {}
        failed: dict[str, str] = {}
        for label, kwargs in queries.items():
            try:
                results[label] = search_trials(
                    statuses=status_list, max_results=MAX_TRIALS_PER_QUERY, **kwargs
                )
            except (SourceError, ValueError) as exc:
                failed[label] = str(exc)
        if not results:
            return error_result(
                ctx,
                self.source,
                start,
                "; ".join(f"{label}: {msg}" for label, msg in failed.items()),
            )

        total = sum(r["total_count"] or 0 for r in results.values())
        fetched = sum(r["returned_count"] for r in results.values())
        truncated = any(r["truncated"] for r in results.values())
        if failed:
            status = "partial_error"
        elif not fetched:
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
                "queries": results,
                "failed_queries": failed,
                "status_filter": status_list or [],
                "truncated": truncated,
                "limitations": [
                    "Trials come directly from the ClinicalTrials.gov API v2; each "
                    "record carries its NCT ID and URL.",
                    "A target symbol or drug name is matched as free text, so a listed "
                    "trial does not necessarily test that target or drug; finding no "
                    "trial is not proof that none exists.",
                ],
            },
            provides=["clinical_trial_evidence"],
            total_count=total,
            fetched_count=fetched,
            truncated=truncated,
            elapsed_seconds=time.monotonic() - start,
        )


def _disease_term_groups(ctx: StepContext) -> list[list[str]]:
    """每個已解析的疾病一組搜尋詞：標準名在前，加上使用者原話（僅限純英文）。"""
    resolved = ctx.board.get("resolved_disease", {})
    refs = [
        ref for ref, entity in ctx.entities.items() if entity.get("entity_type") == "disease"
    ] or list(resolved)
    groups = []
    for ref in refs:
        entry = resolved.get(ref)
        if not entry:
            continue
        mention = ((ctx.entities.get(ref) or {}).get("mention") or "").strip()
        candidates = [entry.get("name")]
        if mention and mention.isascii() and not set('()"') & set(mention):
            candidates.append(mention)
        terms: list[str] = []
        for name in candidates:
            if name and name.lower() not in [t.lower() for t in terms]:
                terms.append(name)
        if terms:
            groups.append(terms)
    return groups


def _or_join(terms: list[str] | None) -> str | None:
    """ClinicalTrials.gov 的 OR 寫法：(A) OR (B)，不加引號以保留同義詞展開。"""
    if not terms:
        return None
    if len(terms) == 1:
        return terms[0]
    return " OR ".join(f"({t})" for t in terms)

class RetrieveTrialStatusAndStopReasonsAdapter(RetrieveClinicalTrialsAdapter):
    """Trial status plus the stop reason exactly as recorded; none is inferred."""

    def run(self, ctx: StepContext) -> StepResult:
        result = super().run(ctx)
        if not isinstance(result.payload, dict):
            return result
        for query in result.payload["queries"].values():
            for trial in query["items"]:
                reason = trial.get("why_stopped")
                trial["stop_reason_recorded"] = bool(reason and reason.strip())
        result.payload["limitations"].append(
            "why_stopped is quoted as recorded on ClinicalTrials.gov; "
            "stop_reason_recorded is false when none exists and no reason is inferred."
        )
        return result.model_copy(update={"provides": []})


MAX_ARTICLES_PER_QUERY = 25


class RetrieveNegativePublicationsAdapter:
    """Europe PMC records mentioning a target or drug together with negative-outcome terms."""

    source = "europe_pmc"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        if ctx.filters:
            return error_result(
                ctx, self.source, start, f"unsupported filter(s): {sorted(ctx.filters)}"
            )
        diseases = _disease_term_groups(ctx) or [None]
        subjects = [
            t.get("symbol") or t.get("name")
            for t in _resolved_for_step(ctx, "target", "resolved_target")
        ] + [
            d.get("name") for d in _resolved_for_step(ctx, "drug", "resolved_drug")
        ]
        queries = {
            f"{subject} | {terms[0] if terms else 'any disease'}": (subject, terms)
            for subject in subjects
            if subject
            for terms in diseases
        }
        if not queries:
            return error_result(
                ctx,
                self.source,
                start,
                "retrieve_negative_publications needs at least one resolved target or drug",
            )
        if len(queries) > MAX_QUERIES:
            return error_result(
                ctx, self.source, start, f"too many queries ({len(queries)} > {MAX_QUERIES})"
            )

        results: dict[str, dict] = {}
        failed: dict[str, str] = {}
        for label, (subject, disease) in queries.items():
            try:
                results[label] = search_negative_literature(
                    subject,
                    disease,
                    max_results=MAX_ARTICLES_PER_QUERY,
                    include_abstract=True,
                )
            except (SourceError, ValueError) as exc:
                failed[label] = str(exc)
        if not results:
            return error_result(
                ctx,
                self.source,
                start,
                "; ".join(f"{label}: {msg}" for label, msg in failed.items()),
            )

        total = sum(r["total_count"] or 0 for r in results.values())
        fetched = sum(r["returned_count"] for r in results.values())
        truncated = any(r["truncated"] for r in results.values())
        if failed:
            status = "partial_error"
        elif not fetched:
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
                "queries": results,
                "failed_queries": failed,
                "truncated": truncated,
                "limitations": [
                    "Candidates are Europe PMC (PubMed) records whose title or abstract "
                    "matched the subject and negative-outcome keywords; a match does not "
                    "confirm that the paper reports a negative result.",
                    "Finding no matching publication is not evidence of efficacy or safety.",
                ],
            },
            total_count=total,
            fetched_count=fetched,
            truncated=truncated,
            elapsed_seconds=time.monotonic() - start,
        )


COMBINATION_TYPES = {"DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT"}
COMPARATOR_ARM_TYPES = {
    "ACTIVE_COMPARATOR",
    "PLACEBO_COMPARATOR",
    "SHAM_COMPARATOR",
    "NO_INTERVENTION",
}


def _trial_payloads(ctx: StepContext) -> list[dict]:
    payloads = [
        p for p in ctx.inputs.values() if isinstance(p, dict) and "queries" in p
    ]
    if not payloads:
        board_payload = ctx.board.get("clinical_trial_evidence")
        if isinstance(board_payload, dict) and "queries" in board_payload:
            payloads = [board_payload]
    return payloads


def _intervention_roles(trial: dict) -> list[dict]:
    arm_types = {g.get("label"): g.get("type") for g in trial.get("arm_groups") or []}
    items = []
    for item in trial.get("interventions") or []:
        labels = item.get("arm_group_labels") or []
        types = {arm_types.get(label) for label in labels} - {None}
        if "EXPERIMENTAL" in types:
            role = "experimental"
        elif types and types <= COMPARATOR_ARM_TYPES:
            role = "comparator"
        else:
            role = "unspecified"
        items.append(
            {
                "name": item.get("name"),
                "type": item.get("type"),
                "role": role,
                "arm_group_labels": list(labels),
            }
        )
    return items


def _combination_arms(trial: dict) -> list[str] | None:
    interventions = trial.get("interventions") or []
    if not any(i.get("arm_group_labels") for i in interventions):
        return None
    counts: dict[str, int] = {}
    for item in interventions:
        if item.get("type") not in COMBINATION_TYPES:
            continue
        for label in item.get("arm_group_labels") or []:
            counts[label] = counts.get(label, 0) + 1
    return sorted(label for label, n in counts.items() if n >= 2)


class RetrieveInterventionsAdapter:
    """Interventions, combinations and arm roles extracted from retrieved trial records."""

    source = "clinicaltrials_gov"

    def run(self, ctx: StepContext) -> StepResult:
        start = time.monotonic()
        if ctx.filters:
            return error_result(
                ctx, self.source, start, f"unsupported filter(s): {sorted(ctx.filters)}"
            )
        payloads = _trial_payloads(ctx)
        if not payloads:
            return error_result(
                ctx,
                self.source,
                start,
                "retrieve_interventions needs trial records from retrieve_clinical_trials",
            )

        trials: dict[str, dict] = {}
        truncated = False
        for payload in payloads:
            truncated = truncated or bool(payload.get("truncated"))
            for label, query in payload["queries"].items():
                for trial in query["items"]:
                    nct_id = trial.get("nct_id")
                    if not nct_id:
                        continue
                    entry = trials.get(nct_id)
                    if entry is None:
                        arms = _combination_arms(trial)
                        trials[nct_id] = {
                            "nct_id": nct_id,
                            "title": trial.get("title"),
                            "status": trial.get("status"),
                            "phases": trial.get("phases") or [],
                            "interventions": _intervention_roles(trial),
                            "combination_arms": arms or [],
                            "is_combination": None if arms is None else bool(arms),
                            "query_labels": [label],
                        }
                    elif label not in entry["query_labels"]:
                        entry["query_labels"].append(label)

        rows = list(trials.values())
        if not rows:
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
                "trial_count": len(rows),
                "trials": rows,
                "truncated": truncated,
                "limitations": [
                    "Interventions come from the retrieved ClinicalTrials.gov records only; "
                    "trials that were truncated or not retrieved are not covered.",
                    "role is derived from the recorded arm group types; it is "
                    "'unspecified' when the record links no arm. is_combination is null "
                    "when no arm information is recorded.",
                ],
            },
            total_count=len(rows),
            fetched_count=len(rows),
            truncated=truncated,
            elapsed_seconds=time.monotonic() - start,
        )