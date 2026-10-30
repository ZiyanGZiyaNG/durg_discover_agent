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

        diseases = [
            d.get("name") for d in _resolved_for_step(ctx, "disease", "resolved_disease")
        ] or [None]
        targets = [
            t.get("symbol") or t.get("name")
            for t in _resolved_for_step(ctx, "target", "resolved_target")
        ]
        drugs = [
            d.get("name") for d in _resolved_for_step(ctx, "drug", "resolved_drug")
        ]

        queries: dict[str, dict] = {}
        for disease in diseases:
            label_disease = disease or "any disease"
            for symbol in targets:
                queries[f"{symbol} | {label_disease}"] = {
                    "condition": disease,
                    "term": symbol,
                }
            for drug in drugs:
                queries[f"{drug} | {label_disease}"] = {
                    "condition": disease,
                    "intervention": drug,
                }
            if not targets and not drugs and disease:
                queries[disease] = {"condition": disease}
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
        diseases = [
            d.get("name") for d in _resolved_for_step(ctx, "disease", "resolved_disease")
        ] or [None]
        subjects = [
            t.get("symbol") or t.get("name")
            for t in _resolved_for_step(ctx, "target", "resolved_target")
        ] + [
            d.get("name") for d in _resolved_for_step(ctx, "drug", "resolved_drug")
        ]
        queries = {
            f"{subject} | {disease or 'any disease'}": (subject, disease)
            for subject in subjects
            if subject
            for disease in diseases
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