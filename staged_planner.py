"""Experimental staged Planner: strategy -> table matching -> Python compile."""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from langchain.agents import create_agent
from pydantic import BaseModel, Field, model_validator

from model_factory import get_chat_model
from natural_language_planner import (
    ModelAttemptTimeout,
    _cleanup_ollama_after_timeout,
    _invoke_with_deadline,
    _normalize_unverified_entities,
)
from planner_acceptance import (
    OPERATION_TABLE,
    PlanAcceptanceResult,
    PlannerAcceptanceCase,
    build_acceptance_rubric,
    evaluate_plan,
)
from schemas import (
    PlannedEntityV2,
    QueryCharacteristicsV2,
    QueryManifestV2,
    RankingSpecV2,
    ResearchPlanStepV2,
    StepInputRef,
)


STRATEGY_PROMPT = """
You are Agent 1's research-strategy planner. Do not retrieve evidence and do not
name or guess software operations. Decompose the question into atomic evidence
and analysis needs. Each need must represent one research action, declare the
data it requires, the output it produces, and prior needs it depends on.

For therapeutic-target questions, keep disease association separate from
translational support: known target-drug evidence and tractability evidence must
be collected before results are stratified. Association is not efficacy;
tractability is not efficacy; missing evidence is unknown rather than negative.

Use stable snake_case English identifiers, input/output data types, safeguards,
filters, ranking metrics and aggregation keys. Use Traditional Chinese only for
purpose descriptions. Do not copy or rewrite the user's question because Python
owns original_question.
"""

MAPPING_PROMPT = """
You match atomic research needs against a human-approved OPERATION_TABLE. For
each need, inspect only its supplied candidates. Select a candidate only when
its meaning covers the need; output its canonical operation, never an alias.
If none is equivalent, return no_equivalent with a proposed snake_case name and
reason. Do not invent an approved operation. Do not modify or omit need IDs.
"""


SourceName = Literal["open_targets", "europe_pmc", "clinicaltrials_gov", "chembl"]


class AtomicResearchNeed(BaseModel):
    need_id: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    operation_search_query: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    entity_refs: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(min_length=1)
    sources: list[SourceName] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    ranking: RankingSpecV2 | None = None
    aggregation: dict[str, Any] | None = None
    safeguards: list[str] = Field(default_factory=list)


class ResearchStrategyDraft(BaseModel):
    research_objective: str = Field(min_length=1)
    entities: list[PlannedEntityV2] = Field(min_length=1)
    needs: list[AtomicResearchNeed] = Field(min_length=1)
    evidence_requirements: list[str] = Field(min_length=1)
    completion_criteria: list[str] = Field(min_length=1)
    global_safeguards: list[str] = Field(min_length=1)
    characteristics: QueryCharacteristicsV2

    @model_validator(mode="after")
    def validate_need_graph(self) -> "ResearchStrategyDraft":
        entity_refs = {entity.entity_ref for entity in self.entities}
        seen: set[str] = set()
        for need in self.needs:
            if need.need_id in seen:
                raise ValueError("need_id values must be unique")
            if set(need.depends_on) - seen:
                raise ValueError(f"need {need.need_id} has non-prior dependencies")
            if set(need.entity_refs) - entity_refs:
                raise ValueError(f"need {need.need_id} has unknown entity_refs")
            seen.add(need.need_id)
        return self


class NeedOperationMatch(BaseModel):
    need_id: str
    match_status: Literal["matched", "no_equivalent"]
    canonical_operation: str | None = None
    proposed_operation_name: str | None = None
    proposal_reason: str | None = None

    @model_validator(mode="after")
    def validate_match(self) -> "NeedOperationMatch":
        if self.match_status == "matched":
            if not self.canonical_operation:
                raise ValueError("matched needs require canonical_operation")
            if self.proposed_operation_name or self.proposal_reason:
                raise ValueError("matched needs cannot include proposal fields")
        else:
            if self.canonical_operation is not None:
                raise ValueError("no_equivalent cannot include canonical_operation")
            if not self.proposed_operation_name or not self.proposal_reason:
                raise ValueError("no_equivalent requires proposal fields")
        return self


class OperationMappingResult(BaseModel):
    matches: list[NeedOperationMatch] = Field(min_length=1)


class StagedPlannerReport(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    planner_mode: Literal["staged"] = "staged"
    question: str
    acceptance_case_id: str | None = None
    status: Literal["running", "passed", "failed", "pending_human_review"]
    strategy: ResearchStrategyDraft | None = None
    operation_candidates: dict[str, list[str]] = Field(default_factory=dict)
    mapping: OperationMappingResult | None = None
    final_plan: QueryManifestV2 | None = None
    semantic_acceptance: PlanAcceptanceResult | None = None
    strategy_duration_seconds: float | None = None
    mapping_duration_seconds: float | None = None
    total_duration_seconds: float
    errors: list[str] = Field(default_factory=list)
    corrected_model_claims: list[str] = Field(default_factory=list)
    timeout_cleanup: dict[str, Any] = Field(default_factory=dict)
    evidence_collection_started: Literal[False] = False
    executed_retrieval_tools: list[str] = Field(default_factory=list)
    generated_at: datetime


def _search_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold().replace("-", "_"))
        if len(token) > 1 and token not in {"and", "for", "the", "per"}
    }


_STOP_WORDS = {"and", "for", "the", "per", "by", "of", "in", "on", "to", "with", "as", "or"}
_SUFFIXES = ("ations", "ation", "ated", "ates", "ate", "ings", "ing", "ies", "ed", "es", "s")


def _stem(token: str) -> str:
    """Crude stemming so disease/diseases and association/associated match."""
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    return token[:-1] if token.endswith("e") and len(token) > 3 else token


def _stem_tokens(text: str) -> set[str]:
    return {
        _stem(token)
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 1 and token not in _STOP_WORDS
    }


@lru_cache(maxsize=1)
def _token_weights() -> dict[str, float]:
    """Inverse-document-frequency weight for each stemmed operation-name token.

    Tokens shared by many operations (disease, target) carry little signal;
    rarer tokens (association, rank) carry more.
    """
    frequency: dict[str, int] = {}
    for canonical, entry in OPERATION_TABLE.items():
        names = [canonical, *entry.equivalent_operations]
        for token in set().union(*(_stem_tokens(name) for name in names)):
            frequency[token] = frequency.get(token, 0) + 1
    total = len(OPERATION_TABLE)
    return {
        token: math.log((total + 1) / (count + 1)) + 1
        for token, count in frequency.items()
    }


def _name_coverage_ranking(query: str) -> list[str]:
    """Rank operations by the weighted share of their own name the query covers.

    The denominator is the operation name, so a long query full of entity and
    database names does not dilute the match, and common name tokens count
    for less than distinctive ones.
    """
    weights = _token_weights()
    query_tokens = _stem_tokens(query)
    scored: list[tuple[float, str]] = []
    for canonical, entry in OPERATION_TABLE.items():
        best = 0.0
        for name in [canonical, *entry.equivalent_operations]:
            name_tokens = _stem_tokens(name)
            total = sum(weights.get(token, 1.0) for token in name_tokens) or 1.0
            matched = sum(
                weights.get(token, 1.0) for token in name_tokens & query_tokens
            )
            best = max(best, matched / total * 10 + matched)
        scored.append((-best, canonical))
    return [canonical for _, canonical in sorted(scored)]


def operation_candidates(
    search_query: str, limit: int = 5, need_id: str = ""
) -> list[str]:
    """Return a deterministic shortlist without deciding the final match.

    Two independent rankings are interleaved so an operation that either one
    places near the top reaches the model: the legacy token score on the
    model's query, and name coverage on the query plus the need ID.
    """
    query_tokens = _search_tokens(search_query)
    normalized_query = "_".join(sorted(query_tokens))
    scored: list[tuple[float, str]] = []
    for canonical, entry in OPERATION_TABLE.items():
        names = [canonical, *entry.equivalent_operations]
        name_tokens = set().union(*(_search_tokens(name) for name in names))
        overlap = len(query_tokens & name_tokens)
        coverage = overlap / max(1, len(query_tokens))
        similarity = max(
            SequenceMatcher(None, normalized_query, name.casefold()).ratio()
            for name in names
        )
        scored.append((coverage * 10 + overlap + similarity, canonical))
    legacy = [canonical for _, canonical in sorted(scored, reverse=True)]
    by_name = _name_coverage_ranking(f"{search_query} {need_id.replace('_', ' ')}")

    merged: list[str] = []
    for first, second in zip(legacy, by_name):
        for name in (first, second):
            if name not in merged:
                merged.append(name)
    return merged[:limit]


def build_mapping_request(
    strategy: ResearchStrategyDraft,
    candidates: dict[str, list[str]],
) -> dict[str, Any]:
    needs = []
    for need in strategy.needs:
        table_entries = []
        for canonical in candidates[need.need_id]:
            entry = OPERATION_TABLE[canonical]
            table_entries.append(
                {
                    "canonical_operation": canonical,
                    "equivalent_operations": entry.equivalent_operations,
                    "meaning": entry.description,
                }
            )
        needs.append(
            {
                "need_id": need.need_id,
                "purpose": need.purpose,
                "required_inputs": need.required_inputs,
                "outputs": need.outputs,
                "candidates": table_entries,
            }
        )
    return {"atomic_needs_with_candidates": needs}

def _derive_inputs(
    depends_on: list[str], steps_by_id: dict[str, ResearchPlanStepV2]
) -> dict[str, StepInputRef]:
    """Reference the first declared output of every upstream step.

    The runtime board stores one payload under every output name of a step,
    so the first output is always a valid reference.
    """
    inputs: dict[str, StepInputRef] = {}
    for step_id in depends_on:
        upstream = steps_by_id[step_id]
        inputs[step_id] = StepInputRef(
            step_id=step_id, output=upstream.outputs[0]
        )
    return inputs



def compile_manifest(
    question: str,
    strategy: ResearchStrategyDraft,
    mapping: OperationMappingResult,
    candidates: dict[str, list[str]],
) -> QueryManifestV2:
    matches = {match.need_id: match for match in mapping.matches}
    expected_ids = {need.need_id for need in strategy.needs}
    if set(matches) != expected_ids or len(mapping.matches) != len(expected_ids):
        raise ValueError("mapping must contain every need_id exactly once")

    steps = []
    steps_by_id: dict[str, ResearchPlanStepV2] = {}
    for need in strategy.needs:
        match = matches[need.need_id]
        if match.match_status == "matched":
            if match.canonical_operation not in candidates[need.need_id]:
                raise ValueError(
                    f"need {need.need_id} selected an operation outside its candidates"
                )
            operation = match.canonical_operation
            operation_status = "approved"
            proposed_name = None
            proposal_reason = None
        else:
            operation = "proposed_new_operation"
            operation_status = "proposed"
            proposed_name = match.proposed_operation_name
            proposal_reason = match.proposal_reason
        step_depends_on = [f"step_{item}" for item in need.depends_on]
        step = ResearchPlanStepV2(
            step_id=f"step_{need.need_id}",
            operation=operation,
            operation_status=operation_status,
            proposed_operation_name=proposed_name,
            proposal_reason=proposal_reason,
            description=need.purpose,
            depends_on=step_depends_on,
            entity_refs=need.entity_refs,
            sources=need.sources,
            outputs=need.outputs,
            filters=need.filters,
            ranking=need.ranking,
            aggregation=need.aggregation,
            safeguards=need.safeguards,
            inputs=_derive_inputs(step_depends_on, steps_by_id),
        )
        steps.append(step)
        steps_by_id[step.step_id] = step

    return QueryManifestV2(
        manifest_id=uuid4(),
        original_question=question,
        research_objective=strategy.research_objective,
        entities=strategy.entities,
        plan_steps=steps,
        evidence_requirements=strategy.evidence_requirements,
        completion_criteria=strategy.completion_criteria,
        global_safeguards=strategy.global_safeguards,
        characteristics=strategy.characteristics,
        generated_at=datetime.now(timezone.utc),
    )


class StagedNaturalLanguagePlanner:
    """Two LLM stages with deterministic table retrieval and compilation."""

    def __init__(self, strategy_agent=None, mapping_agent=None, timeout_cleanup=None):
        owns_agents = strategy_agent is None and mapping_agent is None
        model = None
        if strategy_agent is None or mapping_agent is None:
            model = get_chat_model()
        self.strategy_agent = strategy_agent or create_agent(
            model=model,
            tools=[],
            system_prompt=STRATEGY_PROMPT,
            response_format=ResearchStrategyDraft,
            name="agent1-staged-strategy",
        )
        self.mapping_agent = mapping_agent or create_agent(
            model=model,
            tools=[],
            system_prompt=MAPPING_PROMPT,
            response_format=OperationMappingResult,
            name="agent1-staged-operation-mapper",
        )
        self.timeout_cleanup = timeout_cleanup
        if owns_agents and timeout_cleanup is None:
            if os.getenv("LLM_PROVIDER", "ollama").strip().lower() == "ollama":
                self.timeout_cleanup = _cleanup_ollama_after_timeout

    def run_with_report(
        self,
        question: str,
        acceptance_case: PlannerAcceptanceCase | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> StagedPlannerReport:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        started = perf_counter()
        errors: list[str] = []
        cleanup: dict[str, Any] = {}
        strategy = None
        mapping = None
        plan = None
        acceptance = None
        candidates: dict[str, list[str]] = {}
        corrected: list[str] = []
        strategy_duration = None
        mapping_duration = None
        timeout = float(os.getenv("PLANNER_ATTEMPT_TIMEOUT_SECONDS", "1200"))

        def finish(
            status: Literal["running", "passed", "failed", "pending_human_review"]
        ):
            report = StagedPlannerReport(
                question=question,
                acceptance_case_id=acceptance_case.case_id if acceptance_case else None,
                status=status,
                strategy=strategy,
                operation_candidates=candidates,
                mapping=mapping,
                final_plan=plan if status == "passed" else None,
                semantic_acceptance=acceptance,
                strategy_duration_seconds=strategy_duration,
                mapping_duration_seconds=mapping_duration,
                total_duration_seconds=perf_counter() - started,
                errors=errors,
                corrected_model_claims=corrected,
                timeout_cleanup=cleanup,
                generated_at=datetime.now(timezone.utc),
            )
            if checkpoint_path:
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            return report

        try:
            stage_started = perf_counter()
            result = _invoke_with_deadline(
                self.strategy_agent,
                {"messages": [{"role": "user", "content": question}]},
                timeout,
                self.timeout_cleanup,
            )
            strategy_duration = perf_counter() - stage_started
            payload = result.get("structured_response")
            strategy = ResearchStrategyDraft.model_validate(
                payload.model_dump() if hasattr(payload, "model_dump") else payload
            )
            corrected.extend(_normalize_unverified_entities(strategy))
            candidates = {
                need.need_id: operation_candidates(
                    need.operation_search_query, need_id=need.need_id
                )
                for need in strategy.needs
            }
            # Persist the expensive strategy result before starting mapping so a
            # timeout or forced stop never hides which stage was reached.
            finish("running")
            stage_started = perf_counter()
            request = build_mapping_request(strategy, candidates)
            result = _invoke_with_deadline(
                self.mapping_agent,
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": json.dumps(request, ensure_ascii=False),
                        }
                    ]
                },
                timeout,
                self.timeout_cleanup,
            )
            mapping_duration = perf_counter() - stage_started
            payload = result.get("structured_response")
            mapping = OperationMappingResult.model_validate(
                payload.model_dump() if hasattr(payload, "model_dump") else payload
            )
            finish("running")
            plan = compile_manifest(question, strategy, mapping, candidates)
            proposed = [
                step.proposed_operation_name
                for step in plan.plan_steps
                if step.operation_status == "proposed"
            ]
            if proposed:
                errors.append("proposed operations require human review")
                return finish("pending_human_review")
            if acceptance_case:
                acceptance = evaluate_plan(build_acceptance_rubric(acceptance_case), plan)
                if not acceptance.passed:
                    errors.append("compiled plan failed hidden semantic acceptance")
                    return finish("failed")
            return finish("passed")
        except ModelAttemptTimeout as exc:
            cleanup = exc.cleanup
            errors.append(f"ModelAttemptTimeout: {exc}")
            return finish("failed")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            return finish("failed")
