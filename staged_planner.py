"""Experimental staged Planner: strategy -> table matching -> Python compile."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from langchain.agents import create_agent
from pydantic import BaseModel, Field, field_validator, model_validator

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
each need, inspect every operation in the supplied operation_table. Select an
operation only when its meaning covers the need; output its canonical
operation, never an alias. If none is equivalent, return no_equivalent with a
proposed snake_case name and reason. Do not invent an approved operation. Do
not modify or omit need IDs.

Each operation lists the capabilities it requires and provides. When a need
declares provided_capabilities, prefer an operation whose provides covers them.
"""


SourceName = Literal["open_targets", "europe_pmc", "clinicaltrials_gov", "chembl"]


def _capability_vocabulary() -> list[str]:
    """Data-type names that operations require or provide, without resolved_* forms."""
    names: set[str] = set()
    for entry in OPERATION_TABLE.values():
        names.update(entry.provides)
        names.update(entry.requires_all)
        names.update(entry.requires_any)
    return sorted(name for name in names if not name.startswith("resolved_"))


CAPABILITY_NAMES = _capability_vocabulary()
KNOWN_SAFEGUARDS = sorted(
    {name for entry in OPERATION_TABLE.values() for name in entry.required_safeguards}
)
Capability = Literal[tuple(CAPABILITY_NAMES)]

STRATEGY_VOCABULARY_PROMPT = """
Capabilities: for every need, set required_capabilities (evidence it consumes)
and provided_capabilities (evidence it produces), using only these data-type
names: {capabilities}. Entity resolution is handled by the program, so never
declare resolved_* names.

Safeguards: use these names when they apply: {safeguards}. Create a new
snake_case safeguard only when none of them fits.
"""


def build_strategy_prompt() -> str:
    """Stage 1 prompt plus the closed vocabularies taken from OPERATION_TABLE."""
    return STRATEGY_PROMPT + STRATEGY_VOCABULARY_PROMPT.format(
        capabilities=", ".join(CAPABILITY_NAMES),
        safeguards=", ".join(KNOWN_SAFEGUARDS),
    )


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
    required_capabilities: list[Capability] = Field(default_factory=list)
    provided_capabilities: list[Capability] = Field(default_factory=list)

    @field_validator("required_capabilities", "provided_capabilities", mode="before")
    @classmethod
    def drop_unknown_capabilities(cls, value: Any) -> Any:
        """Keep vocabulary names only, so one stray name cannot fail a whole Stage 1 run."""
        if not isinstance(value, list):
            return value
        return [name for name in value if name in CAPABILITY_NAMES]


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
    program_completions: list[str] = Field(default_factory=list)


def build_mapping_request(strategy: ResearchStrategyDraft) -> dict[str, Any]:
    """Build the mapper input: the whole OPERATION_TABLE plus every atomic need.

    There is no shortlist, so the mapper can always see the correct operation.
    """
    operation_table = [
        {
            "canonical_operation": canonical,
            "equivalent_operations": entry.equivalent_operations,
            "meaning": entry.description,
            "requires": [*entry.requires_all, *entry.requires_any],
            "provides": entry.provides,
        }
        for canonical, entry in OPERATION_TABLE.items()
    ]
    needs = [
        {
            "need_id": need.need_id,
            "purpose": need.purpose,
            "required_inputs": need.required_inputs,
            "outputs": need.outputs,
            "required_capabilities": need.required_capabilities,
            "provided_capabilities": need.provided_capabilities,
        }
        for need in strategy.needs
    ]
    return {"operation_table": operation_table, "atomic_needs": needs}


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


def _resolver_operation(capability: str) -> tuple[str, str] | None:
    """Map a ``resolved_<type>`` capability to its single-entity resolver."""
    if not capability.startswith("resolved_"):
        return None
    entity_type = capability.removeprefix("resolved_")
    operation = f"resolve_{entity_type}"
    entry = OPERATION_TABLE.get(operation)
    if entry is None or capability not in entry.provides:
        return None
    return operation, entity_type


def complete_plan_from_table(
    steps: list[ResearchPlanStepV2],
    entities: list[PlannedEntityV2],
    global_safeguards: list[str],
    notes: list[str],
) -> list[ResearchPlanStepV2]:
    """Fill gaps that the operation table itself defines; add no research content.

    Two completions, both derived only from OPERATION_TABLE and both recorded
    in ``notes``:
    - a resolver step for an entity whose resolved form a step requires;
    - safeguards the table lists as required for the step's operation.
    """
    entity_types = {entity.entity_ref: entity.entity_type for entity in entities}
    mentions = {entity.entity_ref: entity.mention for entity in entities}
    taken_ids = {step.step_id for step in steps}
    taken_outputs = {output for step in steps for output in step.outputs}
    resolver_for: dict[str, str] = {}
    for step in steps:
        if step.operation.startswith("resolve_"):
            for ref in step.entity_refs:
                resolver_for.setdefault(ref, step.step_id)

    provided: set[str] = set()
    position = {step.step_id: index for index, step in enumerate(steps)}
    inserted: list[ResearchPlanStepV2] = []
    for step in steps:
        entry = OPERATION_TABLE.get(step.operation)
        if step.operation_status != "approved" or entry is None:
            continue
        needed = [cap for cap in entry.requires_all if cap not in provided]
        if entry.requires_any and not any(cap in provided for cap in entry.requires_any):
            needed.append(next(
                (cap for cap in entry.requires_any if _resolver_operation(cap)),
                entry.requires_any[0],
            ))
        for capability in needed:
            resolver = _resolver_operation(capability)
            if resolver is None:
                continue
            operation, entity_type = resolver
            for ref in step.entity_refs:
                if entity_types.get(ref) != entity_type:
                    continue
                if ref not in resolver_for:
                    slug = re.sub(r"[^a-z0-9]+", "_", ref.casefold()).strip("_") or "entity"
                    step_id, output = f"step_resolve_{slug}", f"resolved_{slug}"
                    while step_id in taken_ids:
                        step_id += "_x"
                    while output in taken_outputs:
                        output += "_x"
                    new_step = ResearchPlanStepV2(
                        step_id=step_id,
                        operation=operation,
                        description=(
                            f"Resolve {mentions[ref]} to a canonical identifier "
                            "(inserted by program from OPERATION_TABLE prerequisites)"
                        ),
                        entity_refs=[ref],
                        sources=["open_targets"],
                        outputs=[output],
                    )
                    inserted.append(new_step)
                    taken_ids.add(step_id)
                    taken_outputs.add(output)
                    resolver_for[ref] = step_id
                    provided.update(OPERATION_TABLE[operation].provides)
                    notes.append(
                        f"inserted {step_id} ({operation}) for {ref}: "
                        f"{step.step_id} ({step.operation}) requires {capability}"
                    )
                if resolver_for[ref] not in step.depends_on:
                    step.depends_on.append(resolver_for[ref])
              # Wire a required resolved_<type> to the resolver step already in the plan.
        for capability in entry.requires_all:
            resolver = _resolver_operation(capability)
            if resolver is None:
                continue
            for ref in step.entity_refs:
                provider = resolver_for.get(ref)
                if (
                    entity_types.get(ref) == resolver[1]
                    and provider
                    and provider != step.step_id
                    and provider not in step.depends_on
                    and position.get(provider, -1) < position[step.step_id]
                ):
                    step.depends_on.append(provider)
                    notes.append(
                        f"{step.step_id}: depends on existing {provider} "
                        f"({step.operation} requires {capability})"
                    )
        provided.update(entry.provides)

    for step in steps:
        entry = OPERATION_TABLE.get(step.operation)
        if step.operation_status != "approved" or entry is None:
            continue
        # Rename a filter key to the table-declared name that shares its last token.
        for key in [k for k in step.filters if k not in entry.supported_filters]:
            matches = [
                name
                for name in entry.supported_filters
                if name not in step.filters
                and name.rsplit("_", 1)[-1] == key.rsplit("_", 1)[-1]
            ]
            if len(matches) == 1:
                step.filters[matches[0]] = step.filters.pop(key)
                notes.append(f"{step.step_id}: renamed filter {key} -> {matches[0]}")

    for step in [*inserted, *steps]:
        entry = OPERATION_TABLE.get(step.operation)
        if step.operation_status != "approved" or entry is None:
            continue
        absent = [
            safeguard
            for safeguard in entry.required_safeguards
            if safeguard not in step.safeguards and safeguard not in global_safeguards
        ]
        if absent:
            step.safeguards = [*step.safeguards, *absent]
            notes.append(f"{step.step_id}: added table-required safeguards {absent}")
    return [*inserted, *steps]



def wire_needs_by_capability(
    strategy: ResearchStrategyDraft, notes: list[str]
) -> None:
    """Make a need depend on every earlier need that provides a capability it requires."""
    providers: dict[str, list[str]] = {}
    for need in strategy.needs:
        for capability in need.required_capabilities:
            for provider in providers.get(capability, []):
                if provider != need.need_id and provider not in need.depends_on:
                    need.depends_on.append(provider)
                    notes.append(
                        f"step_{need.need_id}: depends on step_{provider} "
                        f"(requires {capability})"
                    )
        for capability in need.provided_capabilities:
            providers.setdefault(capability, []).append(need.need_id)


def compile_manifest(
    question: str,
    strategy: ResearchStrategyDraft,
    mapping: OperationMappingResult,
    completions: list[str] | None = None,
) -> QueryManifestV2:
    notes = completions if completions is not None else []
    strategy = strategy.model_copy(deep=True)
    wire_needs_by_capability(strategy, notes)
    matches = {match.need_id: match for match in mapping.matches}
    expected_ids = {need.need_id for need in strategy.needs}
    if set(matches) != expected_ids or len(mapping.matches) != len(expected_ids):
        raise ValueError("mapping must contain every need_id exactly once")

    steps = []
    steps_by_id: dict[str, ResearchPlanStepV2] = {}
    for need in strategy.needs:
        match = matches[need.need_id]
        if match.match_status == "matched":
            # The mapper sees the whole table, so the only valid choices are
            # operations that exist in it.
            if match.canonical_operation not in OPERATION_TABLE:
                raise ValueError(
                    f"need {need.need_id} selected an operation outside OPERATION_TABLE"
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
        if operation_status == "approved" and need.provided_capabilities:
            provides = set(OPERATION_TABLE[operation].provides)
            if provides and not provides & set(need.provided_capabilities):
                notes.append(
                    f"{step.step_id}: capability mismatch, need provides "
                    f"{need.provided_capabilities} but {operation} provides {sorted(provides)}"
                )
        steps.append(step)
        steps_by_id[step.step_id] = step

    # Complete once, after every need has been compiled into a step.
    steps = complete_plan_from_table(
        steps,
        strategy.entities,
        strategy.global_safeguards,
        notes,
    )
    steps_by_id = {step.step_id: step for step in steps}
    for step in steps:
        step.inputs = _derive_inputs(step.depends_on, steps_by_id)

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
    """Two LLM stages: strategy, then matching against the full table; Python compiles."""

    def __init__(self, strategy_agent=None, mapping_agent=None, timeout_cleanup=None):
        owns_agents = strategy_agent is None and mapping_agent is None
        model = None
        if strategy_agent is None or mapping_agent is None:
            model = get_chat_model()
        self.strategy_agent = strategy_agent or create_agent(
            model=model,
            tools=[],
            system_prompt=build_strategy_prompt(),
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
        corrected: list[str] = []
        completions: list[str] = []
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
                program_completions=completions,
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
            # Persist the expensive strategy result before starting mapping so a
            # timeout or forced stop never hides which stage was reached.
            finish("running")
            stage_started = perf_counter()
            # The mapper receives the whole OPERATION_TABLE; no shortlist step.
            request = build_mapping_request(strategy)
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
            plan = compile_manifest(question, strategy, mapping, completions)
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
