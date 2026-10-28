"""Tool-free blind natural-language planner for Query Manifest v2."""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from time import perf_counter
from typing import Any, Literal
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field, model_validator

from model_factory import get_chat_model, get_ollama_settings
from planner_acceptance import (
    PlanAcceptanceResult,
    PlannerAcceptanceCase,
    RepairFeedback,
    OPERATION_TABLE,
    STANDARD_OPERATION_CATALOG,
    OPERATION_SEMANTICS,
    build_acceptance_rubric,
    build_repair_feedback,
    evaluate_plan,
    normalize_plan_operations,
)
from schemas import (
    PlannedEntityV2,
    QueryCharacteristicsV2,
    QueryManifestV2,
    ResearchPlanStepV2,
)
from planner_examples import build_few_shot_prompt
from operation_rules import (
    OperationRuleEvaluation, evaluate_operation_rules, build_metric_catalog_prompt,
)


PLAN_ONLY_PROMPT = """
You are the Research Planner inside Agent 1. You have no tools and must never
retrieve evidence. Convert the user's complete natural-language research
question into QueryManifestV2.

Plan from the requested research objective, not from a six-class classifier.
The characteristics field is descriptive metadata only. A question may cross
categories or fit them imperfectly.

Requirements:
- Preserve original_question exactly.
- Extract every explicitly named disease, target, drug, tissue, and mechanism.
  Do not invent identifiers. Without verified context, use resolution_status
  'unresolved' and leave entity_id/canonical_name null.
- Create concrete, ordered operations. Every step has a unique `step_id`.
  `depends_on` may contain only earlier `step_id` values, never an output name.
  Example: if step `resolve_disease_1` outputs `resolved_disease_1`, the next
  step must use `depends_on: ["resolve_disease_1"]`, not
  `depends_on: ["resolved_disease_1"]`.
  Use entity_refs to show which top-level entities each step uses.
- `entity_refs` may contain only `entity_ref` values declared in the top-level
  entities list. Never put a prior step's output name in entity_refs. Use
  depends_on for step-to-step data flow and outputs only to name produced data.
- Plan entity resolution before operations that require normalized IDs.
- Assign only sources genuinely needed by each step: open_targets for disease,
  target, association, tractability and integrated evidence; europe_pmc for
  literature; clinicaltrials_gov for trials, status and posted results; chembl
  for compounds, target mapping, assays, bioactivity and mechanisms.
- When the user asks for therapeutic targets, do not equate every
  disease-associated target with a therapeutic target. Plan separate evidence
  collection for disease association and for translational support such as
  known drug evidence or tractability, then stratify the results so those
  evidence layers remain distinguishable.
- Encode thresholds in filters and joins/comparisons/counts/grouping in
  aggregation. When ordering is needed, ranking must use exactly:
  {"primary_metric": "snake_case_metric", "direction": "ascending|descending",
  "secondary_metrics": ["optional_metric"]}. Do not use criteria, sort,
  sort_by, rank_by, or order_by.
- Preserve provenance, retrieval time, pagination, truncation, and per-source
  success/error state in safeguards.
- Missing evidence remains unknown, not negative. Association scores are not
  probabilities, association does not prove efficacy or indication, and
  approval requires an explicit approval source.
- This phase creates a plan only. Never claim evidence was retrieved or that a
  scientific conclusion is established.
- `characteristics` is metadata with exactly this shape:
  {"primary": "one_allowed_enum_value", "secondary": ["zero_or_more_other_allowed_values"]}.
  `primary` must be one of single_hop, multi_hop, evidence_directed,
  filter_and_rank, comparison_and_aggregation, or safety_and_negative_knowledge.
  Do not replace this shape with arbitrary boolean flags.
- Do not generate `generated_at`; Python supplies authoritative runtime time.

Before choosing an operation, check the provided OPERATION_TABLE. If the needed
capability matches a canonical operation or reviewed equivalent, use the
canonical operation. If no equivalent exists, do not invent an approved
operation: use operation='proposed_new_operation', operation_status='proposed',
and provide proposed_operation_name and proposal_reason for human review.

Use concise stable snake_case English operation/key/safeguard identifiers;
never put prose sentences in operation, key, or safeguard identifier fields.
Write descriptions and criteria in clear Traditional Chinese.
"""

PLAN_PATCH_PROMPT = """
You repair an existing QueryManifestV2. You have no tools and must never
retrieve evidence. Return only a QueryManifestPatch, not a complete manifest.

Apply only the supplied concrete errors. Preserve every correct entity, step,
dependency, source, safeguard, and accepted capability. Use remove_* only for
incorrect items and upsert_* only for new or changed items. For an existing
step_id, include the changed fields and Python preserves omitted fields from the
base step. A new step must include every required ResearchPlanStepV2 field.
An entity upsert replaces the full entity with the same entity_ref; otherwise it
appends it. Supply
step_order only when adding/removing/reordering steps, and then list every final
step_id exactly once, or omit it and let Python order the declared dependencies.
Python preserves dependency direction and rejects cycles. Leave unrelated
optional replacement fields null. For an existing step, lists replace the whole
list and objects replace the whole field: include the complete intended ranking
or aggregation object when changing it. Omission preserves; [] explicitly clears.

Do not include schema_version, manifest_id, original_question, generated_at, or
other parent-Plan metadata in the patch. Python owns and preserves those fields.

Do not invent verified IDs or research findings. Do not change the original
question. Analysis steps consume upstream results through depends_on and must
not attach sources. Top-level entities may only represent entities explicitly
named in the question; generated collections remain step outputs. A repair is
still only a Plan and never evidence retrieval.
"""


class QueryManifestPatch(BaseModel):
    """Bounded edits applied by Python to the last schema-valid manifest."""

    remove_entity_refs: list[str] = Field(default_factory=list)
    upsert_entities: list[PlannedEntityV2] = Field(default_factory=list)
    remove_step_ids: list[str] = Field(default_factory=list)
    upsert_steps: list[ResearchPlanStepV2] = Field(default_factory=list)
    step_order: list[str] | None = None
    research_objective: str | None = None
    evidence_requirements: list[str] | None = None
    completion_criteria: list[str] | None = None
    global_safeguards: list[str] | None = None
    characteristics: QueryCharacteristicsV2 | None = None

    @model_validator(mode="after")
    def require_at_least_one_edit(self) -> "QueryManifestPatch":
        edits = (
            self.remove_entity_refs
            or self.upsert_entities
            or self.remove_step_ids
            or self.upsert_steps
            or self.step_order is not None
            or self.research_objective is not None
            or self.evidence_requirements is not None
            or self.completion_criteria is not None
            or self.global_safeguards is not None
            or self.characteristics is not None
        )
        if not edits:
            raise ValueError("repair patch must contain at least one edit")
        return self


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def _extract_patch_payload(text: str) -> dict[str, Any]:
    """Extract one patch object from thinking/fences without another model call."""
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    patch_keys = set(QueryManifestPatch.model_fields)
    for candidate in reversed(candidates):
        if set(candidate).intersection(patch_keys):
            return candidate
    raise ValueError("model response did not contain a QueryManifestPatch JSON object")


def _extract_manifest_payload(text: str) -> dict[str, Any]:
    """Extract one complete manifest object without asking the model to retry."""
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    required = {
        "original_question",
        "research_objective",
        "entities",
        "plan_steps",
        "evidence_requirements",
        "completion_criteria",
        "global_safeguards",
        "characteristics",
    }
    for candidate in reversed(candidates):
        if required.issubset(candidate):
            return candidate
    raise ValueError("model response did not contain a complete QueryManifestV2 JSON object")


def _normalize_dependency_output_refs(payload: dict[str, Any]) -> list[str]:
    """Convert an unambiguous prior output used as a dependency to its step ID."""
    corrections: list[str] = []
    known_step_ids: set[str] = set()
    prior_output_producers: dict[str, list[str]] = {}
    steps = payload.get("plan_steps")
    if not isinstance(steps, list):
        return corrections
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = step.get("step_id")
        dependencies = step.get("depends_on", [])
        if isinstance(dependencies, list):
            normalized: list[Any] = []
            for dependency in dependencies:
                producers = prior_output_producers.get(dependency, [])
                if dependency not in known_step_ids and len(producers) == 1:
                    replacement = producers[0]
                    normalized.append(replacement)
                    corrections.append(
                        f"{step_id}: replaced output dependency {dependency!r} "
                        f"with producer step_id {replacement!r}"
                    )
                else:
                    normalized.append(dependency)
            step["depends_on"] = normalized
        if isinstance(step_id, str):
            known_step_ids.add(step_id)
            outputs = step.get("outputs", [])
            if isinstance(outputs, list):
                for output in outputs:
                    if isinstance(output, str):
                        prior_output_producers.setdefault(output, []).append(step_id)
    return corrections


def _complete_existing_step_upserts(
    payload: dict[str, Any], base_plan: QueryManifestV2 | None
) -> list[str]:
    """Give existing-step upserts true patch semantics; new steps stay strict."""
    if base_plan is None or not isinstance(payload.get("upsert_steps"), list):
        return []
    base_steps = {step.step_id: step.model_dump() for step in base_plan.plan_steps}
    corrections: list[str] = []
    for index, update in enumerate(payload["upsert_steps"]):
        if not isinstance(update, dict):
            continue
        step_id = update.get("step_id")
        if step_id not in base_steps:
            continue
        missing = sorted(set(base_steps[step_id]) - set(update))
        if not missing:
            continue
        payload["upsert_steps"][index] = {**base_steps[step_id], **update}
        corrections.append(
            f"{step_id}: preserved omitted existing-step fields: "
            + ", ".join(missing)
        )
    return corrections


class DirectJsonPlanAgent:
    """One full-Plan model invocation followed by deterministic validation."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def invoke(self, request: dict[str, Any], config: dict | None = None) -> dict:
        schema = json.dumps(QueryManifestV2.model_json_schema(), ensure_ascii=False)
        messages = [
            {
                "role": "system",
                "content": (
                    PLAN_ONLY_PROMPT
                    + "\nReturn exactly one JSON object matching this schema:\n"
                    + schema
                ),
            },
            *request.get("messages", []),
        ]
        response = self._model.invoke(messages, config=config)
        payload = _extract_manifest_payload(_message_text(response))
        corrections = _normalize_dependency_output_refs(payload)
        plan = QueryManifestV2.model_validate(payload)
        plan._reference_corrections.extend(corrections)
        return {"structured_response": plan, "messages": [response]}


class DirectJsonPatchAgent:
    """One model invocation followed by deterministic local JSON validation."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def invoke(self, request: dict[str, Any], config: dict | None = None) -> dict:
        response_schema = QueryManifestPatch.model_json_schema()
        # The wire contract allows partial existing-step updates. Completion
        # against the base precedes strict local validation, including new steps.
        step_schema = response_schema.get("$defs", {}).get("ResearchPlanStepV2", {})
        step_schema["required"] = ["step_id"]
        schema = json.dumps(response_schema, ensure_ascii=False)
        messages = [
            {
                "role": "system",
                "content": (
                    PLAN_PATCH_PROMPT
                    + "\nReturn exactly one JSON object matching this schema:\n"
                    + schema
                ),
            },
            *request.get("messages", []),
        ]
        response = self._model.invoke(messages, config=config)
        payload = _extract_patch_payload(_message_text(response))
        corrections = _complete_existing_step_upserts(
            payload, request.get("base_plan")
        )
        patch = QueryManifestPatch.model_validate(payload)
        return {
            "structured_response": patch,
            "messages": [response],
            "patch_normalizations": corrections,
        }


def _stable_dependency_order(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compile declared edges to an order; never invent or reverse an edge."""
    ids = [step["step_id"] for step in steps]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate step IDs in merged patch")
    known = set(ids)
    for step in steps:
        unknown = set(step.get("depends_on", [])) - known
        if unknown:
            raise ValueError(f"step {step['step_id']} has unknown dependencies: {sorted(unknown)}")
    pending = list(steps)
    ordered: list[dict[str, Any]] = []
    emitted: set[str] = set()
    while pending:
        ready = next((step for step in pending
                      if set(step.get("depends_on", [])) <= emitted), None)
        if ready is None:
            raise ValueError("patch dependencies contain a cycle; correct the declared edges")
        pending.remove(ready)
        ordered.append(ready)
        emitted.add(ready["step_id"])
    return ordered


def apply_manifest_patch(
    base: QueryManifestV2, patch: QueryManifestPatch
) -> QueryManifestV2:
    """Merge a bounded patch and validate the complete graph atomically."""
    payload = base.model_dump()
    removed_entities = set(patch.remove_entity_refs)
    entity_map = {
        item["entity_ref"]: item
        for item in payload["entities"]
        if item["entity_ref"] not in removed_entities
    }
    entity_order = list(entity_map)
    for entity in patch.upsert_entities:
        if entity.entity_ref not in entity_map:
            entity_order.append(entity.entity_ref)
        entity_map[entity.entity_ref] = entity.model_dump()
    payload["entities"] = [entity_map[item] for item in entity_order]

    removed_steps = set(patch.remove_step_ids)
    step_map = {
        item["step_id"]: item
        for item in payload["plan_steps"]
        if item["step_id"] not in removed_steps
    }
    step_order = list(step_map)
    for step in patch.upsert_steps:
        if step.step_id not in step_map:
            step_order.append(step.step_id)
        step_map[step.step_id] = step.model_dump()
    if patch.step_order is not None:
        if len(patch.step_order) != len(set(patch.step_order)):
            raise ValueError("patch step_order contains duplicate step IDs")
        if set(patch.step_order) != set(step_map):
            updates_existing_steps_only = bool(patch.upsert_steps) and not removed_steps and all(
                step.step_id in {item.step_id for item in base.plan_steps}
                for step in patch.upsert_steps
            )
            if not updates_existing_steps_only:
                raise ValueError(
                    "patch step_order must list every final step ID exactly once"
                )
            # Updating existing steps does not require an order declaration.
            # Preserve the authoritative base order rather than discarding an
            # otherwise useful patch because the model supplied stale metadata.
        else:
            step_order = patch.step_order
    payload["plan_steps"] = _stable_dependency_order(
        [step_map[item] for item in step_order]
    )

    for field in (
        "research_objective",
        "evidence_requirements",
        "completion_criteria",
        "global_safeguards",
        "characteristics",
    ):
        value = getattr(patch, field)
        if value is not None:
            payload[field] = value.model_dump() if hasattr(value, "model_dump") else value
    return QueryManifestV2.model_validate(payload)


class RepairScope(BaseModel):
    """Python-authoritative edit boundary derived from actual findings."""

    entity_refs: list[str] = Field(default_factory=list)
    step_ids: list[str] = Field(default_factory=list)
    new_operations: list[str] = Field(default_factory=list)
    top_level_fields: list[str] = Field(default_factory=list)
    allow_new_entities: bool = False
    allow_step_order: bool = False


class DependencyRepairTask(BaseModel):
    """A concrete graph edit compiled from one dependency diagnosis."""

    repair_type: Literal["add_dependency"] = "add_dependency"
    editable_step_id: str
    editable_operation: str
    current_dependencies: list[str] = Field(default_factory=list)
    required_dependency_step_id: str
    required_dependency_operation: str
    required_final_dependencies: list[str] = Field(default_factory=list)
    forbidden_step_ids: list[str] = Field(default_factory=list)
    forbidden_reverse_edge: str
    previously_rejected_step_ids: list[str] = Field(default_factory=list)


def _operations_named_in(values: list[str]) -> set[str]:
    text = "\n".join(values)
    tokens = set(re.findall(r"[a-z][a-z0-9_]*", text))
    return set(STANDARD_OPERATION_CATALOG) & tokens


def build_repair_scope(
    plan: QueryManifestV2,
    acceptance: PlanAcceptanceResult | None,
    rules: OperationRuleEvaluation,
    feedback: RepairFeedback | None = None,
) -> RepairScope:
    """Translate evaluator findings into the only edits a patch may perform."""
    entity_refs: set[str] = set()
    step_ids = {item.step_id for item in rules.blocking_findings if item.step_id}
    new_operations: set[str] = set()
    top_level_fields: set[str] = set()
    allow_new_entities = False
    allow_step_order = False

    for finding in rules.blocking_findings:
        if finding.rule_id == "declared_entities_must_be_question_grounded":
            entity_refs.update(finding.missing)
        if finding.rule_id == "capability_requires_safeguards":
            top_level_fields.add("global_safeguards")

    if acceptance is not None:
        if acceptance.missing_entities:
            allow_new_entities = True
        missing_operation_text = [
            *acceptance.missing_capabilities,
            *acceptance.missing_dependencies,
        ]
        new_operations.update(_operations_named_in(missing_operation_text))
        if acceptance.missing_capabilities:
            allow_step_order = True
        dependency_operations = _operations_named_in([
            edge.split("->", 1)[-1] for edge in acceptance.missing_dependencies
        ])
        step_ids.update(
            step.step_id for step in plan.plan_steps
            if step.operation in dependency_operations
        )
        if acceptance.missing_sources:
            step_ids.update(
                step.step_id for step in plan.plan_steps
                if step.operation.startswith(("resolve_", "retrieve_", "discover_"))
            )
        if acceptance.missing_filters:
            step_ids.update(
                step.step_id for step in plan.plan_steps
                if step.operation.startswith("filter_")
            )
        if acceptance.missing_ranking:
            step_ids.update(
                step.step_id for step in plan.plan_steps
                if step.operation.startswith("rank_")
            )
        if acceptance.missing_aggregation:
            step_ids.update(
                step.step_id for step in plan.plan_steps
                if step.operation.startswith(
                    ("aggregate_", "calculate_", "compare_", "stratify_")
                )
            )
        if acceptance.missing_safeguards:
            top_level_fields.add("global_safeguards")
        if acceptance.forbidden_findings:
            # The current acceptance result does not localize free-text claims.
            # Permit text-bearing fields, but still not unrelated entities.
            step_ids.update(step.step_id for step in plan.plan_steps)
            top_level_fields.update({
                "research_objective", "evidence_requirements", "completion_criteria"
            })

    if new_operations:
        allow_step_order = True
    # Dependency endpoints already present are not permission to add duplicates.
    new_operations -= {step.operation for step in plan.plan_steps}
    scope = RepairScope(
        entity_refs=sorted(entity_refs),
        step_ids=sorted(step_ids),
        new_operations=sorted(new_operations),
        top_level_fields=sorted(top_level_fields),
        allow_new_entities=allow_new_entities,
        allow_step_order=allow_step_order,
    )
    if feedback is None:
        return scope

    actionable_feedback = RepairFeedback(
        missing=feedback.missing,
        incorrect=feedback.incorrect,
    )
    feedback_text = actionable_feedback.model_dump_json()
    named_operations = _operations_named_in([feedback_text])
    explicit_steps = set(re.findall(r"step_id=([a-zA-Z0-9_-]+)", feedback_text))
    relevant_steps = explicit_steps | {
        step.step_id for step in plan.plan_steps
        if step.operation in named_operations
    }
    areas = {
        item.get("area", "")
        for item in [*feedback.missing, *feedback.incorrect]
    }
    selected_groups = {
        _repair_feedback_group(item)
        for item in [*feedback.missing, *feedback.incorrect]
    }
    if "ranking" in selected_groups:
        relevant_steps.update(
            step.step_id for step in plan.plan_steps
            if step.operation.startswith("rank_")
        )
    if "filtering" in selected_groups:
        relevant_steps.update(
            step.step_id for step in plan.plan_steps
            if step.operation.startswith("filter_")
        )
    if "aggregation" in areas:
        relevant_steps.update(
            step.step_id for step in plan.plan_steps
            if step.operation.startswith(
                ("aggregate_", "calculate_", "compare_", "stratify_")
            )
        )
    if "source_coverage" in areas:
        relevant_steps.update(
            step.step_id for step in plan.plan_steps
            if step.operation.startswith(("resolve_", "retrieve_", "discover_"))
        )
    return RepairScope(
        entity_refs=(
            scope.entity_refs
            if "declared_entities_must_be_question_grounded" in areas else []
        ),
        step_ids=sorted(set(scope.step_ids) & relevant_steps),
        new_operations=sorted(set(scope.new_operations) & named_operations),
        top_level_fields=(
            scope.top_level_fields
            if any("safeguard" in area for area in areas) else []
        ),
        allow_new_entities=scope.allow_new_entities and any(
            "entity" in area for area in areas
        ),
        allow_step_order=scope.allow_step_order and bool(
            set(scope.new_operations) & named_operations
        ),
    )


def build_dependency_repair_task(
    plan: QueryManifestV2,
    feedback: RepairFeedback,
    scope: RepairScope,
) -> DependencyRepairTask | None:
    """Resolve one dependency-only diagnosis to exact step IDs and direction."""
    dependency_items = [
        item for item in feedback.missing if item.get("area") == "dependency"
    ]
    if len(dependency_items) != 1:
        return None
    if any(item.get("area") != "dependency" for item in feedback.missing):
        return None
    permitted_transport_errors = {"schema_or_integrity", "non_improving_patch"}
    if any(
        item.get("area") not in permitted_transport_errors
        for item in feedback.incorrect
    ):
        return None

    problem = dependency_items[0].get("problem", "")
    match = re.search(
        r"([a-z][a-z0-9_]*)\s*->\s*([a-z][a-z0-9_]*)", problem
    )
    if match is None:
        return None
    upstream_operation, downstream_operation = match.groups()
    upstream_steps = [
        step for step in plan.plan_steps if step.operation == upstream_operation
    ]
    downstream_steps = [
        step for step in plan.plan_steps if step.operation == downstream_operation
    ]
    if len(upstream_steps) != 1 or len(downstream_steps) != 1:
        return None
    upstream = upstream_steps[0]
    downstream = downstream_steps[0]
    if downstream.step_id not in scope.step_ids:
        return None

    rejected_step_ids = sorted({
        step_id
        for item in feedback.incorrect
        for step_id in re.findall(
            r"(?:upsert_steps|remove_step_ids):([A-Za-z0-9_-]+)",
            item.get("problem", ""),
        )
    })
    final_dependencies = list(downstream.depends_on)
    if upstream.step_id not in final_dependencies:
        final_dependencies.append(upstream.step_id)
    return DependencyRepairTask(
        editable_step_id=downstream.step_id,
        editable_operation=downstream.operation,
        current_dependencies=list(downstream.depends_on),
        required_dependency_step_id=upstream.step_id,
        required_dependency_operation=upstream.operation,
        required_final_dependencies=final_dependencies,
        forbidden_step_ids=[upstream.step_id],
        forbidden_reverse_edge=f"{upstream.step_id} depends_on {downstream.step_id}",
        previously_rejected_step_ids=rejected_step_ids,
    )


def build_dependency_repair_prompt(
    question: str,
    plan: QueryManifestV2,
    task: DependencyRepairTask,
) -> str:
    """Build a small repair prompt without unrelated operation or metric catalogs."""
    step_by_id = {step.step_id: step for step in plan.plan_steps}
    editable_step = step_by_id[task.editable_step_id]
    dependency_graph = [
        {
            "step_id": step.step_id,
            "operation": step.operation,
            "depends_on": list(step.depends_on),
        }
        for step in plan.plan_steps
    ]
    retry_warning = ""
    if task.previously_rejected_step_ids:
        retry_warning = (
            "\n\n上一輪被拒絕，因為修改了這些禁止修改的 step_id：\n"
            + json.dumps(task.previously_rejected_step_ids, ensure_ascii=False)
            + "\n本輪 upsert_steps 中唯一允許出現的 step_id 是："
            + task.editable_step_id
        )
    return (
        f"使用者原始問題：\n{question}\n\n"
        "這一輪只修正一條既有步驟的依賴關係；不要新增 operation、"
        "不要修改 ranking、filters、aggregation、sources 或 step_order。\n\n"
        "目前依賴圖：\n"
        + json.dumps(dependency_graph, ensure_ascii=False, indent=2)
        + "\n\n唯一可修改步驟的目前內容：\n"
        + editable_step.model_dump_json(indent=2)
        + "\n\nPython 已解析完成的精確修正任務：\n"
        + task.model_dump_json(indent=2)
        + retry_warning
        + "\n\n只回傳最小 QueryManifestPatch。upsert_steps 只能有一個項目，"
        "而且只能包含 step_id 與完整的 depends_on；depends_on 必須等於 "
        "required_final_dependencies。其他 patch 欄位請省略。"
    )


def validate_patch_scope(
    base: QueryManifestV2, patch: QueryManifestPatch, scope: RepairScope
) -> None:
    """Reject edits outside evaluator-authorized entities, steps, and fields."""
    known_entities = {item.entity_ref for item in base.entities}
    known_steps = {item.step_id for item in base.plan_steps}
    allowed_entities = set(scope.entity_refs)
    allowed_steps = set(scope.step_ids)
    allowed_new_operations = set(scope.new_operations)
    violations: list[str] = []

    violations.extend(
        f"remove_entity_refs:{item}"
        for item in patch.remove_entity_refs if item not in allowed_entities
    )
    for entity in patch.upsert_entities:
        if entity.entity_ref in known_entities:
            if entity.entity_ref not in allowed_entities:
                violations.append(f"upsert_entities:{entity.entity_ref}")
        elif not scope.allow_new_entities:
            violations.append(f"new_entity:{entity.entity_ref}")
    violations.extend(
        f"remove_step_ids:{item}"
        for item in patch.remove_step_ids if item not in allowed_steps
    )
    for step in patch.upsert_steps:
        if step.step_id in known_steps:
            if step.step_id not in allowed_steps:
                violations.append(f"upsert_steps:{step.step_id}")
        elif step.operation not in allowed_new_operations:
            violations.append(f"new_step:{step.step_id}/{step.operation}")
    if patch.step_order is not None and not scope.allow_step_order:
        violations.append("step_order")
    for field in (
        "research_objective", "evidence_requirements", "completion_criteria",
        "global_safeguards", "characteristics",
    ):
        if getattr(patch, field) is not None and field not in scope.top_level_fields:
            violations.append(field)
    if violations:
        raise ValueError(
            "patch attempted edits outside the authorized repair scope: "
            + ", ".join(sorted(violations))
        )


def _merge_feedback(*items: RepairFeedback | None) -> RepairFeedback:
    merged = RepairFeedback()
    for field in ("missing", "incorrect", "preserve"):
        seen: set[str] = set()
        values: list[dict[str, str]] = []
        for item in items:
            if item is None:
                continue
            for value in getattr(item, field):
                key = repr(sorted(value.items()))
                if key not in seen:
                    seen.add(key)
                    values.append(value)
        setattr(merged, field, values)
    return merged


def _repair_feedback_group(item: dict[str, str]) -> str:
    """Group related repairs without encoding question-specific science."""
    text = (item.get("area", "") + " " + item.get("problem", "")).casefold()
    if "rank" in text or "metric" in text:
        return "ranking"
    if "filter" in text or "constraint" in text:
        return "filtering"
    return "structure"


def select_repair_batch(feedback: RepairFeedback) -> RepairFeedback:
    """Send one coherent repair group, preferring upstream structure first."""
    grouped: dict[str, dict[str, list[dict[str, str]]]] = {
        name: {"missing": [], "incorrect": []}
        for name in ("structure", "filtering", "ranking")
    }
    for field in ("missing", "incorrect"):
        for item in getattr(feedback, field):
            grouped[_repair_feedback_group(item)][field].append(item)
    for group_name in ("structure", "filtering", "ranking"):
        group = grouped[group_name]
        if group["missing"] or group["incorrect"]:
            return RepairFeedback(
                missing=group["missing"],
                incorrect=group["incorrect"],
                preserve=feedback.preserve,
            )
    return feedback


def build_repair_operation_catalog(
    plan: QueryManifestV2, feedback: RepairFeedback
) -> str:
    """Show repair calls only operations present in or named by the diagnosis."""
    feedback_text = feedback.model_dump_json()
    selected = {step.operation for step in plan.plan_steps}
    selected.update(
        operation for operation in STANDARD_OPERATION_CATALOG
        if operation in feedback_text
    )
    selected = {item for item in selected if item in OPERATION_TABLE}
    return (
        "Relevant authoritative OPERATION_TABLE entries for this repair:\n"
        + "\n".join(
            _operation_line(item, include_meaning=True) for item in sorted(selected)
        )
        + "\nUse no other approved operation unless it is explicitly named in the repair scope."
    )


def _plan_issue_count(
    acceptance: PlanAcceptanceResult | None, rules: OperationRuleEvaluation
) -> int:
    count = len(rules.blocking_findings)
    if acceptance is None:
        return count
    for field in (
        "missing_entities", "missing_capabilities", "missing_dependencies",
        "missing_sources", "missing_safeguards", "missing_filters",
        "missing_ranking", "missing_aggregation", "forbidden_findings",
        "proposed_operations",
    ):
        count += len(getattr(acceptance, field))
    return count

def _operation_line(item: str, include_meaning: bool) -> str:
    entry = OPERATION_TABLE[item]
    parts = [
        f"canonical={item}",
        f"equivalents={entry.equivalent_operations or 'none'}",
        f"kind={entry.operation_kind}",
    ]
    for name in (
        "requires_all", "requires_any", "provides", "supported_filters",
        "required_safeguards", "reserved_outputs",
    ):
        value = getattr(entry, name)
        if value:
            parts.append(f"{name}={value}")
    if entry.requires_aggregation:
        parts.append("requires_aggregation=true")
    line = "- " + "; ".join(parts)
    if include_meaning:
        line += f"; meaning={OPERATION_SEMANTICS[item]}"
    return line


FULL_OPERATION_CATALOG_PROMPT = """
Agent 1 standard operation catalog with capability boundaries (choose only what
the question needs):
{operations}

This is the authoritative OPERATION_TABLE, not a per-question answer. Reviewed
equivalents must be replaced by their canonical operation. A genuinely new
operation must be marked proposed and cannot be executed before human review.
""".format(
    operations="\n".join(
        _operation_line(item, include_meaning=True)
        for item in STANDARD_OPERATION_CATALOG
    )
)

OPERATION_GROUPS = {
    "entity_resolution": [
        item for item in STANDARD_OPERATION_CATALOG if item.startswith("resolve_")
    ],
    "evidence_retrieval": [
        item
        for item in STANDARD_OPERATION_CATALOG
        if item.startswith("retrieve_") or item.startswith("discover_")
    ],
    "filtering": [
        item for item in STANDARD_OPERATION_CATALOG if item.startswith("filter_")
    ],
    "ranking": [
        item for item in STANDARD_OPERATION_CATALOG if item.startswith("rank_")
    ],
    "comparison": [
        item for item in STANDARD_OPERATION_CATALOG if item.startswith("compare_")
    ],
    "joining_and_analysis": [
        item
        for item in STANDARD_OPERATION_CATALOG
        if item.startswith(
            (
                "aggregate_",
                "calculate_",
                "classify_",
                "define_",
                "integrate_",
                "intersect_",
                "join_",
                "normalize_",
                "stratify_",
                "check_",
            )
        )
    ],
}

COMPACT_BOUNDARY_OPERATIONS = {
    "aggregate_category_distribution",
    "discover_targets",
    "filter_phase2_not_pdac_approved",
    "rank_targets",
    "retrieve_approved_indications",
    "retrieve_target_drugs",
    "retrieve_tractability",
    "stratify_target_evidence_layers",
}


def build_operation_catalog_prompt(mode: str | None = None) -> str:
    """Build a reversible full or grouped-compact Planner table view."""

    selected = (
        mode
        or os.getenv("PLANNER_OPERATION_CATALOG_MODE", "grouped_compact")
    ).strip().lower()
    if selected == "full":
        return FULL_OPERATION_CATALOG_PROMPT
    if selected != "grouped_compact":
        raise NaturalLanguagePlannerError(
            "PLANNER_OPERATION_CATALOG_MODE must be 'grouped_compact' or 'full'"
        )
    sections = []
    for group, operations in OPERATION_GROUPS.items():
        lines = "\n".join(
            _operation_line(item, include_meaning=False) for item in operations
        )
        sections.append(f"[{group}]\n{lines}")
    boundaries = "\n".join(
        _operation_line(item, include_meaning=True)
        for item in sorted(COMPACT_BOUNDARY_OPERATIONS)
    )
    return (
        "Agent 1 authoritative OPERATION_TABLE, grouped compact view. All "
        "canonical names and reviewed equivalents are listed; only easily "
        "confused operations include expanded boundaries. Choose only what the "
        "question needs.\n\n"
        + "\n\n".join(sections)
        + "\n\n[critical_boundaries]\n"
        + boundaries
        + "\n\nUse a canonical operation when an equivalent exists. If no equivalent "
        "exists, mark proposed_new_operation for human review."
    )


# Backward-compatible public name used by existing diagnostics/tests.
OPERATION_CATALOG_PROMPT = FULL_OPERATION_CATALOG_PROMPT


class NaturalLanguagePlannerError(RuntimeError):
    pass


class ModelAttemptTimeout(TimeoutError):
    def __init__(self, message: str, cleanup: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.cleanup = cleanup or {}


class ModelCallDiagnostic(BaseModel):
    """One underlying model call, including structured-output retries."""

    call_index: int = Field(ge=1)
    run_id: str
    parent_run_id: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    status: Literal["running", "completed", "error", "timed_out"]
    input_messages: list[dict[str, Any]] = Field(default_factory=list)
    raw_response: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None


def _message_dump(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json")
    return {"type": type(message).__name__, "content": str(message)}


class ModelCallTraceHandler(BaseCallbackHandler):
    """Thread-safe callback trace retained even if the outer attempt times out."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._calls: dict[str, ModelCallDiagnostic] = {}

    def on_chat_model_start(
        self, serialized, messages, *, run_id: UUID, parent_run_id=None, **kwargs
    ) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._calls[str(run_id)] = ModelCallDiagnostic(
                call_index=len(self._calls) + 1,
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                started_at=now,
                status="running",
                input_messages=[
                    _message_dump(message)
                    for batch in messages
                    for message in batch
                ],
            )

    def on_llm_end(self, response, *, run_id: UUID, **kwargs) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            call = self._calls.get(str(run_id))
            if call is None:
                return
            call.finished_at = now
            call.duration_seconds = (now - call.started_at).total_seconds()
            call.status = "completed"
            if hasattr(response, "model_dump"):
                call.raw_response = response.model_dump(mode="json")
            elif hasattr(response, "dict"):
                call.raw_response = response.dict()
            else:
                call.raw_response = {"value": str(response)}

    def on_llm_error(self, error, *, run_id: UUID, **kwargs) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            call = self._calls.get(str(run_id))
            if call is None:
                return
            call.finished_at = now
            call.duration_seconds = (now - call.started_at).total_seconds()
            call.status = "error"
            call.error_type = type(error).__name__
            call.error_message = str(error)

    def mark_running_calls_timed_out(self) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            for call in self._calls.values():
                if call.status == "running":
                    call.finished_at = now
                    call.duration_seconds = (now - call.started_at).total_seconds()
                    call.status = "timed_out"
                    call.error_type = "OuterAttemptTimeout"
                    call.error_message = "Outer Planner deadline elapsed before callback completion."

    def snapshot(self) -> list[ModelCallDiagnostic]:
        with self._lock:
            return [call.model_copy(deep=True) for call in self._calls.values()]


class PlannerAttemptDiagnostic(BaseModel):
    attempt: int = Field(ge=1, le=5)
    started_at: datetime
    duration_seconds: float = Field(ge=0)
    schema_valid: bool
    generation_mode: Literal["full_plan", "local_patch"] = "full_plan"
    applied_patch: QueryManifestPatch | None = None
    repair_scope: RepairScope | None = None
    promoted_to_repair_base: bool = False
    semantic_acceptance: PlanAcceptanceResult | None = None
    operation_rule_evaluation: OperationRuleEvaluation | None = None
    candidate_plan: QueryManifestV2 | None = None
    raw_candidate_payload: dict[str, Any] | None = None
    model_calls: list[ModelCallDiagnostic] = Field(default_factory=list)
    repair_feedback_sent_next: RepairFeedback | None = None
    errors: list[str] = Field(default_factory=list)
    corrected_model_claims: list[str] = Field(default_factory=list)
    regressed_components: list[str] = Field(default_factory=list)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    model_total_duration_ns: int | None = Field(default=None, ge=0)
    model_eval_duration_ns: int | None = Field(default=None, ge=0)
    model_eval_count: int | None = Field(default=None, ge=0)
    model_generation_timeout: bool = False
    ollama_stop_requested: bool = False
    ollama_stopping_detected: bool = False
    ollama_stopping_timeout: bool = False
    ollama_cleanup_succeeded: bool = False
    ollama_cleanup_error: str | None = None
    repeated_repair_failure: bool = False
    evidence_collection_started: Literal[False] = False
    executed_retrieval_tools: list[str] = Field(default_factory=list)


class PlannerRunReport(BaseModel):
    schema_version: Literal["1.1"] = "1.1"
    question: str
    acceptance_case_id: str | None = None
    few_shot_mode: str = "off"
    operation_catalog_mode: Literal["grouped_compact", "full"]
    repair_mode: Literal["local_patch", "full_rewrite"] = "local_patch"
    blind_first_attempt: bool = True
    resumed_from_plan: bool = False
    resume_source: str | None = None
    initial_plan: QueryManifestV2 | None = None
    status: Literal["running", "passed", "failed"]
    final_plan: QueryManifestV2 | None = None
    attempts: list[PlannerAttemptDiagnostic] = Field(min_length=1, max_length=5)
    first_attempt_passed: bool
    repair_count: int = Field(ge=0, le=5)
    total_duration_seconds: float = Field(ge=0)
    evidence_collection_started: Literal[False] = False
    executed_retrieval_tools: list[str] = Field(default_factory=list)
    generated_at: datetime


def _ollama_executable() -> str | None:
    configured = os.getenv("OLLAMA_EXECUTABLE", "").strip()
    if configured:
        return configured
    discovered = shutil.which("ollama")
    if discovered:
        return discovered
    fallback = Path("D:/Ollama/ollama.exe")
    return str(fallback) if fallback.exists() else None


def _ollama_process_state(executable: str, model: str) -> tuple[bool, bool]:
    completed = subprocess.run(
        [executable, "ps"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    model_running = any(
        line.split() and line.split()[0] == model
        for line in output.splitlines()
    )
    return model_running, "stopping" in output.casefold()


def _cleanup_ollama_after_timeout() -> dict[str, Any]:
    """Request model stop and verify it leaves Ollama's running-model list."""

    result: dict[str, Any] = {
        "ollama_stop_requested": False,
        "ollama_stopping_detected": False,
        "ollama_stopping_timeout": False,
        "ollama_cleanup_succeeded": False,
        "ollama_cleanup_error": None,
    }
    executable = _ollama_executable()
    if executable is None:
        result["ollama_cleanup_error"] = "ollama executable not found"
        return result
    model, _ = get_ollama_settings()
    grace_seconds = float(os.getenv("PLANNER_OLLAMA_STOP_GRACE_SECONDS", "90"))
    poll_seconds = max(
        0.1, float(os.getenv("PLANNER_OLLAMA_STOP_POLL_SECONDS", "5"))
    )
    result["ollama_stop_requested"] = True
    try:
        try:
            subprocess.run(
                [executable, "stop", model],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=min(30.0, grace_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired:
            result["ollama_stopping_detected"] = True

        deadline = perf_counter() + grace_seconds
        while True:
            running, stopping = _ollama_process_state(executable, model)
            result["ollama_stopping_detected"] |= stopping
            if not running:
                result["ollama_cleanup_succeeded"] = True
                return result
            if perf_counter() >= deadline:
                result["ollama_stopping_timeout"] = True
                return result
            time.sleep(min(poll_seconds, max(0.0, deadline - perf_counter())))
    except Exception as exc:
        result["ollama_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        return result


def _invoke_with_deadline(
    agent,
    request: dict,
    timeout_seconds: float,
    timeout_cleanup=None,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> dict:
    """Run one model request in a daemon thread with a hard wall-clock deadline."""

    result_queue: queue.Queue = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            if callbacks:
                result_queue.put((True, agent.invoke(request, config={"callbacks": callbacks})))
            else:
                result_queue.put((True, agent.invoke(request)))
        except BaseException as exc:  # Forward backend exceptions to the caller.
            result_queue.put((False, exc))

    worker = Thread(target=invoke, daemon=True, name="planner-model-attempt")
    worker.start()
    try:
        succeeded, value = result_queue.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        cleanup = timeout_cleanup() if timeout_cleanup is not None else None
        raise ModelAttemptTimeout(
            f"model attempt exceeded {timeout_seconds:.1f} seconds",
            cleanup,
        ) from exc
    if not succeeded:
        raise value
    return value


def _usage_from_result(result: dict[str, Any]) -> dict[str, int | None]:
    """Read LangChain/Ollama usage metadata when the backend provides it."""

    messages = result.get("messages") or []
    message = messages[-1] if messages else None
    usage = getattr(message, "usage_metadata", None) or {}
    response = getattr(message, "response_metadata", None) or {}
    return {
        "input_tokens": usage.get("input_tokens") or response.get("prompt_eval_count"),
        "output_tokens": usage.get("output_tokens") or response.get("eval_count"),
        "total_tokens": usage.get("total_tokens"),
        "model_total_duration_ns": response.get("total_duration"),
        "model_eval_duration_ns": response.get("eval_duration"),
        "model_eval_count": response.get("eval_count"),
    }


def _normalize_unverified_entities(plan: QueryManifestV2) -> list[str]:
    corrected = []
    for entity in plan.entities:
        if (
            entity.resolution_status != "unresolved"
            or entity.entity_id is not None
            or entity.canonical_name is not None
        ):
            corrected.append(
                f"{entity.entity_ref}: removed unverified identity authority"
            )
        entity.entity_id = None
        entity.canonical_name = None
        entity.resolution_status = "unresolved"
    return corrected


class NaturalLanguageResearchPlanner:
    """Generate a plan blindly, then score it with a hidden rubric."""

    def __init__(
        self,
        agent=None,
        patch_agent=None,
        timeout_cleanup=None,
        capture_model_calls: bool | None = None,
    ) -> None:
        owns_agent = agent is None
        self._capture_model_calls = (
            owns_agent if capture_model_calls is None else capture_model_calls
        )
        self._agent = agent or DirectJsonPlanAgent(get_chat_model())
        self._patch_agent = patch_agent
        if owns_agent and patch_agent is None:
            self._patch_agent = DirectJsonPatchAgent(get_chat_model())
        self._timeout_cleanup = timeout_cleanup
        if owns_agent and timeout_cleanup is None:
            if os.getenv("LLM_PROVIDER", "ollama").strip().lower() == "ollama":
                self._timeout_cleanup = _cleanup_ollama_after_timeout

    def run_with_report(
        self,
        question: str,
        acceptance_case: PlannerAcceptanceCase | None = None,
        checkpoint_path: str | Path | None = None,
        max_attempts: int = 3,
        initial_plan: QueryManifestV2 | None = None,
        resume_source: str | None = None,
    ) -> PlannerRunReport:
        question = question.strip()
        if max_attempts not in (1, 2, 3, 4, 5):
            raise ValueError("max_attempts must be between 1 and 5")
        few_shot_mode = os.getenv("PLANNER_FEW_SHOT_MODE", "off").strip().lower()
        demonstration_prompt = build_few_shot_prompt(few_shot_mode)
        if not question:
            raise NaturalLanguagePlannerError("question cannot be empty")

        rubric = build_acceptance_rubric(acceptance_case) if acceptance_case else None
        operation_catalog_mode = os.getenv(
            "PLANNER_OPERATION_CATALOG_MODE", "grouped_compact"
        ).strip().lower()
        operation_catalog_prompt = build_operation_catalog_prompt(
            operation_catalog_mode
        )
        repair_mode = os.getenv("PLANNER_REPAIR_MODE", "local_patch").strip().lower()
        if repair_mode not in {"local_patch", "full_rewrite"}:
            raise NaturalLanguagePlannerError(
                "PLANNER_REPAIR_MODE must be 'local_patch' or 'full_rewrite'"
            )
        attempts: list[PlannerAttemptDiagnostic] = []
        feedback: RepairFeedback | None = None
        previous_plan: QueryManifestV2 | None = None
        previous_accepted: set[str] = set()
        previous_issue_count: int | None = None
        base_feedback: RepairFeedback | None = None
        repair_scope: RepairScope | None = None
        seen_patch_failure_fingerprints: set[str] = set()
        run_started = perf_counter()
        final_plan = None
        attempt_timeout = float(os.getenv("PLANNER_ATTEMPT_TIMEOUT_SECONDS", "1200"))
        total_timeout = float(os.getenv("PLANNER_TOTAL_TIMEOUT_SECONDS", "3600"))
        if attempt_timeout <= 0 or total_timeout <= 0:
            raise NaturalLanguagePlannerError("Planner timeout values must be positive")

        if initial_plan is not None:
            if initial_plan.original_question != question:
                raise NaturalLanguagePlannerError(
                    "resume Plan original_question does not match the requested question"
                )
            seeded_rules = evaluate_operation_rules(initial_plan)
            seeded_acceptance = (
                evaluate_plan(rubric, initial_plan) if rubric is not None else None
            )
            seed_passed = seeded_rules.passed and (
                seeded_acceptance is None or seeded_acceptance.passed
            )
            if seed_passed:
                raise NaturalLanguagePlannerError(
                    "resume Plan already passes; no repair is required"
                )
            seeded_feedback = (
                build_repair_feedback(seeded_acceptance)
                if seeded_acceptance is not None else RepairFeedback()
            )
            seeded_feedback.incorrect.extend(
                {
                    "area": item.rule_id,
                    "problem": (
                        (f"step_id={item.step_id}; " if item.step_id else "")
                        + item.problem
                    ),
                    "required_correction": item.required_correction,
                }
                for item in seeded_rules.blocking_findings
            )
            previous_plan = initial_plan.model_copy(deep=True)
            previous_accepted = set(
                seeded_acceptance.accepted_capabilities
                if seeded_acceptance is not None else []
            )
            previous_issue_count = (
                _plan_issue_count(seeded_acceptance, seeded_rules)
                if seeded_acceptance is not None
                else len(seeded_rules.blocking_findings)
            )
            feedback = select_repair_batch(seeded_feedback)
            base_feedback = feedback
            repair_scope = build_repair_scope(
                previous_plan, seeded_acceptance, seeded_rules, feedback
            )

        def make_report(status: Literal["running", "passed", "failed"]):
            return PlannerRunReport(
                question=question,
                acceptance_case_id=(
                    acceptance_case.case_id if acceptance_case else None
                ),
                operation_catalog_mode=operation_catalog_mode,
                repair_mode=repair_mode,
                few_shot_mode=few_shot_mode,
                resumed_from_plan=initial_plan is not None,
                blind_first_attempt=initial_plan is None,
                resume_source=resume_source,
                initial_plan=initial_plan,
                status=status,
                final_plan=final_plan,
                attempts=attempts,
                first_attempt_passed=(
                    initial_plan is None and status == "passed" and len(attempts) == 1
                ),
                repair_count=sum(a.generation_mode == "local_patch" for a in attempts)
                if initial_plan is not None else max(0, len(attempts) - 1),
                total_duration_seconds=perf_counter() - run_started,
                generated_at=datetime.now(timezone.utc),
            )

        def save_checkpoint(status: Literal["running", "passed", "failed"]):
            if checkpoint_path is None:
                return
            path = Path(checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(make_report(status).model_dump_json(indent=2), encoding="utf-8")

        for attempt_number in range(1, max_attempts + 1):
            remaining_total = total_timeout - (perf_counter() - run_started)
            if remaining_total <= 0:
                break
            current_timeout = min(attempt_timeout, remaining_total)
            use_local_patch = (
                repair_mode == "local_patch"
                and
                (attempt_number > 1 or initial_plan is not None)
                and previous_plan is not None
                and self._patch_agent is not None
            )
            dependency_repair_task = (
                build_dependency_repair_task(previous_plan, feedback, repair_scope)
                if (
                    use_local_patch
                    and previous_plan is not None
                    and feedback is not None
                    and repair_scope is not None
                )
                else None
            )
            # Blind by construction: the hidden rubric is never serialized here.
            # A dependency-only repair receives a minimal graph context instead of
            # operation/metric catalogs that cannot help with this edit.
            if dependency_repair_task is not None and previous_plan is not None:
                prompt = build_dependency_repair_prompt(
                    question, previous_plan, dependency_repair_task
                )
            else:
                selected_catalog_prompt = (
                    build_repair_operation_catalog(previous_plan, feedback)
                    if use_local_patch and previous_plan is not None and feedback is not None
                    else operation_catalog_prompt
                )
                prompt = (
                    f"使用者原始問題：\n{question}\n\n"
                    + selected_catalog_prompt
                    + "\n\n" + build_metric_catalog_prompt()
                    + demonstration_prompt
                )
            if feedback is not None and dependency_repair_task is None:
                previous_text = (
                    previous_plan.model_dump_json(indent=2)
                    if previous_plan is not None
                    else "上一輪沒有可通過 schema 的候選 Plan。"
                )
                if use_local_patch:
                    prompt += (
                        "\n\n這是不可直接覆寫的基礎 Plan：\n"
                        + previous_text
                        + "\n\nPython 授權的修改範圍：\n"
                        + repair_scope.model_dump_json(indent=2)
                        + "\n\n只回傳 QueryManifestPatch。保留 preserve 中已通過的項目，"
                        "只修正 missing 與 incorrect；不要回傳完整 Plan：\n"
                        + feedback.model_dump_json(indent=2)
                    )
                else:
                    prompt += (
                        "\n\n上一輪候選 Plan：\n"
                        + previous_text
                        + "\n\n上一輪 Plan 的具體缺失與錯誤如下。以候選 Plan 為基礎"
                        "修改，保留 preserve 中已通過的項目，只修正 missing 與 "
                        "incorrect；不要從頭改寫或假設未提供的驗收答案：\n"
                        + feedback.model_dump_json(indent=2)
                    )
            started_at = datetime.now(timezone.utc)
            attempt_started = perf_counter()
            diagnostic = PlannerAttemptDiagnostic(
                attempt=attempt_number,
                started_at=started_at,
                duration_seconds=0,
                schema_valid=False,
                generation_mode="local_patch" if use_local_patch else "full_plan",
                repair_scope=(
                    repair_scope.model_copy(deep=True)
                    if use_local_patch and repair_scope is not None else None
                ),
            )
            call_trace = ModelCallTraceHandler() if self._capture_model_calls else None
            try:
                invocation_request = {
                    "messages": [{"role": "user", "content": prompt}]
                }
                if use_local_patch:
                    invocation_request["base_plan"] = previous_plan
                result = _invoke_with_deadline(
                    self._patch_agent if use_local_patch else self._agent,
                    invocation_request,
                    current_timeout,
                    self._timeout_cleanup,
                    callbacks=[call_trace] if call_trace else None,
                )
                for key, value in _usage_from_result(result).items():
                    setattr(diagnostic, key, value)
                structured = result.get("structured_response")
                if structured is None:
                    raise ValueError("LLM returned no structured_response")
                structured_reference_corrections = list(
                    getattr(structured, "reference_corrections", [])
                )
                payload = (
                    structured.model_dump()
                    if hasattr(structured, "model_dump")
                    else structured
                )
                if isinstance(payload, dict):
                    diagnostic.raw_candidate_payload = payload
                if use_local_patch:
                    diagnostic.corrected_model_claims.extend(
                        result.get("patch_normalizations", [])
                    )
                    patch = QueryManifestPatch.model_validate(payload)
                    diagnostic.applied_patch = patch.model_copy(deep=True)
                    if repair_scope is None:
                        raise ValueError("local patch has no authorized repair scope")
                    validate_patch_scope(previous_plan, patch, repair_scope)
                    plan = apply_manifest_patch(previous_plan, patch)
                else:
                    plan = QueryManifestV2.model_validate(payload)
                diagnostic.schema_valid = True
                if plan.original_question != question:
                    raise ValueError("original_question must exactly match user input")
                diagnostic.corrected_model_claims.extend(_normalize_unverified_entities(plan))
                diagnostic.corrected_model_claims.extend(
                    structured_reference_corrections
                )
                diagnostic.corrected_model_claims.extend(plan.reference_corrections)
                diagnostic.corrected_model_claims.extend(
                    normalize_plan_operations(plan)
                )
                # Runtime metadata is program authority, never model authority.
                plan.generated_at = datetime.now(timezone.utc)
                diagnostic.candidate_plan = plan.model_copy(deep=True)
                rule_evaluation = evaluate_operation_rules(plan)
                diagnostic.operation_rule_evaluation = rule_evaluation
                rule_feedback = [
                    {
                        "area": item.rule_id,
                        "problem": (
                            (f"step_id={item.step_id}; " if item.step_id else "")
                            + item.problem
                        ),
                        "required_correction": item.required_correction,
                    }
                    for item in rule_evaluation.blocking_findings
                ]
                proposed = [
                    step.proposed_operation_name
                    for step in plan.plan_steps
                    if step.operation_status == "proposed"
                ]
                if rubric is not None:
                    acceptance = evaluate_plan(rubric, plan)
                    diagnostic.semantic_acceptance = acceptance
                    current_accepted = set(acceptance.accepted_capabilities)
                    diagnostic.regressed_components = sorted(
                        previous_accepted - current_accepted
                    )
                    if not acceptance.passed or not rule_evaluation.passed:
                        candidate_feedback = build_repair_feedback(acceptance)
                        candidate_feedback.incorrect.extend(rule_feedback)
                        candidate_feedback.incorrect.extend(
                            {
                                "area": "regression",
                                "problem": f"上一輪已通過但本輪遺失：{item}",
                                "required_correction": "恢復該能力，並保留本輪已通過項目。",
                            }
                            for item in diagnostic.regressed_components
                        )
                        issue_count = _plan_issue_count(acceptance, rule_evaluation)
                        promotes_base = (
                            previous_plan is None
                            or (
                                not diagnostic.regressed_components
                                and previous_issue_count is not None
                                and issue_count < previous_issue_count
                            )
                        )
                        if promotes_base:
                            diagnostic.promoted_to_repair_base = True
                            previous_plan = plan.model_copy(deep=True)
                            previous_accepted = current_accepted
                            previous_issue_count = issue_count
                            feedback = select_repair_batch(candidate_feedback)
                            base_feedback = feedback
                            repair_scope = build_repair_scope(
                                previous_plan, acceptance, rule_evaluation, feedback
                            )
                        else:
                            rejection = RepairFeedback(incorrect=[{
                                "area": "non_improving_patch",
                                "problem": (
                                    "本輪 patch 未減少問題，或破壞了已通過能力；"
                                    "已退回上一個較佳 Plan。"
                                ),
                                "required_correction": (
                                    "以授權範圍內的更小修改處理仍未解決問題，"
                                    "不得再次改動 preserve 項目。"
                                ),
                            }])
                            regression_feedback = RepairFeedback(
                                incorrect=[
                                    item for item in candidate_feedback.incorrect
                                    if item.get("area") == "regression"
                                ]
                            )
                            feedback = _merge_feedback(
                                base_feedback, regression_feedback, rejection
                            )
                            diagnostic.errors.append(
                                "local patch rejected because it did not improve the repair base"
                            )
                        diagnostic.repair_feedback_sent_next = feedback
                        if not acceptance.passed:
                            diagnostic.errors.append("plan failed hidden semantic acceptance")
                        if not rule_evaluation.passed:
                            diagnostic.errors.append("plan failed general operation rules")
                    else:
                        final_plan = plan
                else:
                    if proposed or not rule_evaluation.passed:
                        diagnostic.errors.append(
                            "plan requires review or repair"
                        )
                        candidate_feedback = RepairFeedback(
                            incorrect=[
                                {
                                    "area": "proposed_operation",
                                    "problem": f"新 operation 等待人工審核：{item}",
                                    "required_correction": (
                                        "停止自動執行，待人工審核並更新 OPERATION_TABLE。"
                                    ),
                                }
                                for item in proposed
                                if item
                            ] + rule_feedback
                        )
                        issue_count = len(rule_evaluation.blocking_findings) + len(proposed)
                        promotes_base = (
                            previous_plan is None
                            or (
                                previous_issue_count is not None
                                and issue_count < previous_issue_count
                            )
                        )
                        if promotes_base:
                            diagnostic.promoted_to_repair_base = True
                            previous_plan = plan.model_copy(deep=True)
                            previous_issue_count = issue_count
                            feedback = select_repair_batch(candidate_feedback)
                            base_feedback = feedback
                            repair_scope = build_repair_scope(
                                previous_plan, None, rule_evaluation, feedback
                            )
                        else:
                            feedback = _merge_feedback(
                                base_feedback,
                                RepairFeedback(incorrect=[{
                                    "area": "non_improving_patch",
                                    "problem": "本輪 patch 未減少阻擋問題；已退回較佳 Plan。",
                                    "required_correction": "只修改仍未解決的授權範圍。",
                                }]),
                            )
                            diagnostic.errors.append(
                                "local patch rejected because it did not improve the repair base"
                            )
                        diagnostic.repair_feedback_sent_next = feedback
                    else:
                        final_plan = plan
            except Exception as exc:
                diagnostic.errors.append(f"{type(exc).__name__}: {exc}")
                if isinstance(exc, ModelAttemptTimeout):
                    if call_trace:
                        call_trace.mark_running_calls_timed_out()
                    diagnostic.model_generation_timeout = True
                    for key, value in exc.cleanup.items():
                        if hasattr(diagnostic, key):
                            setattr(diagnostic, key, value)
                exception_feedback = RepairFeedback(
                    incorrect=[
                        {
                            "area": "schema_or_integrity",
                            "problem": str(exc),
                            "required_correction": "修正指定錯誤並保留原始問題。",
                        }
                    ]
                )
                feedback = _merge_feedback(base_feedback or feedback, exception_feedback)
                diagnostic.repair_feedback_sent_next = feedback
            finally:
                if call_trace:
                    diagnostic.model_calls = call_trace.snapshot()
                diagnostic.duration_seconds = perf_counter() - attempt_started
                if (
                    use_local_patch
                    and not diagnostic.schema_valid
                    and diagnostic.errors
                    and not diagnostic.model_generation_timeout
                ):
                    failure_fingerprint = "\n".join(sorted(diagnostic.errors))
                    if failure_fingerprint in seen_patch_failure_fingerprints:
                        diagnostic.repeated_repair_failure = True
                        diagnostic.errors.append(
                            "repeated_repair_failure: the same invalid repair was rejected twice"
                        )
                    else:
                        seen_patch_failure_fingerprints.add(failure_fingerprint)
                attempts.append(diagnostic)
                timed_out = diagnostic.model_generation_timeout
                save_checkpoint("failed" if timed_out else "running")
            if final_plan is not None:
                break
            if timed_out:
                break
            if diagnostic.repeated_repair_failure:
                break

        status = "passed" if final_plan is not None else "failed"
        report = make_report(status)
        save_checkpoint(status)
        return report

    def run(
        self,
        question: str,
        acceptance_case: PlannerAcceptanceCase | None = None,
    ) -> QueryManifestV2:
        report = self.run_with_report(question, acceptance_case)
        if report.final_plan is None:
            last_errors = report.attempts[-1].errors
            raise NaturalLanguagePlannerError(
                f"QueryManifestV2 failed after {report.repair_count} repair attempts: "
                + (last_errors[-1] if last_errors else "unknown acceptance failure")
            )
        return report.final_plan
