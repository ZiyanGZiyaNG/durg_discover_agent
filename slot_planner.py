"""Slot planner: the model fills a small table, Python assembles the Plan.

The model answers only what needs language understanding:

* entities - what the question names and its type;
* paths    - which entity types the question walks through (disease -> target);
* ranking, checks, evidence - conditions the question states explicitly.

Everything else in a ``QueryManifestV2`` (step IDs, ``depends_on``, inputs,
outputs, sources, safeguards) is filled from ``OPERATION_TABLE`` and its path
index, so the planner and the executor always read the same names.
"""

from __future__ import annotations

import json
import os
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from planner_acceptance import (
    ANALYSIS_INDEX,
    ATTRIBUTE_INDEX,
    CHECK_INDEX,
    COMBINE_INDEX,
    DEFAULT_HOPS,
    DEFAULT_PRODUCERS,
    FILTER_MEANINGS,
    FILTER_RULES,
    FILTER_STEPS,
    FILTER_VALUES,
    HOP_INDEX,
    OPERATION_TABLE,
    PER_ENTITY_HOPS,
    RANK_INDEX,
    RESOLVE_INDEX,
    RANKING_KEY_EQUIVALENCES,
    PlanAcceptanceResult,
    PlannerAcceptanceCase,
    evaluate_plan,
)
from operation_rules import (
    _METRIC_REQUIREMENTS,
    OperationRuleEvaluation,
    _grounding_text,
    evaluate_operation_rules,
    mention_is_grounded,
)
from schemas import (
    PlannedEntityV2,
    QueryCharacteristicsV2,
    QueryManifestV2,
    ResearchPlanStepV2,
    StepInputRef,
)


SlotEntityType = Literal["disease", "target", "drug"]
# Filters and evidence may also be about the association edge itself.
SlotRowType = Literal["disease", "target", "drug", "association"]


class SlotEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mention: str = Field(min_length=1)
    type: Literal["disease", "target", "drug", "mechanism", "tissue"]
    variants: list[str] = Field(default_factory=list)
    # Other names the question uses for the same entity ("胰臟癌 ... PDAC").
    aliases: list[str] = Field(default_factory=list)
    # "context": named to scope the question (a family, pathway or tissue
    # group); it is recorded but not resolved or walked.
    role: Literal["named", "context"] = "named"

    @model_validator(mode="after")
    def only_context_can_be_mechanism_or_tissue(self) -> "SlotEntity":
        if self.type in {"mechanism", "tissue"} and self.role != "context":
            raise ValueError(f"{self.type} entities are context: set role to context")
        return self


class SlotPath(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: str = Field(min_length=1)
    types: list[SlotEntityType] = Field(min_length=2)
    # Named entities on a middle node, e.g. "PDAC drugs (such as erlotinib)".
    via: list[str] = Field(default_factory=list)
    # A named entity the path ends at: the one edge between start and end.
    end: str | None = None


class SlotRanking(BaseModel):
    model_config = ConfigDict(extra="forbid")

    on_type: SlotEntityType
    metric: str = Field(min_length=1)
    direction: Literal["descending", "ascending"] = "descending"


class SlotCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1)
    entity: str = Field(min_length=1)


class SlotEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attribute: str = Field(min_length=1)
    of_type: SlotRowType


class SlotFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    on_type: SlotRowType
    value: Any = True


class SlotAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)


class SlotCombine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1)
    paths: list[int] = Field(min_length=1)  # indexes into QuerySlots.paths


class QuerySlots(BaseModel):
    """The small table the model fills; nothing in it is a Plan field."""

    model_config = ConfigDict(extra="forbid")

    entities: list[SlotEntity] = Field(min_length=1)
    paths: list[SlotPath] = Field(default_factory=list)
    combine: list[SlotCombine] = Field(default_factory=list)
    filters: list[SlotFilter] = Field(default_factory=list)
    ranking: list[SlotRanking] = Field(default_factory=list)
    checks: list[SlotCheck] = Field(default_factory=list)
    evidence: list[SlotEvidence] = Field(default_factory=list)
    analyses: list[SlotAnalysis] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def drop_empty_unknown_fields(cls, value: Any) -> Any:
        # An empty extra list ("filters": []) carries no claim; a non-empty one
        # is a condition this table cannot express yet and must stay an error.
        if isinstance(value, dict):
            return {
                key: item
                for key, item in value.items()
                if key in cls.model_fields or item not in (None, [], {}, "")
            }
        return value


class SlotError(ValueError):
    """The table is inconsistent with the question or the operation table."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().casefold()


def _find_entity(slots: QuerySlots, mention: str) -> int | None:
    """Index of the entity a path/check names: exact, else unique containment."""
    wanted = _norm(mention)
    for index, entity in enumerate(slots.entities):
        if wanted in {_norm(name) for name in [entity.mention, *entity.aliases]}:
            return index
    contained = [
        index
        for index, entity in enumerate(slots.entities)
        if wanted in _norm(entity.mention) or _norm(entity.mention) in wanted
    ]
    return contained[0] if len(contained) == 1 else None


def _merge_groups(slots: QuerySlots) -> list[list[int]]:
    """Paths of the same shape from different named entities.

    Their results must live in one per-entity step: two separate steps would
    publish the same capability and the second would overwrite the first.
    """
    by_shape: dict[tuple, list[int]] = {}
    for position, path in enumerate(slots.paths):
        end = _find_entity(slots, path.end) if path.end else None
        by_shape.setdefault((tuple(path.types), end), []).append(position)
    groups = []
    for members in by_shape.values():
        starts = {_find_entity(slots, slots.paths[i].start) for i in members}
        if len(starts - {None}) >= 2:
            groups.append(members)
    return groups


_THRESHOLD_OPERATORS = {"gt", "gte", "lt", "lte"}
# Stated thresholds the adapters apply themselves (no filter step).
_THRESHOLD_FILTERS = {"max_genetic_somatic_score"}


def _is_threshold(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and set(value) <= _THRESHOLD_OPERATORS
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value.values())
    )


def _relation(path: SlotPath) -> str:
    """Relation of a path's first hop: "exact" for an edge to a named end."""
    if path.end:
        return "exact"
    operation = DEFAULT_HOPS.get((path.types[0], path.types[1]))
    return OPERATION_TABLE[operation].path_role.name if operation else "default"


def _group_resolver(path: SlotPath) -> tuple[str, str]:
    return (path.types[-1], "pairs") if path.end else (path.types[0], "many")


def validate_slots(question: str, slots: QuerySlots) -> list[str]:
    """Deterministic checks on the small table; each problem names one row."""

    problems: list[str] = []
    grounded_question = _grounding_text(question)
    mentions = [_norm(entity.mention) for entity in slots.entities]
    if len(set(mentions)) != len(mentions):
        problems.append("entities: the same mention is listed twice")
    for entity in slots.entities:
        for name in [entity.mention, *entity.aliases]:
            if not mention_is_grounded(name, grounded_question):
                problems.append(
                    f"entities: {name!r} is not written in the question; "
                    "copy the mention exactly"
                )
        if entity.variants and entity.type != "target":
            problems.append(f"entities: only targets have variants ({entity.mention!r})")
        for variant in entity.variants:
            if _grounding_text(variant) not in grounded_question:
                problems.append(
                    f"entities: variant {variant!r} of {entity.mention!r} is not "
                    "written in the question"
                )
        variant_key = "variant" if entity.variants else "default"
        if entity.role == "named" and (entity.type, variant_key) not in RESOLVE_INDEX:
            problems.append(
                f"entities: no resolver for {entity.type} ({variant_key})"
            )

    used: set[int] = set()
    reached: set[str] = set()
    starts: set[str] = set()
    for path in slots.paths:
        index = _find_entity(slots, path.start)
        if index is None:
            problems.append(f"paths: start {path.start!r} is not one of the entities")
            continue
        used.add(index)
        entity = slots.entities[index]
        starts.add(entity.type)
        if path.types[0] != entity.type:
            problems.append(
                f"paths: path from {path.start!r} must begin with {entity.type!r}, "
                f"not {path.types[0]!r}"
            )
        for before, after in zip(path.types, path.types[1:]):
            if (before, after) not in DEFAULT_HOPS:
                problems.append(f"paths: no operation walks {before} -> {after}")
            reached.add(after)
        if path.end:
            end_index = _find_entity(slots, path.end)
            if end_index is None:
                problems.append(f"paths: end {path.end!r} is not one of the entities")
            else:
                used.add(end_index)
                end_type = slots.entities[end_index].type
                if len(path.types) != 2 or path.types[-1] != end_type:
                    problems.append(
                        f"paths: a path ending at {path.end!r} is one hop ending at {end_type!r}"
                    )
                elif (path.types[0], end_type, "exact") not in HOP_INDEX:
                    problems.append(
                        f"paths: no operation reads the edge {path.types[0]} -> {end_type}"
                    )
                elif (end_type, "pair") not in RESOLVE_INDEX:
                    problems.append(f"paths: no resolver for a {path.types[0]}-{end_type} pair")
        middle = set(path.types[1:-1])
        via_types = set()
        for name in path.via:
            via_index = _find_entity(slots, name)
            if via_index is None:
                problems.append(f"paths: via {name!r} is not one of the entities")
                continue
            used.add(via_index)
            via_types.add(slots.entities[via_index].type)
            if slots.entities[via_index].type not in middle:
                problems.append(
                    f"paths: via {name!r} is a {slots.entities[via_index].type}, "
                    "not a middle node of the path"
                )
        for via_type in via_types:
            if (entity.type, f"with_{via_type}s") not in RESOLVE_INDEX:
                problems.append(
                    f"paths: no resolver for a {entity.type} together with {via_type}s"
                )

    for group in _merge_groups(slots):
        types = slots.paths[group[0]].types
        shape = " -> ".join(types)
        names = [slots.paths[i].start for i in group]
        if len(types) != 2:
            problems.append(
                f"paths: {names} all walk {shape}; only a single hop can keep "
                "several entities apart in one step"
            )
        elif (*types, _relation(slots.paths[group[0]])) not in PER_ENTITY_HOPS:
            problems.append(
                f"paths: {names} all walk {shape}, and no operation keeps their "
                "results apart, so they would overwrite each other"
            )
        elif _group_resolver(slots.paths[group[0]]) not in RESOLVE_INDEX:
            problems.append(f"paths: no resolver for several {types[0]} entities")
        for i in group:
            index = _find_entity(slots, slots.paths[i].start)
            if index is not None and slots.entities[index].variants:
                problems.append(
                    f"paths: {slots.paths[i].start!r} has variants and cannot be "
                    "merged with other entities"
                )

    # What each row type carries, so a ranking metric is checked against the
    # rows it orders rather than against the plan as a whole.
    carried: dict[str, set[str]] = {}
    for path in slots.paths:
        for before, after in zip(path.types, path.types[1:]):
            operation = DEFAULT_HOPS.get((before, after))
            if operation:
                carried.setdefault(after, set()).update(OPERATION_TABLE[operation].provides)
    for item in slots.evidence:
        operation = ATTRIBUTE_INDEX.get((item.of_type, item.attribute))
        if operation:
            carried.setdefault(item.of_type, set()).update(OPERATION_TABLE[operation].provides)

    combine_kinds = {name for name, _ in COMBINE_INDEX}
    for combine in slots.combine:
        if any(i < 0 or i >= len(slots.paths) for i in combine.paths):
            problems.append(
                f"combine: path index {combine.paths} is outside paths "
                f"(0 to {len(slots.paths) - 1})"
            )
        elif combine.kind not in combine_kinds:
            problems.append(f"combine: unknown kind {combine.kind!r}")
        elif combine.kind == "join_path":
            if len(combine.paths) != 1:
                problems.append("combine: join_path joins exactly one path")
                continue
            types = slots.paths[combine.paths[0]].types
            if ("join_path", tuple(types)) not in COMBINE_INDEX:
                problems.append(f"combine: no operation joins {' -> '.join(types)}")
                continue
            # A joined row keeps the evidence of every edge on its path.
            for before, after in zip(types, types[1:]):
                carried.setdefault(types[-1], set()).update(
                    OPERATION_TABLE[DEFAULT_HOPS[(before, after)]].provides
                )
        else:
            shapes = {tuple(slots.paths[i].types) for i in combine.paths}
            starts = {_find_entity(slots, slots.paths[i].start) for i in combine.paths}
            if len(combine.paths) < 2 or len(starts) != len(combine.paths):
                problems.append(
                    f"combine: {combine.kind} needs paths from at least two different entities"
                )
            elif len(shapes) != 1:
                problems.append(f"combine: {combine.kind} needs paths of the same shape")
            elif (combine.kind, next(iter(shapes))) not in COMBINE_INDEX:
                problems.append(
                    f"combine: no operation does {combine.kind} over "
                    f"{' -> '.join(next(iter(shapes)))}"
                )

    edges = any(path.end for path in slots.paths)
    for item in slots.filters:
        if item.name not in FILTER_MEANINGS:
            problems.append(f"filters: {item.name!r} is not a filter the table supports")
        elif item.on_type not in reached and not (item.on_type == "association" and edges):
            problems.append(f"filters: no path reaches {item.on_type!r} rows to filter")
        rule = FILTER_RULES.get(item.name)
        if rule is not None:
            if item.on_type not in FILTER_STEPS:
                problems.append(f"filters: {item.on_type} rows have no filter step")
            if "condition" in rule:
                if item.value is not True:
                    problems.append(f"filters: {item.name} takes value true")
            elif not _is_threshold(item.value):
                problems.append(
                    f'filters: {item.name} needs a threshold such as {{"gt": 0.7}}, '
                    f"not {item.value!r}"
                )
        elif item.name in _THRESHOLD_FILTERS and not _is_threshold(item.value):
            problems.append(
                f'filters: {item.name} needs a threshold such as {{"gte": 0.5}}, '
                f"not {item.value!r}"
            )
        allowed = FILTER_VALUES.get(item.name)
        if allowed is not None:
            values = item.value if isinstance(item.value, list) else [item.value]
            unknown = [v for v in values if v not in allowed]
            if unknown or not values:
                problems.append(
                    f"filters: {item.name} values {unknown or values} are not among {allowed}"
                )
    for analysis in slots.analyses:
        if analysis.name not in ANALYSIS_INDEX:
            problems.append(f"analyses: {analysis.name!r} is not an analysis the table has")

    for ranking in slots.ranking:
        if ranking.on_type not in reached:
            problems.append(
                f"ranking: no path reaches {ranking.on_type!r} rows to rank"
            )
        if (ranking.on_type, "default") not in RANK_INDEX:
            problems.append(f"ranking: {ranking.on_type!r} rows cannot be ranked")
        if ranking.metric not in _METRIC_REQUIREMENTS:
            problems.append(f"ranking: {ranking.metric!r} is not a reviewed metric")
        elif (
            ranking.on_type in reached
            and _METRIC_REQUIREMENTS[ranking.metric] not in carried.get(ranking.on_type, set())
        ):
            problems.append(
                f"ranking: {ranking.on_type} rows do not carry {ranking.metric}; it "
                "comes from an earlier hop, so the path must be joined (combine join_path)"
            )

    for check in slots.checks:
        index = _find_entity(slots, check.entity)
        if index is None:
            problems.append(f"checks: {check.entity!r} is not one of the entities")
            continue
        used.add(index)
        entity_type = slots.entities[index].type
        if (entity_type, check.kind) not in CHECK_INDEX:
            problems.append(
                f"checks: {check.kind!r} cannot be checked for a {entity_type}"
            )

    for item in slots.evidence:
        if (item.of_type, item.attribute) not in ATTRIBUTE_INDEX:
            problems.append(
                f"evidence: {item.attribute!r} is not available for {item.of_type}"
            )
        elif item.of_type in reached | starts or (item.of_type == "association" and edges):
            pass
        else:
            named = [
                i for i, e in enumerate(slots.entities)
                if e.type == item.of_type and e.role == "named"
            ]
            if named:
                used.update(named)  # evidence about the named entities themselves
            else:
                problems.append(f"evidence: no path or named entity yields {item.of_type!r} rows")

    if not slots.paths and not slots.evidence:
        problems.append("paths: the table needs a path, or evidence about named entities")
    for index, entity in enumerate(slots.entities):
        if entity.role == "context":
            continue
        if index not in used:
            problems.append(
                f"entities: {entity.mention!r} is not used by any path or check"
            )
    return problems


def _hop_for(before: str, after: str, slots: QuerySlots) -> str:
    """The default hop, unless a stated filter needs a variant that supports it."""
    default = DEFAULT_HOPS[(before, after)]
    wanted = {item.name for item in slots.filters if item.on_type == after}
    missing = wanted - set(OPERATION_TABLE[default].supported_filters)
    if not missing:
        return default
    variants = [
        operation
        for (start, end, _), operation in HOP_INDEX.items()
        if (start, end) == (before, after)
        and missing <= set(OPERATION_TABLE[operation].supported_filters)
    ]
    return variants[0] if len(variants) == 1 else default


class _Builder:
    """Collect steps in order; every field comes from OPERATION_TABLE."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.by_id: dict[str, dict[str, Any]] = {}
        self.notes: list[str] = []

    def add(
        self,
        operation: str,
        depends_on: list[str],
        entity_refs: list[str],
        description: str,
        filters: dict[str, Any] | None = None,
        ranking: dict[str, Any] | None = None,
        aggregation: dict[str, Any] | None = None,
    ) -> str:
        entry = OPERATION_TABLE[operation]
        number = len(self.steps) + 1
        step_id = f"{operation}_{number}"
        output = (
            f"{entry.provides[0]}_{number}"
            if entry.provides
            else f"{operation}_result_{number}"
        )
        depends_on = list(dict.fromkeys(depends_on))
        step = {
            "step_id": step_id,
            "operation": operation,
            "description": description,
            "depends_on": depends_on,
            "entity_refs": entity_refs,
            "sources": list(entry.default_sources),
            "outputs": [output],
            "filters": filters or {},
            "ranking": ranking,
            "aggregation": aggregation,
            "safeguards": list(entry.required_safeguards),
            "provides": list(entry.provides),
            "inputs": {
                dependency: StepInputRef(
                    step_id=dependency, output=self.by_id[dependency]["outputs"][0]
                )
                for dependency in depends_on
            },
        }
        self.steps.append(step)
        self.by_id[step_id] = step
        return step_id

    def producers(self, capability: str) -> list[str]:
        return [
            step["step_id"]
            for step in self.steps
            if capability in OPERATION_TABLE[step["operation"]].provides
        ]

    def upstream_capabilities(self, step_ids: list[str]) -> set[str]:
        found: set[str] = set()
        pending = list(step_ids)
        seen: set[str] = set()
        while pending:
            step_id = pending.pop()
            if step_id in seen:
                continue
            seen.add(step_id)
            step = self.by_id[step_id]
            found.update(OPERATION_TABLE[step["operation"]].provides)
            pending.extend(step["depends_on"])
        return found


def assemble_plan(question: str, slots: QuerySlots) -> tuple[QueryManifestV2, list[str]]:
    """Build a QueryManifestV2 from a validated table. Returns (plan, notes)."""

    problems = validate_slots(question, slots)
    if problems:
        raise SlotError(problems)

    builder = _Builder()
    entities: list[PlannedEntityV2] = []
    refs_by_entity: dict[int, list[str]] = {}
    resolve_step: dict[int, str] = {}
    type_counts: dict[str, int] = {}

    def new_ref(entity_type: str) -> str:
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        return f"{entity_type}_{type_counts[entity_type]}"

    # 1. Every entity gets its resolve step; entities whose same-shape paths
    #    are merged share one per-entity resolve.
    groups = _merge_groups(slots)
    group_of_entity: dict[int, int] = {}
    group_of_path: dict[int, int] = {}
    for number, group in enumerate(groups):
        for i in group:
            group_of_path[i] = number
            group_of_entity[_find_entity(slots, slots.paths[i].start)] = number
    for index, entity in enumerate(slots.entities):
        mentions = (
            [f"{entity.mention} {variant}" for variant in entity.variants]
            if entity.variants
            else [entity.mention]
        )
        refs = []
        for mention in mentions:
            ref = new_ref(entity.type)
            refs.append(ref)
            entities.append(
                PlannedEntityV2(
                    entity_ref=ref,
                    entity_type=entity.type,
                    mention=mention,
                    resolution_status="unresolved",
                )
            )
        refs_by_entity[index] = refs
    group_refs: dict[int, list[str]] = {}

    def resolve_together(members: list[int], key: tuple[str, str]) -> list[str]:
        operation = RESOLVE_INDEX[key]
        refs = [ref for i in members for ref in refs_by_entity[i]]
        names = "」「".join(slots.entities[i].mention for i in members)
        step_id = builder.add(
            operation, [], refs,
            f"分別解析「{names}」。{OPERATION_TABLE[operation].description}",
        )
        for i in members:
            resolve_step[i] = step_id
        return refs

    # Same-shape paths share one resolver (with their shared end, if any).
    for number, group in enumerate(groups):
        first = slots.paths[group[0]]
        members = sorted({_find_entity(slots, slots.paths[i].start) for i in group})
        if first.end:
            members.append(_find_entity(slots, first.end))
        group_refs[number] = resolve_together(members, _group_resolver(first))
    # A path resolves its named middle entities, or its named end, with its
    # start, using the table's combined resolver.
    for path in slots.paths:
        first = _find_entity(slots, path.start)
        if first in resolve_step:
            continue
        if path.via:
            members = [first] + [_find_entity(slots, name) for name in path.via]
            via_type = slots.entities[members[1]].type
            resolve_together(members, (slots.entities[first].type, f"with_{via_type}s"))
        elif path.end:
            last = _find_entity(slots, path.end)
            resolve_together([first, last], (slots.entities[last].type, "pair"))
    # Several remaining named entities of one type resolve together.
    remaining: dict[str, list[int]] = {}
    for index, entity in enumerate(slots.entities):
        if index not in resolve_step and entity.role == "named" and not entity.variants:
            remaining.setdefault(entity.type, []).append(index)
    for entity_type, members in remaining.items():
        if len(members) >= 2 and (entity_type, "many") in RESOLVE_INDEX:
            resolve_together(members, (entity_type, "many"))
    for index, entity in enumerate(slots.entities):
        if index in resolve_step or entity.role == "context":
            continue
        operation = RESOLVE_INDEX[
            (entity.type, "variant" if entity.variants else "default")
        ]
        resolve_step[index] = builder.add(
            operation, [], refs_by_entity[index],
            f"解析「{entity.mention}」。{OPERATION_TABLE[operation].description}",
        )

    # 2. Every hop is looked up in the hop index.
    rows_of: dict[str, list[str]] = {}  # entity type -> steps yielding those rows
    tails: dict[int, str] = {}  # variant entity -> last step of its branch
    hop_memo: dict[tuple[str, str], str] = {}
    group_hop: dict[int, str] = {}
    path_labels: list[str] = []
    path_steps: list[list[str]] = []
    for path_index, path in enumerate(slots.paths):
        index = _find_entity(slots, path.start)
        entity = slots.entities[index]
        path_labels.append(f"「{entity.mention}」{' → '.join(path.types)}")
        if path_index in group_of_path:
            number = group_of_path[path_index]
            if number not in group_hop:
                before, after = path.types
                operation = PER_ENTITY_HOPS[(before, after, _relation(path))]
                role = OPERATION_TABLE[operation].path_role
                group_hop[number] = builder.add(
                    operation,
                    [resolve_step[index]],
                    group_refs[number],
                    f"取得{role.meaning}。{OPERATION_TABLE[operation].description}",
                    filters=dict(role.fixed_filters),
                )
                rows_of.setdefault(after, []).append(group_hop[number])
                if path.end:
                    rows_of.setdefault("association", []).append(group_hop[number])
            path_steps.append([group_hop[number]])
            continue
        resolve_role = OPERATION_TABLE[builder.by_id[resolve_step[index]]["operation"]].path_role
        current = resolve_step[index]
        path_steps.append([])
        via_refs: dict[str, list[str]] = {}
        for name in path.via:
            via_index = _find_entity(slots, name)
            via_refs.setdefault(slots.entities[via_index].type, []).extend(
                refs_by_entity[via_index]
            )
        for position, (before, after) in enumerate(zip(path.types, path.types[1:])):
            operation = (
                HOP_INDEX[(before, after, "exact")] if path.end else _hop_for(before, after, slots)
            )
            key = (current, operation)
            if key in hop_memo:
                current = hop_memo[key]
                path_steps[-1].append(current)
                continue
            role = OPERATION_TABLE[operation].path_role
            filters = dict(role.fixed_filters)
            if position == 0 and resolve_role.scope_filter:
                if resolve_role.scope_filter not in OPERATION_TABLE[operation].supported_filters:
                    raise SlotError(
                        [f"{operation} cannot keep the {resolve_role.scope_filter} scope"]
                    )
                filters[resolve_role.scope_filter] = list(entity.variants)
            if position == 0:
                entity_refs = list(refs_by_entity[index])
                if path.end:
                    entity_refs += refs_by_entity[_find_entity(slots, path.end)]
            else:
                entity_refs = via_refs.get(before, [])
            current = builder.add(
                operation,
                [current],
                entity_refs,
                f"取得{role.meaning}。{OPERATION_TABLE[operation].description}",
                filters=filters,
            )
            hop_memo[key] = current
            path_steps[-1].append(current)
            rows_of.setdefault(after, []).append(current)
            if path.end:
                rows_of.setdefault("association", []).append(current)
        if resolve_role.scope_grouping and len(entity.variants) >= 2:
            tails[index] = current

    # 3. Evidence attributes attach to the steps that yield those rows.
    for item in slots.evidence:
        operation = ATTRIBUTE_INDEX[(item.of_type, item.attribute)]
        role = OPERATION_TABLE[operation].path_role
        upstream = rows_of.get(item.of_type, [])
        entity_refs: list[str] = []
        if not upstream:
            # Evidence about a named entity itself, e.g. a named drug's mechanism.
            named = [
                i for i, e in enumerate(slots.entities)
                if e.type == item.of_type and e.role == "named"
            ]
            upstream = list(dict.fromkeys(resolve_step[i] for i in named))
            entity_refs = [ref for i in named for ref in refs_by_entity[i]]
        step_id = builder.add(
            operation,
            upstream,
            entity_refs,
            f"取得{role.meaning}。{OPERATION_TABLE[operation].description}",
            filters=dict(role.fixed_filters),
        )
        rows_of.setdefault(f"{item.of_type}:evidence", []).append(step_id)
        for follow_up in role.then:
            follow_role = OPERATION_TABLE[follow_up].path_role
            meaning = f"{follow_role.meaning}。" if follow_role and follow_role.meaning else ""
            follow_id = builder.add(
                follow_up,
                upstream,
                entity_refs,
                f"{meaning}{OPERATION_TABLE[follow_up].description}",
            )
            rows_of.setdefault(f"{item.of_type}:evidence", []).append(follow_id)
        for entity_index, tail in list(tails.items()):
            if tail in upstream:
                tails[entity_index] = step_id

    # 4. Combines: the combined result becomes the row table for its end type.
    for combine in slots.combine:
        path = slots.paths[combine.paths[0]]
        if combine.kind != "join_path":
            operation = COMBINE_INDEX[(combine.kind, tuple(path.types))]
            role = OPERATION_TABLE[operation].path_role
            inputs = list(dict.fromkeys(path_steps[i][-1] for i in combine.paths))
            refs = [
                ref
                for i in combine.paths
                for ref in refs_by_entity[_find_entity(slots, slots.paths[i].start)]
            ]
            refs = list(dict.fromkeys(refs))
            step_id = builder.add(
                operation,
                inputs,
                [],
                f"{role.meaning}。{OPERATION_TABLE[operation].description}",
                filters=dict(role.fixed_filters),
                aggregation={
                    role.aggregation_key: {"entity_refs": refs, "per_entity_steps": inputs}
                } if role.aggregation_key else None,
            )
            rows_of[path.types[-1]] = [step_id]
            for follow_up in role.then:
                follow_role = OPERATION_TABLE[follow_up].path_role
                builder.add(
                    follow_up,
                    [step_id],
                    [],
                    f"{follow_role.meaning}。{OPERATION_TABLE[follow_up].description}",
                    aggregation={
                        follow_role.aggregation_key: {
                            "entity_refs": refs,
                            "within_step": step_id,
                            "combined_score": False,
                        }
                    } if follow_role.aggregation_key else None,
                )
            continue
        operation = COMBINE_INDEX[(combine.kind, tuple(path.types))]
        role = OPERATION_TABLE[operation].path_role
        edges = path_steps[combine.paths[0]]
        start = _find_entity(slots, path.start)
        aggregation = {
            role.aggregation_key: {
                "path": " -> ".join(path.types),
                "edge_steps": edges,
                "start_entity_refs": refs_by_entity[start],
            }
        } if role.aggregation_key else None
        step_id = builder.add(
            operation,
            edges,
            [],
            f"{role.meaning}。{OPERATION_TABLE[operation].description}",
            filters=dict(role.fixed_filters),
            aggregation=aggregation,
        )
        rows_of[path.types[-1]] = [step_id]

    # 5. Analyses the question names; their prerequisites come from the table.
    def ensure(capability: str, wanted_by: str) -> str:
        found = builder.producers(capability)
        if found:
            return found[-1]
        candidates = sorted(
            op for op, entry in OPERATION_TABLE.items() if capability in entry.provides
        )
        operation = DEFAULT_PRODUCERS.get(capability) or (
            candidates[0] if len(candidates) == 1 else None
        )
        if operation is None:
            raise SlotError([
                f"analyses: {wanted_by} needs {capability}; the paths must provide it "
                f"(one of {candidates})"
            ])
        step_id = add_with_prerequisites(operation)
        builder.notes.append(f"{wanted_by} needs {capability}; added {step_id} from the table")
        return step_id

    def add_with_prerequisites(operation: str) -> str:
        entry = OPERATION_TABLE[operation]
        role = entry.path_role
        if role is not None and role.needs_stated_filter and not any(
            item.name == role.needs_stated_filter for item in slots.filters
        ):
            raise SlotError([
                f"analyses: {operation} needs a stated {role.needs_stated_filter} "
                "threshold; the question gives none and the project has not defined one"
            ])
        depends = [ensure(capability, operation) for capability in entry.requires_all]
        if entry.requires_any:
            have = [
                builder.producers(capability)[-1]
                for capability in entry.requires_any
                if builder.producers(capability)
            ]
            depends += have or [ensure(entry.requires_any[0], operation)]
        # An analysis ranks by its default metric, so the evidence that metric
        # is registered against must be upstream too, even when another
        # producer already satisfied requires_any.
        if role is not None and role.kind == "analysis" and role.default_metric:
            needed = _METRIC_REQUIREMENTS.get(role.default_metric)
            if (
                needed
                and needed not in entry.provides
                and needed not in builder.upstream_capabilities(depends)
            ):
                depends.append(ensure(needed, operation))
        depends = list(dict.fromkeys(depends))
        ranking = None
        if role is not None and role.kind == "analysis" and role.default_metric:
            ranking = {
                "primary_metric": role.default_metric,
                "direction": "descending",
                "secondary_metrics": [],
            }
        aggregation = None
        if role is not None and role.aggregation_key:
            aggregation = {role.aggregation_key: {"within_steps": depends}}
        meaning = f"{role.meaning}。" if role is not None and role.meaning else ""
        return builder.add(
            operation,
            depends,
            [],
            f"{meaning}{entry.description}",
            filters=dict(role.fixed_filters) if role is not None else {},
            ranking=ranking,
            aggregation=aggregation,
        )

    # Conditions that need a filter step: one step per row type, after the
    # evidence each condition's field needs.
    rule_filters: dict[str, list[SlotFilter]] = {}
    for item in slots.filters:
        if item.name in FILTER_RULES:
            rule_filters.setdefault(item.on_type, []).append(item)
    for on_type, items in rule_filters.items():
        operation = FILTER_STEPS[on_type]
        depends = list(rows_of.get(on_type, []))
        conditions: dict[str, Any] = {}
        for item in items:
            rule = FILTER_RULES[item.name]
            depends.append(ensure(rule["requires"], operation))
            conditions[rule["field"]] = rule.get("condition", item.value)
        step_id = builder.add(
            operation,
            depends,
            [],
            f"依題目條件篩選{on_type}。{OPERATION_TABLE[operation].description}",
            filters=conditions,
        )
        rows_of[on_type] = [step_id]

    for analysis in slots.analyses:
        add_with_prerequisites(ANALYSIS_INDEX[analysis.name])

    # Other stated filters go on the latest step whose operation supports them.
    for item in slots.filters:
        if item.name in FILTER_RULES:
            continue
        target = next(
            (
                step["step_id"]
                for step in reversed(builder.steps)
                if item.name in OPERATION_TABLE[step["operation"]].supported_filters
            ),
            None,
        )
        if target is None:
            raise SlotError([f"filters: no step on the paths supports {item.name}"])
        value = {"in": item.value} if isinstance(item.value, list) else item.value
        builder.by_id[target]["filters"][item.name] = value

    # 6. Rankings order the rows (and any evidence) gathered for that type.
    ranked: dict[str, str] = {}

    def add_rank(on_type: str, metric: str, direction: str) -> str:
        operation = RANK_INDEX[(on_type, "default")]
        upstream = rows_of.get(on_type, []) + rows_of.get(f"{on_type}:evidence", [])
        step_id = builder.add(
            operation,
            upstream,
            [],
            f"依 {metric} 排序{on_type}。{OPERATION_TABLE[operation].description}",
            ranking={
                "primary_metric": metric,
                "direction": direction,
                "secondary_metrics": [],
            },
        )
        ranked[on_type] = step_id
        return step_id

    for ranking in slots.ranking:
        add_rank(ranking.on_type, ranking.metric, ranking.direction)

    # 7. Checks: find the upstream capability in the table; add a ranking only
    #    when the check needs one the question did not spell out.
    for check in slots.checks:
        index = _find_entity(slots, check.entity)
        entity = slots.entities[index]
        operation = CHECK_INDEX[(entity.type, check.kind)]
        entry = OPERATION_TABLE[operation]
        upstream: list[str] = []
        for capability in entry.requires_all:
            found = builder.producers(capability)
            if not found:
                rank_operation = RANK_INDEX.get((entity.type, "default"))
                rank_entry = OPERATION_TABLE.get(rank_operation or "")
                if rank_entry and capability in rank_entry.provides and entity.type in rows_of:
                    metric = rank_entry.path_role.default_metric
                    found = [add_rank(entity.type, metric, "descending")]
                    builder.notes.append(
                        f"{check.kind} needs {capability}; added {found[0]} "
                        f"ranked by the table default {metric}"
                    )
                else:
                    raise SlotError(
                        [f"checks: {check.kind} needs {capability}, which no path provides"]
                    )
            upstream.extend(found)
        aggregation = None
        if entry.path_role.aggregation_key:
            within = upstream[-1]
            ranking = builder.by_id[within].get("ranking") or {}
            aggregation = {
                entry.path_role.aggregation_key: {
                    "entity_refs": refs_by_entity[index],
                    "within_step": within,
                    "metric": ranking.get("primary_metric", "ranked_position"),
                }
            }
        builder.add(
            operation,
            upstream + [resolve_step[index]],
            refs_by_entity[index],
            f"核對「{entity.mention}」：{entry.path_role.meaning}。{entry.description}",
            aggregation=aggregation,
        )

    # 8. Variant scope: results of each named variant stay apart at the end.
    for index, tail in tails.items():
        entity = slots.entities[index]
        grouping = OPERATION_TABLE[
            builder.by_id[resolve_step[index]]["operation"]
        ].path_role.scope_grouping
        step = builder.by_id[tail]
        step["aggregation"] = {
            **(step["aggregation"] or {}),
            grouping: {"variants": list(entity.variants), "entity_refs": refs_by_entity[index]},
        }

    steps = [ResearchPlanStepV2.model_validate(step) for step in builder.steps]
    safeguards = sorted({item for step in steps for item in step.safeguards})
    multi_hop = any(len(path.types) > 2 for path in slots.paths) or any(
        combine.kind == "intersect" for combine in slots.combine
    )
    secondary: list[str] = []
    if slots.evidence or any(c.kind == "approved_indication" for c in slots.checks):
        secondary.append("evidence_directed")
    if any(c.kind == "rank_position" for c in slots.checks) or any(
        combine.kind == "intersect" for combine in slots.combine
    ):
        secondary.append("comparison_and_aggregation")
    elif slots.ranking:
        secondary.append("filter_and_rank")

    checks_text = "".join(
        f"；核對「{check.entity}」的 {check.kind}" for check in slots.checks
    ) + ("；連接路徑上的每條邊" if slots.combine else "")
    plan = QueryManifestV2(
        manifest_id=uuid4(),
        original_question=question,
        research_objective="沿 " + "、".join(path_labels) + " 規劃查詢" + checks_text + "。",
        entities=entities,
        plan_steps=steps,
        evidence_requirements=[
            f"{step.operation}: {OPERATION_TABLE[step.operation].description}"
            for step in steps
            if OPERATION_TABLE[step.operation].operation_kind == "retrieval"
        ],
        completion_criteria=[
            "每個步驟回傳結果或明確標示缺失；缺失不等於否定。",
            "保留每筆紀錄的來源、分頁與截斷資訊。",
        ],
        global_safeguards=safeguards or ["missing_not_negative"],
        characteristics=QueryCharacteristicsV2(
            primary="multi_hop" if multi_hop else "single_hop",
            secondary=secondary,
        ),
        generated_at=datetime.now(timezone.utc),
    )
    return plan, builder.notes


# ---------------------------------------------------------------- acceptance


class SlotAcceptanceResult(BaseModel):
    passed: bool
    missing: list[str] = Field(default_factory=list)
    unexpected: list[str] = Field(default_factory=list)


def _entity_key(expected: QuerySlots, entity: SlotEntity) -> str:
    for index, item in enumerate(expected.entities):
        a, b = _norm(item.mention), _norm(entity.mention)
        if item.type == entity.type and (a in b or b in a):
            return f"entity#{index}"
    return f"{entity.type}:{entity.mention}"


def _slot_keys(reference: QuerySlots, slots: QuerySlots) -> set[str]:
    keys: set[str] = set()
    for entity in slots.entities:
        variants = ",".join(sorted(_norm(v) for v in entity.variants))
        role = " (context)" if entity.role == "context" else ""
        keys.add(f"entity {_entity_key(reference, entity)} {entity.type} [{variants}]{role}")

    def entity_of(mention: str) -> str:
        index = _find_entity(slots, mention)
        if index is None:
            return f"?{mention}"
        return _entity_key(reference, slots.entities[index])

    for path in slots.paths:
        extra = ""
        if path.via:
            extra += " via " + ",".join(sorted(entity_of(name) for name in path.via))
        if path.end:
            extra += f" end {entity_of(path.end)}"
        keys.add(f"path {entity_of(path.start)} {'->'.join(path.types)}{extra}")
    for item in slots.filters:
        value = sorted(map(str, item.value)) if isinstance(item.value, list) else item.value
        keys.add(f"filter {item.name} {item.on_type} {value}")
    for analysis in slots.analyses:
        keys.add(f"analysis {analysis.name}")
    for ranking in slots.ranking:
        keys.add(f"ranking {ranking.on_type} {ranking.metric} {ranking.direction}")
    for check in slots.checks:
        keys.add(f"check {check.kind} {entity_of(check.entity)}")
    for item in slots.evidence:
        keys.add(f"evidence {item.attribute} {item.of_type}")
    for combine in slots.combine:
        joined = [
            "->".join(slots.paths[i].types) if 0 <= i < len(slots.paths) else f"?{i}"
            for i in combine.paths
        ]
        keys.add(f"combine {combine.kind} {' + '.join(sorted(joined))}")
    return keys


def _describe(reference: QuerySlots, key: str) -> str:
    for index, entity in enumerate(reference.entities):
        key = key.replace(f"entity#{index}", repr(entity.mention))
    return key


def evaluate_slots(expected: QuerySlots, actual: QuerySlots) -> SlotAcceptanceResult:
    """Compare the model's table with the question set's labels."""
    want = _slot_keys(expected, expected)
    got = _slot_keys(expected, actual)
    missing = sorted(_describe(expected, key) for key in want - got)
    unexpected = sorted(_describe(expected, key) for key in got - want)
    return SlotAcceptanceResult(
        passed=not missing and not unexpected, missing=missing, unexpected=unexpected
    )


def expected_slots(case: PlannerAcceptanceCase) -> QuerySlots | None:
    return QuerySlots.model_validate(case.expected_slots) if case.expected_slots else None


# ---------------------------------------------------------------- model step


def build_slot_prompt() -> str:
    """Model instructions, generated from the path index so they cannot drift."""

    hops = "\n".join(
        f"    {before} -> {after}: {OPERATION_TABLE[op].path_role.meaning}"
        for (before, after), op in sorted(DEFAULT_HOPS.items())
    )
    checks = "\n".join(
        f"    {name} (a named {entity_type}): {OPERATION_TABLE[op].path_role.meaning}"
        for (entity_type, name), op in sorted(CHECK_INDEX.items())
    )
    attributes = "\n".join(
        f"    {name} (of {entity_type} rows): {OPERATION_TABLE[op].path_role.meaning}"
        for (entity_type, name), op in sorted(ATTRIBUTE_INDEX.items())
    )
    combines = "\n".join(
        f"    {name} (paths of {' -> '.join(types)}): {OPERATION_TABLE[op].path_role.meaning}"
        for (name, types), op in sorted(COMBINE_INDEX.items())
    )
    filters = "\n".join(
        f"    {name}: {meaning}"
        + (f" (values: {', '.join(FILTER_VALUES[name])})" if name in FILTER_VALUES else "")
        for name, meaning in sorted(FILTER_MEANINGS.items())
    )
    analyses = "\n".join(
        f"    {name} (on {OPERATION_TABLE[op].path_role.subject_type} rows): "
        f"{OPERATION_TABLE[op].path_role.meaning}"
        for name, op in sorted(ANALYSIS_INDEX.items())
    )
    metrics = ", ".join(sorted(_METRIC_REQUIREMENTS))
    example = json.dumps(
        {
            "entities": [{"mention": "asthma", "type": "disease", "variants": []}],
            "paths": [{"start": "asthma", "types": ["disease", "target"]}],
            "combine": [],
            "filters": [],
            "ranking": [
                {
                    "on_type": "target",
                    "metric": "overall_association_score",
                    "direction": "descending",
                }
            ],
            "checks": [],
            "evidence": [],
            "analyses": [],
        },
        ensure_ascii=False,
    )
    return f"""You read one biomedical research question and fill a small table.
Do not write a plan, step IDs, dependencies, outputs, sources or safeguards; a
program builds those from the table. Do not retrieve anything.

Return exactly one JSON object with these keys and nothing else:
{{"entities": [...], "paths": [...], "combine": [...], "filters": [...], "ranking": [...], "checks": [...], "evidence": [...], "analyses": [...]}}

entities: every disease, target (gene/protein) or drug the question names.
  {{"mention": "<copied exactly from the question>", "type": "disease"|"target"|"drug", "variants": [], "aliases": [], "role": "named"}}
  - List each entity once. If the question names the same entity again with
    another name, put that name in aliases.
  - A bracketed alias written right after a name stays in the same mention:
    "X（Y, Z）" is one entity whose mention is "X（Y, Z）".
  - A family, pathway or tissue group named only to scope the question
    (e.g. "the kinase family", "a pathway", "normal tissues") is a context
    entity: {{"mention": "...", "type": "target"|"mechanism"|"tissue", "role": "context"}}.
  - A gene named with mutations is one target. mention is the gene only;
    list each mutation as written in variants, e.g. {{"mention": "BRAF", "type": "target", "variants": ["V600E"]}}.
paths: what the question asks to list, as the entity types walked from a named entity.
  {{"start": "<mention of a named entity>", "types": ["<start type>", "<next type>", ...]}}
  Allowed hops:
{hops}
  Use one hop unless the question walks through an intermediate type.
  A question that only asks facts about named entities needs no path; put
  those facts in evidence with of_type set to the named entities' type.
  A question with two separate asks from the same entity has two paths.
  Named entities given as examples of a middle node ("drugs for X, such as
  A and B") go in via: {{"start": "X", "types": [...], "via": ["A", "B"]}}.
  A question about the link between two named entities (the association of
  A with B) is one hop that ends at B: {{"start": "A", "types": [...], "end": "B"}}.
  The same question about several named entities with one named end has one
  such path per entity.
  A named entity the question only asks about inside a result ("is X the
  strongest?", "is X one of them?") is not a path start; use checks.
combine: what to do with path results beyond listing them.
  {{"kind": "<kind>", "paths": [<index into paths, starting at 0>]}}
  Kinds:
{combines}
  Use join_path when the question asks for the last rows of a multi-hop path
  on the basis of the earlier links, so every edge must stay attached.
  Use intersect when the question asks what several named entities have in
  common: give each entity its own path, then combine those paths.
filters: conditions the question states on result rows.
  {{"name": "<filter>", "on_type": "<row type, or association for the link itself>", "value": true}}
  Filters:
{filters}
ranking: only when the question asks to order results or how strong they are.
  {{"on_type": "<type of the rows ordered>", "metric": "<one of: {metrics}>", "direction": "descending"|"ascending"}}
checks: a yes/no or position question about a named entity.
  {{"kind": "<kind>", "entity": "<mention of the named entity>"}}
  Kinds:
{checks}
evidence: extra facts the question asks for about result rows.
  {{"attribute": "<attribute>", "of_type": "<type of the rows>"}}
  Attributes:
{attributes}
analyses: a named analysis the question asks for; the program adds the
  evidence each one needs.
  {{"name": "<analysis>"}}
  Analyses:
{analyses}
Use [] for anything the question does not ask. Do not add thresholds, rankings,
checks or evidence that the question does not state.

Example (not from the evaluation set)
Question: Which targets are associated with asthma, and which are most strongly associated?
Answer: {example}
"""


_LIST_FIELDS = (
    "entities", "paths", "combine", "filters", "ranking", "checks", "evidence", "analyses",
)


_METRIC_SYNONYMS = {
    alias: canonical
    for canonical, aliases in RANKING_KEY_EQUIVALENCES.items()
    for alias in aliases
    if alias != canonical
}


def _normalize_slot_payload(payload: dict[str, Any]) -> list[str]:
    """Unambiguous form fixes only; each one is recorded.

    * a single object written where a list belongs is wrapped in a list;
    * a reviewed metric synonym becomes its canonical name;
    * a gene written together with its listed mutations ("KRAS G12C／G12D")
      keeps only the gene, which is the text before the first mutation.
    """
    notes = []
    for field in _LIST_FIELDS:
        if isinstance(payload.get(field), dict):
            payload[field] = [payload[field]]
            notes.append(f"{field}: wrapped a single object in a list")
    for item in payload.get("ranking") or []:
        if isinstance(item, dict) and item.get("metric") in _METRIC_SYNONYMS:
            canonical = _METRIC_SYNONYMS[item["metric"]]
            notes.append(f"ranking: {item['metric']} -> {canonical}")
            item["metric"] = canonical
    for entity in payload.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        mention, variants = entity.get("mention"), entity.get("variants") or []
        if not isinstance(mention, str) or not variants:
            continue
        positions = [mention.find(v) for v in variants if isinstance(v, str) and v in mention]
        if positions:
            gene = mention[: min(positions)].strip(" ／/,，、-")
            if gene:
                notes.append(f"entities: {mention!r} -> gene {gene!r} with variants {variants}")
                entity["mention"] = gene
                for path in payload.get("paths") or []:
                    if isinstance(path, dict) and path.get("start") == mention:
                        path["start"] = gene
    return notes


def _extract_slot_payload(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            found.append(value)
    for candidate in reversed(found):
        if {"entities", "paths"}.issubset(candidate):
            return candidate
    raise ValueError("model response did not contain a slot table JSON object")


class DirectJsonSlotAgent:
    """One model call per attempt; Python parses and validates the table."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def invoke(self, request: dict[str, Any], config: dict | None = None) -> dict:
        from natural_language_planner import _message_text

        messages = [
            {"role": "system", "content": build_slot_prompt()},
            *request.get("messages", []),
        ]
        response = self._model.invoke(messages, config=config)
        return {"text": _message_text(response), "messages": [response]}


class SlotAttempt(BaseModel):
    attempt: int
    duration_seconds: float
    raw_response: str | None = None
    slots: QuerySlots | None = None
    errors: list[str] = Field(default_factory=list)
    normalizations: list[str] = Field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None


class SlotPlannerReport(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    planner_mode: Literal["slots"] = "slots"
    question: str
    acceptance_case_id: str | None = None
    model: str
    status: Literal["passed", "failed"]
    attempts: list[SlotAttempt] = Field(default_factory=list)
    slots: QuerySlots | None = None
    assembly_notes: list[str] = Field(default_factory=list)
    final_plan: QueryManifestV2 | None = None
    plan_step_count: int | None = None
    slot_acceptance: SlotAcceptanceResult | None = None
    plan_acceptance: PlanAcceptanceResult | None = None
    operation_rules: OperationRuleEvaluation | None = None
    execution_compile_problems: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    total_duration_seconds: float
    evidence_collection_started: Literal[False] = False
    executed_retrieval_tools: list[str] = Field(default_factory=list)
    generated_at: datetime


def configured_model_label() -> str:
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    name = {
        "ollama": os.getenv("OLLAMA_MODEL", ""),
        "anthropic": os.getenv("ANTHROPIC_MODEL", ""),
        "nim": os.getenv("NIM_MODEL", ""),
    }.get(provider, "")
    return f"{provider}:{name}" if name else provider


def slot_reasoning_effort() -> str | None:
    """Reasoning effort for the table call on NIM (default "low").

    The table is small, so long hidden reasoning only adds latency: on the
    configured NIM model, low effort matched the question-set labels in 17 of
    18 runs at 4-17 s typical, versus up to 120 s with default reasoning.
    Set SLOT_PLANNER_REASONING_EFFORT=default to send no setting.
    """
    if os.getenv("LLM_PROVIDER", "ollama").strip().lower() != "nim":
        return None
    effort = os.getenv("SLOT_PLANNER_REASONING_EFFORT", "low").strip().lower()
    return None if effort in {"", "default", "none"} else effort


def check_assembled_plan(
    plan: QueryManifestV2,
    acceptance_case: PlannerAcceptanceCase | None = None,
) -> tuple[OperationRuleEvaluation, list[str], PlanAcceptanceResult | None]:
    """Operation rules, execution compile check and (optionally) the plan rubric."""
    from execution.compiler import PlanRefusedError, compile_plan

    rules = evaluate_operation_rules(plan)
    try:
        compile_plan(plan)
        compile_problems: list[str] = []
    except PlanRefusedError as exc:
        compile_problems = exc.problems
    rubric = evaluate_plan(acceptance_case, plan) if acceptance_case else None
    return rules, compile_problems, rubric


class SlotPlanner:
    """Fill the table with one model call (plus one fix-up), then assemble."""

    def __init__(self, agent=None, timeout_cleanup=None, model_label: str | None = None):
        effort = None
        if agent is None:
            from model_factory import get_chat_model

            model = get_chat_model()
            effort = slot_reasoning_effort()
            if effort:
                model = model.bind(reasoning_effort=effort)
            agent = DirectJsonSlotAgent(model)
            if timeout_cleanup is None and os.getenv("LLM_PROVIDER", "ollama").strip().lower() == "ollama":
                from natural_language_planner import _cleanup_ollama_after_timeout

                timeout_cleanup = _cleanup_ollama_after_timeout
        self.agent = agent
        self.timeout_cleanup = timeout_cleanup
        self.model_label = model_label or (
            configured_model_label() + (f" (reasoning_effort={effort})" if effort else "")
        )

    def run_with_report(
        self,
        question: str,
        acceptance_case: PlannerAcceptanceCase | None = None,
        checkpoint_path: str | Path | None = None,
        max_attempts: int = 2,
    ) -> SlotPlannerReport:
        from natural_language_planner import (
            ModelAttemptTimeout,
            _invoke_with_deadline,
            _usage_from_result,
        )

        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        timeout = float(os.getenv("SLOT_PLANNER_ATTEMPT_TIMEOUT_SECONDS", "600"))
        started = perf_counter()
        attempts: list[SlotAttempt] = []
        messages: list[dict[str, str]] = [{"role": "user", "content": question}]
        slots: QuerySlots | None = None
        errors: list[str] = []

        def report(status: Literal["passed", "failed"], **extra: Any) -> SlotPlannerReport:
            result = SlotPlannerReport(
                question=question,
                acceptance_case_id=acceptance_case.case_id if acceptance_case else None,
                model=self.model_label,
                status=status,
                attempts=attempts,
                slots=slots,
                errors=errors,
                total_duration_seconds=perf_counter() - started,
                generated_at=datetime.now(timezone.utc),
                **extra,
            )
            if checkpoint_path:
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            return result

        for number in range(1, max_attempts + 1):
            attempt_started = perf_counter()
            attempt = SlotAttempt(attempt=number, duration_seconds=0.0)
            attempts.append(attempt)
            try:
                result = _invoke_with_deadline(
                    self.agent, {"messages": messages}, timeout, self.timeout_cleanup
                )
            except ModelAttemptTimeout as exc:
                attempt.duration_seconds = perf_counter() - attempt_started
                attempt.errors.append(f"ModelAttemptTimeout: {exc}")
                errors.append("model attempt timed out")
                return report("failed")
            attempt.duration_seconds = perf_counter() - attempt_started
            usage = _usage_from_result(result)
            attempt.input_tokens = usage["input_tokens"]
            attempt.output_tokens = usage["output_tokens"]
            attempt.raw_response = result.get("text")
            try:
                payload = _extract_slot_payload(attempt.raw_response or "")
                attempt.normalizations = _normalize_slot_payload(payload)
                candidate = QuerySlots.model_validate(payload)
                attempt.slots = candidate
                problems = validate_slots(question, candidate)
            except (ValueError, ValidationError) as exc:
                problems = [str(exc)]
                candidate = None
            if not problems:
                slots = candidate
                break
            attempt.errors.extend(problems)
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": attempt.raw_response or ""},
                {
                    "role": "user",
                    "content": "The table has these problems. Return the whole "
                    "table again with only these fixed:\n- " + "\n- ".join(problems),
                },
            ]
            report("failed")  # checkpoint after each attempt
        if slots is None:
            errors.append("no valid slot table")
            return report("failed")

        # Score the reading first, so it is reported even when assembly stops.
        slot_result = None
        expected = expected_slots(acceptance_case) if acceptance_case else None
        if expected is not None:
            slot_result = evaluate_slots(expected, slots)
        try:
            plan, notes = assemble_plan(question, slots)
        except SlotError as exc:
            errors.extend(exc.problems)
            return report("failed", slot_acceptance=slot_result)
        rules, compile_problems, rubric = check_assembled_plan(plan, acceptance_case)
        if not rules.passed:
            errors.append("assembled plan failed operation rules")
        if compile_problems:
            errors.append("assembled plan does not compile for execution")
        if slot_result is not None and not slot_result.passed:
            errors.append("slot table does not match the question set labels")
        if rubric is not None and not rubric.passed:
            errors.append("assembled plan failed the plan rubric")
        status = "failed" if errors else "passed"
        return report(
            status,
            assembly_notes=notes,
            final_plan=plan,
            plan_step_count=len(plan.plan_steps),
            slot_acceptance=slot_result,
            plan_acceptance=rubric,
            operation_rules=rules,
            execution_compile_problems=compile_problems,
        )
