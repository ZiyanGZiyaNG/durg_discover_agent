"""Run a saved QueryManifestV2 plan and print a short report.

python -m execution PLAN.json [--disease-id MONDO_0021040] [--show-calls]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from execution import adapters
from execution.compiler import FAILED, PlanRefusedError, run_plan
from schemas import QueryManifestV2

TOOLS = (
    "resolve_entity",
    "discover_disease_targets",
    "get_disease_target_evidence_by_ids",
    "get_target_drug_evidence_by_ids",
    "get_target_tractability_by_id",
)


def _logged(name, tool, log):
    def call(*args, **kwargs):
        shown = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        log.append((name, f"{name}({', '.join(shown)})"))
        return tool(*args, **kwargs)

    return call


def _record_tool_calls(log):
    """Wrap every retrieval function so each call the compiler makes is listed."""
    for name in TOOLS:
        setattr(adapters, name, _logged(name, getattr(adapters, name), log))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m execution", description=__doc__)
    parser.add_argument("plan", type=Path, help="QueryManifestV2 JSON file")
    parser.add_argument(
        "--disease-id",
        help="use this ID for the disease entity instead of resolving its name",
    )
    parser.add_argument(
        "--show-calls",
        action="store_true",
        help="list every tool call the compiler made",
    )
    args = parser.parse_args(argv)

    plan = QueryManifestV2.model_validate(json.loads(args.plan.read_text()))
    if args.disease_id:
        for entity in plan.entities:
            if entity.entity_type == "disease":
                entity.entity_id = args.disease_id

    calls: list[tuple[str, str]] = []
    if args.show_calls:
        _record_tool_calls(calls)

    try:
        trace = run_plan(plan)
    except PlanRefusedError as exc:
        print("Plan refused before any retrieval:")
        for problem in exc.problems:
            print(f"  - {problem}")
        return 2

    print(args.plan.name)
    print(f"{'step':<36}{'operation':<36}{'status':<16}{'sec':>5}")
    for result in trace:
        print(
            f"{result.step_id:<36}{result.operation:<36}"
            f"{result.status:<16}{result.elapsed_seconds:>5.1f}"
        )
    print(f"total {sum(r.elapsed_seconds for r in trace):.1f} s")

    if args.show_calls:
        counts: dict[str, int] = {}
        for name, _ in calls:
            counts[name] = counts.get(name, 0) + 1
        summary = ", ".join(f"{name} x{n}" for name, n in counts.items())
        print()
        print(f"tool calls: {len(calls)}  ({summary})")
        for number, (_, shown) in enumerate(calls, 1):
            print(f"{number:>3}  {shown}")

    payload = trace[-1].payload
    if isinstance(payload, dict) and payload.get("targets"):
        print()
        print(
            f"{'#':>3}  {'target':<10}{'score':>7}  {'layer':<26}{'drugs':>6}{'tract':>6}"
        )
        for row in payload["targets"]:
            print(
                f"{row['ranked_position']:>3}  {row['target_symbol']:<10}"
                f"{row['overall_association_score']:>7.3f}  "
                f"{row.get('target_translation_layer', '-'):<26}"
                f"{row.get('known_drug_count', '-'):>6}"
                f"{row.get('tractable_bucket_count', '-'):>6}"
            )
        if payload.get("rule_version"):
            print(
                f"layer rule: {payload['rule_version']} (pending research-team approval)"
            )

    return 1 if any(r.status in FAILED for r in trace) else 0


if __name__ == "__main__":
    sys.exit(main())
