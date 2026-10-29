"""Run a saved QueryManifestV2 plan and print a short report.

python -m execution PLAN.json [--disease-id MONDO_0021040]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from schemas import QueryManifestV2
from execution.compiler import FAILED, PlanRefusedError, compile_plan, run_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m execution", description=__doc__)
    parser.add_argument("plan", type=Path, help="QueryManifestV2 JSON file")
    parser.add_argument(
        "--disease-id",
        help="use this ID for the disease entity instead of resolving its name",
    )

    parser.add_argument(
    "--dry-run",
    action="store_true",
    help="print the tool calls this plan compiles to, without running any",
    )

    args = parser.parse_args(argv)

    plan = QueryManifestV2.model_validate(json.loads(args.plan.read_text()))
    if args.disease_id:
        for entity in plan.entities:
            if entity.entity_type == "disease":
                entity.entity_id = args.disease_id


    if args.dry_run:
        try:
            calls = compile_plan(plan)
        except PlanRefusedError as exc:
            print("Plan refused before any retrieval:")
            for problem in exc.problems:
                print(f"  - {problem}")
            return 2
        print(json.dumps([call.model_dump() for call in calls], ensure_ascii=False, indent=2))
        return 0
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
