"""Plan-only comparison: slot planner (plan-query-slots) vs plan-query.

Runs each selected acceptance case through both planners with the configured
model and records wall time and correctness. No evidence is retrieved.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from planner_acceptance import load_acceptance_cases
from slot_planner import SlotPlanner, check_assembled_plan, configured_model_label

CATEGORY_1 = [
    "pdac_known_targets",
    "kras_associated_diseases",
    "kras_variant_drugs",
    "olaparib_targets_indications",
]


def _plan_checks(plan, case) -> dict:
    if plan is None:
        return {"plan_rubric": None, "operation_rules": None, "compiles": None, "steps": None}
    rules, compile_problems, rubric = check_assembled_plan(plan, case)
    return {
        "plan_rubric": rubric.passed,
        "operation_rules": rules.passed,
        "compiles": not compile_problems,
        "steps": len(plan.plan_steps),
        "operations": [step.operation for step in plan.plan_steps],
    }


def run_slots(case, output: Path, max_attempts: int) -> dict:
    report = SlotPlanner().run_with_report(
        case.question,
        acceptance_case=case,
        checkpoint_path=output / f"{case.case_id}.slots.report.json",
        max_attempts=max_attempts,
    )
    return {
        "planner": "plan-query-slots",
        "status": report.status,
        "seconds": round(report.total_duration_seconds, 2),
        "model_calls": len(report.attempts),
        "slot_labels_match": report.slot_acceptance.passed if report.slot_acceptance else None,
        "slot_missing": report.slot_acceptance.missing if report.slot_acceptance else [],
        "slot_unexpected": report.slot_acceptance.unexpected if report.slot_acceptance else [],
        "errors": report.errors,
        **_plan_checks(report.final_plan, case),
    }


def run_baseline(case, output: Path, max_attempts: int) -> dict:
    from natural_language_planner import NaturalLanguageResearchPlanner

    report = NaturalLanguageResearchPlanner().run_with_report(
        case.question,
        acceptance_case=case,
        checkpoint_path=output / f"{case.case_id}.plan_query.report.json",
        max_attempts=max_attempts,
    )
    last = report.final_plan or next(
        (a.candidate_plan for a in reversed(report.attempts) if a.candidate_plan),
        None,
    )
    return {
        "planner": "plan-query",
        "status": report.status,
        "seconds": round(report.total_duration_seconds, 2),
        "model_calls": sum(max(1, len(a.model_calls)) for a in report.attempts),
        "attempts": len(report.attempts),
        "first_attempt_passed": report.first_attempt_passed,
        "errors": [error for a in report.attempts for error in a.errors][-3:],
        **_plan_checks(last, case),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", nargs="+", default=CATEGORY_1)
    parser.add_argument(
        "--planners", nargs="+", choices=["slots", "baseline"], default=["slots", "baseline"]
    )
    parser.add_argument("--slot-attempts", type=int, default=2)
    parser.add_argument("--baseline-attempts", type=int, default=4)
    args = parser.parse_args()
    load_dotenv()

    cases = {
        case.case_id: case
        for case in load_acceptance_cases(
            Path(__file__).parent / "tests/fixtures/planner_acceptance_cases.json"
        )
    }
    if args.cases == ["all"]:
        args.cases = list(cases)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary = {
        "model": configured_model_label(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "evidence_collection_started": False,
        "runs": [],
    }
    runners = {"slots": run_slots, "baseline": run_baseline}
    for case_id in args.cases:
        case = cases[case_id]
        for planner in args.planners:
            attempts = args.slot_attempts if planner == "slots" else args.baseline_attempts
            try:
                row = runners[planner](case, output, attempts)
            except Exception as exc:  # keep the batch going; record the failure
                row = {"planner": planner, "status": "crashed", "errors": [f"{type(exc).__name__}: {exc}"]}
            row = {"case_id": case_id, "evaluation_set": case.evaluation_set, **row}
            summary["runs"].append(row)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(row, ensure_ascii=False), flush=True)
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
