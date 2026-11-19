"""Replay Stage 2 (mapping) on saved Stage 1 strategies, then compile and score.

Stage 1 is fixed: each saved strategy is reused, so the number of needs and
their wording cannot change between runs; only the mapping model call is
repeated. Every replay is saved as a staged report named
<configuration>_<case_id>_<run>.report.json, so score_staged_reports.py and
score_staged_acceptance.py pick it up as a new column. Existing files are kept,
so an interrupted batch can be restarted with the same command.

Example:
  python replay_mapping.py --source ftcomp --configuration replay_a --repeat 5
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from natural_language_planner import ModelAttemptTimeout, _invoke_with_deadline
from planner_acceptance import build_acceptance_rubric, evaluate_plan, load_acceptance_cases
from score_staged_acceptance import FIXTURE
from score_staged_reports import SUFFIX, load_runs
from staged_planner import (
    OperationMappingResult,
    ResearchStrategyDraft,
    StagedNaturalLanguagePlanner,
    StagedPlannerReport,
    build_mapping_request,
    compile_manifest,
)


def replay_once(planner, saved: dict, case, timeout: float) -> StagedPlannerReport:
    """Run only the mapping stage for one saved strategy and return a staged report."""
    strategy = ResearchStrategyDraft.model_validate(saved["strategy"])
    started = perf_counter()
    errors: list[str] = []
    completions: list[str] = []
    cleanup: dict = {}
    mapping = plan = acceptance = None
    mapping_seconds = None
    status = "failed"
    try:
        request = build_mapping_request(strategy)
        stage_started = perf_counter()
        result = _invoke_with_deadline(
            planner.mapping_agent,
            {"messages": [{"role": "user", "content": json.dumps(request, ensure_ascii=False)}]},
            timeout,
            planner.timeout_cleanup,
        )
        mapping_seconds = perf_counter() - stage_started
        payload = result.get("structured_response")
        mapping = OperationMappingResult.model_validate(
            payload.model_dump() if hasattr(payload, "model_dump") else payload
        )
        plan = compile_manifest(saved["question"], strategy, mapping, completions)
        if any(step.operation_status == "proposed" for step in plan.plan_steps):
            errors.append("proposed operations require human review")
            status = "pending_human_review"
        else:
            acceptance = evaluate_plan(build_acceptance_rubric(case), plan)
            if acceptance.passed:
                status = "passed"
            else:
                errors.append("compiled plan failed hidden semantic acceptance")
    except ModelAttemptTimeout as exc:
        cleanup = exc.cleanup
        errors.append(f"ModelAttemptTimeout: {exc}")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return StagedPlannerReport(
        question=saved["question"],
        acceptance_case_id=case.case_id,
        status=status,
        strategy=strategy,
        mapping=mapping,
        final_plan=plan if status == "passed" else None,
        semantic_acceptance=acceptance,
        strategy_duration_seconds=saved.get("strategy_duration_seconds"),
        mapping_duration_seconds=mapping_seconds,
        total_duration_seconds=perf_counter() - started,
        errors=errors,
        corrected_model_claims=saved.get("corrected_model_claims", []),
        timeout_cleanup=cleanup,
        generated_at=datetime.now(timezone.utc),
        program_completions=completions,
    )


def latest_source_reports(plans_dir: Path, source: str, wanted: list[str] | None) -> dict:
    """Newest saved report with a strategy, per acceptance case, for one configuration."""
    chosen: dict[str, dict] = {}
    for run in sorted(load_runs(plans_dir), key=lambda r: r["generated_at"]):
        if run["configuration"] != source or run["case"] == "(no case)":
            continue
        if wanted and run["case"] not in wanted:
            continue
        saved = json.loads((plans_dir / run["file"]).read_text(encoding="utf-8"))
        if saved.get("strategy"):
            chosen[run["case"]] = saved
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plans-dir", default="outputs/plans")
    parser.add_argument("--source", required=True, help="Configuration whose saved strategies are reused.")
    parser.add_argument("--configuration", required=True, help="Name of the new column.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cases", nargs="*", help="Acceptance case ids; default: every case in the source.")
    args = parser.parse_args()
    if args.configuration == args.source:
        raise SystemExit("--configuration must differ from --source")
    load_dotenv()
    plans_dir = Path(args.plans_dir)
    cases = {case.case_id: case for case in load_acceptance_cases(FIXTURE)}
    sources = latest_source_reports(plans_dir, args.source, args.cases)
    if not sources:
        raise SystemExit(f"no saved strategies found for configuration {args.source!r}")
    timeout = float(os.getenv("PLANNER_ATTEMPT_TIMEOUT_SECONDS", "1200"))
    planner = StagedNaturalLanguagePlanner()
    for case_id, saved in sorted(sources.items()):
        for number in range(1, args.repeat + 1):
            target = plans_dir / f"{args.configuration}_{case_id}_r{number}{SUFFIX}"
            if target.exists():
                print(f"keep    {target.name}")
                continue
            report = replay_once(planner, saved, cases[case_id], timeout)
            target.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            print(f"{report.status:<22} {target.name} ({report.total_duration_seconds:.0f}s)")


if __name__ == "__main__":
    main()