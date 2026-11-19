"""Re-score saved staged-planner reports against the hidden acceptance rubric.

No model is called. Each saved strategy and mapping is compiled twice by the
current compiler, without and with program-side completion, and both plans go
through evaluate_plan. Runs that stopped at pending_human_review are scored
too, so every run that reached mapping gets a gap count.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import staged_planner
from planner_acceptance import PlanAcceptanceResult, evaluate_plan, load_acceptance_cases
from score_staged_reports import COLUMN, load_runs
from staged_planner import OperationMappingResult, ResearchStrategyDraft, compile_manifest

FIXTURE = Path(__file__).parent / "tests" / "fixtures" / "planner_acceptance_cases.json"


def compile_plan(question: str, strategy, mapping, complete: bool):
    """Compile with the current compiler; complete=False skips program-side completion."""
    if complete:
        return compile_manifest(question, strategy, mapping, [])
    with patch.object(
        staged_planner, "complete_plan_from_table", lambda steps, *_: steps
    ):
        return compile_manifest(question, strategy, mapping, [])


def count_gaps(result: PlanAcceptanceResult) -> dict[str, int]:
    """Count rubric gaps once each.

    A dependency is counted only when both of its operations are present;
    otherwise the missing operation already accounts for it.
    """
    absent = set(result.missing_capabilities)
    wiring = [
        edge
        for edge in result.missing_dependencies
        if not absent.intersection(edge.split("->"))
    ]
    return {
        "op": len(result.missing_capabilities),
        "dep": len(wiring),
        "safeguard": len(result.missing_safeguards),
        "filter": len(result.missing_filters),
        "rank": len(result.missing_ranking),
        "agg": len(result.missing_aggregation),
        "source": len(result.missing_sources),
        "entity": len(result.missing_entities),
        "forbidden": len(result.forbidden_findings),
    }


def score_run(run: dict, plans_dir: Path, cases: dict) -> None:
    """Add 'raw' and 'completed' gap counts to a run, or the reason it has none."""
    report = json.loads((plans_dir / run["file"]).read_text(encoding="utf-8"))
    case = cases.get(report.get("acceptance_case_id"))
    if case is None or not report.get("strategy") or not report.get("mapping"):
        run["skip"] = "no case, strategy or mapping"
        return
    try:
        strategy = ResearchStrategyDraft.model_validate(report["strategy"])
        mapping = OperationMappingResult.model_validate(report["mapping"])
        for label, complete in (("raw", False), ("completed", True)):
            plan = compile_plan(report["question"], strategy, mapping, complete)
            run[label] = count_gaps(evaluate_plan(case, plan))
    except Exception as exc:  # A saved report that no longer compiles is a finding.
        run["skip"] = f"{type(exc).__name__}: {exc}"[:80]


def print_summary(runs: list[dict], configurations: list[str]) -> None:
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        if "completed" in run:
            cells[(run["case"], run["configuration"])].append(run)
    cases = sorted({case for case, name in cells if name in configurations})
    width = max(len(name) for name in [*cases, "TOTAL"]) + 2
    print("Rubric gaps: without completion -> with completion (0 = passes, apart from proposed operations)")
    print("case".ljust(width) + "".join(name.rjust(COLUMN) for name in configurations))
    totals = {name: [0, 0] for name in configurations}
    for case in cases:
        line = case.ljust(width)
        for name in configurations:
            group = cells.get((case, name), [])
            if not group:
                line += "-".rjust(COLUMN)
                continue
            raw = sum(sum(run["raw"].values()) for run in group)
            done = sum(sum(run["completed"].values()) for run in group)
            totals[name][0] += raw
            totals[name][1] += done
            repeat = f" (n={len(group)})" if len(group) > 1 else ""
            line += f"{raw}->{done}{repeat}".rjust(COLUMN)
        print(line)
    print(
        "TOTAL".ljust(width)
        + "".join(f"{totals[n][0]}->{totals[n][1]}".rjust(COLUMN) for n in configurations)
    )


def print_details(runs: list[dict], configurations: list[str]) -> None:
    print("\nconfiguration    case                               run  proposed  raw  completed  remaining gaps")
    for run in sorted(runs, key=lambda r: (r["case"], r["generated_at"])):
        if run["configuration"] not in configurations:
            continue
        head = f"{run['configuration']:<16} {run['case']:<34} {run['run']:<4} {run['proposed']:<9} "
        if "completed" not in run:
            print(head + f"not scored ({run.get('skip', run['outcome'])})")
            continue
        remaining = " ".join(f"{key}{count}" for key, count in run["completed"].items() if count)
        print(
            head
            + f"{sum(run['raw'].values()):<4} {sum(run['completed'].values()):<10} "
            + (remaining or "none")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plans-dir", default="outputs/plans")
    parser.add_argument("--configurations", nargs="*")
    args = parser.parse_args()
    plans_dir = Path(args.plans_dir)
    runs = load_runs(plans_dir)
    if not runs:
        raise SystemExit(f"no staged reports found in {plans_dir}")
    cases = {case.case_id: case for case in load_acceptance_cases(FIXTURE)}
    for run in runs:
        score_run(run, plans_dir, cases)
    found: dict[str, None] = {}
    for run in sorted(runs, key=lambda r: r["generated_at"]):
        found.setdefault(run["configuration"])
    configurations = args.configurations or list(found)
    unknown = [name for name in configurations if name not in found]
    if unknown:
        raise SystemExit(f"unknown configurations: {unknown}; found {list(found)}")
    print_summary(runs, configurations)
    print_details(runs, configurations)


if __name__ == "__main__":
    main()