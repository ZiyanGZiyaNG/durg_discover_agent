"""Run the whole staged Planner (Stage 1 + Stage 2) several times per case.

Unlike replay_mapping.py, nothing is reused: every run draws a fresh strategy,
so the spread across runs shows how stable Stage 1 is. Each run is saved as
<configuration>_<case_id>_r<n>.report.json, which score_staged_reports.py and
score_staged_acceptance.py pick up as a new column. Runs are interleaved
(r1 of every case, then r2, ...) so a stopped batch still holds balanced data,
and the same command resumes it: finished runs are kept, runs that were cut
off or failed on a connection error are redone.

Only development cases may be run unless --allow-holdout is given, so the
internal holdout set stays unseen while the Planner is being tuned.

Example:
  python repeat_staged.py --configuration s1_a --repeat 5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from planner_acceptance import load_acceptance_cases
from score_staged_acceptance import FIXTURE
from score_staged_reports import SUFFIX, classify_outcome
from staged_planner import StagedNaturalLanguagePlanner

DEFAULT_CASES = (
    "cldn182_pdac_evidence_types",
    "kras_variant_drugs",
    "olaparib_targets_indications",
    "pdac_drugs_other_cancers",
    "pdac_score_tractability_filter",
)


def is_finished(path: Path) -> bool:
    """True when a saved run reached a real end state and can be kept."""
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if report.get("status") == "running":
        return False
    return classify_outcome(report) != "infra_error"


def run_batch(planner, cases: list, plans_dir: Path, configuration: str, repeat: int, log=print) -> None:
    """Interleave runs across cases and write one report file per run."""
    plans_dir.mkdir(parents=True, exist_ok=True)
    for number in range(1, repeat + 1):
        for case in cases:
            target = plans_dir / f"{configuration}_{case.case_id}_r{number}{SUFFIX}"
            if is_finished(target):
                log(f"keep    {target.name}")
                continue
            report = planner.run_with_report(
                case.question, acceptance_case=case, checkpoint_path=target
            )
            needs = len(report.strategy.needs) if report.strategy else 0
            log(
                f"{report.status:<22} {target.name} "
                f"({report.total_duration_seconds:.0f}s, {needs} needs)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plans-dir", default="outputs/plans")
    parser.add_argument("--configuration", required=True, help="Name of the new column.")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    parser.add_argument("--allow-holdout", action="store_true")
    args = parser.parse_args()
    by_id = {case.case_id: case for case in load_acceptance_cases(FIXTURE)}
    unknown = [name for name in args.cases if name not in by_id]
    if unknown:
        raise SystemExit(f"unknown cases: {unknown}")
    cases = [by_id[name] for name in args.cases]
    held_out = [c.case_id for c in cases if c.evaluation_set != "development"]
    if held_out and not args.allow_holdout:
        raise SystemExit(f"holdout cases need --allow-holdout: {held_out}")
    load_dotenv()
    run_batch(
        StagedNaturalLanguagePlanner(), cases, Path(args.plans_dir), args.configuration, args.repeat
    )


if __name__ == "__main__":
    main()