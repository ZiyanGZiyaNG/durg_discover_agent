"""Score saved staged-planner reports without calling a model.

Reads <plans-dir>/*.report.json, keeps planner_mode == "staged", and prints
matched needs per case and configuration plus one detail line per run.
The configuration is the file-name part before the acceptance case id, e.g.
ftcomp_kras_variant_drugs.report.json -> configuration "ftcomp".
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SUFFIX = ".report.json"
INFRA_MARKERS = ("ConnectionError", "RateLimitError", "APITimeoutError")
COLUMN = 16


def classify_outcome(report: dict) -> str:
    """Name where a run stopped; infrastructure failures are not model failures."""
    errors = " | ".join(report.get("errors", []))
    if any(marker in errors for marker in INFRA_MARKERS):
        return "infra_error"
    if "ModelAttemptTimeout" in errors:
        return "timeout"
    if report.get("strategy") is None:
        return "invalid_strategy"
    if report.get("mapping") is None:
        return "invalid_mapping"
    if report.get("status") == "pending_human_review":
        return "proposed_operation"
    if report.get("status") == "failed":
        if "hidden semantic acceptance" in errors:
            return "acceptance_failed"
        return "compile_failed"
    return str(report.get("status", "unknown"))


def split_name(stem: str, case_id: str | None) -> tuple[str, str]:
    """Split '<configuration>_<case_id>_<run>' into (configuration, run)."""
    if not case_id or case_id not in stem:
        return stem, ""
    before, _, after = stem.partition(case_id)
    before, after = before.strip("_"), after.strip("_")
    if before:
        return before, after
    return after or "unlabeled", ""


def load_runs(plans_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(plans_dir.glob(f"*{SUFFIX}")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(report, dict) or report.get("planner_mode") != "staged":
            continue
        case_id = report.get("acceptance_case_id")
        configuration, run = split_name(path.name.removesuffix(SUFFIX), case_id)
        needs = (report.get("strategy") or {}).get("needs", [])
        matches = (report.get("mapping") or {}).get("matches", [])
        runs.append(
            {
                "configuration": configuration,
                "case": case_id or "(no case)",
                "run": run,
                "needs": len(needs),
                "matched": sum(m.get("match_status") == "matched" for m in matches),
                "proposed": sum(
                    m.get("match_status") == "no_equivalent" for m in matches
                ),
                "outcome": classify_outcome(report),
                "seconds": report.get("total_duration_seconds") or 0.0,
                "completions": len(report.get("program_completions", [])),
                "generated_at": str(report.get("generated_at", "")),
                "file": path.name,
            }
        )
    return runs


def print_summary(runs: list[dict], configurations: list[str]) -> None:
    """Matched needs / total needs; infrastructure failures are left out."""
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        if run["outcome"] != "infra_error":
            cells[(run["case"], run["configuration"])].append(run)
    cases = sorted({case for case, name in cells if name in configurations})
    width = max(len(name) for name in [*cases, "TOTAL"]) + 2
    print("case".ljust(width) + "".join(name.rjust(COLUMN) for name in configurations))
    totals = {name: [0, 0] for name in configurations}
    for case in cases:
        line = case.ljust(width)
        for name in configurations:
            group = cells.get((case, name), [])
            if not group:
                line += "-".rjust(COLUMN)
                continue
            matched = sum(run["matched"] for run in group)
            needs = sum(run["needs"] for run in group)
            totals[name][0] += matched
            totals[name][1] += needs
            repeat = f" (n={len(group)})" if len(group) > 1 else ""
            line += f"{matched}/{needs}{repeat}".rjust(COLUMN)
        print(line)
    print(
        "TOTAL".ljust(width)
        + "".join(f"{totals[n][0]}/{totals[n][1]}".rjust(COLUMN) for n in configurations)
    )


def print_details(runs: list[dict], configurations: list[str]) -> None:
    print("\nconfiguration    case                               run  matched  proposed  completions  seconds  outcome")
    for run in sorted(runs, key=lambda r: (r["case"], r["generated_at"])):
        if run["configuration"] not in configurations:
            continue
        print(
            f"{run['configuration']:<16} {run['case']:<34} {run['run']:<4} "
            f"{run['matched']}/{run['needs']:<6} {run['proposed']:<9} "
            f"{run['completions']:<12} {run['seconds']:<8.0f} {run['outcome']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plans-dir", default="outputs/plans")
    parser.add_argument(
        "--configurations",
        nargs="*",
        help="Columns to show, in order. Default: every configuration, oldest first.",
    )
    args = parser.parse_args()
    runs = load_runs(Path(args.plans_dir))
    if not runs:
        raise SystemExit(f"no staged reports found in {args.plans_dir}")
    first_seen: dict[str, str] = {}
    for run in sorted(runs, key=lambda r: r["generated_at"]):
        first_seen.setdefault(run["configuration"], run["generated_at"])
    configurations = args.configurations or list(first_seen)
    unknown = [name for name in configurations if name not in first_seen]
    if unknown:
        raise SystemExit(f"unknown configurations: {unknown}; found {list(first_seen)}")
    print_summary(runs, configurations)
    print_details(runs, configurations)


if __name__ == "__main__":
    main()