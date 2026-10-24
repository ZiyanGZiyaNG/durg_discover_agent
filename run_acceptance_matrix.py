from pathlib import Path
from planner_acceptance import load_acceptance_cases
from agent import run_natural_language_plan_with_report
import json

cases = load_acceptance_cases(Path("tests/fixtures/planner_acceptance_cases.json"))
dev = [c for c in cases if c.evaluation_set == "development"]
out_dir = Path("outputs/plans/matrix_run2")
out_dir.mkdir(parents=True, exist_ok=True)
max_attempts = 4

for case in dev:
    report_path = out_dir / f"{case.case_id}.report.json"
    if report_path.exists():
        done = json.loads(report_path.read_text(encoding="utf-8"))
        if done.get("status") in ("passed", "failed"):
            print(case.case_id, "已完成，跳過")
            continue
    try:
        report = run_natural_language_plan_with_report(
            case.question, case.case_id, report_path, max_attempts
        )
        print(case.case_id, report.status, round(report.total_duration_seconds), "s")
    except Exception as exc:
        print(case.case_id, "出錯：", type(exc).__name__, str(exc)[:100])