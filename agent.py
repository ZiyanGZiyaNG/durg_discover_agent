"""CLI entry point for the preserved assistant and new evidence pipeline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pydantic import ValidationError
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from rich.console import Console
from rich.markdown import Markdown
from uuid import uuid4
from datetime import datetime, timezone

from agent1 import (
    BioinformaticsEvidenceAgent,
    package_agent1_output,
    package_discovery_output,
)
from agent2 import TargetEvidenceInterpreterAgent
from evidence_synthesizer import EvidenceSynthesizerSubAgent
from evidence_pipeline import pipeline_as_json
from expert_review import build_expert_review_view
from model_factory import get_chat_model
from natural_language_planner import (
    NaturalLanguagePlannerError,
    NaturalLanguageResearchPlanner,
    PlannerRunReport,
)
from planner_acceptance import load_acceptance_cases
from schemas import (
    Agent1RunOutput,
    DiseaseDiscoveryInput,
    DiseaseTargetEvidence,
    DiseaseTargetInput,
    PlannedExecutionOutput,
    QueryManifestV2,
)
from research_planner import Agent1ResearchPlanner, ResearchPlannerError
from staged_planner import StagedNaturalLanguagePlanner, StagedPlannerReport
from slot_planner import SlotPlanner, SlotPlannerReport
from query_opentargets import EntityResolutionError
from tools import DISEASE_TARGET_TOOLS, DRUG_PROPERTY_TOOLS


load_dotenv()
console = Console()


class Agent1EvidenceInputError(RuntimeError):
    """Raised when Agent 2 does not receive a valid Agent 1 run output."""

DISEASE_TARGET_PROMPT = """
You are a disease biology and target-identification specialist. Use Open Targets
tools for ontology IDs, Ensembl IDs, disease-target associations, traceable
pair-level evidence, and known drugs. Return precise, structured, source-aware
answers. Do not invent missing data.
"""

DRUG_PROPERTIES_PROMPT = """
You are a drug chemistry and pharmacology specialist. Use ChEMBL tools for ADMET,
structural, and physicochemical properties. Explain retrieved values without
turning them into clinical recommendations.
"""

SUPERVISOR_PROMPT = """
You coordinate a research-use drug discovery assistant.
- disease_target_worker handles disease IDs, target associations, pair evidence,
  and known target drugs using Open Targets.
- drug_properties_worker handles ADMET and chemical properties using ChEMBL.
Delegate only when needed and synthesize a source-aware answer. Never invent
database results or present the output as clinical advice.
"""


def build_supervisor():
    model = get_chat_model()
    disease_agent = create_agent(
        model=model,
        tools=DISEASE_TARGET_TOOLS,
        system_prompt=DISEASE_TARGET_PROMPT,
        name="disease-target-worker",
    )
    drug_agent = create_agent(
        model=model,
        tools=DRUG_PROPERTY_TOOLS,
        system_prompt=DRUG_PROPERTIES_PROMPT,
        name="drug-properties-worker",
    )

    @tool
    def disease_target_worker(query: str) -> str:
        """Delegate disease and target research questions to Open Targets."""
        result = disease_agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        return result["messages"][-1].content

    @tool
    def drug_properties_worker(query: str) -> str:
        """Delegate drug property questions to ChEMBL."""
        result = drug_agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        return result["messages"][-1].content

    return create_agent(
        model=model,
        tools=[disease_target_worker, drug_properties_worker],
        system_prompt=SUPERVISOR_PROMPT,
        name="supervisor",
    )


def run_agent(query: str) -> str:
    result = build_supervisor().invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content


def run_agent1_as_json(
    disease: str,
    target: str,
    max_evidence_records: int = 500,
    with_synthesizer: bool = False,
) -> str:
    """Run Agent 1 alone and return its stable evidence JSON."""
    request = DiseaseTargetInput(disease=disease, target=target)
    evidence = BioinformaticsEvidenceAgent().run(
        request, max_evidence_records=max_evidence_records
    )
    expert_review = build_expert_review_view(evidence)
    interpretation = None
    if with_synthesizer:
        interpretation = EvidenceSynthesizerSubAgent().run(expert_review)
    return package_agent1_output(
        evidence,
        expert_review_view=expert_review,
        model_interpretation=interpretation,
    ).model_dump_json(indent=2)


def run_target_discovery_as_json(disease: str, top_n: int = 10) -> str:
    """Run deterministic disease-only target discovery without Ollama."""
    request = DiseaseDiscoveryInput(disease=disease, top_n=top_n)
    result = BioinformaticsEvidenceAgent().discover_targets(request)
    return package_discovery_output(result).model_dump_json(indent=2)


def load_agent1_evidence(path: str) -> DiseaseTargetEvidence:
    """Strictly validate metadata and evidence from a completed Agent 1 run."""
    evidence_path = Path(path)
    if not evidence_path.is_file():
        raise Agent1EvidenceInputError(
            f"Agent 1 evidence file does not exist: {evidence_path}"
        )
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Agent1EvidenceInputError(
            f"Agent 1 evidence file is not valid UTF-8 JSON: {exc}"
        ) from exc
    try:
        validated = Agent1RunOutput.model_validate(payload)
    except ValidationError as exc:
        raise Agent1EvidenceInputError(
            "Agent 2 requires output generated by Agent 1 with producer='agent1', "
            "a valid run_id, generated_at, and agent_1_evidence. "
            f"Validation failed: {exc}"
        ) from exc
    return validated.agent_1_evidence


def run_agent2_as_json(evidence_path: str) -> str:
    """Run Agent 2 alone using evidence previously produced by Agent 1."""
    evidence = load_agent1_evidence(evidence_path)
    interpretation = TargetEvidenceInterpreterAgent().run(evidence)
    return interpretation.model_dump_json(indent=2)


def emit_json(content: str, output_path: str | None) -> None:
    """Print JSON or save it as UTF-8 when an output path is supplied."""
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + "\n", encoding="utf-8")
        console.print(f"Saved JSON to {path.resolve()}")
        return
    console.print_json(content)


def discovery_output_path(
    disease: str,
    top_n: int,
    requested_output: str | None,
) -> str:
    """Generate the discovery filename from the disease input."""
    safe_disease = re.sub(r"[^A-Za-z0-9]+", "_", disease.strip()).strip("_")
    if not safe_disease:
        safe_disease = "disease"
    filename = f"{safe_disease}_top{top_n}_targets.json"
    if not requested_output:
        return filename
    requested_path = Path(requested_output)
    output_directory = (
        requested_path
        if requested_path.is_dir() or requested_path.suffix == ""
        else requested_path.parent
    )
    return str(output_directory / filename)


def plan_output_path(disease: str, requested_output: str | None) -> str:
    safe_disease = re.sub(r"[^A-Za-z0-9]+", "_", disease.strip()).strip("_")
    filename = f"{safe_disease or 'disease'}_research_plan.json"
    if not requested_output:
        return str(Path("plans") / filename)
    requested_path = Path(requested_output)
    directory = (
        requested_path
        if requested_path.is_dir() or requested_path.suffix == ""
        else requested_path.parent
    )
    return str(directory / filename)


def run_research_plan_as_json(
    disease: str,
    target: str | None,
    top_n: int,
    disease_id: str | None = None,
    target_id: str | None = None,
) -> str:
    plan = Agent1ResearchPlanner().run(
        disease, target, top_n, disease_id, target_id
    )
    return plan.model_dump_json(indent=2)


def run_natural_language_plan_with_report(
    question: str,
    acceptance_case_id: str | None = None,
    checkpoint_path: str | Path | None = None,
    max_attempts: int = 4,
    initial_plan: QueryManifestV2 | None = None,
    resume_source: str | None = None,
) -> PlannerRunReport:
    """Create and diagnose a blind QueryManifestV2 run without retrieval."""
    acceptance_case = None
    if acceptance_case_id:
        fixture = Path(__file__).parent / "tests" / "fixtures" / "planner_acceptance_cases.json"
        cases = {case.case_id: case for case in load_acceptance_cases(fixture)}
        if acceptance_case_id not in cases:
            raise ValueError(f"unknown acceptance case: {acceptance_case_id}")
        acceptance_case = cases[acceptance_case_id]
        if acceptance_case.question != question:
            raise ValueError(
                "--question must exactly match the selected acceptance case question"
            )
    return NaturalLanguageResearchPlanner().run_with_report(
        question,
        acceptance_case=acceptance_case,
        checkpoint_path=checkpoint_path,
        max_attempts=max_attempts,
        initial_plan=initial_plan,
        resume_source=resume_source,
    )


def run_natural_language_plan_as_json(
    question: str,
    acceptance_case_id: str | None = None,
) -> str:
    report = run_natural_language_plan_with_report(question, acceptance_case_id)
    if report.final_plan is None:
        raise NaturalLanguagePlannerError(
            "blind Plan failed hidden acceptance; inspect the diagnostic report"
        )
    return report.final_plan.model_dump_json(indent=2)


def run_staged_plan_with_report(
    question: str,
    acceptance_case_id: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> StagedPlannerReport:
    """Run the reversible staged strategy/table-matching experiment."""
    acceptance_case = None
    if acceptance_case_id:
        fixture = Path(__file__).parent / "tests" / "fixtures" / "planner_acceptance_cases.json"
        cases = {case.case_id: case for case in load_acceptance_cases(fixture)}
        if acceptance_case_id not in cases:
            raise ValueError(f"unknown acceptance case: {acceptance_case_id}")
        acceptance_case = cases[acceptance_case_id]
        if acceptance_case.question != question:
            raise ValueError(
                "--question must exactly match the selected acceptance case question"
            )
    return StagedNaturalLanguagePlanner().run_with_report(
        question,
        acceptance_case=acceptance_case,
        checkpoint_path=checkpoint_path,
    )


def _acceptance_case_for(question: str, acceptance_case_id: str | None):
    if not acceptance_case_id:
        return None
    fixture = Path(__file__).parent / "tests" / "fixtures" / "planner_acceptance_cases.json"
    cases = {case.case_id: case for case in load_acceptance_cases(fixture)}
    if acceptance_case_id not in cases:
        raise ValueError(f"unknown acceptance case: {acceptance_case_id}")
    case = cases[acceptance_case_id]
    if case.question != question:
        raise ValueError(
            "--question must exactly match the selected acceptance case question"
        )
    return case


def run_slot_plan_with_report(
    question: str,
    acceptance_case_id: str | None = None,
    checkpoint_path: str | Path | None = None,
    max_attempts: int = 2,
) -> SlotPlannerReport:
    """Model fills the small table; Python assembles the Plan from the table."""
    return SlotPlanner().run_with_report(
        question,
        acceptance_case=_acceptance_case_for(question, acceptance_case_id),
        checkpoint_path=checkpoint_path,
        max_attempts=max_attempts,
    )


def run_planned_agent1_as_json(
    disease: str,
    target: str | None,
    top_n: int,
    disease_id: str | None = None,
    target_id: str | None = None,
) -> str:
    plan = Agent1ResearchPlanner().run(
        disease, target, top_n, disease_id, target_id
    )
    output = PlannedExecutionOutput(
        run_id=uuid4(),
        generated_at=datetime.now(timezone.utc),
        research_plan=plan,
        execution_status="collection_policy_not_configured",
        executed_tools=[],
        message=(
            "Research Plan validated. Evidence retrieval did not start because "
            "safe per-source collection policies have not yet been configured."
        ),
    )
    return output.model_dump_json(indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NCKU drug discovery assistant")
    subparsers = parser.add_subparsers(dest="command")

    chat = subparsers.add_parser("chat", help="Use the preserved chat assistant")
    chat.add_argument("query", nargs="+")

    agent1 = subparsers.add_parser(
        "agent1", help="Run Agent 1 evidence collection only"
    )
    agent1.add_argument("--disease", required=True)
    agent1.add_argument("--target", required=True)
    agent1.add_argument("--output", help="Optional UTF-8 JSON output path")
    agent1.add_argument(
        "--max-evidence-records",
        type=int,
        default=500,
        help="Maximum underlying clinical_precedence evidence rows (default: 500)",
    )
    agent1.add_argument(
        "--with-synthesizer",
        action="store_true",
        help="Also run the internal no-tool Evidence Synthesizer via Ollama",
    )

    discovery = subparsers.add_parser(
        "discover-targets",
        help="Find top disease-associated targets using Open Targets overall score",
    )
    discovery.add_argument("--disease", required=True)
    discovery.add_argument(
        "--top",
        type=int,
        default=10,
        choices=range(1, 101),
        metavar="1-100",
        help="Number of targets to return; v1 default is 10",
    )

    for command_name, command_help in (
        ("plan", "Use Qwen to create and validate an Agent 1 Research Plan"),
        ("agent1-planned", "Create a plan and run the policy-gated Agent 1 workflow"),
    ):
        planned = subparsers.add_parser(command_name, help=command_help)
        planned.add_argument("--disease", required=True)
        planned.add_argument(
            "--disease-id",
            help="Explicit ontology ID required when the disease name is ambiguous",
        )
        planned.add_argument("--target")
        planned.add_argument(
            "--target-id",
            help="Optional explicit Ensembl ID; requires --target",
        )
        planned.add_argument("--top", type=int, default=10, choices=range(1, 11))
        planned.add_argument(
            "--output",
            default=r"D:\NCKU_Drug_Discovery\outputs\plans",
            help="Output directory; filename is generated from --disease",
        )
    plan_query = subparsers.add_parser(
        "plan-query",
        help="Create QueryManifestV2 from a natural-language question; no retrieval",
    )
    plan_query.add_argument("--question", required=True)
    plan_query.add_argument(
        "--acceptance-case",
        help="Score against one hidden rubric and diagnose repairable failures",
    )
    plan_query.add_argument(
        "--output",
        help="Optional UTF-8 JSON file; prints to the terminal when omitted",
    )
    plan_query.add_argument(
        "--max-attempts",
        type=int,
        choices=range(1, 6),
        default=4,
        help="Maximum full-plan plus repair attempts (default: 4)",
    )
    plan_query.add_argument(
        "--resume-report",
        help=(
            "Continue from the latest promoted schema-valid candidate in an "
            "earlier diagnostic report"
        ),
    )
    staged_query = subparsers.add_parser(
        "plan-query-staged",
        help="Experimental strategy -> operation-table -> compiled Plan; no retrieval",
    )
    staged_query.add_argument("--question", required=True)
    staged_query.add_argument("--acceptance-case")
    staged_query.add_argument("--output")
    slot_query = subparsers.add_parser(
        "plan-query-slots",
        help=(
            "Model fills entities/path/conditions; Python assembles the Plan "
            "from the operation table; no retrieval"
        ),
    )
    slot_query.add_argument("--question", required=True)
    slot_query.add_argument("--acceptance-case")
    slot_query.add_argument("--output")
    slot_query.add_argument(
        "--max-attempts",
        type=int,
        choices=range(1, 4),
        default=2,
        help="Model attempts at the table: one fill plus fix-ups (default: 2)",
    )
    discovery.add_argument(
        "--output",
        help=(
            "Optional output location; the filename is generated from "
            "--disease and --top"
        ),
    )

    agent2 = subparsers.add_parser(
        "agent2", help="Run Agent 2 interpretation from Agent 1 JSON"
    )
    agent2.add_argument("--evidence", required=True, help="Agent 1 JSON path")
    agent2.add_argument("--output", help="Optional UTF-8 JSON output path")

    evidence = subparsers.add_parser(
        "evidence", help="Run Agent 1 and Agent 2 for a disease-target pair"
    )
    evidence.add_argument("--disease", required=True)
    evidence.add_argument("--target", required=True)
    evidence.add_argument("--output", help="Optional UTF-8 JSON output path")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "plan-query-slots":
        checkpoint_path = (
            Path(args.output).with_suffix(".report.json") if args.output else None
        )
        report = run_slot_plan_with_report(
            args.question, args.acceptance_case, checkpoint_path, args.max_attempts
        )
        if args.output:
            console.print(f"Saved diagnostics: {checkpoint_path}")
            if report.status == "passed" and report.final_plan is not None:
                Path(args.output).write_text(
                    report.final_plan.model_dump_json(indent=2), encoding="utf-8"
                )
                console.print(f"Saved Plan: {args.output}")
        else:
            console.print_json(report.model_dump_json())
        if report.status != "passed":
            raise SystemExit(1)
        return
    if args.command == "plan-query-staged":
        checkpoint_path = (
            Path(args.output).with_suffix(".report.json") if args.output else None
        )
        report = run_staged_plan_with_report(
            args.question, args.acceptance_case, checkpoint_path
        )
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            report_path = output_path.with_suffix(".report.json")
            report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            console.print(f"Saved diagnostics: {report_path}")
            if report.final_plan is not None:
                output_path.write_text(
                    report.final_plan.model_dump_json(indent=2), encoding="utf-8"
                )
                console.print(f"Saved Plan: {output_path}")
        else:
            console.print_json(report.model_dump_json())
        if report.status != "passed":
            raise SystemExit(1)
        return
    if args.command == "plan-query":
        try:
            checkpoint_path = (
                Path(args.output).with_suffix(".report.json")
                if args.output
                else None
            )
            initial_plan = None
            if args.resume_report:
                if checkpoint_path and Path(args.resume_report).resolve() == checkpoint_path.resolve():
                    raise ValueError("resume source and output report must use different paths")
                prior_report = PlannerRunReport.model_validate_json(
                    Path(args.resume_report).read_text(encoding="utf-8")
                )
                if prior_report.question != args.question:
                    raise ValueError("--resume-report question does not match --question")
                initial_plan = prior_report.final_plan
                if initial_plan is None:
                    initial_plan = next(
                        (
                            attempt.candidate_plan
                            for attempt in reversed(prior_report.attempts)
                            if attempt.promoted_to_repair_base
                            and attempt.candidate_plan is not None
                        ),
                        prior_report.initial_plan,
                    )
                if initial_plan is None:
                    raise ValueError(
                        "--resume-report contains no promoted schema-valid candidate"
                    )
            report = run_natural_language_plan_with_report(
                args.question,
                args.acceptance_case,
                checkpoint_path,
                args.max_attempts,
                initial_plan,
                args.resume_report,
            )
        except Exception as exc:
            error = {
                "status": "natural_language_planner_failed",
                "stage": "plan_only",
                "reason": str(exc),
                "evidence_collection_started": False,
            }
            console.print_json(json.dumps(error, ensure_ascii=False))
            raise SystemExit(1) from exc
        content = (
            report.final_plan.model_dump_json(indent=2)
            if report.final_plan is not None
            else None
        )
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            report_path = output_path.with_suffix(".report.json")
            report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            console.print(f"Saved diagnostics: {report_path}")
            if content is not None:
                output_path.write_text(content, encoding="utf-8")
                console.print(f"Saved Plan: {output_path}")
        else:
            console.print_json(report.model_dump_json(indent=2))
        if report.status != "passed":
            raise SystemExit(1)
        return
    if args.command in {"plan", "agent1-planned"}:
        try:
            content = (
                run_research_plan_as_json(
                    args.disease,
                    args.target,
                    args.top,
                    args.disease_id,
                    args.target_id,
                )
                if args.command == "plan"
                else run_planned_agent1_as_json(
                    args.disease,
                    args.target,
                    args.top,
                    args.disease_id,
                    args.target_id,
                )
            )
        except Exception as exc:
            failure_stage = (
                "agent1_research_planner"
                if isinstance(exc, ResearchPlannerError)
                else "verified_entity_resolution_or_planner_initialization"
            )
            error = {
                "status": "research_planner_failed",
                "stage": failure_stage,
                "reason": str(exc),
                "evidence_collection_started": False,
            }
            if isinstance(exc, EntityResolutionError):
                error["resolution_status"] = "ambiguous"
                error["entity"] = exc.entity
                error["query"] = exc.query
                error["candidates"] = exc.candidates
            console.print_json(json.dumps(error, ensure_ascii=False))
            raise SystemExit(2) from exc
        emit_json(content, plan_output_path(args.disease, args.output))
        return
    if args.command == "discover-targets":
        output_path = discovery_output_path(args.disease, args.top, args.output)
        emit_json(
            run_target_discovery_as_json(args.disease, args.top),
            output_path,
        )
        return
    if args.command == "agent1":
        emit_json(
            run_agent1_as_json(
                args.disease,
                args.target,
                args.max_evidence_records,
                args.with_synthesizer,
            ),
            args.output,
        )
        return
    if args.command == "agent2":
        try:
            content = run_agent2_as_json(args.evidence)
        except Agent1EvidenceInputError as exc:
            console.print(f"[bold red]AGENT1_EVIDENCE_REQUIRED:[/bold red] {exc}")
            raise SystemExit(2) from exc
        emit_json(content, args.output)
        return
    if args.command == "evidence":
        emit_json(pipeline_as_json(args.disease, args.target), args.output)
        return
    if args.command == "chat":
        console.print(Markdown(run_agent(" ".join(args.query))))
        return
    console.print(
        "Use `python agent.py discover-targets --disease \"...\" --top 10`, "
        "`python agent.py agent1 --disease \"...\" --target \"...\"`, "
        "`python agent.py agent2 --evidence \"agent1.json\"`, or "
        "`python agent.py evidence --disease \"...\" --target \"...\"`."
    )


if __name__ == "__main__":
    main()
