# Agent 1 Handoff

## Purpose

Agent 1 is the project's evidence-acquisition boundary. It plans research, resolves biomedical entities, retrieves evidence through approved sources, validates provenance and completeness, and prepares a Stage 1 package for Agent 2. Agent 2 must not query external sources directly.

## Current status

### Implemented and tested

- Disease-only target discovery and top-N Open Targets ranking.
- Exact disease-target pair lookup using verified IDs.
- Overall and per-datasource association scores.
- Partial underlying Open Targets evidence retrieval.
- Provenance, pagination, truncation, limitations, and expert-review views.
- Query Manifest v1 and plan-first Query Manifest V2.
- Tool-free Natural-language Planner with hidden semantic acceptance.
- Operation contracts, dependency and metric validation, bounded repairs, timeouts, checkpoints, and diagnostic reports.
- Historical Planner regression catalog.

Offline baseline on 2026-09-15: **150 tests passed**.

### Experimental

- Real-model Qwen/Ollama planning and local repair.
- `plan-query-staged`.
- Resume-from-report repair.

### Designed but incomplete

- Planner support for the execution compiler: the compiler and adapters for all 64 operations exist, but the Planner does not yet emit `inputs` references.
- Offline tests for the Europe PMC and ClinicalTrials.gov query modules, and a production-ready ChEMBL adapter (`query_chembl.py` is partial/legacy).
- Per-source bounded collection policies.
- Final `Stage1EvidencePackage`.
- Agent 2 `EvidenceRequest` supplemental retrieval.
- Parquet/DuckDB persistence and structured claims/review flags.

## Non-negotiable boundaries

- The LLM cannot verify or invent entity IDs.
- Open Targets scores are not probabilities of causality, efficacy, or success.
- Missing or incomplete retrieval is not negative evidence.
- Raw facts, deterministic derivations, and model interpretation remain separate.
- Preserve identifiers, sources, time, pagination, truncation, and failures.
- Do not add numerical confidence before team approval.
- Unknown operations remain blocked until human approval.
- Do not add case-specific rules merely to pass one acceptance example.

## Contributor workflow

1. Install dependencies and copy `.env.example` to `.env`.
2. Run `python -m pytest -q -p no:cacheprovider`.
3. Read `PROJECT_STATUS.md` and all files in `docs/`.
4. Select one backlog item with explicit acceptance criteria.
5. Add offline regression tests before costly model runs.
6. Record runtime, attempts, tool calls, timeouts, record counts, and artifacts without committing credentials.