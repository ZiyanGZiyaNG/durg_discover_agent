# Agent 1 Architecture

## Target workflow

```text
Research question
  -> deterministic or complex planning route
  -> Query Manifest
  -> structural and semantic validation
  -> deterministic entity resolution
  -> execution compiler
  -> bounded source tools
  -> evidence validation
  -> persistent evidence store
  -> deterministic review
  -> structured synthesis
  -> Stage1EvidencePackage
  -> Agent 2
```

The Natural-language Planner has no retrieval tools. It creates a plan and Python validates it. A separate execution layer must map approved operations to tool calls.

| Layer | Main files | Status |
| --- | --- | --- |
| CLI | `agent.py` | Implemented |
| Natural-language planning | `natural_language_planner.py` | Implemented; real-model experimental |
| Staged planning | `staged_planner.py` | Experimental |
| Operation contracts | `operation_rules.py` | Implemented |
| Acceptance | `planner_acceptance.py` | Implemented |
| Open Targets retrieval | `query_opentargets.py` | Partial |
| ChEMBL retrieval | `query_chembl.py` | Partial/legacy |
| Europe PMC adapter | `query_europepmc.py`, `execution/external_source_adapters.py` | Implemented and live-tested; offline tests still to be written |
| ClinicalTrials.gov adapter | `query_clinicaltrials.py`, `execution/external_source_adapters.py` | Implemented and live-tested; offline tests still to be written |
| Validation/review | `expert_review.py`, `evidence_pipeline.py` | Partial |
| V2 execution compiler | 2026-10-31 | Done: pre-execution checks, compile/dry-run, execution trace and adapters for all 64 operations; the Planner does not yet emit `inputs` references |
| Stage 1 handoff | schemas in progress | Not finalized |

## Execution policy

- Use deterministic templates for common disease-only and exact-pair questions.
- Use QueryManifestV2 for genuinely multi-step questions.
- Resolve identifiers before dependent retrieval.
- Run independent source calls concurrently only after prerequisites pass.
- Enforce record, page, byte, retry, and wall-time limits per source.
- Record success, partial, empty, error, and not-executed states.
- Keep `no_records` separate from a supported negative conclusion.

## Proposed persistence

Use Parquet plus DuckDB. Preserve source-native records under a stable metadata envelope. Keep source records, scores, run metadata, structured claims, and review flags separate. Exclude numerical model confidence until the team approves a calculation.