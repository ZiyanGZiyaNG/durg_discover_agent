# Decisions Required from the Research Team

Engineering contributors must not silently decide these items.

## Evidence collection

- Record, page, byte, and time limits per source.
- Required first-round versus supplemental evidence.
- Maximum Agent 2 supplemental requests.
- Source priorities and partial-failure behavior.
- Scientific sufficiency for Stage 1 completion.

## Interpretation

- Approved claim and review-flag taxonomy.
- Taxonomy versioning.
- Evidence-quality weights.
- Whether numerical confidence is needed and its calculation.
- Rules for conflicts, missing evidence, negative findings, and expert escalation.

## Target and drug analysis

- Selection thresholds beyond the initial top 10.
- How association and translational/drug support are combined.
- Definitions of tractability, clinical maturity, selectivity, and acceptable ADMET evidence.
- Sources required for approval claims.
- Approve or replace the provisional `provisional_v0` translation layers
  (`execution/analysis.py`): `translational_support` means at least one known-drug
  record or at least one tractability assessment with value true; `undetermined`
  means a required evidence source is missing for the target; otherwise
  `disease_association_only`. No thresholds or scores are applied.
- Whether `retrieve_target_drugs` must also query ChEMBL. The planner lists
  `chembl` as a source, but the adapter currently queries Open Targets only.

## Data governance

- Required source release/API/schema metadata.
- Snapshot retention and deduplication.
- Redistribution and licensing rules.
- Public-repository policy for raw records and model diagnostics.
- Retention of prompts, model output, and expert annotations.

## Recommended defaults pending approval

- Partition evidence by datasource and run.
- Preserve immutable snapshots plus a latest/deduplicated view.
- Keep nulls in stable schemas.
- Preserve source-native records under common metadata.
- Use structured claims without numerical confidence in v1.
- Treat live exploration as informal until Agent 1 validation promotes it.