# Agent 1 Improvement Backlog

## P0 — Collaboration baseline

- Keep this repository and `D:\NCKU_Drug_Discovery` as the canonical source.
- Never commit `.env`, `.venv`, installers, caches, or build dependencies.
- Remove user-specific absolute paths from reusable scripts.
- Split runtime and development dependencies and add a lock.
- Add CI for offline tests.

## P1 — Close the Plan-to-evidence loop

- Done (2026-10-31): the QueryManifestV2 execution compiler (`execution/`) with adapters for all 64 operations.
- Extend operation contracts with executor, schemas, prerequisites, limits, retry, timeout, and cache metadata.
- Implement one adapter interface for all four sources.
- Define bounded collection and completion statuses.
- Finalize `Stage1EvidencePackage`.
- Implement bounded Agent 2 supplemental requests.

## P2 — Evidence persistence

- Add Parquet and DuckDB.
- Preserve per-run snapshots.
- Provide a latest/deduplicated view without deleting history.
- Keep stable fields nullable; do not remove fields per row.
- Preserve source-native schemas linked by shared identifiers.
- Add versioned claim and review-flag taxonomies after approval.

## P3 — Planner reliability

- Route fixed patterns to deterministic manifests.
- Keep model calls bounded by Python.
- Repair upstream structure, then filtering, then ranking.
- Run a representative real-model test of the revised dependency prompt.
- Measure model digest, latency, attempts, tokens, timeouts, and regressions.

## P4 — Scientific validation

- Test known, low-ranked, missing, ambiguous, truncated, and partial-failure cases.
- Test multi-disease, drug-absence, trial-without-results, literature ambiguity, and ChEMBL mapping.
- Separate structural, execution, and scientific acceptance.
- Benchmark volume, storage, query latency, and reruns.