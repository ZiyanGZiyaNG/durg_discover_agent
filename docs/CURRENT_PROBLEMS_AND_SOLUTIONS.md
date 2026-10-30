# Current Problems and Recommended Solutions

## 1. Natural-language plans are only partly executable

**Current state:** An execution compiler now exists (`execution/`). Approved plans are checked before any retrieval (operation approved, adapter exists, filter/ranking fields exist, required data is provided by an earlier step), compiled to an ordered tool-call list (`python -m execution PLAN.json --dry-run`), and run with typed output. Entity resolution and step-to-step data flow work. Adapters cover 54 of 64 operations; 15 of 18 acceptance cases have every required operation implemented. The Planner does not yet emit `inputs` references.

**Solution:** Add adapters for the remaining operations by capability (generic, not drug- or disease-specific), and update Planner examples to emit `inputs`.

**Done when:** Every acceptance case compiles to tool calls or is refused with a clear reason, and the adapter set covers the operations the Planner actually proposes.

## 2. Source support is uneven

**Current state:** Open Targets is the most developed source. ChEMBL is partial/legacy; Europe PMC and ClinicalTrials.gov do not yet have production-level adapters.

**Solution:** Use one adapter interface with source-native payloads and a common metadata envelope. Every adapter must implement validation, pagination, timeout, retry, truncation, provenance, and explicit completion status.

## 3. Collection limits are not scientifically finalized

**Current state:** Planned execution can stop at `collection_policy_not_configured`.

**Solution:** The research team defines record, page, byte, and time limits per source. Engineering encodes them as versioned policies.

**Important:** `no_records`, `partial_timeout`, and `partial_truncated` are different states. None automatically proves absence of evidence.

## 4. Real-model planning is slow and variable

**Observed:** Qwen/Ollama runs have ranged from minutes to stalled runs. Timeouts, checkpointing, cleanup, direct one-call generation, and local patches are implemented, but the latest dependency-only prompt still needs a representative real-model test.

**Solution:** Route fixed questions to deterministic manifests. Use the LLM only for genuinely multi-step planning. Repair only the failing portion, preserve accepted capabilities, and replay historical failures offline before another real-model run.

## 5. Planner acceptance is not scientific acceptance

**Current state:** Validators can prove schema, operation, dependency, filter, ranking, and safeguard compliance. They cannot prove biomedical adequacy.

**Solution:** Keep three gates: structural validation, execution validation, and domain-expert scientific review.

## 6. Output is still mostly standalone JSON

**Problem:** Cross-run analysis, deduplication, and selective agent queries are awkward.

**Solution:** Introduce Parquet plus DuckDB. Preserve immutable run snapshots, source-native schemas, run metadata, and a latest/deduplicated view. Store structured claims and review flags separately from raw evidence.

## 7. Dependencies are not fully reproducible

**Current state:** Most dependencies use minimum versions rather than exact versions. `pytest` is not in the runtime requirements.

**Solution:** Separate runtime and development dependencies, define supported Python versions, and generate a reviewed lock file. Record Ollama version and model digest for real-model results.

## 8. Some scripts and examples contain local absolute paths

**Problem:** Another computer may not use drive D or the same user directory.

**Solution:** Resolve paths relative to the repository or CLI arguments. Treat absolute-path examples as local examples, not required layout.

## 9. Public-repository data governance remains open

**Problem:** Raw records, model prompts, diagnostics, and external datasets may have source-specific licensing or privacy considerations.

**Solution:** The research team approves a publication policy before adding new raw datasets. Never commit API keys, `.env`, credentials, private prompts, or restricted records.