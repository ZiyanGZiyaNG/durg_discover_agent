# Current Problems and Recommended Solutions

## 1. Natural-language plans are not yet automatically executable

**Current state:** Query Manifest V2 planning and validation are mature, but there is no complete compiler from approved operations to retrieval tool calls.

**Solution:** Implement an execution compiler that reads the authoritative operation table and maps each approved operation to an adapter, validated input, bounded execution policy, and typed output.

**Done when:** A validated plan can execute against mocked adapters, records an execution trace, and refuses unknown operations without starting retrieval.

## 2. Source support is uneven

**Current state:** Open Targets is the most developed source. ChEMBL is partial/legacy; Europe PMC and ClinicalTrials.gov do not yet have production-level adapters.

**Solution:** Use one adapter interface with source-native payloads and a common metadata envelope. Every adapter must implement validation, pagination, timeout, retry, truncation, provenance, and explicit completion status.

## 3. Collection limits are not scientifically finalized

**Current state:** Planned execution can stop at `collection_policy_not_configured`.

**Solution:** The research team defines record, page, byte, and time limits per source. Engineering encodes them as versioned policies.

**Important:** `no_records`, `partial_timeout`, and `partial_truncated` are different states. None automatically proves absence of evidence.

## 4. Real-model planning is slow and variable

**Observed:** Qwen/Ollama runs have ranged from minutes to stalled runs. Timeouts, checkpointing, cleanup, direct one-call generation, and local patches are implemented, but the latest dependency-only prompt still needs a representative real-model test.

**Solution:** Python first normalizes what OPERATION_TABLE already decides (missing edges, reversed edges, step order, required safeguards, duplicated analysis sources; see `dependency_inference.py` and `contract_normalization.py`). Route fixed questions to deterministic manifests. Use the LLM only for genuinely multi-step planning. Repair only the failing portion, preserve accepted capabilities, and replay historical failures offline before another real-model run.

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