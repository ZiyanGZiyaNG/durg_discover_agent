# NCKU Open-Model Drug Discovery Assistant

An independent continuation of the NCKU drug-discovery assistant. The original
Open Targets and ChEMBL workers are preserved, while the paid Claude dependency
is replaced by local Ollama by default.

## Current scope

The first two research agents are implemented:

1. **Bioinformatics Evidence Agent** resolves a disease and target, queries Open
   Targets, and produces stable JSON with the overall association score,
   datasource scores, IDs, provenance, retrieval time, and limitations.
2. **Target Evidence Interpretation Agent** interprets only that JSON and returns
   a structured research report. It does not invent a confidence score because
   the research team has not defined the scoring framework.

The original ChEMBL drug-properties worker remains available and unchanged in
scope. Outputs are for research use and are not clinical recommendations.

## Contributor handoff

Before changing Agent 1, read:

- [Project status](PROJECT_STATUS.md)
- [Agent 1 handoff](docs/AGENT1_HANDOFF.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Improvement backlog](docs/IMPROVEMENT_BACKLOG.md)
- [Research-team decisions](docs/DECISIONS_REQUIRED.md)
- [Acceptance criteria](docs/ACCEPTANCE_CRITERIA.md)
- [Current problems and solutions](docs/CURRENT_PROBLEMS_AND_SOLUTIONS.md)
- [User guide](docs/USER_GUIDE.md)
- [Model configuration and replacement](docs/MODEL_CONFIGURATION.md)

The GitHub repository is the shared canonical source. Generated files under
`outputs/` are run artifacts and Planner diagnostics, not source code.
## Architecture

```text
Natural language chat ──> Supervisor ──> Open Targets worker
                                  └────> ChEMBL drug-properties worker

Disease + target ──> Agent 1: deterministic evidence JSON
                 └─> Agent 2: constrained structured interpretation
```

## Windows setup with Ollama

1. Install Python 3.11 or newer.
2. Install Ollama from <https://ollama.com/download>.
3. Open PowerShell and download the development model:

   ```powershell
   ollama pull qwen3:14b
   ```

4. Create and activate a Python virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   Copy-Item .env.example .env
   ```

5. Verify Ollama:

   ```powershell
   ollama list
   ollama run qwen3:8b "Reply with OK"
   ```

### Portable Windows fallback

If the installer reports `Failed to expand shell folder constant
"localappdata"`, download the official `ollama-windows-amd64.zip` package,
extract it, and start the server from a PowerShell window:

```powershell
.\ollama.exe serve
```

Keep that window open while using the assistant. In a second PowerShell window:

```powershell
.\ollama.exe pull qwen3:8b
```

This project includes three presets tuned for the current 32 GB RAM and 8 GB
NVIDIA GPU laptop:

| Profile | Model | Context | Use |
| --- | --- | ---: | --- |
| `lightweight` | `qwen3:8b` | 4096 | Fast deployment/fallback |
| `development` | `qwen3:14b` | 4096 | Default feature development |
| `validation` | `qwen3:32b` | 2048 | Slow quality comparison only |

The 32B profile relies heavily on system RAM. Close memory-heavy applications
before using it. It is deliberately assigned a smaller context to reduce the
risk of exhausting the machine's 32 GB RAM.

## Run the two-agent evidence pipeline

```powershell
python agent.py evidence `
  --disease "pancreatic ductal adenocarcinoma" `
  --target "KRAS"
```

The command returns one JSON object containing `agent_1_evidence` and
`agent_2_interpretation`.

## Run each evidence agent separately

Discover the first 10 candidate targets from a disease name. This mode uses
Open Targets overall association score only for initial ordering and does not
download the underlying evidence records:

```powershell
python agent.py discover-targets `
  --disease "Alzheimer's disease" `
  --top 10 `
  --output "D:\NCKU_Drug_Discovery\outputs"
```

The discovery filename is generated from the disease input. For example,
`--disease "pancreatic cancer" --top 10` produces
`pancreatic_cancer_top10_targets.json`. If an old JSON filename is supplied to
`--output`, only its parent directory is used so the filename cannot retain a
different disease name.

Run Agent 1 and save its evidence JSON:

```powershell
python agent.py agent1 `
  --disease "pancreatic ductal adenocarcinoma" `
  --target "KRAS" `
  --max-evidence-records 500 `
  --output "agent1_output.json"
```

Agent 1 now expands the `clinical_precedence` datasource score into its
underlying Open Targets evidence rows. The evidence field list is generated
from the live GraphQL schema, so release-specific clinical report fields are
preserved without LLM reconstruction. The JSON reports total/fetched counts,
whether the records were truncated, and an estimated complete byte size. Raise
`--max-evidence-records` only after reviewing that estimate.

Run Agent 2 from the saved Agent 1 JSON:

```powershell
python agent.py agent2 `
  --evidence "agent1_output.json" `
  --output "agent2_output.json"
```

The `--output` option is optional. Without it, JSON is printed in the terminal.

## LLM Research Planner

The deterministic commands remain available and do not require an LLM. Use
Qwen through Ollama to create a schema-validated plan without collecting the
planned evidence:

```powershell
python agent.py plan `
  --disease "pancreatic cancer" `
  --disease-id "MONDO_0005192" `
  --top 10 `
  --output "D:\NCKU_Drug_Discovery\outputs\plans"
```

Add `--target "KRAS"` for specified disease-target planning. Use
`agent1-planned` to enter the policy-gated execution flow. It currently returns
`collection_policy_not_configured` without bulk retrieval because safe limits
for the four sources have not yet been established.

For phase-1 free-text Plan-only testing, use `plan-query`. When an acceptance
case is selected, its rubric remains hidden on the first attempt. Optional
`PLANNER_FEW_SHOT_MODE=synthetic_v1` adds two synthetic planning demonstrations;
`off` (default) restores the baseline. Examples live in `planner_examples.py`

Schema-valid failed Plans are repaired with bounded `QueryManifestPatch`
responses by default. Python applies only explicit entity, step, ordering, or
top-level-field edits to the previous candidate and then reruns the complete
schema, operation rules, and hidden acceptance checks. This avoids regenerating
already-correct content. Python derives an authorized repair scope from the
actual findings and rejects edits to unrelated entities, steps, or top-level
fields. A candidate becomes the next repair base only when it preserves every
previously accepted capability and reduces the total blocking-issue count;
otherwise repair rolls back to the better base. Invalid patches add their error
to the still-unresolved feedback instead of replacing it. Repair prompts include
only operations already in the Plan or explicitly named by the findings, while
the blind first Plan still receives the configured complete catalog. Set
`PLANNER_REPAIR_MODE=full_rewrite` to restore the former whole-Plan repair flow
for comparison; `local_patch` is the default.
Patch responses intentionally omit parent-Plan metadata such as
`schema_version`, `manifest_id`, `original_question`, and `generated_at`; Python
preserves those authoritative values. Internal structured-output retries are
avoided entirely for patch calls: Python invokes the chat model exactly once,
extracts one JSON object from the final text (after any thinking text), and
validates it locally. One invalid response therefore returns to the bounded
outer repair loop instead of repeatedly calling the model inside one attempt.
and are experimental drafts, not expert-approved research standards. They
assert no biomedical results and do not alter model weights or the 18 cases.

Run `python compare_few_shot.py --output outputs/plans/few_shot_comparison_v1`
for two development questions, two repetitions and two modes (eight runs).
The experiment fixes the catalog to full and uses the same 16384-token context
for both modes (process-
local override, not a change to the development profile). The initial 4096-token
baseline was aborted after server logs confirmed context shifting during
generation, so it is excluded from scoring. It evaluates first attempts without
repair feedback, alternates mode order between repetitions, saves each report,
and stops the batch on a model timeout. Each run retains existing time limits.
No holdout case or retrieval tool is used. Passing the rubric still requires
manual semantic review; these two questions cannot establish general quality.
Use `--modes off` for a single diagnostic baseline without a paired run.

Planner diagnostic reports use schema 1.1. Each outer attempt contains a
`model_calls` list recording every underlying chat-model request observed by
LangChain callbacks: start/end time, duration, actual message payload, raw
response, status and error. This distinguishes hidden-rubric repair attempts
from structured-output retries inside one attempt. On an outer timeout, any
currently running call is marked `timed_out`; all earlier completed calls remain
in the checkpoint. These local reports can be large and contain the user's full
question, so review before sharing them.

`QueryManifestV2.generated_at` is runtime metadata and now defaults in Python;
the Planner is explicitly told not to generate it. The prompt also shows the
fixed `characteristics` object and clarifies that `entity_refs` can only reuse
top-level declared references. The first two changes removed their observed
schema errors in a KRAS rerun, but prompt wording alone did not stop the model
from treating prior output names as entity references. See the corresponding
`RESULT.md` before changing the manifest contract or adding normalization.

The approved safe normalization now removes such a value only if it is a
uniquely identified output of a direct or transitive upstream step. The existing
dependency chain retains data flow and every removal is reported. Arbitrary,
downstream, unrelated, or ambiguous references still fail validation. In the first real
KRAS rerun this reduced five retrying calls to one completed schema-valid call;
the Plan then failed the separate semantic rubric, so normalization does not
turn weak research content into a passing Plan.

For the standard `plan-query` repair flow, when an acceptance
case is selected, the first attempt is blind: the case rubric is held by the
Python evaluator and is not included in the model prompt. A sibling
`.report.json` records each attempt, timing, available token metadata, missing
or incorrect items, repairs, and zero retrieval tools. The Plan file is written
only after hidden acceptance passes.

Ranking instructions in `QueryManifestV2` use a fixed contract:
`primary_metric`, `direction` (`ascending` or `descending`), and optional
`secondary_metrics`. Legacy `sort_by`/`rank_by`/`order_by` values and ordered
`sort` lists are normalized into that contract. An unordered `criteria` list is
rejected because it does not identify the primary metric or direction.

The Planner uses one authoritative operation table containing canonical names,
human-reviewed equivalents, meanings, and approval status. A reviewed alias is
normalized to its canonical operation. If no equivalent capability exists, the
Planner must emit `proposed_new_operation` with a proposed name and reason. Such
a Plan is blocked for human review and cannot proceed to execution until the
operation table is updated.

After schema validation, every candidate Plan is also checked against the
general `operation_rules.py` rule table. Seven rules block acceptance or require
repair. Operation-specific entity types, required upstream capabilities,
provided capabilities, safeguards, aggregation requirements, reserved outputs,
and supported filters now come from the authoritative operation table instead
of parallel rule-specific registries. A filter, ranking, or aggregation may use
a reviewed metric only when the current operation or an upstream dependency
provides its required evidence; unsupported filters and unregistered ranking
metrics require review. Three deliberately conservative checks only warn: explicit
aliases split into multiple entities, explicitly separate comparison groups
being collapsed, and numeric thresholds not stated in the question. Findings
name the affected step, the missing item, and the required correction. They do
not retrieve data or silently rewrite research operations.

`tests/test_planner_regression_catalog.py` is the fast offline regression
catalog. It records minimal versions of failures observed in real Planner runs,
classifies the layer responsible, and pairs blocking cases with reasonable Plan
alternatives that must remain accepted. Add a case here only for a previously
observed general failure; case-specific scientific expectations remain in the
acceptance fixtures.

Top-level entities must also be grounded in the original question after
case/width/whitespace/punctuation normalization. Explicit full names,
abbreviations and parenthetical aliases remain valid. A generated collection or
step output cannot be declared as an entity; downstream data flow must use
`depends_on`.

Full-Plan and patch generation each use one direct model call followed by local
Python JSON extraction and validation. This prevents structured-output
middleware from repeatedly asking the model to repair the same schema error
inside one attempt. `depends_on` accepts earlier `step_id` values, never output
names. Python converts an output-name dependency only when it has exactly one
prior producer; unknown, ambiguous, and future references remain errors.

Reviewed operation capability contracts also prevent one canonical operation
from claiming another operation's reserved output. Evidence-layer
stratification and category-distribution operations must describe their actual
aggregation method, but the global rule does not impose a case-specific JSON
key. Category-distribution aggregation cannot claim the reviewed
`target_translation_layer` output. Approval-status metrics require upstream
approval evidence, not merely a target-drug relation.

`PLANNER_OPERATION_CATALOG_MODE=grouped_compact` is an experimental reversible
prompt view: it groups all canonical operations and reviewed equivalents while
expanding only easily confused boundaries. Python still validates against the
complete table. Set the value to `full` to restore every expanded operation
description without changing code or acceptance data.

The reversible `plan-query-staged` experiment avoids asking one model call to
both design the research and recall the whole operation table. Stage 1 produces
atomic evidence/analysis needs without operation names. Python searches the
complete authoritative table and supplies at most five candidates per need to
Stage 2, which must choose a canonical candidate or propose a new operation for
human review. Python then compiles `QueryManifestV2` and runs the same hidden
acceptance checks. The original `plan-query` path remains available unchanged.

Each model attempt has a 20-minute wall-clock limit and each question has a
60-minute total limit by default. A checkpoint report is rewritten after every
completed attempt, so a stalled later repair cannot discard earlier candidate
Plans. Ollama keeps the model loaded for 30 minutes to avoid unnecessary unloads.
If generation exceeds its limit, the runner requests `ollama stop`, monitors the
model for up to 90 seconds, and records whether `Stopping` was observed, cleanup
succeeded, or stopping itself timed out. It never starts retrieval after such a
failure. Override only when needed with `PLANNER_ATTEMPT_TIMEOUT_SECONDS`,
`PLANNER_TOTAL_TIMEOUT_SECONDS`, and the documented Ollama cleanup settings.
The `plan-query` command permits one to five total attempts and defaults to four,
so an initial Plan can be followed by separate structure, filtering, and ranking
repair batches without removing the total wall-clock bound. Use
`--max-attempts 3` to restore the previous limit.

Repair feedback is grouped before it reaches the model. Upstream structure and
dependency defects are handled first, filtering controls second, and ranking
controls last. The authorized patch scope is narrowed to the selected group.
When a patch only updates existing steps, a stale incomplete `step_order` is
ignored and the base order is preserved; incomplete ordering remains an error
when steps are actually added or removed.

Patch dependency order is compiled by Python using a stable topological sort.
The model declares dependency edges; missing step references and cycles remain
errors. Existing-step updates preserve omitted fields, while supplied objects
and lists replace that field completely. The prompt exposes the validator's
reviewed metric registry. Resumed reports retain their seed and source and are
reported separately from blind first-attempt results. These flow-review changes
have not yet been tested (user requested implementation only).

```powershell
python agent.py plan-query `
  --acceptance-case pdac_known_targets `
  --question "胰臟癌（pancreatic ductal adenocarcinoma, PDAC）目前有哪些已知的治療標的？" `
  --output "D:\NCKU_Drug_Discovery\outputs\plans\pdac_known_targets_blind.json"
```

To compare the staged route on the same case:

```powershell
python agent.py plan-query-staged `
  --acceptance-case pdac_known_targets `
  --question "胰臟癌（pancreatic ductal adenocarcinoma, PDAC）目前有哪些已知的治療標的？" `
  --output "D:\NCKU_Drug_Discovery\outputs\plans\pdac_known_targets_staged_v1.json"
```

If a free-text disease or target does not have exactly one matching canonical
name, planning stops before Qwen and prints candidate IDs. Review the intended
scope, then rerun with `--disease-id`; use `--target-id` when a target is also
ambiguous. The explicit ID is verified against Open Targets before planning.
Agent 2 strictly requires an Agent 1 output containing `producer: "agent1"`, a
valid UUID `run_id`, `generated_at`, and schema-valid `agent_1_evidence`.
Handwritten evidence JSON or the older metadata-free format is rejected before
the local model is started.

## Use the preserved chat assistant

```powershell
python agent.py chat "What are the top targets for pancreatic ductal adenocarcinoma?"
```

## Change the local model

Select a profile in `.env`:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_PROFILE=development
```

Or select one for the current PowerShell session and optionally download it:

```powershell
.\ollama-profile.ps1 lightweight
.\ollama-profile.ps1 development -Pull
.\ollama-profile.ps1 validation -Pull
```

`OLLAMA_MODEL` and `OLLAMA_NUM_CTX` can still override a profile when running
experiments. Start with `development`, compare important results with
`validation`, and switch to `lightweight` after the workflow is stable.

The current development override uses `OLLAMA_NUM_CTX=8192` and
`OLLAMA_NUM_PREDICT=4096`. The first leaves room for the full operation catalog,
structured-output schema, reasoning and final JSON; the second bounds reasoning
plus answer generation so Ollama cannot context-shift indefinitely. Thinking is
not disabled. The 35-minute outer attempt timeout remains a final safety guard.

### Model storage

Keep large Ollama models on the D drive because the C drive has limited free
space. Before starting Ollama, set a persistent Windows user variable and then
restart Ollama:

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", "D:\\OllamaModels", "User")
```

Moving an existing model store is intentionally not automated; verify the new
location works before removing the original files.

## Optional Claude compatibility

Claude is no longer required. To compare with the original paid backend:

```powershell
pip install -r requirements-anthropic.txt
```

Then set:

```dotenv
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-key
ANTHROPIC_MODEL=claude-sonnet-4-6
```

## Tests

```powershell
pip install pytest
pytest
```

Unit tests do not require Ollama or live biomedical APIs. A live end-to-end run
requires Ollama plus internet access to Open Targets.

## Evidence and scoring policy

- Database values and provenance are collected by deterministic code.
- The LLM interprets supplied evidence but may not fabricate missing data.
- Open Targets association scores are not presented as clinical confidence.
- The confidence score remains `null` with status
  `pending_research_team_definition` until the team approves a scoring method.
