# Project Framework and Coordination Status

Last framework confirmation: 2026-09-15

## Canonical source directory

The shared canonical source is:

```text
https://github.com/bao-kai/ncku-drug-discovery-agent1
```

The current local checkout is `D:\NCKU_Drug_Discovery`. Source code, tests,
and documentation must be updated through this repository. `outputs/` contains
generated artifacts, not a second source tree. Older copies are historical only.
## Confirmed scope

Only Agent 1 and Agent 2 are in the current project scope. Later mechanism and
decision agents are deferred.

### Agent 1: Evidence Acquisition Agent

Agent 1 is the only agent allowed to use external data-retrieval tools. It:

- supports a specified disease and target;
- supports disease-only target exploration;
- ranks automatic target discovery by the Open Targets overall association
  score and selects the top 10 targets;
- uses Open Targets, Europe PMC, ClinicalTrials.gov, and ChEMBL;
- retrieves a controlled ChEMBL summary during the initial collection;
- performs deeper ChEMBL or other evidence retrieval when Agent 2 submits a
  structured evidence request;
- validates identifiers, provenance, pagination, truncation, and schemas;
- preserves raw evidence separately from derived summaries; and
- produces the Stage 1 evidence package consumed by Agent 2.

The previous Evidence Interpreter is part of Agent 1 and is renamed the
`Evidence Synthesizer`. It is not Agent 1B and must not occupy the Agent 2 role.
The Evidence Synthesizer may be implemented as an internal sub-agent. It has no
external retrieval tools and may only interpret evidence already retrieved and
validated by Agent 1. Its model-generated interpretation must remain separate
from database facts and raw evidence.

Agent 1 internal responsibilities are:

```text
Research Planner
-> Tool Runner
-> Evidence Validator
-> Evidence Synthesizer
```

### Agent 2: Target-Drug Evidence Analysis Agent

Agent 2 has no external retrieval tools. It:

- accepts only schema-valid Agent 1 evidence packages;
- analyzes disease-target-drug evidence;
- organizes mechanisms of action, activity, potency, selectivity evidence,
  molecular properties, clinical maturity, and available ADMET evidence;
- distinguishes database facts, derived comparisons, unknowns, and model
  interpretations;
- submits a structured `EvidenceRequest` to Agent 1 when required evidence is
  missing; and
- produces a `TargetDrugLinkCard` without inventing evidence or scores.

## Confirmed top-10 definition

"Top 10" means the top 10 candidate targets in disease-only exploration,
ranked initially by the Open Targets overall association score. It does not
mean limiting every target to only 10 underlying evidence records.

## ChEMBL responsibility

ChEMBL is a data source used through Agent 1. Agent 2 never queries ChEMBL
directly.

The collection strategy is:

```text
Initial Agent 1 run: controlled ChEMBL summary
Agent 2 review: identify missing evidence
Agent 2 -> Agent 1: structured EvidenceRequest
Agent 1: focused deep retrieval
Agent 2: final TargetDrugLinkCard
```

## Shared development rules

1. Check this file and the current schemas before changing Agent 1 or Agent 2.
2. Do not modify the older `work` project copy.
3. Keep database facts and LLM interpretation in separate fields.
4. Preserve raw identifiers, source references, retrieval time, and limitations.
5. Do not reinterpret Open Targets scores as probabilities or clinical success
   rates.
6. Do not add a project-defined confidence score until the research team defines
   its calculation.
7. Prefer backward-compatible schema additions; use a new schema version for
   semantic or structural breaking changes.
8. Update this status file when an agreed framework decision changes.
9. Agent 1 outputs must preserve complete raw records and also provide a
   deterministic expert-review view so domain experts can decide the future
   database classification without reading an unstructured raw JSON dump.

## Next design work

- Define tool input/output contracts for the four Agent 1 data sources.
- Define `Stage1EvidencePackage` for specified-target and top-10 discovery modes.
- Define Agent 2 `EvidenceRequest`.
- Define `TargetDrugLinkCard`.
- Define bounded collection and Agent 1/Agent 2 handoff stopping conditions.

## Implemented disease-only discovery

Agent 1 now provides a deterministic `discover-targets` CLI mode. It resolves
the disease, requests the selected number of associated targets, sorts by Open
Targets overall association score, and emits schema-valid candidate target
records with datasource scores and provenance. The v1 default is the confirmed
top 10. This discovery step intentionally does not download underlying evidence
records and does not use Ollama.

## Implemented Agent 1 Research Planner framework

Agent 1 now has a Qwen/Ollama `Research Planner` with no retrieval tools. The
`plan` command deterministically resolves disease/target identities first,
then produces a schema-validated plan. Disease-only planning uses the current
top-10 discovery result and creates one subplan per target. Open Targets,
Europe PMC, ClinicalTrials.gov, and ChEMBL are all available; the Planner
selects the sources needed for the question using only registered tool names.
LLM changes to verified IDs, mismatches between selected and planned sources,
invalid tool mappings, and terms incorrectly labeled as
verified are rejected; the model receives at most two repair attempts.

The existing `discover-targets` and `agent1` commands remain deterministic and
do not require an LLM. `agent1-planned` is a separate entry point. Until
per-source collection policies are established through data-volume testing,
it validates and saves the plan, then stops with
`collection_policy_not_configured` without evidence retrieval.

Free-text entity resolution no longer permits the Research Planner to silently
select a non-exact first search hit. Ambiguous disease or target text stops
before Qwen and returns candidate IDs; the user can explicitly select with
`--disease-id` or `--target-id`. Verified entity records are reattached by
Python after planning, so the LLM cannot clear or alter query text, IDs,
canonical names, symbols, synonyms, or provenance.

## Implemented Query Manifest contract

`Agent1ResearchPlan` schema version 1.1 now contains a mandatory pre-retrieval
`QueryManifest`. It records the original question, one of the six required final
query categories, verified entities, query path, target expansion strategy,
evidence requirements, selected sources, filters, ranking instructions, and the
Planner's selection reason. The current implemented inputs use `single_hop` for
disease-only discovery and `evidence_directed` for a specified disease-target
pair. Disease-only plans retain both `select_then_expand` and
`expand_all_top10`; Qwen chooses and explains the strategy. Python rejects
changes to verified disease/target identities, invalid paths, or incompatible
strategies before any retrieval can begin.

## Planner acceptance phase 1

All 18 natural-language research questions supplied by the research team are
now formal Planner acceptance cases. Phase 1 evaluates the generated plan only
and must not execute retrieval. Each case declares the expected entities,
research intent, required operations and dependencies, candidate data sources,
filters, ranking, aggregation, and safeguards. The six query characteristics
are descriptive metadata for indexing and coverage; they do not substitute for
the plan graph and are not the Planner's training target.

`QueryManifestV2` is the plan-first acceptance contract for these complex
questions. It represents unresolved or verified entities, ordered plan steps,
dependencies, source assignments, outputs, filters, ranking, aggregation,
completion criteria, and safeguards. The deterministic acceptance evaluator
checks plan semantics without calling Open Targets, Europe PMC,
ClinicalTrials.gov, ChEMBL, Ollama, or any other retrieval service.

The `plan-query` CLI is now the phase-1 natural-language entry point. It sends
the complete question to a tool-free Planner and validates the response as
`QueryManifestV2`; it does not resolve entity IDs or call research databases.
Unverified entities remain explicitly unresolved for a later execution phase.
Offline Planner and acceptance tests pass. The first live qwen3:14b structured
output completed in about five minutes and produced a schema-valid file, but it
failed the semantic acceptance case: the model invented a disease ID and
verification status, used incompatible operation names, and omitted required
ranking and safeguards. Plan-only validation removes any model-supplied
identity authority by clearing IDs/canonical names and forcing
`resolution_status=unresolved`.

Acceptance is now blind on the first attempt: the selected case is compiled
into a hidden rubric containing required capabilities, forbidden conditions,
optional paths, and reviewed operation equivalences, but that rubric is never
serialized into the Planner prompt. Only the user's question and generic
planning rules are visible. After hidden scoring, a repair attempt receives
only the concrete missing items and incorrect claims from its own prior Plan;
the complete expected Plan and research intent remain hidden. At most two
repairs are allowed.

Every attempt records wall time, schema validity, semantic gaps, corrected
unverified identity claims, repair feedback, available backend token/duration
metadata, and proof that evidence collection remained disabled. The CLI saves
this as a sibling `.report.json` file and saves a Plan only when hidden
acceptance passes. The earlier 25-minute guided run remains historical evidence
of model latency, not a blind capability result. No live model batch has been
run under the new blind protocol yet.

The first blind live run (`pdac_known_targets`, 2026-08-25) completed three
schema-valid attempts in 1455.53 seconds and failed hidden semantic acceptance.
Attempt 1 lacked recognized target discovery/ranking capabilities, both required
safeguards, and the association-score ranking key. Attempt 2 improved source and
safeguard coverage but lost the recognized disease-resolution capability and
still lacked the ranking key. Attempt 3 regressed toward the first attempt.
No retrieval tool ran and no Plan file was accepted. The diagnostic report did
not retain each rejected candidate Plan, so this result cannot yet distinguish
a genuinely missing research step from an unregistered but semantically
equivalent operation name. Rejected candidate retention is required before the
next blind run.

The pre-rerun repair work is now complete. Every rejected attempt retains its
normalized candidate Plan and raw structured payload, plus actual and
unrecognized operation names. All blind prompts include the same general Agent
1 operation catalog; the catalog does not identify which operations a specific
question should use. Ranking keys use a controlled equivalence map.

Repair attempts now receive the prior candidate Plan, concrete missing and
incorrect items, and an explicit preserve list for capabilities that already
passed. The evaluator records regressions when a later attempt loses a
previously accepted capability. Unknown operation names remain visible for
review instead of silently disappearing or automatically being treated as
equivalent. These changes passed the full offline suite; no new live model run
or retrieval was performed during the repair.

The approved `blind_v2` rerun completed on 2026-08-27 in 753.37 seconds (three
schema-valid attempts, zero retrieval). Candidate retention showed that attempts
2 and 3 did contain disease resolution, disease-target discovery, target
ranking, correct dependencies, and both required safeguards. Hidden acceptance
still failed for two evaluator representation gaps: `discover_targets_per_disease`
was present in the general catalog but not registered as equivalent to
`discover_targets`, and the model encoded `overall_association_score` as the
value of `ranking.sort_by` while the evaluator checked only ranking dictionary
keys. This is a confirmed false negative, not evidence that the Planner lacked
the research strategy. Total time improved by about 48% from blind_v1, and no
accepted capability regressed during repair.

The confirmed false-negative gaps are fixed: `discover_targets_per_disease` is
now a reviewed equivalent of `discover_targets`; ranking validation recognizes
canonical fields supplied through `sort_by`, `rank_by`, or `order_by`; and
Python overwrites model-generated `generated_at` values with the current UTC
runtime time. Offline replay of blind_v2 attempt 2 now passes every hidden
rubric field with zero missing items and zero retrieval. The original report is
preserved unchanged as historical evidence; no additional Qwen run was needed.

The subsequent `blind_v3` end-to-end run on 2026-08-27 was manually stopped
after more than 40 minutes. Ollama remained in `Stopping...` for over ten
minutes while the waiting Python process accumulated no additional CPU time,
indicating a stalled response-finalization/unload path rather than productive
thinking. No Plan or report file was written, no retrieval ran, and blind_v1/v2
artifacts were unchanged. Before retrying, the runner needs a per-attempt
timeout and incremental diagnostic checkpoint so a stalled later repair cannot
discard completed earlier attempts.

The timeout/checkpoint safeguard is now implemented. Defaults are 720 seconds
per model attempt and 1800 seconds per question, chosen from observed healthy
attempts (up to about 697 seconds) and healthy three-attempt runs (about 24
minutes). A hard wall-clock timeout stops further repair attempts, returns a
diagnostic failure, and never starts retrieval. The sibling report is written
after every completed attempt and again at final status, retaining candidate
Plans even if a later model call stalls. Environment overrides are documented
in `.env.example`.

The guarded `blind_v4` rerun completed successfully on 2026-08-27 in 945.30
seconds. Attempt 1 took 664.26 seconds and independently selected disease
resolution plus per-disease target discovery; it lacked target ranking, the
discovery-to-ranking dependency, both required safeguards, and the association
score ranking field. Attempt 2 received only its prior Plan and diagnostics,
preserved both accepted capabilities, added `rank_targets`, and passed every
hidden rubric field in 281.04 seconds with no regression. Python wrote the
current UTC `generated_at`. The final Plan and report were saved, and retrieval
remained zero. This is the first successful end-to-end blind acceptance result
for `pdac_known_targets` under the corrected protocol.

On 2026-08-29, three unchanged-program stability runs all completed normally
but failed final acceptance after two repairs. Every repair added the required
ranking operation, dependency, safeguards, and `overall_association_score`, but
used `ranking.criteria` or `ranking.sort`, which the evaluator did not
consistently recognize. This isolated a Plan-contract mismatch rather than a
missing research concept. `QueryManifestV2` now defines ranking with
`primary_metric`, `direction`, and `secondary_metrics`. Legacy `sort_by`,
`rank_by`, `order_by`, and ordered `sort` forms are normalized; `criteria` is
rejected as ambiguous and must be repaired. The Planner prompt and deterministic
evaluator use the same contract.

On 2026-08-30, the 18 acceptance cases were reviewed as a capability matrix
rather than six training classes. Twelve cases are marked `development` and six
capability-balanced cases are marked `internal_holdout`. The reviewed rules now
separate disease association from drug/tractability evidence, bound no-drug
claims by source and date, treat shared-disease scores as comparison rather
than synthetic ranking, require classify/count before mechanism ranking, and
flag regulatory and specialized expression confirmation gaps. The authoritative
human-readable map is `PLANNER_CAPABILITY_MATRIX_V1.md`. This changed acceptance
specifications only; it did not change the Planner prompt or run the model.

The first blind development run under the reviewed two-layer target rubric
failed after 1144.53 seconds. Attempt 1 produced only disease resolution and
associated-target discovery; it omitted drug evidence, tractability, evidence
stratification, ranking, ChEMBL, and the required safeguards. Attempt 2 hit the
720-second limit before returning a candidate. This was a genuine planning gap,
not an evaluator alias problem. The general Planner policy now defines
"therapeutic target" as distinct disease-association and translational-support
evidence layers and prohibits prose sentences in identifier fields. It does not
include PDAC-specific steps or the hidden case rubric.

On 2026-09-02, the operation registry was redesigned after project review. One
authoritative `OPERATION_TABLE` now owns canonical operation IDs, reviewed
equivalents, Chinese capability boundaries, and approval state. Python
normalizes reviewed aliases to canonical operations. A capability absent from
the table must be emitted as `proposed_new_operation` with a proposed name and
reason; it blocks acceptance and execution until human review adds it to the
table. The former catalog, equivalence, and semantics views are derived from
this table rather than maintained independently.

Planner limits were widened to 1200 seconds per model attempt and 3600 seconds
per question. Ollama `keep_alive` defaults to 30 minutes. The previous daemon
thread deadline only stopped Python from waiting and could leave Ollama stuck
in `Stopping`; timeout cleanup now requests `ollama stop`, polls for model
removal for up to 90 seconds, and records generation timeout, stop request,
Stopping detection, cleanup success, cleanup error, or stopping timeout in the
attempt report. The server is not automatically killed or restarted.

The operation-table prompt now supports a reversible experiment controlled by
`PLANNER_OPERATION_CATALOG_MODE`. `grouped_compact` retains every canonical ID
and reviewed equivalent, groups them by capability, and expands only eight
confusion-prone boundaries; `full` restores all 64 expanded descriptions. The
compact view is 5374 characters versus 7057 for full (23.8% shorter). In the
first real compact run, attempt 1 finished in 475.95 seconds versus 645.08
seconds for the preceding full-table run, but still produced only disease
resolution and target discovery. Attempt 2 hit 1200 seconds. Ollama remained in
`Stopping` throughout the 90-second cleanup grace period, so the report
correctly recorded `ollama_stopping_timeout=true`; the model disappeared later.
The compact experiment improved latency for one first attempt but did not yet
improve Plan completeness.

On 2026-09-03, the default operation-catalog mode was restored to `full` to
prioritize planning quality. A separate, reversible `plan-query-staged` route
was added instead of replacing `plan-query`. Its first model stage creates
atomic research needs without operation names. Python then searches the entire
authoritative `OPERATION_TABLE` and gives the second model at most five
candidate operations per need. The second model may select only a supplied
canonical candidate; otherwise it must propose a new operation and execution
is held for human review. Python owns `original_question`, compiles
`QueryManifestV2`, and performs the existing hidden semantic acceptance. Both
stages remain Plan-only and report zero retrieval tools. Intermediate strategy
and mapping checkpoints are saved before the next expensive stage.

Automated coverage now includes candidate retrieval, rejection of candidates
outside the shortlist, the human-review path for new operations, exact question
preservation, JSON mapping input, checkpoint output, and hidden acceptance.
The full suite passes 67 tests; the only warning is the pre-existing inability
to write pytest's optional cache directory.

The first real staged run used the `pdac_known_targets` development case. Both
model stages completed without timeout: strategy took 387.06 seconds, operation
mapping took 231.72 seconds, and total runtime was 618.81 seconds. No retrieval
tool ran. The hidden acceptance failed because the strategy omitted explicit
disease resolution, target-drug evidence, and final target ranking; it also
declared required inputs without graph dependencies and omitted the required
score/safeguard/aggregation semantics. The mapper correctly selected only
shortlisted canonical operations, but cannot repair research needs that Stage 1
never created. This result supports keeping staged planning as an experiment
and adding a generic strategy-quality gate before operation mapping, rather
than replacing the existing planner now.

## 2026-09-04 few-shot experiment

Superseding the proposed additional strategy gate: keep the full-table single
Planner and test two synthetic demonstrations first. `planner_examples.py`
contains exact-pair/source-ranking and shared-target/filter-ranking plans with
brief rationales. They are experimental drafts, not expert-approved standards.
No acceptance case is embedded or reassigned; no weights are trained.
`PLANNER_FEW_SHOT_MODE=off|synthetic_v1` controls a reversible prompt addition;
the default remains off. Reports record the selected mode. `compare_few_shot.py`
uses development cases only, two repetitions in alternating mode order and
first attempts only (no repair), with timeout-triggered batch termination.

The initial real baseline at context 4096 was explicitly aborted and excluded:
Ollama logged a 3737-token prompt and a context shift after approximately 355
generated tokens (2045 discarded). Cleanup succeeded without Stopping timeout.
This is an observed context-retention confound, not proof that every earlier
planning failure had the same cause. Both arms of the new comparison use
16384 context tokens without changing the saved profile or model thinking.
Real comparison artifacts: `outputs/plans/few_shot_comparison_ctx16k_v1`.
Do not infer a few-shot improvement until matched runs and semantic review
are complete. No retrieval is enabled.
Automated regression: 73 tests passed (optional pytest cache disabled).
The batch has ended under its timeout guard: 3/8 runs attempted, two completed
Plans and one timeout; five not started. PDAC baseline failed at 698.02 seconds;
few-shot failed at 674.35 seconds. Few-shot added stratification and safeguards
but omitted upstream drug/tractability retrieval and ranking, and invented an
unrequested >0.3 score threshold. KRAS baseline exceeded 1200 seconds; stopping
remained incomplete through the 90-second grace period (total 1290.58 seconds).
The stop guard correctly prevented further requests. No server kill/restart
was performed. The first-attempt label excludes hidden-rubric repairs, but not
the underlying structured-output agent's possible format retries. The server
log showed multiple underlying requests during the KRAS attempt. Only one
matched pair exists, so stability/generalization and speed benefits are not
established. Detailed semantic review is in the comparison folder's REVIEW.md.
Subsequent API verification showed the model had unloaded; this does not change
the recorded cleanup timeout. The terminated batch was not auto-restarted.

### 2026-09-05 model-call diagnostics

Planner reports are upgraded additively to schema 1.1. Every real Planner
attempt now attaches a thread-safe callback recorder and stores each underlying
chat-model call separately: actual input messages, raw response, timing, status,
parent/run IDs and error details. This exposes structured-output retries that
were previously visible only as multiple Ollama server requests. When the outer
deadline elapses, an unfinished call is marked `timed_out`; completed calls are
retained before the checkpoint is written. Injected test agents do not capture
by default, but the behavior can be enabled explicitly. The diagnostic path
does not alter prompts, operation selection, acceptance rules, thinking,
timeouts or retrieval policy. Raw traces remain local and may contain the full
user question. Automated regression: 75 tests passed.

The real no-example KRAS diagnostic reproduced the timeout and identified the
retry chain. Five underlying calls occurred inside one outer attempt: four
completed in 269.66, 190.81, 291.81 and 228.98 seconds; the fifth was timed out
at 309.54 seconds. Calls 1-4 failed because `characteristics`/`generated_at`
were missing, `characteristics.primary` was missing, `primary` was an object
rather than an allowed string, and resolution output names were used as entity
references. No candidate reached semantic acceptance. The confirmed immediate
cause is structured-output correction retries, not hidden-acceptance repair.
The last candidate also omitted explicit mechanism retrieval and invented an
inhibitor-only filter and minimum-three-drugs criterion. Detailed findings are
in `outputs/plans/kras_model_call_diagnostic_v1/DIAGNOSIS.md`. No contract fix
has been applied yet.

The approved minimal contract edit was subsequently implemented: runtime
defaults `generated_at`, and the Planner prompt explicitly defines the
`characteristics` shape and the entity_refs/depends_on/outputs boundary. All 77
automated tests pass. The real KRAS rerun still timed out. It produced three
underlying calls (307.41 s, 371.22 s, then timed out at 612.00 s), versus five
in the preceding diagnostic. `generated_at` and `characteristics` errors were
eliminated, but the model repeatedly used prior output names as `entity_refs`.
Its later candidate added explicit mechanism retrieval yet also used drug-list
outputs as entity refs, misclassified KRAS variants as mechanism entities, and
introduced unsupported completion/extra-scope claims. No Plan reached
hidden acceptance and no retrieval ran. See
`outputs/plans/kras_contract_fix_v1/RESULT.md`. The prompt-only reference fix is
not sufficient; no further schema/normalization change has been made.

Safe reference normalization was then approved and implemented. During
`QueryManifestV2` validation, an unknown `entity_refs` value is removed only
when it is uniquely produced by a directly depended-on prior step; `depends_on`
preserves the data-flow edge and the correction is written to the attempt
report. Arbitrary, indirect or ambiguous unknown refs remain errors. Python
does not insert operations or infer replacement entities. Automated regression:
81 tests passed.

The real no-example KRAS rerun completed in one underlying call (289.63 s),
passed schema validation, and reached hidden semantic acceptance without a
timeout. Four direct output refs were safely removed and recorded. Semantic
acceptance still failed: KRAS variants were typed as mechanisms rather than
targets, three safeguards were absent, and the target-variant filter and
group-by-variant aggregation were missing. All required operations,
dependencies and sources were present. No retrieval ran. See
`outputs/plans/kras_reference_normalization_v1/RESULT.md`. This separates the
resolved serialization blocker from the remaining Plan-quality work.

The controlled few-shot comparison then resumed. Six valid first Plans were
evaluated: two KRAS pairs and one PDAC pair. All used one underlying call and
passed schema validation; none passed semantic acceptance. KRAS few-shot
consistently corrected entity types and two general safeguards, reducing rubric
gaps from 6-7 to 3, but always missed variant-specific safeguard/filter/
grouping. One run added unrequested ranking; the other did not. PDAC few-shot
added stratification/ranking labels but omitted upstream drug and tractability
retrieval, split one bilingual disease mention into two entities, and used
unsupported filters/ranking. Mean KRAS times were 298.29 s off versus 334.37 s
few-shot; the single PDAC pair was 392.43 s off versus 311.87 s few-shot, so no
speed conclusion is supported. The default remains off. Full analysis:
`outputs/plans/FEW_SHOT_EVALUATION_AFTER_CONTRACT_FIX_V1.md`.

## Implemented exact disease-target pair retrieval

Specified disease-target evidence collection now resolves the disease and
target identifiers first and filters Open Targets `associatedTargets` directly
with the verified Ensembl target ID. It no longer searches only the first 100
disease-associated targets, so a valid lower-ranked pair such as pancreatic
cancer and SERPINE1 (PAI-1) is not incorrectly reported as missing merely
because it falls outside that rank window. Exact-pair results do not claim a
rank within the disease target list.

### 2026-09-06 general operation-quality rules

The approved first version of the general operation rule table is implemented
in `operation_rules.py` and runs after `QueryManifestV2` schema validation but
before a Plan can be accepted. It contains five blocking/repair rules and three
warning-only rules. Blocking rules cover missing upstream inputs, incompatible
operation/entity types, analysis steps substituting source declarations for
retrieval, incomplete therapeutic-target evidence stratification, and missing
capability-specific safeguards. Warning rules conservatively flag explicit
aliases split into multiple entities, lost explicit variant comparison groups,
and numeric thresholds not present in the original question. Each finding is
stored in the attempt report and blocking findings become precise repair
feedback. No rule contains an answer to a specific acceptance case, no Plan is
silently research-rewritten, and warnings alone do not block acceptance.

This change did not modify the Planner prompt, enable few-shot examples, call a
real model, or retrieve evidence. Automated regression: 91 tests passed.

The first real-model check after this change used `pdac_known_targets`, the
full operation catalog, no few-shot examples, and Plan-only mode. The model
returned no structured candidate within the 1200-second attempt limit, so
neither schema validation, the new general rules, nor hidden semantic
acceptance could run. One underlying model call was recorded as timed out. The
runner requested `ollama stop`, detected `Stopping`, and recorded that unloading
did not complete within the configured 90-second grace period; total elapsed
time was 1290.90 seconds. A subsequent `ollama ps` check showed no loaded model.
No repair attempt or evidence retrieval ran. Diagnostics:
`outputs/plans/pdac_operation_rules_v1.report.json`.

At the user's request, the same real-model check was repeated with the per-attempt
limit widened from 1200 to 2100 seconds (35 minutes); the `.env` setting now
retains that reversible value. All other conditions remained unchanged. The
single underlying model call again returned no structured candidate and used
the full 2100-second limit. Stop was requested, `Stopping` was detected, and
the 90-second cleanup grace period also elapsed, for 2190.88 seconds total.
A subsequent `ollama ps` check showed no loaded model. Schema validation,
general operation rules, hidden acceptance, repair, and evidence retrieval did
not run because there was no candidate Plan. Diagnostics:
`outputs/plans/pdac_operation_rules_35m_v1.report.json`.

Ollama server-log diagnosis identified the cause. The actual request occupied
3883 of the configured 4096 context tokens before generation. After roughly 203
generated tokens, Ollama began repeated context shifts that discarded 2045
tokens at a time. With no `num_predict` bound, the second run generated about
14,256 tokens at roughly 6.58 tokens/second until the outer deadline. Thus the
model was actively generating rather than computationally frozen, but it was
progressively losing the original instructions and schema. This makes longer
wall-clock limits alone ineffective.

The immediate local fix keeps the full operation table and thinking, overrides
the development context to 8192, and adds a 4096-token `num_predict` limit to
`ChatOllama`. The 2100-second outer timeout remains. Runtime inspection confirms
`qwen3:14b`, `num_ctx=8192`, and `num_predict=4096`. No real-model rerun has yet
been performed after this fix. Automated regression: 94 tests passed.

The first real-model rerun with `num_ctx=8192` and `num_predict=4096` completed
and produced an accepted Plan instead of context-shifting indefinitely. Total
elapsed time was 1547.19 seconds across three outer attempts. Attempt 1 was
schema-valid but failed semantic and general-rule checks. Attempt 2 passed the
hidden semantic rubric but was blocked by general operation rules. Attempt 3
passed both. Underlying model-call counts were 1, 2, and 1 respectively; all
completed without an outer timeout. No evidence retrieval ran. Plan and report:
`outputs/plans/pdac_ctx8192_predict4096_v1.json` and its `.report.json`.

The accepted Plan is not yet judged fully research-safe. It retains an
unrequested `min_evidence_score=0.5`, correctly surfaced by the warning-only
`constraints_require_authority` rule. It also places `tractability_score` as a
secondary ranking metric in the discovery step before tractability evidence is
retrieved; this dependency defect is not currently detected by the rule table.
These are quality gaps in the acceptance/rule specification, not schema or
generation failures.

The reviewed general rule `analysis_metric_requires_available_evidence` is now
implemented as a hard error. It inspects known metrics in filters, ranking and
aggregation, maps each reviewed metric to its required evidence capability, and
requires that capability from the current operation or any transitive upstream
dependency. It does not guess the meaning of unknown custom metrics, avoiding
false blocking until those metrics are reviewed. Re-evaluating the accepted
PDAC Plan now correctly blocks `discover_targets_per_disease` because it uses
`tractability_score` before `retrieve_tractability`; the independent 0.5
authority finding remains warning-only. The rule table now contains six
blocking/repair rules and three warnings. Automated regression: 97 tests passed.

The next real-model PDAC rerun completed in 2242.70 seconds across three outer
attempts and seven underlying model calls. The final candidate passed the hidden
rubric and all current general rules. It removed the unsupported 0.5 threshold
and uses `tractability_score` only in the final ranking after the upstream
tractability retrieval, so the newly protected dependency is correct. No
context-shift loop, outer timeout, or evidence retrieval occurred. Artifacts:
`outputs/plans/pdac_metric_dependency_rule_v1.json` and its `.report.json`.

Manual review found a separate general integrity gap. The model declared
`pdac_associated_targets` as a top-level target entity even though it is a
generated collection/output name, not an entity explicitly mentioned by the
user. Downstream steps then used it in `entity_refs` rather than relying solely
on `depends_on`. Because the fabricated reference was predeclared, the existing
unknown-reference validator did not reject it. The Plan is mechanically
accepted but is not yet judged fully correct; a grounded-entity rule is still
needed.

The hard rule `declared_entities_must_be_question_grounded` is now implemented.
It accepts literal mentions, abbreviations and parenthetical aliases present in
the question after conservative text normalization, while rejecting entities
whose mention is absent or whose reference/mention duplicates a step output.
The prior `pdac_associated_targets` Plan is correctly blocked. The rule table
now has seven blocking/repair rules and three warnings. Automated regression:
100 tests passed.

The subsequent real PDAC rerun ended after 1722.36 seconds and three outer
attempts without an accepted Plan. All candidates were schema-valid, and none
recreated the fabricated collection entity; the third candidate contained only
the explicitly mentioned PDAC disease and passed all current general rules.
Hidden semantic acceptance still failed because the model inserted
`aggregate_category_distribution` between stratification and ranking without
creating the required target-translation-layer aggregation, and ranking was not
directly dependent on stratification. Manual review found this was not merely a
strict direct-edge false negative: the category-distribution operation did not
substitute for evidence-layer stratification, and the Plan also ranked by an
unretrieved `drug_approval_status`. The unsupported 0.5 threshold remained as a
warning. No evidence retrieval or timeout occurred. Diagnostics:
`outputs/plans/pdac_grounded_entities_rule_v1.report.json`.

The operation rules were extended after reviewing that failed candidate. The
existing metric-evidence rule now maps `drug_approval_status`, `approval_status`
and `approved_indication_count` to `approval_evidence`, provided by
`retrieve_approved_indications`. A new hard rule,
`operation_must_honor_capability_contract`, requires reviewed transformation
operations to provide a non-empty aggregation definition and prevents one
operation from claiming a reserved output belonging to another. It deliberately
does not require a case-specific aggregation key: for example,
`stratify_target_evidence_layers` may use any semantically clear key, while
`aggregate_category_distribution` may not claim the reviewed
`target_translation_layer` output as its own.

Offline replay of the real failed Plan now produces three separate blocking
findings: missing aggregation structure on stratification,
category aggregation impersonating that output, and ranking by
`drug_approval_status` without approval evidence. The independent unsupported
0.5 threshold remains a warning. The rule table now has eight blocking/repair
rules and three warnings. Automated regression: 105 tests passed.

The real-model PDAC rerun after generalizing the capability contract finished
without an accepted Plan after 3542.51 seconds. Attempt 1 was schema-valid but
failed both hidden acceptance and general rules. Attempt 2 was schema-valid and
passed the complete hidden acceptance rubric, but the general rules correctly
blocked fabricated collection entities, sources attached to analysis steps, and
a stratification step without an aggregation definition. Attempt 3 received
the repair feedback but exceeded the 2100-second per-attempt limit before
returning structured output. The timeout path requested `ollama stop`, detected
`Stopping`, and stopped waiting after the 90-second cleanup grace period. A
subsequent `ollama ps` showed no loaded model, so there is no persistent stuck
process. No evidence retrieval ran. Diagnostics:
`outputs/plans/pdac_generalized_capability_rules_v1.report.json`.

The repair flow now defaults to bounded local patches after the first
schema-valid candidate. The repair model returns only entity removals/upserts,
step removals/upserts, an optional complete final step order, or explicit
top-level field replacements. Python merges the patch atomically into the prior
Plan and reruns the full schema, operation rules, and hidden acceptance; a patch
cannot bypass any validator. If no schema-valid candidate exists, full Plan
generation remains necessary. `PLANNER_REPAIR_MODE=full_rewrite` restores the
former behavior for reversible comparison. Diagnostics now record the run's
repair mode and whether each attempt generated a full Plan or local patch.
The patch flow was subsequently hardened before real-model testing. Python now
derives a per-attempt repair scope from hidden-acceptance and operation-rule
findings, and rejects changes to unflagged entities, steps, and top-level fields.
Only a candidate that preserves all previously accepted capabilities and
strictly reduces the blocking-issue count is promoted as the next repair base;
non-improving candidates remain visible in diagnostics but repair rolls back to
the better Plan. Patch/schema failures are accumulated with the base Plan's
unresolved feedback instead of replacing it. Patch prompts use a relevant
operation-table subset consisting of operations already in the Plan plus those
explicitly named by diagnostics; first-pass planning still uses the configured
full catalog. Diagnostics record the authorized scope and whether a candidate
was promoted. Automated regression: 113 tests passed, including adding all
missing operations through a scoped patch and completing the hidden acceptance
case. No real-model run has yet been performed with the hardened patch path.

The first real-model run of the hardened patch flow used the PDAC known-targets
acceptance case and ended after 2392.62 seconds without a final Plan. Attempt 1
returned a schema-valid full Plan in 239.05 seconds. Attempt 2 correctly entered
`local_patch`, but the structured-output middleware made 24 underlying calls
and hit the 2100-second limit. Cleanup requested an Ollama stop, detected
`Stopping`, completed within the grace period, and left no model loaded. No
evidence retrieval ran.

These were not 24 independent research repairs. The model produced a substantial
patch on its first call, but copied the parent Plan's `schema_version=2.0` while
`QueryManifestPatch` required `1.0`, and omitted the patch's redundant required
`original_question`. The middleware repeatedly tried to repair those same two
validation errors instead of returning control to the outer loop. Offline
normalization of only those metadata fields showed that the first patch passed
the complete hidden semantic rubric. It still had two genuine blocking issues:
the original discovery step retained premature `tractability_score`, and
`missing_not_negative` remained absent. The patch content therefore improved
substantially; the immediate blocker is the patch response contract and
unbounded internal structured-output retries, not the repair-scope guard.
Diagnostics: `outputs/plans/pdac_local_patch_repair_v1.report.json`.

The confirmed patch-contract failure is now fixed. `QueryManifestPatch` no
longer asks the model to reproduce `schema_version` or `original_question` (and
never exposes parent `manifest_id` or `generated_at`); Python already preserves
all parent-Plan metadata. The patch agent now uses a LangChain `ToolStrategy`
with `handle_errors=False`, so a malformed structured response returns to the
bounded outer repair loop after one model call instead of triggering unbounded
internal retries. Offline replay proves the first raw patch from the failed real
run now parses unchanged, remains within its authorized scope, and passes the
complete hidden acceptance rubric. Its two remaining blocking rule findings are
preserved for the next outer patch: premature `tractability_score` on discovery
and the missing `missing_not_negative` safeguard. Automated regression: 115
tests passed. A new real-model run has not yet been performed after this fix.

The next PDAC real-model run tested the metadata-free patch contract and
`handle_errors=False`. It still ended without a final Plan, but completed all
three outer attempts in 1919.43 seconds rather than waiting for the total
deadline. Attempt 1 produced a schema-valid full Plan in 286.95 seconds.
Attempts 2 and 3 were local patches lasting 728.85 and 903.62 seconds. No
attempt timed out, stopping cleanup was unnecessary, and no retrieval ran.

The metadata failure is resolved, but the one-call expectation is not: the two
patch attempts made five and six underlying calls. Inspection shows these were
not Pydantic validation-error retries controlled by `ToolStrategy.handle_errors`.
Qwen repeatedly returned ordinary JSON/thinking text without completing the
artificial structured-output tool call, so the general agent execution loop
continued. Both outer patches then failed whole-Plan validation because
generated upstream collections such as `pdac_targets` were placed in downstream
`entity_refs`. This is semantically a data-flow reference, not a top-level
entity. Offline removal of non-declared output references from attempt 3 made it
pass the complete hidden acceptance rubric; the only remaining blocking rule
was the original discovery step's premature `tractability_score`, with the
unsupported 0.5 threshold remaining warning-only. The next architectural fixes
should therefore use a direct one-shot structured model invocation for patch
generation (not an agent loop) and generalize safe output-reference cleanup from
direct dependencies to transitive upstream dependencies. Diagnostics:
`outputs/plans/pdac_patch_no_internal_retry_v1.report.json`.

The direct patch architecture and transitive output-reference normalization are
now implemented and validated by a successful real-model PDAC run. Patch
generation bypasses the agent executor: one direct chat-model call returns text,
Python extracts one JSON object after any thinking content, and validates it
locally. Each of the three outer attempts made exactly one underlying model
call. Attempt 1 produced a schema-valid full Plan in 290.30 seconds but failed
hidden acceptance and safeguards. Attempt 2 produced a local patch in 621.61
seconds, passed hidden acceptance, and was promoted; general rules still blocked
sources on the stratification analysis step and missing `missing_not_negative`.
Attempt 3 produced one local patch in 163.26 seconds and passed schema, hidden
acceptance, and all blocking general rules. Total duration was 1075.18 seconds
(about 17.9 minutes), with no timeout, cleanup, or evidence retrieval.

Safe reference normalization removed only undeclared names proven to be unique
outputs of direct or transitive upstream steps; arbitrary, downstream,
unrelated, and ambiguous references remain errors. Manual review found two
items not currently blocked: discovery uses `association_type=therapeutic`,
whose datasource support and research meaning need review, and final ranking
uses the unregistered custom metric `drug_evidence_count`. These should be
assessed for generality before adding rules. Automated regression: 117 tests
passed. Artifacts: `outputs/plans/pdac_direct_patch_transitive_refs_v1.json` and
its `.report.json`.

Operation semantics have now been consolidated into the authoritative operation
table. Each reviewed operation can declare its kind, allowed entity types,
required and provided upstream capabilities, required safeguards, supported
filters, aggregation requirement, and reserved outputs. General validation reads
these contracts rather than maintaining separate operation-name registries. The
previous case-specific stratification rule was therefore removed, while exact
pair retrieval now explicitly requires both resolved disease and resolved target.
Unsupported filter keys and unregistered ranking metrics are blocked for review,
covering the previously observed `association_type=therapeutic` and
`drug_evidence_count` issues without adding one rule per symptom. The Planner's
full catalog prompt exposes the same compact contracts. Automated regression:
119 tests passed (one harmless pytest cache permission warning).

The subsequent real-model PDAC run did not produce a final Plan. Its single
outer attempt reached the 2100-second limit after 2174.52 seconds; stopping was
detected and cleanup succeeded, and no evidence retrieval ran. The operation
contracts were not the direct validation failure. Within that one outer attempt,
the structured-output middleware completed three model calls (269.81, 821.15,
and 625.50 seconds). Each candidate used the prior step's output name
`resolved_disease_1` in `depends_on`; the schema requires the prior step ID
`resolve_disease_1`. Pydantic rejected the same non-prior dependency three times,
so the attempt exhausted its deadline. This demonstrates that the widened time
limit and stopping cleanup work, but the full-Plan path still permits costly
internal structured-output retries and the prompt does not make the distinction
between step IDs and output names explicit enough. Diagnostic artifact:
`outputs/plans/pdac_operation_contract_table_v1.report.json`.

The repeated dependency failure is now fixed at both the prompt and execution
layers. The prompt explicitly distinguishes `step_id`, `depends_on`, and
`outputs`, with a concrete correct/incorrect example. Full-Plan generation no
longer uses the agent structured-output retry loop; like patch generation, it
makes one direct model call, extracts one JSON object, and lets Python validate
it. Python also performs one narrowly safe normalization: an output name used
in `depends_on` is replaced by its producer `step_id` only when that producer is
unique and appears earlier. Unknown, ambiguous, and future references are still
rejected. Automated regression: 121 tests passed (one harmless pytest cache
permission warning). The next real-model run has not yet been started.

A formal offline Planner regression catalog has been added to accelerate rule
development without additional model calls. Its first version contains eight
minimal historical failures: missing resolution dependency, datasource attached
to analysis, missing-is-unknown safeguard omission, premature tractability
ranking, unsupported `association_type` filter, unregistered
`drug_evidence_count`, generated collection declared as an entity, and entity
type/resolution mismatch. Four accepted alternatives guard against overfitting:
the complete reference Plan, evidence stratification without optional ranking,
an equivalent aggregation shape, and ranking with genuinely available upstream
metrics. Every case records the responsible layer so case-specific scientific
expectations are not promoted into global rules. The catalog's 13 checks passed;
the complete suite now passes 134 tests in 1.60 seconds, with only the existing
harmless pytest cache permission warning. No LLM or evidence retrieval was used.

The first representative real-model test of the phased rollout used
`pdac_known_targets`. It failed after three outer attempts and 1673.51 seconds,
but confirmed the new bounded full-Plan architecture: attempts 1-3 each made
exactly one model call, with no hidden structured-output retries. Attempt 1
produced a schema-valid and repairable Plan in 608.42 seconds; all required
operation semantics were present, while stratification lacked target-drug input
and aggregation, analysis steps carried sources, and ranking used an unregistered
metric. Attempt 2 returned an invalid patch because its `step_order` named final
steps it had not actually added. Attempt 3 added target-drug retrieval and fixed
stratification, leaving only the rank step's datasource and unregistered
`disease_association_strength` metric. No evidence retrieval ran. The existing
offline suite already covers incomplete patch `step_order`, so no new global
rule was added. Tests 2 and 3 of the representative batch were intentionally not
started to avoid repeating the unresolved repair-allocation problem. Artifacts:
`outputs/plans/phase2_pdac_known_targets_v1.report.json`; no final Plan was
written.

Repair allocation has now been revised based on that representative failure.
Python sends one coherent feedback batch at a time, prioritizing upstream
structure/dependencies, then filtering, then ranking, and narrows patch authority
to the selected batch. A stale incomplete `step_order` no longer discards a
patch that only updates existing steps: the base order is preserved. It remains
a hard error when topology actually changes, or when no step edit accompanies
the incomplete order. Because three repair groups can follow the initial Plan,
the `plan-query` CLI now defaults to four total attempts, accepts one through
five via `--max-attempts`, and retains the existing total wall-clock deadline.
The core planner API keeps its three-attempt default for compatibility unless a
caller explicitly requests more. Automated regression: 137 tests passed in
1.51 seconds with only the existing harmless cache warning. No new scientific
rule was added and no further real-model test has yet run.

## Flow review after resumed v7 (implementation only; not tested)

The later v7 report failed after 1272.38 seconds. Its final patch declared the
requested stratification-to-ranking dependency, but the retained list order put
ranking before stratification. This was an ordering compilation problem rather
than evidence of an incorrect dependency direction. Earlier v4 passed the then
current evaluator but contained a placeholder aggregation; do not treat that
artifact as research-quality approval. v5 and v6 also failed; their reports
preserve the history. The most recent previous test count was 146, before the
following changes, and is not validation of these changes.

At the user's explicit request, the current changes were inspected as source
only. No tests, model invocations, or retrieval were executed in this review.
Patch merge now performs stable topological ordering of declared dependencies,
rejecting unknown references and cycles without inventing edges. Partial-step
wire schema now matches omitted-field preservation; changed objects/lists replace
their field in full. Metric names and evidence prerequisites are exposed from
the same registry used by validation. Operation matching uses exact tokens;
dependency repair targets the downstream operation and does not authorize new
copies of already-present operations. Normalization records are retained.

Resumed reports now store the source and seed Plan, count actual Patch calls,
and do not claim blind first-attempt success. CLI resumption refuses to overwrite
its source report and can retain a seed when no resumed candidate was promoted.
The empty-method check no longer rejects an otherwise specified method merely
because an optional nested list is empty. This check still does not prove the
scientific adequacy of a nonempty aggregation. Research-quality review, full
integration verification, and broader acceptance cases remain outstanding.

## Dependency-only repair prompt revision (automated tests complete)

The previously untested flow changes now pass the full automated suite: 146
pre-existing tests passed. A resumed real-model v8 run then failed safely after
two local Patch attempts and 953.12 seconds. Both attempts edited
`stratify_target_evidence_layers_1` even though the authorized downstream edit
target was `rank_targets_1`, and both tried the reverse dependency direction.
The scope guard rejected both patches; no final Plan or evidence retrieval was
produced. The run did not time out or become stuck in stopping. Diagnostic:
`outputs/plans/phase2_pdac_known_targets_resumed_v8.report.json`.

Dependency-only repair prompting has now been narrowed without adding a
scientific rule. Python resolves one missing operation edge to exact upstream
and downstream step IDs, supplies the current and required final dependency
lists, names forbidden reverse edits, and sends only the dependency graph plus
the editable step. Operation and metric catalogs are omitted when they cannot
help that repair. A retry explicitly names any step ID rejected in the prior
attempt. Two identical invalid local-Patch failures now stop early and are
recorded as `repeated_repair_failure` instead of consuming all remaining calls.
Four regression tests cover direction resolution, reduced context, adaptive
retry feedback, and repeated-failure termination. The complete suite now passes
150 tests in 1.73 seconds, with only the existing harmless pytest cache warning.
The revised dependency prompt has not yet been tested with the real model.




## Execution compiler: representative plans compile and run (2026-10-29)

`compile_plan` turns an approved Query Manifest into an ordered list of tool
calls without running anything, and `python -m execution PLAN.json --dry-run`
prints that list. Plans are refused before retrieval when an operation has no
adapter, a filter or ranking field is not produced upstream, or a step needs data
that no earlier step provides. Three representative plans compile in tests
(known targets, target filtering, drug targets with approved-indication check).
A live run of `phase2_pdac_known_targets_batched_v4.json` completed all six
steps in 19.2 seconds and produced a ten-target ranking.

Coverage is partial: adapters exist for 37 of 64 operations, and 9 of 18
acceptance cases have every required operation implemented. Of the remaining 27
operations, 16 need only Open Targets data (variant resolution, cross-indication
drugs and cancers, datasource and evidence-type comparison, safety and
normal-tissue expression, kinase classification and coverage rates). The other
11 depend on trial or publication data (ClinicalTrials.gov, Europe PMC), which
has no query module yet.
Added since the last update: `retrieve_associated_diseases`, `rank_diseases`,
`compare_disease_rank` (generic replacement for the PDAC-specific
`compare_pdac_rank`), `retrieve_genetic_evidence`,
`retrieve_genetic_and_somatic_evidence`, `filter_supported_without_clinical_drug`,
`filter_first_in_class_candidates` and `rank_opportunities`. Genetic and somatic
evidence use the Open Targets `genetic_association` and `somatic_mutation`
datatype scores. "Strong" evidence has no project-wide threshold, so the filter
requires one in the plan. "No drug" means no drug record in Open Targets known
drugs on the evaluation date, and a failed drug lookup is reported as
unevaluable, not as no drug. The Planner does not yet emit `inputs` references.
The full suite passes N tests with no failures. The target-translation layer
rule is still provisional and awaits research-team approval.