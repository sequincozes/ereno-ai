# Feedback policy and multi-round campaigns (E10)

## Goal

The Defender agent acts as the DEFENDER stage of the intent-driven pipeline; the
FEEDBACK stage comes right after and closes the Red×Blue loop. It receives the
`DetectionReport` from the DETECTOR stage and the `DefensePlan` from the DEFENDER
stage of this round and decides, deterministically: whether the campaign continues and, if
so, what the next `IntentSpec` is. No LLM takes part in this stage —
same central guardrail of the pipeline: the model only produces the intent or the
defense plan, everything downstream is deterministic Python
(`core/feedback_policy.py::decide_feedback`).

`IntentLoopOrchestrator.run(prompt)` still resolves **one** round.
`IntentLoopOrchestrator.run_campaign(prompt, rounds=N)` chains up to `N`
rounds: from the second on, the `IntentSpec` no longer comes from a new
interpretation of the prompt — it comes from the previous round's
`FeedbackDecision.next_intent`.

## Output contract

`domain/feedback_decision.py::FeedbackDecision` (`extra="forbid"`,
`frozen=True`):

- `should_continue` / `stop_reason` — always agree with each other (validator).
- `objective_metric` (`"f1"` or `"recall"`, or `None`) and `objective_value` —
  what was measured in this round.
- `best_value_so_far` / `best_round` — the best value of the campaign so far.
- `improvement` — `previous_best_value − this_round_value` (positive =
  improved).
- `round` / `run_id` / `parent_run_id` — the lineage of this decision.
- `defense_priority` — the priority of the round's `DefensePlan` (it comes from the
  plan, not recalculated: the E5 gate already guarantees it agrees with
  `priority_from_report`).
- `next_intent` — present if and only if `should_continue`.
- `rationale` — deterministic sentence with the numbers that justify the
  decision.

Persisted as `feedback.json` in the round's directory, by the same
`_stage(...)` that persists the other six stages — no special I/O.

## Objective metric per effect

| `desired_effect` | metric minimized |
|---|---|
| `lower_f1` | `f1` |
| `lower_recall` | `recall` |
| `mimic_normal_traffic` | `recall` |
| `increase_attack_activity` | none |
| `increase_resource_pressure` | none |

The last two effects have no evasion metric in the `DetectionReport`
to minimize — inventing one would be pretending to measure. The policy stops in the
first round with `no_objective_metric`.

## Escalation ladder

Only two levers actually change the variant compiled by
`core/intent_compiler.py` (see the `feedback_policy.py` module for the full
analysis), so the ladder uses exactly those, "deepen before
widening":

1. `intensity`: `low` → `medium` → `high`.
2. `restrictions.max_fields_changed`: `+1` per round, up to the number of
   candidate fields of the effect **on that attack** — it reuses
   `resolve_candidate_paths` to compute the ceiling, the same gate the compiler
   and the `IntentAgent` use. It varies per catalog: 12 in
   `masquerade_fault`/`lower_recall`, 15 in `random_replay`, only 2 in
   `injection` (see `docs/attack_capabilities.md`). The contract's ceiling
   (`domain.intent_spec.MAX_FIELDS_CHANGED_CEILING`, today 16) is only a
   sanity lock above that number — never the number itself.

Dictated values (`restrictions.target_values`) stay outside the intensity
ladder, but consume the quota: the compiler only gives the intensity
`max_fields_changed - len(target_values)` fields. When that remainder is zero, step
1 is skipped and the ladder goes straight to 2, because raising the intensity would
compile the same variant. Before this rule, intent examples 1a and 1b
(2026-11-03, `docs/intent_examples.md`) measured in round 2 the same trace as
round 1, byte for byte, and stopped at `no_improvement`. The LLM tends to set
`max_fields_changed` equal to the number of dictated fields, so that case is
the common one when there are dictated values, not an exception.

The `seed` is not a lever: `intent_compiler._select_field_paths` shuffles the
candidates with `random.Random(intent.seed)` and cuts at `[:limit]`, so
keeping the seed fixed makes `limit+1` a superset of the previous set
— the escalation stays monotonic ("the same fields, plus one") and the whole
campaign reproducible.

`desired_effect` is not a lever either: `lower_f1`, `lower_recall` and
`mimic_normal_traffic` share the same candidate fields and the same
direction in the compiler — the policy never rewrites `objective`,
`base_attack` or `desired_effect`.

Every derived intent is revalidated with `IntentSpec.model_validate(...)`
(not `model_copy(update=...)`, which in Pydantic v2 does not re-run validators)
and goes through `validate_intent_capability` before becoming `next_intent` —
defense in depth: the policy never returns an intent that the
compiler would reject.

## Stop conditions

First one that matches, in this order:

1. `no_objective_metric` — effect with no evasion metric.
2. `objective_reached` — `target_metric` given and reached.
3. `no_improvement` — plateau: `gain < min_delta` (compared to the **best**
   value already measured in the campaign, not to that of the immediately preceding round,
   because each compile starts from the baseline).
4. `levers_exhausted` — scientific reason ("there is nothing left to scale")
   comes before the round budget.
5. `max_rounds_reached` — round ceiling.
6. `continue_campaign` — proceeds with `next_intent`.

## Campaign lineage

`LoopRecord` gained two additive fields (E10): `round` (default `1`) and
`parent_run_id` (default `None`), with the invariant "only round 1 has no
parent". `schema_version` stays `1`: both fields are additive with a
default, so a record written before E10 loads normally
(`tests/test_loop_record_store.py::test_records_persisted_before_the_e10_lineage_fields_still_load`)
and the on-disk format (`{"loop_records": [...]}`) does not change. A campaign of
N rounds is N rows in `outputs/loop_records.json`, chained by
`parent_run_id` — not a new campaign contract.

## INTENT stage in rounds 2+

When `run_campaign` supplies `intent_override`, the `intent` stage does not
call the `IntentLike` — it persists the same `intent.json` and is recorded as
**`skipped`**, never `succeeded`: the status must keep telling the
truth about what ran. `succeeded` would read as "the LLM ran again", which
would erase exactly the property E10 delivers (the Red side is
deterministic Python from round 2 on).

## Safety and scope

- No second LLM call per campaign round — the `IntentAgent`
  only runs in round 1.
- The policy applies nothing: `next_intent` is a proposal that still goes
  through the compiler and the normal gates in the next round.
- The `DefensePlan` still has no execution field — E10 does not change that.

## Stage failure

A failure in FEEDBACK (or in any earlier stage of the round) marks the
corresponding `LoopStage` as `failed` and interrupts **the round** —
`run_campaign` then interrupts **the campaign**, without raising an exception: the
records already completed (each persisted individually) are
returned, and the cause stays in the `LoopStage` that failed.

## Campaign resume

Acceptance criterion of the Loop layer: *"resume does not duplicate records"*.

```bash
uv run adversarial-ids --engine intent --resume <RUN_ID>             # any round of the campaign
uv run adversarial-ids --engine intent --resume <RUN_ID> --rounds 5  # overrides the recorded ceiling
```

A campaign stops without having finished in three ways, and the resume
(`IntentLoopOrchestrator.resume_campaign`, resolved by
`agents/orchestrator/campaign_resume.py::plan_resume`) handles each one from
the ledger:

| the campaign's tip | action | what runs |
|---|---|---|
| has a `failed` stage (rate limit, JAR timeout, volume gate) | `retry` | the same round again, with a new `run_id` and `retry_of` pointing to the one that failed |
| finished and the policy said continue (token budget, process died in the middle of the next round) | `continue` | the next round, with the `next_intent` from `feedback.json` |
| finished and the policy ended it (`should_continue` false) | `done` | nothing — returns `()` and the ledger does not change |

It is the `done` case that makes the resume not duplicate: resuming a finished campaign,
or resuming the same one twice, writes nothing. The resume never undoes a
verdict of the policy, and `--rounds` does not undo it either: a round that stopped by
`max_rounds_reached` has no `next_intent`, so there is nowhere to continue from.

Some rules:

- **The tip comes from the ledger.** Any `run_id` of the campaign works. Resuming
  from a round that already has a child would create a second branch of the same
  campaign, which is precisely the forbidden duplicate. The campaign is the
  connected component by `parent_run_id` and `retry_of`, and the tip is the round with the highest
  `round`, with the most recent one in the ledger breaking ties.
- **The round that failed stays in the ledger.** The ledger is append-only and it is
  the truth about the first attempt. The new attempt has the same `round` and the
  same parent, and it is `retry_of` that distinguishes it from a duplicate.
- **The validated intent is reused.** Redoing a round that fell at the
  DEFENDER does not call the `IntentLike` again: the `intent.json` of the round that failed
  is persisted and the stage becomes `skipped`, with the origin in the message. Only
  a round 1 that failed at INTENT itself calls the LLM again. With the account's TPM
  (~6.9k tokens per call against 8k, see `docs/pilot_e0.md`), that decides whether the
  retry fits in the minute.
- **The `history` is rebuilt.** The `objective_value` of each round on the path
  are read from the `feedback.json` files. A campaign interrupted and resumed makes the
  same decisions as the same campaign without interruption
  (`test_a_resumed_campaign_decides_exactly_like_an_uninterrupted_one`).
- **The budget counts only the resume call.** Resuming a campaign cut off by the
  ceiling is the explicit decision to spend more. Adding the previous spend to it
  would cut it off again right after the first round.
- **The detector is the campaign's.** Without `--detector`, the CLI uses the one in the
  campaign's most recent `detector_manifest.json`. A different detector is
  refused (`CampaignResumeError`), because it would join two experiments in the same
  ledger. The `generator_mode` is not recorded anywhere, so that
  check does not cover it: resume with the same mode.
- **Caller errors raise, stage failures do not.** Unknown `run_id`,
  orchestrator without a ledger, deleted artifact (`outputs/` cleaned by
  `scripts/init_state.py`) or a record that predates `max_rounds` without an explicit `--rounds`
  raise `CampaignResumeError`, and the CLI exits with code 2. After
  resuming, the campaign follows the rules of `run_campaign`.
- **Resume point before the agents.** The CLI resolves the resume point
  before building the agents, so discovering that there is nothing to resume does not
  require `GROQ_API_KEY`.

The ledger gained two additive fields with a default, `max_rounds` and `retry_of`, in the
same pattern as `round`/`parent_run_id`; `schema_version` stays `1`. In addition,
`append_loop_record` now refuses a `run_id` that is already in the ledger.
A round that the process lost midway leaves a directory in
`outputs/intent_loop/<run_id>/` (with `events.jsonl`) with no row in the ledger. That
directory is an orphan and the resume ignores it.

## Cached mode limitation

In `generator_mode="cached"` the baseline dataset and the compiled candidate's
are the same file — the objective metric does not move between
rounds, so a cached campaign stops at round 2 with
`no_improvement` (unless `feedback_min_delta=0.0`, used in tests
to exercise more rounds without depending on real measurement). Use
`generator_mode="jar"` for a physically measurable campaign.

## Out of scope and future work

- **Feature-driven lever**: mapping `DetectionReport.top_features`
  to `allowed_fields`/`forbidden_fields` would let the escalation aim at (or
  avoid) the fields tied to the features that weigh most in the
  detector's decision. There is today no dataset-feature → config-field
  mapping in the repository; adding it is the natural extension of this epic.
- `LoopStage.error` remains the only free-text field, carrying
  both failure messages and the "inherited intent" message — renaming
  it to something like `detail` is a contract break with no functional gain by
  itself.
- **Resource pressure metric**: `increase_resource_pressure` has no
  entry in `_OBJECTIVE_METRIC_BY_EFFECT` and no `AttackCapability` announces it
  (see `docs/attack_capabilities.md`) — `DetectionReport` carries
  nothing that measures resource pressure without inventing an arbitrary measurement
  (`latency_ms` measures the local Python process, not the GOOSE bus). An honest
  metric would come from `DatasetBundle` (volume/rate of attack rows),
  not from `DetectionReport` — it would require a second source in the signature of
  `decide_feedback`, a contract change outside the scope of this epic.

## Tests

- `tests/test_new_contracts.py` — `FeedbackDecision` (versioning,
  freezing, JSON round-trip, consistency validators) and the
  `LoopRecord` lineage (`round`/`parent_run_id`), alongside the other
  frozen contracts.
- `tests/test_feedback_policy.py` — objective metric per effect; intensity→width
  ladder; ceiling from the candidate-field allowlist;
  determinism (same input → same decision); the escalated intent still
  compiles; precedence of the five stop reasons, including the cases in
  which two reasons hold at the same time.
- `tests/test_intent_loop_orchestrator.py` — the seven full stages
  in a single round; `feedback.json` persisted; `run_campaign` stopping at the
  plateau in cached mode; running up to the ceiling when epsilon is zero;
  calling the `IntentLike` only once per campaign; aborting when a
  stage fails; escalating the intensity of round 2; failure of the
  FEEDBACK stage itself.
- `tests/test_cli_intent_engine.py` — `--rounds` forwarded, rejection of
  non-positive `--rounds`, `--iterations` (from the legacy loop) ignored in the
  intent engine, per-round summary, non-zero exit when any round
  fails.
- `tests/test_campaign_resume.py` — resume: retry of a round that failed at
  DEFENDER (without a new LLM call) and at INTENT (with one), retry of a middle round,
  continuation after the token ceiling and after a dead process, decisions
  identical to those of the campaign without interruption, and the cases that write nothing
  (finished campaign, resuming from the middle, resuming twice); caller errors
  and the CLI's `--resume` flag.
- `tests/test_loop_record_store.py` — a `LoopRecord` written before E10
  (without `round`/`parent_run_id`) still loads with the defaults.
