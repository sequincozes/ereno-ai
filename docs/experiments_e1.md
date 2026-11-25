# E1 — Search and feedback (RQ1–RQ2): protocol

Fixed on 2026-10-14, **before** the runs. Changing any item below after
seeing a test result invalidates the comparison; changes made after the
pilot (which does not enter the results) are recorded in the "Pilot" section.

## Question

- **RQ1:** with the same proposal budget, does an adaptive injector (LLM) find
  a larger valid evasion than the reference configuration, random search and
  the deterministic policy?
- **RQ2:** does giving the injector the Defender's plan, in addition to the
  metrics, improve the search?

## Conditions

| Id | Strategy | Who proposes the next candidate | LLM |
|----|----------|---------------------------------|-----|
| B0 | Reference | nobody — the family baseline, no search | no |
| B1 | Random search | uniform draw in the search box (below) | no |
| B2 | Open-loop deterministic control | the ladder of `core/feedback_policy.py` (intensity low→medium→high, then +1 field), without the plateau rule | no |
| B3 | Attacker with metrics | LLM sees the family catalog, the box, the candidate history and the search metrics | yes |
| B4 | Attacker with full feedback | B3 + the Defender's last validated `DefensePlan` | yes (2 calls/round) |

B0 is the starting point, not a competitor with the same number of queries.
Without the plateau rule B2 **looks at no metric at all**: it is a fixed
schedule of levers — open-loop deterministic control, not a "policy with
feedback". B2 runs **without** the policy's plateau rule (`no_improvement`): it saves
rounds in the production loop, but here it would cut B2's budget and the
comparison would stop being at equal budget. An exhausted ladder ends the
campaign earlier (which is recorded); the comparison uses the best result up
to each budget consumed.

B3 and B4 use the **same** model, temperature, base prompt, tool
(`submit_intent_spec`) and gates; the only difference is the defense-plan block.
Temperature: **0.7** for both (it is what gives variation between repetitions).
The LLM still produces only the intent — compilation, box and validity are
deterministic, as in the pipeline.

## Fixed

- **Families:** `masquerade_fault`, `random_replay`, `inverse_replay`,
  `high_stnum` (post-E0 `.v2` catalogs). `injection` was dropped on 2026-10-15,
  after the references and before any campaign: evasion ceiling (see
  "References"). If the pilot shows that the Groq quota cannot hold all
  four, we cut to three families —
  never to fewer than 5 repetitions, which is the minimum for the per-campaign
  bootstrap to say anything. `grayhole` stays out
  of E1 (borderline case).
- **Detector:** Random Forest, `protocol_features="deltas"`, E6 preprocessing,
  no selection or undersampling. Trained once per family on the training
  seeds with the reference config and **frozen** (`experiments/protocol.py::FrozenIds`).
- **Seeds:** training 101–103, calibration 201–202, search 301–303, test 401–405
  (`SeedPlan`). Three search seeds, not two: the search objective is a mean
  per seed, and with two the noise between seeds competes with the difference
  between candidates. No strategy sees a test number.
- **Alarm budgets:** 1, 10 and 100 false alarms per 10 thousand normal
  messages, calibrated on the normal traffic of the calibration seeds. **Primary point: 10.**
- **Search budget:** 10 proposals per campaign. An invalid proposal
  (does not compile, leaves the box, generation fails, volume floor, attack
  undetectable by the oracle) **counts** against the budget.
- **Repetitions:** 5 per (family, strategy). Repetition `r` uses `seed=r` in
  B1's draw and in B2's `IntentSpec` (which shuffles the field order); in
  B3/B4 the variation comes from the LLM sampling.

## Search box

The same space for all strategies:

- the editable fields of the family catalog (all of them, not just the evasion ones);
- the catalog bounds; a side with no bound in the catalog is limited to
  a **physical bound** (approved on 2026-10-15; scales measured on the benign
  traffic of training seeds 101–103, `scripts/run_e1.py::PHYSICAL_BOUNDS`). It only affects
  `masquerade_fault`:
  - `fault.durationMs` ≤ **1000 ms**. In the generator the field is not the
    fault duration: it is a delay added to the copy's timestamp. The publisher's
    heartbeat reaches 1.031 s; a larger delay would make the copy cross the
    next legitimate message (reordering, another attack);
  - `analog.deltaAbs` ≤ **10**: the largest |current| in the benign traffic (10.31), from the
    simulated real faults. The generator applies the same deviation to current
    and voltage (|v| ≤ 0.14), a coupling the box does not separate — a limitation;
  - `trapArea.multiplier` ≤ **2.5**, the reference maximum. Real faults
    move the trapezoid sums 0.71–1.41× the median; the reference already
    exceeds that, and the box does not stretch beyond it.

  Without a physical bound, an open side would be 4 × the reference (of the pair's `max`);
- at most **4** fields changed per candidate;
- `{min,max}` pairs with strict `min < max` (ERENO refuses `min == max`).

A proposal outside the box is invalid (`out_of_box`).

## Validity of a candidate

In order; the first failure gives the reason:

1. it compiles (`compile_attack_candidate`) and fits in the box;
2. it generates on the search seeds;
3. volume floor: ≥ 50 attack rows and prevalence ≥ 1% on each seed
   (the same floor as E4);
4. **the attack remains detectable by an oracle**: an RF with the same
   preprocessing, trained on the candidate's own trace on seed 301 and
   evaluated on 302, needs `AP ≥ p + λ·(1 − p)`, with `p` the attack
   prevalence on 302. **λ = 0.5**, fixed on 2026-10-15 from the references
   (the four E1 families have observed λ ≥ 0.9998; see "References").
   If any reference does not separate from normal even with the
   oracle, the problem is the criterion, not the family — and the criterion is revised
   before any run.

Item 4 is the defense against "evasion by disappearance": a config that erases the
attack (e.g. `fault.prob = 0`) evades any detector without exposing any
blind spot. Evasion of the frozen detector only counts if a detector that **saw**
the variant still separates it from normal. Each family's reference is checked by the
same criterion; if the reference fails, the family leaves E1 with the reason.

## Measures

- **Search objective** (what B2–B4 see and what selects): **soft
  evasion** = 1 − mean score of the frozen detector on the attack messages,
  averaged over the three search seeds. Tie-break: lower mean AP, then the oldest
  proposal. *Changed on 2026-10-15, after the pilot and before any main
  run* (see "Pilot"); until then it was evasion at the budget of 10.
- **Selection and test:** at the end of the campaign, the best valid candidate is
  regenerated on the 5 test seeds and measured against the frozen detector. A campaign
  with no valid candidate records an absent test (and enters the validity rate).
- **Per campaign (unit of analysis):** test soft evasion (primary metric)
  and test evasion at the three budgets (mean of the 5 seeds), AP, false
  alarms per 10k, validity (fraction of the proposals made — an exhausted
  ladder did not propose anything invalid), tokens, LLM calls, time.
- **Curve:** best soft search evasion up to query *k*, per strategy and
  family.
- **Pre-defined comparisons** (paired by family × repetition, bootstrap
  of 1000 resamples of campaigns, 95% percentile CI; "separates" only if the CI
  excludes zero): B3−B1, B3−B2, **B4−B3** (RQ2), B1−B0 and B2−B0 as context.
  The aggregate bootstrap is **stratified by family** (it resamples
  repetitions within each family): the families have very different search spaces
  (`injection` has 2 fields, `masquerade_fault` 11), and a free
  resample would make the aggregate depend on how many times each
  family was drawn. Each family is reported next to the aggregate.

## What the LLM sees

- **B3:** the family catalog (same rendering as the `IntentAgent`, only the
  family), the search box, and the history of the campaign's previous proposals:
  fields/values changed, validity (and reason) and the search metrics
  (evasion at the three budgets, AP).
- **B4:** everything in B3 + the last `DefensePlan` that passed the E5 gate,
  produced by the `DefenderAgent` over a `DetectionReport` of the last valid
  candidate. A plan refused by the gate = a round without a plan, recorded.

### The Defender's evidence changes with the candidate

B4's `DetectionReport` carries the counts at the primary point, summed over the
search seeds, and as `top_features` the **mean absolute SHAP of the attack class
computed only over the attack messages the detector let through**
(false negatives at the primary point) — "what made the detector call these
messages normal". When the candidate has no false negative, SHAP is
computed over all of its attack messages. The report's `split` field says which
of the two populations was used.

Why not the Gini importances of the frozen RF: the detector does not change during
the campaign, so they would be **the same in every round**. The Defender
would receive the same evidence for each candidate, the plan would tend to repeat, and
a "B4 = B3" would measure the experiment design, not the value of defensive
feedback. SHAP over the false negatives changes with the candidate because the
messages that evade change with it.

## Cost and quota

Free Groq tier: 8000 TPM, 1000 RPD, and TPD per model (not exposed in the headers).
The runner spaces out the calls, accounts for tokens per call and, on a
daily-quota error, saves the ledger and resumes later — the campaign is resumable and never
repeats a proposal already recorded.

Rule: **a provider failure is never a data point.** Quota (daily or per
minute), a prompt that is too large and, since 2026-10-21, network or server errors
(`Connection error`, 5xx, timeout) are read before any conclusion: the
runner retries and, if it persists, stops without recording the proposal. Until 10-21 the
network error passed as "the LLM did not call the tool" and counted as an invalid
proposal — it happened once, in B3 of `high_stnum` rep5, proposal 6.
By the rule (applied by cause, not by outcome), that campaign's ledger
was cut before it and proposals 6–10 redone, because the following ones saw
the failure in the history. The original was kept alongside, as `rep5.jsonl.bak-20261021`
(and `results/…/rep5.json.bak-20261021`). No other E1 ledger has the error.

**Defender JSON refused by Groq (2026-10-22).** In the first real B4
(`masquerade_fault` rep1), 3 of the 7 Defender calls on
`gpt-oss-20b` came back as `400 json_validate_failed` with an empty
`failed_generation`: the provider refused the generation before a plan existed for the
gate to judge. In the pilot, still on the 120b, 8 of 8 were accepted. Leaving this
as a "refused plan" would remove the plan from ~40% of B4's rounds because of a flaw in the
JSON mode, not because of a Defender decision, and would dilute exactly what
B4−B3 measures. Rule since then: this refusal is retried up to 2 times
(temperature 0.2), and only then does the round go without a plan; the retries are
recorded (`provider_retries`) and count toward the calls. So that the 20 B4
campaigns follow a single rule, rep1 was archived
(`ledger/…/B4/rep1.jsonl.bak-20261022`, `results/…/rep1.json.bak-20261022`) and
is redone. In the same change, the text of a normal response stopped being read
as a provider error: a plan containing 0.9502 cannot pass as a 502.

**B4 cut to 2 families (2026-10-22).** The 20 B4 campaigns cost ~6
days of 120b quota, and the schedule cannot accommodate that. B4 runs only on
`masquerade_fault` and `high_stnum` × 5 reps (10 campaigns, ~3 days): the two
families that E3 had already fixed from E1's B1 (largest soft evasion and largest
evasion@10), chosen before any B4 and without looking at B3. B4−B3 is
paired and stratified only on those two; `report` already does this, because each
comparison uses only the families where both sides exist.

## Outputs

`outputs/experiments/e1/`:
- `ledger/<family>/<strategy>/rep<r>.jsonl` — one line per proposal;
- `ids/<family>.json` — training and thresholds of the frozen detector;
- `results.json` — test of the selected candidate per campaign + B0.

## Pilot

The pilot (1 family, 1 repetition per strategy) serves to measure time and
tokens and to fix the LLM model and the scale. It does not enter the results.

It decides, in this order, with the tokens measured per call:

1. do the five families fit in the daily quota in an acceptable time? If so, they stay;
2. if not, the Defender moves to another model (`openai/gpt-oss-20b`), which has its own
   daily quota — the attacker stays the same in B3 and B4, which is
   what the B4−B3 comparison requires;
3. if it still does not fit, three families, keeping 5 repetitions.

**Decision (10-15):** step 2. Pilot projection for 4 families × 5
repetitions: ~1.85M tokens, ~9 days on a single model at the estimated daily ceiling
(200k/model, not confirmed). The Defender goes to `openai/gpt-oss-20b`
(default of `--defender-model-id`); the attacker stays on `openai/gpt-oss-120b`
in both conditions. The four families stay (~6 days).

### What the pilot showed (2026-10-15, `masquerade_fault`, rep 1)

| | proposals | valid | calls | tokens | evasion@10 |
|---|---|---|---|---|---|
| B1 | 10 | 10 | 0 | 0 | 0.000 |
| B2 | 6 (ladder exhausted) | 6 | 0 | 0 | 0.000 |
| B3 | 10 | 9 | 10 | 27k | 0.000 |
| B4 | 10 | 8 | 17 | 65k | 0.000 |

- **Per-budget evasion stuck at zero.** The 34 valid proposals had
  evasion 0 at the three budgets and oracle AP 1.0. The cause is the detector, not the
  search: the RF gives a score of exactly 0 to practically all normal traffic, so
  the 10 and 100 alarm thresholds calibrate at 0.0 (the 1-alarm one, at 0.02) and
  any tree vote for "attack" fires. With this objective, every B×B
  comparison would give zero by construction.
- **The mean attack score moves.** Recomputed over the same proposals
  (cached traces): 1.00 on the reference, down to 0.44 on the variants with
  `incrementStNumOnFault=False` or `sqnumMode=normal`, ~0.83 on a line that
  B3 found (minimal analog deviation, multiplier ~1). Hence the switch of the
  objective to soft evasion. Per-budget evasion is still reported: it is
  what says whether any message actually gets through.
- **Measured cost:** ~35–80 s per proposal (3 generations + oracle); B3 ≈ 2.7k
  tokens per call, B4 ≈ 3.8k (17 calls: 10 from the attacker, 7 from the Defender;
  plan reuse saved 2). Gates working: the LLM dictated 5 fields
  once (`no_intent`) and inverted a pair another time (`compile_failed`); Defender
  with 8/8 plans accepted.

## Implementation

- `experiments/search.py` — box (`SearchBox`), validity chain
  (`CandidateEvaluator`, `Oracle`), B1 (`RandomSearch`), B2 (`OpenLoopLadder`,
  over `core/feedback_policy.py::escalate_intent`), ledger and resumable
  campaign, selection and test. No LLM.
- `agents/attacker/` — B3/B4 (`LlmAttacker`), prompt in `prompts/attacker.md`.
  Every strategy proposes an `IntentSpec` (B1 as `target_values`), so
  compiler, box and validity are the same code for all four.
- `scripts/run_e1.py` — `references` (frozen IDS, B0, oracle AP on the
  references → λ), `campaign` (with `--pilot` for the pilot) and `report`.
- Provider errors are read before the proposal is counted: a per-minute limit
  waits and retries (does not count); a daily quota or a prompt that is too large stop
  the campaign without recording the proposal in progress. A response with no tool
  call, or refused by the gates, **counts** as an invalid proposal
  (`no_intent`).
- B4 reuses the plan when the last valid candidate has not changed since
  the previous query to the Defender: same evidence, no extra call.
- B2 has at most 6 rungs (4 in `injection`) up to the box width; the
  curve repeats the best value after exhaustion.

## References (2026-10-15)

`scripts/run_e1.py references`, truly frozen detector (training
seeds 101–103, thresholds calibrated on 201–202):

| Family | search evasion@10 | test evasion@10 | test AP | oracle AP | p | observed λ |
|---|---|---|---|---|---|---|
| masquerade_fault | 0.000 | 0.000 | 1.000 | 1.000 | 0.375 | 1.000 |
| random_replay | 0.000 | 0.000 | 1.000 | 1.000 | 0.500 | 1.000 |
| inverse_replay | 0.000 | 0.000 | 1.000 | 1.000 | 0.500 | 1.000 |
| high_stnum | 0.008 | 0.007 | 1.000 | 1.000 | 0.500 | 1.000 |
| injection | 0.981 | 0.980 | 0.346 | 0.363 | 0.048 | 0.331 |

- Test soft evasion (the objective adopted after the pilot) on the reference:
  masquerade 0.000, random_replay 0.002, inverse_replay 0.001, high_stnum
  0.007 (injection 0.656).
- The four E1 families start from perfect detection: B0 is at the floor, and
  any evasion a strategy finds is measurable.
- `injection` evades 98% **with no search at all**: no strategy can
  improve on B0, and the comparison between them would measure noise. It was also the
  only case that would pull λ down — not even an RF trained on it exceeds
  AP 0.36, consistent with an attack that clones legitimate messages. It leaves E1
  as a finding in its own right (detector blind spot on the reference), not as a failure.

## Results (2026-10-30)

All campaigns closed: B0–B3 on the 4 families × 5 reps, B4 on the cut
(masquerade_fault and high_stnum × 5 reps; see "Cost and quota").
`scripts/run_e1.py report` → `outputs/experiments/e1/results.json`.

Test soft evasion (mean of the campaigns):

| Family | B0 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| high_stnum | 0.007 | 0.166 | 0.102 | 0.231 | 0.223 |
| inverse_replay | 0.001 | 0.180 | 0.027 | 0.191 | — |
| masquerade_fault | 0.000 | 0.515 | 0.514 | 0.537 | 0.539 |
| random_replay | 0.002 | 0.092 | 0.062 | 0.189 | — |

Paired differences (family-stratified bootstrap, 95% CI):

| | soft evasion | evasion@10 |
|---|---|---|
| B1 − B0 | +0.236 [+0.216, +0.257] | +0.061 [+0.050, +0.072] |
| B2 − B0 | +0.174 [+0.157, +0.191] | +0.017 [+0.012, +0.023] |
| B3 − B1 | **+0.049 [+0.011, +0.088]** | +0.030 [+0.018, +0.042] |
| B3 − B2 | **+0.111 [+0.067, +0.160]** | +0.074 [+0.064, +0.084] |
| B4 − B3 | −0.003 [−0.036, +0.033] (does not separate) | −0.008 [−0.023, +0.008] (does not separate) |

- The LLM attacker (B3) beats random search and the open ladder. Seeing the
  Defender's plan (B4) adds nothing.
- Evasion@10 only leaves 0 in high_stnum (B3 0.290, B4 0.274) and inverse_replay
  (B3 0.082). In masquerade and random_replay it stays at 0 in all
  strategies: soft evasion exists, but it does not cross the 10/10k threshold.
- **Why B4 ≈ B3.** The gate accepted 87 of 90 B4 plans (proposals 0
  have no candidate and do not count; the 3 refusals were a missing
  `validation_test`). The plans are almost all the same: 81 of 94 actions are
  `detector_threshold_tuning` to reduce false positives, and 78 of 87 have
  priority `low`. The plan tells the attacker little about what the detector looks at.
- The `json_validate_failed` retries of the 20b (`d4ec71ce`) recovered all the
  plans that failed at the provider, including a new case, "max completion
  tokens reached".
- B2's 5 reps are not independent (fixed seeds, the ladder converges): B2 is
  deterministic per family.
- By E3's 10-20 rule, B4 − B3 without separation would pick B3 for the LLM
  arm. The 10-22 cut (`docs/experiments_e3.md`), fixed before this
  result, had already picked B3.

## Open items before the main runs

- token pilot and the scale decision above.
