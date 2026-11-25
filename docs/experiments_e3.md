# E3 — Applied defense and arms race (RQ3): protocol

Fixed on 2026-10-20, **before** any execution and while E1's B3/B4 was still
running; the five open decisions of the draft were closed by the proposal (see
"Decisions"). The same rule as E1 applies: changing an item after seeing a test
number invalidates the comparison, and post-pilot changes are recorded in the
"Pilot" section.

## Question

E1 measures only the attacker, against a frozen detector. E3 closes the loop:
in each round the attacker looks for evasion against the current detector, and
the defense is **applied** — the detector is retrained with what the attacker
found.

- **RQ3a (does the defense close the gap?):** does the variant that evaded the
  detector of round *r* stop evading the detector of round *r+1*?
- **RQ3b (does the defense accumulate?):** does the detector of the last round
  keep catching the variants of the earlier rounds (no forgetting)?
- **RQ3c (who wins the race?):** does the evasion the attacker manages against
  the current detector fall over the rounds?
- **RQ3d (how much does hardening cost?):** does the hardened detector lose
  performance on the reference attack or on normal traffic?

## Design

A **lineage** = (family, attacker arm, repetition). Each lineage is
independent: the D₁ of a repetition depends on its own A₀.

```
D₀ = frozen detector from E1 (same training, same thresholds)
for r = 0 … R−1:
    attacker campaign against D_r (frozen during the campaign)  → A_r
    D_{r+1} = retraining with reference + variants of A_0 … A_r
              + recalibration of the thresholds on the calibration normal traffic
final evaluation: cross matrix D_i × A_j on the test seeds
```

- **R = 4** attack rounds (A₀…A₃) and 5 detectors (D₀…D₄).
- **Round 0 is E1.** D₀ is E1's `FrozenIds` and the round-0 campaign is the E1
  campaign of the same arm and repetition (same seeds, same detector, same
  budget). It is reused from the ledger, not run again: E3 only pays for
  rounds 1…R−1.
- During a campaign the detector does not change — within the round, evasion is
  measured against a fixed target, exactly as in E1.

## The applied defense

**Adversarial training**, deterministic — it is the only technique in the
Defender's catalog (`config/defense_techniques.py`, `detector_retraining`) that
runs without human intervention; network rules, segmentation or GOOSE
authentication cannot be applied to an RF automatically and honestly.

- **Which variants go in:** the **k = 3** valid candidates with the highest
  search objective in round *r* (the selected one and the next two; fewer if the
  campaign had fewer valid ones; the 20 B1 campaigns in E1 have ≥ 3 each). Only
  candidates that passed the oracle (λ = 0.5): a variant that "evades because it
  disappeared" teaches the defense nothing.
- **On which seeds:** each variant is regenerated on **training seeds 101–103**
  — never on the search or test ones. No message that the attacker saw or that
  measures the result enters the training.
- **Attack rows only.** The benign part is a function of the seed (ERENO
  `c36bee2`): the normal traffic of the variant's trace on seed 101 is identical
  to that of the reference on 101. Adding it would duplicate the normal traffic
  with every variant; only the rows labeled as attack go in. This also holds
  down the training memory, which would grow ~3× per round with whole traces
  (the 15.6 GB machine has already killed a process in E1).
- **Cumulative:** D_{r+1} sees the reference and the variants of **all** rounds
  up to *r*. Without that, "forgetting" would be an effect of the design, not of
  the detector.
- **Fixed pipeline:** RF, `protocol_features="deltas"`, E6 preprocessing, no
  selection or undersampling — the same as E1. Only the training **data**
  changes. This is the project's guardrail applied to E3: nothing that looks at
  the metric chooses preprocessing, threshold or features
  (`core/undersampling_policy.py`).
- **Recalibration:** D_{r+1}'s thresholds are recalibrated on the normal traffic
  of calibration seeds 201–202, by the same rule as E1. It is the defender's
  operating point, not an adjustment to the variant: the calibration never sees
  attack.

### The role of the Defender (agent)

For each round's selected variant, the `DefenderAgent` (`gpt-oss-20b`, as in B4)
produces a `DefensePlan` over its `DetectionReport` against D_r on the search
seeds (E1's `candidate_report`), going through the E5 gates. The plan **does
not decide** the retraining (which is fixed above); it is recorded and
measured:

- did the plan recommend retraining? It counts `detector_retraining` or
  `dataset_enrichment`, the two catalog techniques that describe the applied
  defense (agreement with the defense);
- are the features it cites as evidence among the 5 whose importance changes
  the most (Gini, in absolute value) between D_r and D_{r+1}? (fraction of the
  cited ones; ties in with E4, SHAP vs. cited features).

Discarded: letting the plan choose *which* candidates enter the retraining. It
would be more "multi-agent", but it puts an LLM inside the detector's training —
which the project's central guardrail avoids (the LLM only produces intent or
plan). The plans are a separate step (`run_e3.py plans`), after the campaigns:
they do not interfere with the race and they spend the 20b's quota, not the
attacker's.

## Attacker arms

| Arm | Attacker per round | Families | Repetitions | Cost |
|---|---|---|---|---|
| E3-B1 | random search (E1's B1) | all 4 | 5 | no tokens |
| E3-LLM | the E1 winner between B3 and B4 → **B3, cut down on 2026-10-22** | 2: `masquerade_fault`, `high_stnum` → **`high_stnum`** | 5 → **2** | Groq |

- **E3-B1** is the main arm: cheap, complete, and it answers RQ3a–d on its own.
- **E3-LLM families**, fixed before B3/B4 and only from E1's B1: the one with
  the highest soft evasion (`masquerade_fault`, 0.515) and the one with the
  highest evasion@10 (`high_stnum`, 0.189). Two families fit in ~4–5 days of
  quota; four would need ~8–9. E3-LLM is read family by family next to E3-B1.
- **E3-LLM** answers whether an adaptive attacker **reacts** to the defense. Rule
  fixed now, before E1's result: B4 enters if B4−B3 "separates" in B4's favor in
  E1's aggregate; otherwise B3 (cheaper). E1's minimum of 5 repetitions applies
  here.
- **Memory between rounds (E3-LLM):** at the start of round *r*, the selected
  variants of the earlier rounds are re-measured against D_r on the search
  seeds and enter the attacker's history ("isto funcionava e foi corrigido" —
  "this worked and was fixed"). This is what lets the attacker learn from the
  defense. E3-B1 has no memory by construction.
- Budget per round: 10 proposals, the same box and the same validity chain as
  E1.

## Controls

- **No defense:** the attacker keeps going against D₀ for the same number of
  rounds (that is, running the extra campaigns against E1's detector). It
  separates "the attacker improved because the defense changed" from "the
  attacker improved because it had more queries". Only in E3-B1, and there it
  comes almost for free: random search does not read history, so the proposals
  of the no-defense lineage are **the same** as those of the defended one (same
  RNG per family, repetition, round and index) and the search traces are
  already cached — only the detector that scores them changes. In E3-LLM the
  control would double the quota; the comparison there is against round 0 and
  against the E3-B1 of the same family.
- **Placebo retraining:** D'_{r+1} receives as many traces as D_{r+1} received
  of variants, but from the reference config on extra training seeds (111, 112,
  …, up to 146 with R = 4), and the attack rows are sampled (fixed seed) down
  to the **same number** that D_{r+1} received. It separates "the detector
  improved because it saw the variants" from "the detector improved because it
  saw more data". Only in E3-B1, and only evaluated (it goes into the matrix),
  never attacked: it answers the closure (RQ3a) and the cost (RQ3d), not the
  race.

## Measures

Everything on **test seeds 401–405**, none seen by the attacker or the
retraining. Primary metric: **soft evasion** (the same as E1). Evasion@10 and AP
alongside.

- **Cross matrix** `E[i][j]` = soft evasion of A_j against D_i, for
  i = 0…R and j = 0…R−1. It is E3's main figure.
- **RQ3a, closure:** `E[r][r] − E[r+1][r]` per round (the gap the defense
  closed).
- **RQ3b, forgetting:** `E[R][j] − E[j+1][j]` for each j < R−1
  (positive = the final detector went back to letting through what it had
  already fixed).
- **RQ3c, race:** the sequence `E[r][r]`, r = 0…R−1 — what the attacker manages
  against the detector of the moment. Pre-defined comparison: round R−1
  against round 0, and against the no-defense control in the same round.
- **RQ3d, cost:** at the primary point (10 alarms/10k), recall on the reference
  attack and false alarms per 10k on the test normal traffic, for D₀…D_R. Here
  the reference attack's **precision, recall and F1** also come in: the
  reference prevalence is fixed, so F1 compares detectors without the
  distortion it has in the comparison between variants (see E1, "Measures").
  **Secondary, recorded after the pilot** (see "After the pilot"): false alarms
  per 10k on the **variant trace's** normal traffic, `F[r+1][r] − F[r][r]`
  (`trace_alarm_cost`), and the same against the placebo.
- **Defender:** fraction of plans accepted by the gate, fraction that recommends
  `detector_retraining`, overlap between cited features and the importance
  change (top-5).
- **Cost:** tokens, calls, generation and retraining time per round.

**Comparisons** (paired by family × repetition, bootstrap of 1000 resamples
stratified by family, 95% percentile CI, "separates" only if it excludes zero —
the same as E1): mean closure; `E[R−1][R−1] − E[0][0]`; `E[R−1][R−1]` with
defense − without defense; mean forgetting; reference recall D_R − D₀. Each
family is reported alongside the aggregate.

## Possible results and what each one says

- Closure > 0 and the race falling: the defense hardens and the attacker finds
  less and less — the co-evolution favors the defense.
- Closure > 0 but the race stable: the defense fixes what it sees, the attacker
  finds another gap every round — "whack-a-mole", and each family's catalog says
  how many gaps there are.
- Closure ≈ 0: retraining is not enough for that family (the features do not
  separate the variant) — evidence for another defense, not for more rounds.
- Forgetting > 0: retraining with the new pushes out the old.

All four are publishable; none is "the experiment failed".

## Estimated cost

- Proposal: ~35–80 s (3 generations + oracle, measured in E1). Retraining: 3
  reference traces + the accumulated attack rows, per lineage and round.
- **E3-B1** (3 new rounds × 4 families × 5 reps × 10 proposals = 600
  proposals): ~12–27 h of generation, no tokens (the 2026-10-22 pilot gives
  ~25–30 h with the matrices; see "After the pilot"). The no-defense control
  reuses the search traces (same proposals): it costs the oracle and the
  scoring, not the generation. The retraining generates 9 traces per round and
  lineage (3 variants × 3 training seeds); the placebo, 36 reference traces per
  family (seeds 111–146, shared among the repetitions). Runs overnight, like
  E1's B1/B2, with `JAVA_TOOL_OPTIONS=-Xmx2g`.
- **Defender plans:** one per round and defended lineage — 80 in E3-B1
  (4 × 5 × 4) and 40 in E3-LLM, ~5k tokens each on `gpt-oss-20b`.
- **E3-LLM** with B3 (~28k tokens per campaign on `gpt-oss-120b`): 2 families ×
  5 reps × 3 rounds = 30 campaigns ≈ 840k tokens ≈ 4–5 days of quota;
  4 families ≈ 8–9 days. With B4 (~38k on the 120b + ~27k on the Defender's
  `gpt-oss-20b`, per campaign) the 120b's bill goes up ~35%.
- It only starts **after** E1's B3/B4 (same ERENO runtime and same quota).

## Implementation

- `experiments/arms_race.py`:
  - `Lineage` and `DetectorLineage`: the lineage ledgers, `D_r`, the placebo `D'_r` and the attacker's memory.
  - `top_variants`.
  - The matrix: `score_detector` and `matrix_values`.
  - The measures: `race_measures`, `placebo_closure`, `reference_cost` and `trace_alarm_cost`.
  - `saturated_budgets`: reads the budget with no operating point as NaN (`metrics.is_saturated`).
  - The Defender's: `importance_shift` and `plan_measures`.
  - No model is persisted, because retraining `D_r` is deterministic.
- `FrozenIds.extra_attack` / `IdsEvaluator.fit_traces(attack_only_paths=…)`:
  attack-only rows in the training, with an optional cap (placebo). Empty = E1,
  unchanged.
- `ProposalContext.race_round`/`prior`: round 0 draws exactly what E1 drew; the
  following ones draw again. The LLM attacker sees the earlier rounds in a
  dedicated section of the message, which does not exist in E1.
- `scripts/run_e3.py` has the commands `campaign`, `matrix`, `plans` (all three resumable, with exit code 3 when the quota is exhausted) and `report`. Outputs in `outputs/experiments/e3/`:
  - `ledger/<family>/<lineage>/rep<r>/round<k>.jsonl`
  - `ids/…/D<k>.json`
  - `results/…/round<k>.json`
  - `matrix/<family>/<lineage>/rep<r>.json`
  - `plans/…/round<k>.json`
  - `results.json`
- Tests in `tests/test_experiment_arms_race.py` (no Java or Groq).

## Pilot

1 family (`high_stnum`, the one that evaded the most in E1's B1), arm E3-B1, 1
repetition, R = 2. It measures retraining time and memory, checks that D₁
closes A₀'s gap (if not even that happens, the problem is the design and not
the family) and that the placebo is distinguishable. It does not enter the
results.

## After the pilot (2026-10-22)

The pilot ran on 2026-10-22 (`outputs/experiments/e3/pilot/`). The design
worked: D₁ closes A₀ (soft evasion from 0.186 to 0.0004 on the test) and the
equally sized placebo does not close it (0.212), so the gain comes from the
variants, not from more data. Against D₁, the B1 attacker only reached 0.027.
Cost on the reference at the 10/10k point: recall from 0.993 to 0.984 and
alarms from 7.7 to 14.1 per 10k (D₀ → D₂). Time: ~5.5 min per campaign round
and ~19 min for the matrix (5 detectors, up to 1.2M rows), without running out
of memory. That gives ~25–30 h for the complete E3-B1.

The pilot showed two things the protocol had not foreseen. Both were recorded
**before** the E3-B1 and E3-LLM campaigns:

1. **A saturated operating point becomes NaN, not evasion 1.0.** The retraining
   only adds attack rows. As a result, the training imbalance grows every round
   (1:1, 4:1, 7:1 in the pilot, ~13:1 in D₄), and the calibration threshold
   rises with it: at 1/10k it went from 0.85 to 0.93 and, in D₂, reached 1.00,
   with **zero possible alarms**. The "evasion 1.0" at that point comes from the
   threshold, not from the attack.
   - **Rule:** when the alarm rate in the calibration is 0
     (`metrics.is_saturated`), the rates of that budget are NaN and the block
     carries `saturated: true`. The counts stay.
   - **Where it applies:**
     - in the matrix and in the `report` (`saturated_budgets`; a matrix written
       before the fix is corrected on read);
     - in the cost on the reference;
     - in the history the LLM attacker reads (it shows "n/d" (n/a), and not
       "1.000", which would make the attacker think it evaded everything);
     - in the `plans` step: if the primary point saturated, the Defender does
       not receive a report with recall 0. The round is left out of the
       Defender's measure, without counting as a refused plan.
   - **What does not change:** soft evasion, the primary metric, does not use a
     threshold. No campaign, selection or oracle changes.
   - **Caveat:** if the 10/10k point saturates in D₃ or D₄, the evasion@10 and
     the RQ3d of that lineage and round are left without a value (NaN). This is
     reported as a limit of retraining that only adds attack, not hidden.
   - **Discarded:** balancing the training (undersampling or class weight). It
     would change the detector's pipeline between rounds, which E3's guardrail
     forbids (only the **data** changes).

2. **Alarm cost on the variant's trace (RQ3d, secondary).** The deltas of a
   normal message (`stDiff`, `sqDiff`, `timestampDiff`…) are computed against
   the previous message of the flow, which may be an attack message. Thus a
   detector can "close" a variant by also alarming on the normal traffic around
   it. In the pilot, at the 10/10k point and on the normal traffic of A₀'s
   traces:
   - D₀ alarmed **7%** (704/10k);
   - D₁ alarms **20%** (1986/10k);
   - the placebo D'₁ alarms 5%.

   Part of the closure, therefore, is paid for with alarms in the attack's
   context, and the cost measured on the reference's normal traffic (RQ3d) does
   not see this. It is now reported, without changing anything in the race:
   - `trace_alarm_cost` = `F[r+1][r] − F[r][r]`, with F = false alarms/10k on
     A_r's trace normal traffic;
   - the difference against the placebo, with the same paired bootstrap as the
     others.

   It stays **secondary**: RQ3d's primary remains the reference's normal
   traffic, which is the operator's "real" normal. This one measures what the
   retraining costs near the attack.

**Deviation in E3-LLM (2026-10-22).** The first B3 × high_stnum campaign loaded
`metrics.py` one minute before rule 1 went in. In three rounds the attacker saw
"evasão@1 = 1,000" (evasion@1 = 1.000) in every proposal: rep1 r3, rep2 r2 and
rep2 r3, all with the 1/10k point saturated. Those rounds were archived with the
suffix `.bak-20261022-saturado`, together with rep2's D3 card, and redone with
the new code. In the redone ledgers the 1/10k point is NaN, and the attacker
sees "n/d" (n/a). Rounds 1 (and rep1's 2) had no saturated point and were left
as they were. Soft evasion on the test, before and after:
- rep1 r3: 0.017 → 0.006;
- rep2 r2: 0.038 → 0.029.

## Decisions (2026-10-20)

The user delegated all five to the draft's proposal:

1. Applied defense = adversarial training with k = 3 variants per round.
2. The Defender's plan is only recorded and measured; it does not choose
   variants.
3. R = 4 rounds.
4. E3-LLM on 2 families (`masquerade_fault`, `high_stnum`), by the rule above.
5. The placebo goes in, in E3-B1, only evaluated.

## E3-LLM cut (2026-10-22)

The deadline does not accommodate the 4–5 days of quota of the complete
E3-LLM, and E1's B4 was cut down on the same day (`docs/experiments_e1.md`,
"Cost and quota"). Decided before any B4 result and before any E3 round:

- **Arm B3, fixed now**, instead of waiting for the B4−B3 rule. It breaks the
  2026-10-20 rule, and is therefore recorded: the choice is by cost (~28k tokens
  per campaign against ~65k for B4), not by result, which does not exist yet.
- **1 family, `high_stnum`** (the one with the highest evasion@10 in E1's B1),
  **2 repetitions** (reps 1 and 2) and R = 4. Round 0 is E1's B3 of those reps.
  That makes 6 new campaigns, ~170k tokens on the 120b (~1 day of quota), plus 8
  Defender plans on the 20b.
- **Reading: case study, not statistics.** With 2 lineages, the `report`'s
  paired aggregate has no power. E3-LLM is read through each lineage's D_i × A_j
  matrices, next to the E3-B1 of `high_stnum` on the same reps, and it answers
  descriptively whether the LLM attacker reacts to the defense (the memory
  between rounds). RQ3a–d with a confidence interval come only from E3-B1.
- No no-defense control, as was already the case in E3-LLM.

Commands (after E1's B4, never together with another campaign):

```bash
uv run python scripts/run_e3.py campaign --arm B3 --family high_stnum --reps 1 2
uv run python scripts/run_e3.py matrix   --arm B3 --family high_stnum --reps 1 2
uv run python scripts/run_e3.py plans    --arm B3 --family high_stnum --reps 1 2
```

## E3-LLM results (2026-10-30)

Campaigns (2026-10-23) and matrices (2026-10-30) closed, B3 × high_stnum × reps
1–2, R = 4. The `plans` step has not run yet. Matrices in
`outputs/experiments/e3/matrix/high_stnum/B3/rep{1,2}.json`.

Soft evasion on the test, row = detector D_i, column = attack A_j:

| | A0 | A1 | A2 | A3 |
|---|---|---|---|---|
| **rep1** D0 | 0.235 | 0.247 | 0.160 | 0.084 |
| D1 | 0.000 | *0.175* | 0.125 | 0.072 |
| D2 | 0.000 | 0.002 | *0.024* | 0.007 |
| D3 | 0.000 | 0.002 | 0.009 | *0.006* |
| D4 | 0.000 | 0.002 | 0.008 | 0.004 |
| **rep2** D0 | 0.236 | 0.235 | 0.185 | 0.194 |
| D1 | 0.000 | *0.195* | 0.156 | 0.164 |
| D2 | 0.000 | 0.014 | *0.029* | 0.025 |
| D3 | 0.000 | 0.011 | 0.017 | *0.015* |
| D4 | 0.000 | 0.010 | 0.013 | 0.011 |

In italics, D_r × A_r: the attacker's response to that round's detector.

- **D_{r+1} closes A_r.** D1 zeroes A0; D2 takes A1 to 0.002 / 0.014.
- **The attacker does not keep up.** The diagonal falls from ~0.18–0.20 to ~0.025
  and then ~0.01. The race ends in the defense's favor in the 2nd round.
- **The retraining generalizes forward.** D2 already covers A2 and A3, which it
  never saw: after A1, the attacker found no new direction in the box.
- **Specialization cost (rep1 only).** Against D0, A3 evades 0.084, against
  0.235 for A0: by adapting to the retrained detector, the attacker got worse
  against the original. In rep2 this does not show up (0.194).
- **Cost to the defense.**
  - False alarms per 10k on the reference: ~7.7 in D0, 10–15 in D2/D3.
  - In D4, the 10/10k point saturated in both reps: evasion@10 is NaN (rule 1
    of "After the pilot").
  - Evasion@10 falls less than the soft one: against D2/D3 it stays at
    0.04–0.13.
  - On the variants' traces, the false alarms/10k on the normal traffic are in
    the thousands (`trace_alarm_cost`, rule 2).

Descriptive reading, as fixed in the cut: the RQ3 confidence intervals come
from E3-B1.

## E3-B1 plans cut (2026-11-01)

The full `plans --arm B1` step asks for 80 plans (4 families × 5 reps × 4
rounds) on `gpt-oss-20b`. With B4's retries, that is ~5–6k tokens per plan,
~450–500k in total: almost 3 days of quota (200k/day), which the execution week
(2026-10-30 → 2026-11-06) does not accommodate. Decided before any E3-B1
campaign:

- **E3-B1 plans only on `high_stnum`**, the 5 reps, R = 4: 20 plans,
  ~120k tokens, one day of quota. It is the same family as E3-LLM, so the
  Defender measures of the two arms stay comparable.
- **What changes:** only the Defender measures (acceptance, fraction that
  recommends `detector_retraining`, overlap between cited features and
  importance change). On the other three families these measures are absent,
  and the `report` already skips rounds without a plan.
- **What does not change:** the race, the placebo and the no-defense control
  (RQ3a–d) on the 4 families. The plan is only measured and never enters the
  retraining.

```bash
uv run python scripts/run_e3.py plans --arm B1 --family high_stnum
```

## E3-B1 campaigns (2026-11-02)

`campaign --arm B1` (2026-11-01 ~20:50 → 2026-11-02 06:08, ~9.5 h) and
`campaign --arm B1 --no-defense` (2026-11-02 ~10:05 → 12:40, ~2.6 h) closed:
4 families × 5 reps × rounds 1–3, 10 proposals per round, validity 100%,
0 tokens. A generator timeout (inverse_replay, no defense, rep3 round 3)
was absorbed by the retry. With the cut rule (drop the control only if B1
went past ~15 h), the control ran on all 4 families.

Soft evasion on round r's test, mean of the 5 reps. With defense, against the
retrained D_r; without defense, against D0 (the log prints "D_r", but the
no-defense lineage always uses index 0, `arms_race.py`). A0 is E1's B1, against
D0.

| Family | A0 | r1 def / without | r2 def / without | r3 def / without |
|---|---|---|---|---|
| masquerade_fault | 0.515 | 0.139 / 0.547 | 0.100 / 0.519 | 0.027 / 0.563 |
| high_stnum | 0.166 | 0.045 / 0.185 | 0.021 / 0.152 | 0.014 / 0.180 |
| inverse_replay | 0.180 | 0.006 / 0.164 | 0.001 / 0.176 | 0.000 / 0.164 |
| random_replay | 0.092 | 0.010 / 0.173 | 0.006 / 0.062 | 0.001 / 0.146 |

Per rep, with defense (r1 / r2 / r3):

| Family | rep1 | rep2 | rep3 | rep4 | rep5 |
|---|---|---|---|---|---|
| masquerade_fault | .284/.034/.001 | .161/.016/.004 | .190/.006/.087 | .036/.427/.044 | .022/.016/.001 |
| high_stnum | .027/.034/.006 | .020/.011/.018 | .067/.010/.021 | .030/.012/.011 | .079/.036/.014 |
| inverse_replay | .004/.001/.001 | .008/.001/.000 | .001/.000/.001 | .004/.004/.000 | .011/.001/.000 |
| random_replay | .002/.000/.001 | .002/.000/.001 | .000/.014/.000 | .008/.017/.001 | .036/.000/.000 |

- **The drop comes from the retraining, not from search noise.** Without
  defense, random search stays at E1's level in all three rounds (masquerade
  ~0.52–0.56, high_stnum ~0.15–0.19, inverse ~0.16–0.18). With defense, it falls
  by an order of magnitude already at D1 and keeps falling.
- **Search noise is large in random_replay.** Without defense, the per-round
  mean oscillates 0.062–0.173 (rep4: 0.375 in r1); this is why the control is
  per rep and per round, not against A0.
- **One attacker recovery:** masquerade rep4, 0.036 at D1 → 0.427 at D2
  → 0.044 at D3. Without it, r2's mean would be 0.018.
- **B1 × B3 on high_stnum (reps 1–2):** in r1 the LLM evades ~7× more
  (0.175 / 0.195 against 0.027 / 0.020); from r2 on the arms tie
  (~0.01–0.03).

## E3-B1 matrices (2026-11-03)

`matrix --arm B1` (2026-11-02 evening → 2026-11-03) closed, exit 0: 4 families
× 5 reps, D0…D4 and placebo D'1…D'4 against A0…A3, in
`outputs/experiments/e3/matrix/<family>/B1/rep{1..5}.json`.

**Check against the campaigns.** In the 60 D_r × A_r cells (r = 1–3), the
matrix's soft evasion is identical to the campaign's test one (maximum
difference 0). The retraining is deterministic, as predicted (no model
persisted).

Soft evasion on the test, mean of the 5 reps. Row = detector, column = attack;
in italics, the diagonal D_r × A_r.

| Family | Det. | A0 | A1 | A2 | A3 | FA/10k ref. |
|---|---|---|---|---|---|---|
| masquerade_fault | D0 | *0.515* | 0.172 | 0.380 | 0.466 | 2.7 |
| | D1 | 0.000 | *0.138* | 0.159 | 0.195 | 5.6 |
| | D2 | 0.000 | 0.000 | *0.100* | 0.116 | 6.4 |
| | D3 | 0.000 | 0.000 | 0.000 | *0.027* | 6.1 |
| | D4 | 0.000 | 0.000 | 0.000 | 0.000 | 6.2 |
| | D'1…D'4 | 0.60–0.62 | 0.20–0.21 | 0.43–0.44 | 0.54–0.55 | |
| high_stnum | D0 | *0.166* | 0.141 | 0.134 | 0.146 | 7.7 |
| | D1 | 0.002 | *0.045* | 0.027 | 0.027 | 14.2 |
| | D2 | 0.001 | 0.007 | *0.020* | 0.018 | 14.7 |
| | D3 | 0.001 | 0.005 | 0.013 | *0.014* | 14.5 |
| | D4 | 0.001 | 0.004 | 0.011 | 0.010 | 12.4 |
| | D'1…D'4 | 0.18–0.19 | 0.15 | 0.14 | 0.16 | |
| inverse_replay | D0 | *0.180* | 0.049 | 0.121 | 0.096 | 9.7 |
| | D1 | 0.001 | *0.006* | 0.006 | 0.004 | 9.1 |
| | D2 | 0.001 | 0.000 | *0.001* | 0.002 | 9.1 |
| | D3 | 0.000 | 0.000 | 0.000 | *0.001* | 9.2 |
| | D4 | 0.000 | 0.000 | 0.000 | 0.000 | 9.3 |
| | D'1…D'4 | 0.16–0.19 | 0.05–0.06 | 0.11–0.13 | 0.08–0.10 | |
| random_replay | D0 | *0.092* | 0.132 | 0.048 | 0.070 | 4.3 |
| | D1 | 0.000 | *0.009* | 0.008 | 0.009 | 6.1 |
| | D2 | 0.000 | 0.000 | *0.006* | 0.002 | 6.1 |
| | D3 | 0.000 | 0.000 | 0.000 | *0.001* | 3.9 |
| | D4 | 0.000 | 0.000 | 0.000 | 0.000 | 5.0 |
| | D'1…D'4 | 0.11–0.12 | 0.15–0.17 | 0.04–0.05 | 0.08 | |

- **D_{r+1} closes A_r.** In masquerade, inverse and random, the cell right below
  the diagonal is ≤ 0.002. In high_stnum, D1 zeroes A0 (0.002), but the following
  ones stay at 0.007 / 0.013 / 0.010: they reduce, without zeroing.
- **The placebo closes nothing.** With the same volume of traces from the
  reference config, D'_r stays at D0's level or a little above (masquerade A0:
  0.60 against 0.515). D_r's gain comes from the variants, not from more data.
  It is the pilot's result, now on 4 families × 5 reps.
- **Forward generalization depends on the family.** In inverse and random,
  D1 already covers A2/A3 (≤ 0.009). In high_stnum, it covers largely (0.027). In
  masquerade, it does not: D1 still leaves 0.16–0.20 on A2/A3, and only the whole
  row (D3/D4) zeroes. Masquerade is the family in which the B1 attacker finds
  new directions for more rounds.
- **Attacker specialization cost.** Against D0, masquerade's A1 evades
  0.172 and inverse's A1 0.049, against 0.515 / 0.180 for A0. In high_stnum this
  barely happens (0.141–0.146). Same reading as B3 rep1.
- **Cost to the defense (RQ3d, reference).** False alarms/10k go up with
  the retraining in masquerade (2.7 → ~6) and high_stnum (7.7 → 12–15). In
  inverse and random they stay stable (9.1–9.7 and 3.9–6.1).
- **Evasion@10.** It is 0 in all masquerade and random cells (it already was in
  E1). In inverse, it goes from 0.02–0.06 (D0) to 0. In high_stnum, it falls from
  ~0.15 to 0.02–0.10 in D1/D2, but **rises again** in D3/D4 (up to 0.078),
  while soft evasion keeps falling. Probable reading, not yet checked: the
  threshold calibrated on the reference's normal traffic rises when the
  retrained detector alarms more, and the residual attack stays below it. In
  B1's D4, the 10/10k point did not saturate (unlike B3).

### Without defense (2026-11-03)

`matrix --arm B1 --no-defense` (~1 h, exit 0): only D0 against A0…A3 of the
no-defense lineage, in `outputs/experiments/e3/matrix/<family>/B1-nodef/rep{1..5}.json`.

**Check.** In the 60 D0 × A_r cells (r = 1–3), the soft evasion is identical
to the no-defense campaign's test one (maximum difference 0). In the 20
D0 × A0 cells, it is identical to the with-defense matrix's (same D0, same A0).

Soft evasion against D0, mean of the 5 reps. In the "with def." column, the A_r
of the defended lineage against the same D0 (D0 row of the table above).

| Family | A0 | A1 without / with def. | A2 without / with def. | A3 without / with def. | Evasion@10 without def. (A0…A3) |
|---|---|---|---|---|---|
| masquerade_fault | 0.515 | 0.547 / 0.172 | 0.519 / 0.380 | 0.563 / 0.466 | 0 in all |
| high_stnum | 0.166 | 0.185 / 0.141 | 0.152 / 0.134 | 0.180 / 0.146 | 0.19 / 0.22 / 0.16 / 0.21 |
| inverse_replay | 0.180 | 0.164 / 0.049 | 0.176 / 0.121 | 0.164 / 0.096 | 0.06 / 0.05 / 0.06 / 0.05 |
| random_replay | 0.092 | 0.173 / 0.132 | 0.062 / 0.048 | 0.146 / 0.070 | 0 in all |

FA/10k on the reference: D0's (2.7 / 7.7 / 9.7 / 4.3), as expected.

- **Without defense, there is no race.** The attacker always optimizes against
  the same D0 and stays at A0's level in all three rounds (this is the
  no-defense campaign's diagonal, now checked). Random search does not improve
  on A0 consistently. The maximum gain is in random_replay (0.173 in r1), pulled
  up by rep4 (0.375).
- **The specialization cost is confirmed.** Against the same D0, the A_r of the
  defended lineage evade less than those of the no-defense lineage in all 12
  cells. The difference is large in masquerade A1 (0.172 against 0.547) and
  inverse A1 (0.049 against 0.164). Escaping the retrained D_r pushes the attack
  to regions that D0 already catches.

E3-B1 matrices complete (with defense, placebo and without defense).

## E3-LLM plans (2026-11-03)

`plans --arm B3 --family high_stnum --reps 1 2`, with Defender `openai/gpt-oss-20b`.
Exit 0, 8 plans in `outputs/experiments/e3/plans/high_stnum/B3/rep{1,2}/round{0..3}.json`.
No point saturated. Each call spent ~3.7k input tokens and 0.9–1.5k output
tokens, ~41k in total.

| rep / round | priority | actions (technique ← cited evidence) |
|---|---|---|
| 1 / 0 | medium | threshold_tuning ← recall; dataset_enrichment ← accuracy |
| 1 / 1 | medium | threshold_tuning ← recall, fn |
| 1 / 2 | — | **refused**: response cut off at 2048 output tokens, no `validation_test` |
| 1 / 3 | low | threshold_tuning ← fp |
| 2 / 0 | medium | threshold_tuning ← recall; operator_alerting ← fn |
| 2 / 1 | medium | threshold_tuning ← recall, fn |
| 2 / 2 | low | threshold_tuning ← recall |
| 2 / 3 | low | threshold_tuning ← fn (accepted after a Groq 400 `json_validate_failed`) |

- **Acceptance of 7/8.** The refusal came from the parse (the JSON did not
  close), not from the evidence gate. No plan was refused for citing invented
  evidence.
- **The Defender only cites metrics, never features.** The reports carry
  `stDiff`, `sqDiff`, `SqNum` and `timeFromLastChange` at the top of the
  importances, and none of the 7 plans cites any of them. All recommend
  threshold adjustment, and only two add another technique. It is a generic
  Defender: the plan responds to the low recall, not to the attack's mechanism.
- **The priority follows the report.** It stays medium in rounds 0–1 and drops
  to low in rounds 2–3, as the retrained detector catches more. Since the
  validator requires it to match `priority_from_report`, this is a consequence
  of the metric, not a judgment by the LLM.

## E3-B1 plans (2026-11-03)

`plans --arm B1 --family high_stnum`, with Defender `openai/gpt-oss-20b`. It ran
without stopping for quota: 20 plans in
`outputs/experiments/e3/plans/high_stnum/B1/rep{1..5}/round{0..3}.json`.
Log in `outputs/experiments/e3_b1_plans_hs.log`. No point saturated. It was
~3.7k input tokens and 0.9–2.0k output tokens per plan, ~100k in total. There
were Groq 400 `json_validate_failed` errors, all resolved on retry.

- **Acceptance of 20/20** (in B3 it was 7/8).
- **Techniques, 24 actions in total:** `detector_threshold_tuning` in 19
  (present in 19 of the 20 plans), `dataset_enrichment` in 3,
  `operator_alerting` in 1 and `device_quarantine` in 1. No plan recommends
  `detector_retraining`.
- **Only one feature citation in 20 plans:** `device_quarantine ← stDiff`,
  in rep2 round 1. The rest cites only metrics (recall, fn, fp, precision,
  accuracy). It is the same generic Defender as in B3 and E1.
- **Low priority in all 20 plans.** In B3 it was medium in rounds
  0–1. This is consistent with B3's attacker evading more (~7× in r1): the
  priority comes from `priority_from_report`, so it follows the metric and not
  the arm.

## E3 report (2026-11-05)

`run_e3.py report`, exit 0. The result is in
`outputs/experiments/e3/results.json`, and the log in
`outputs/experiments/e3_report.log`. The intervals are 95%, by paired bootstrap
stratified by family. B1 has n = 20 (4 families × 5 reps); B3
has n = 2 (only high_stnum).

| measure (soft evasion) | B1 | B3 |
|---|---|---|
| mean closure `E[r][r] − E[r+1][r]` | +0.080 [+0.071, +0.090] | +0.107 [+0.106, +0.108] |
| mean forgetting | −0.001 [−0.001, −0.000] | −0.002 [−0.003, −0.001] |
| race `E[3][3] − E[0][0]` | −0.228 [−0.251, −0.206] | −0.225 [−0.229, −0.221] |
| closure − placebo | +0.203 [+0.175, +0.239] | — |
| `E[3][3]` with defense − without defense | −0.253 [−0.277, −0.233] | — |

In evasion@10/10k (B1): closure +0.019 [+0.014, +0.024], forgetting
+0.006 [+0.005, +0.007], race −0.049 [−0.059, −0.038], closure − placebo
+0.090 [+0.076, +0.106] and with − without defense −0.051 [−0.060, −0.043]. In
B3, the closure was +0.166 and the race −0.239.

Cost on the reference (B1, D_R − D_0): recall −0.008 [−0.009, −0.007], F1
−0.004 and +2.1 [+1.6, +2.6] false alarms/10k on the normal traffic.

- **RQ3a/b: the defense closes, and it is not an effect of "more training".**
  The closure is positive in both arms and beats the placebo by +0.20 of soft
  evasion. The race ends ~0.23 below the starting point, and the detector with
  defense ends 0.25 below the no-defense control.
- **RQ3c: there is almost no forgetting.** In soft evasion the forgetting even
  separates from zero, but it is negative and on the order of 0.001: the final
  D_R catches the old variants even better than the D_{r+1} that closed them.
  In evasion@10 it is positive (+0.006). It is small, but it indicates that the
  final D_R lets slightly more through at the operational threshold than the
  detector that closed each round.
- **RQ3d: the cost on the reference is small.** On the variants' traces' normal
  traffic, however, it is large and heavy-tailed. The alarm cost on the trace
  is +92 [+44, +166]/10k in B1 and +579/10k in B3, and is concentrated in round 0
  of high_stnum: D1 on A0's trace adds +245 to +1300/10k across the 5 B1 reps,
  and +1234/+1300 in B3. There was one outlier in masquerade rep2 (+3043). The
  placebo has a negative cost in almost all lineages, which explains why
  "cost − placebo" = +354 is larger than the cost itself.
- **B3 versus B1.** B3 closes a little more (+0.107 against +0.080) and
  ends in the same place (−0.225 against −0.228). But B3 has n = 2, in a single
  family: the tight interval comes from there being two units, not from
  precision. It is not good for comparing arms. In B3 the last point of the
  alarm cost is NaN because the primary point saturated.
- **Defender: 28 plans (8 from B3, 20 from B1), 27 accepted (96%).**
  "Recommend retraining 0.148" is 4 of 27: the measure counts `detector_retraining`
  **or** `dataset_enrichment` (`arms_race.RETRAINING_TECHNIQUES`), and all 4
  are `dataset_enrichment`. "Mean overlap 1.0" comes from **one** plan only
  (B1 rep2 round 1, `device_quarantine ← stDiff`), the only one that cites a
  feature. With n = 1, the mean says nothing about the Defender reading the
  importance change. What the data show is that it almost never cites a feature.

## E3-LLM extension to 5 repetitions (2026-11-09)

**Reason:** the review pointed out that E3's LLM arm, with n=2, does not
support a comparison with B1. The 2026-10-22 cut was therefore extended to
B3 × high_stnum × reps 1–5, with the same code, protocol and family. Reps 1–2
were not redone.

**Execution:**
- Command: `campaign --arm B3 --family high_stnum --reps 3 4 5`. It exited 0
  in a single day, with ~25k tokens per round, ~230k in total.
- Then, `matrix` of the same reps and `report` (backup of the previous one in
  `results.json.bak-20261109`).

**`E[r][r]` (soft evasion, test)**, rounds r0–r3:

| rep | B3 | B1 |
|---|---|---|
| 1 | 0.235 / 0.175 / 0.024 / 0.006 | 0.186 / 0.027 / 0.034 / 0.006 |
| 2 | 0.236 / 0.195 / 0.029 / 0.015 | 0.155 / 0.020 / 0.011 / 0.018 |
| 3 | 0.247 / 0.083 / 0.032 / 0.032 | 0.201 / 0.067 / 0.010 / 0.021 |
| 4 | 0.229 / 0.024 / 0.003 / 0.032 | 0.158 / 0.030 / 0.012 / 0.011 |
| 5 | 0.208 / 0.035 / 0.023 / 0.007 | 0.132 / 0.079 / 0.036 / 0.014 |
| mean | 0.231 / 0.103 / 0.022 / 0.018 | 0.166 / 0.045 / 0.020 / 0.014 |

**B3 alone (n=5):**
- closure: +0.085 [+0.066, +0.104];
- forgetting: −0.002;
- race: −0.213 [−0.223, −0.202];
- alarm cost on the trace: +383 [+181, +619] per 10k.

**B3 − B1 (paired by rep, high_stnum):**
- `E[0][0]`: +0.065 [+0.052, +0.077]. Separates.
- `E[1][1]`: +0.058 [−0.013, +0.132]. Does not separate: reps 1–2 found a new
  direction (0.18–0.20) and reps 3–5 did not (0.02–0.08).
- `E[2][2]`: +0.002, and `E[3][3]`: +0.004. Neither separates.
- Closure: +0.032 [+0.012, +0.051]. Race: −0.060 [−0.077, −0.044]. Both come
  from the higher starting point, not from a better defense.
- Alarm cost on the trace: +165 [+2, +367] per 10k. It separates by a narrow
  margin, and depends almost entirely on rep1 (797).

**B3 mean matrix:**
- D1 still leaves 0.07–0.10 on A1–A3.
- D2 holds A2/A3 at 0.022–0.023.
- A3 against D0 stays at 0.164 (A0: 0.231).

**Reading:** the LLM's advantage over B1 does not survive the retraining. It
exists in E1 (D0), becomes uncertain against D1 and disappears from D2 on.
