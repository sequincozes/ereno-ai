# Pilot E0 — variant validity per attack family

E0 is the first experiment and produces no result about the IDS:
it decides **which attack families can be used** in the following
experiments, and at what cost. The plan asks for three things — generate the baseline and
variants of `masquerade_fault`, `random_replay` and `grayhole`; confirm that
changing the parameters alters the traffic while preserving the attack; estimate time and
tokens to fix the budget — and explicitly asks that a failed family have its
cause recorded *before* the scope shrinks.

It is run by `scripts/run_pilot.py`:

```bash
uv run python scripts/run_pilot.py                      # the three families (~90s, needs the JAR)
uv run python scripts/run_pilot.py --attack grayhole --replicates 5
uv run python scripts/run_pilot.py --llm-probe          # includes the token cost of the intent stage
uv run python scripts/run_pilot.py --llm-probe-only     # token cost only, without generating a trace
uv run python scripts/run_pilot.py --attack grayhole --max-fields 11 --label campos11  # separate run
```

Output: `outputs/pilot/pilot_report.json` (plain JSON with `schema_version`; it is
an experiment artifact, nothing in `src/` reads it) and, per family,
`outputs/pilot/<ataque>/` with each trace, the config that generated it and a sample of
attack rows for manual inspection. Exit code 0 if all requested families
passed, 1 if any failed.

## The variants do not come from an LLM

Each variant comes from an `IntentSpec` compiled by `core/intent_compiler.py`,
one per intensity (`low`/`medium`/`high`), under the same `desired_effect`. This is the
mechanism the feedback policy (E10) uses from round 2 onward — that is, the
pilot measures exactly the variation generator the campaign will use, without
spending a model call and without introducing LLM variability into the measurement.

The token cost comes in through another path: `--llm-probe` makes **one** real call
to the `IntentAgent` per family just to record consumption (`--llm-probe-only`
measures without generating any trace). The calls are spaced 90s apart because the prompt
with the injected capability catalog weighs 6.66k tokens and the account's `on_demand` tier
has a TPM of 8000 — two calls in the same window become `rate_limit_exceeded`,
which shows up from the outside as "the LLM did not call submit_intent_spec".

## Three measures, and what each one lets through

**1. Volume gate (E4).** `build_dataset_bundle` over each trace: classes
present, counts, prevalence of the attack class. It catches an empty or broken trace,
one without the expected class, or one too rare to measure. It says nothing about
whether the attack is active.

**2. Signature.** Standardized difference of means between the attack rows and
the normal rows **of the same trace**, per feature. A trace can have 50% of its
rows labeled as attack and still be indistinguishable from normal traffic:
the label comes from the generator, not from physics. The report gives the peak, how many
features exceed the threshold, and the ranking — because a huge separation in a single
column and a moderate separation in eight are different evidence about the same
attack.

This is statistical evidence that the class is **distinguishable**, not proof of
preserved semantics. That is why the script also saves
`<tag>_sample.csv`: inspecting the messages is part of E0 and no statistic
replaces it.

**3. Drift against the generator's noise.** This is the measure that justifies the pilot
existing, and the one that took the most work.

## The generator is not deterministic (without `randomSeed`)

> **Update (Phase 2.0, 2026-09-26):** this held because the creators
> drew from `Math.random()`, outside the seeded RNG. Phase 2.0 routed those
> draws through `ConfigLoader.RNG` and made `GeneratorRunner` write a
> `randomSeed` into the action config — with a fixed seed, the same config now
> produces **the same bytes** (verified: `seed=123` gave an identical hash in two
> runs). The text below describes the behavior **without** `randomSeed`, which
> remains the default outside the intent-driven pipeline. See
> `docs/phase2_plan.md`.

The same configuration, run twice **without `randomSeed`**, produces **different
bytes** (measured on 2026-09-17 on `random_replay`: two distinct `content_hash`
values for the same JSON). Direct consequence: "the variant's hash is
different from the baseline's hash" is not evidence of anything — it would differ
either way.

That is why the pilot generates the baseline `--replicates` times (default 3) and measures the
separation **between replicates of the same config**. That is the generator's noise, and it is the
floor that any drift needs to exceed.

The first version used a single floor: the maximum of that separation over all
features. It did not work. The maximum over ~40 columns is the maximum of noisy
estimates; it itself oscillates between runs, and with it the family's verdict
oscillated too — on 2026-09-18, `grayhole` came out 3/3 in one run and 1/3 in the
next, with nothing having changed between the two.

The current version compares **column by column**: the noise of `timestampDiff` is the floor
of `timestampDiff`. A feature counts as moved when its drift exceeds
`--min-drift` (absolute floor, default 0.05 — without it, a column with almost no noise
would pass with any shift) **and** `--drift-margin` times the noise of its
own column (default 2x). The variant is distinct from the baseline when at least
`--min-moved-features` columns (default 1) move. With this, the per-family verdict
started repeating across identical runs.

Two deliberate exclusions: identity columns and absolute clock
(`always_drop_for`, the same drop as the preprocessor) are left out of the
measurement, otherwise the "separation" would measure the generation block and not the attack; and a
column that the generator itself shifts entirely between replicates goes into
`unusable_features` — it cannot serve as evidence for anything.

The default of `--protocol-features` here is `deltas`, and not the `drop` used by the rest of the
repo: without `stDiff`/`sqDiff`/`tDiff`/`timestampDiff` there is nothing left on which replay
and grayhole could have a signature.

## Result of the 2026-09-18 run

Three families, effect `lower_recall`, three intensities, three baseline
replicates, 18 generations: 88s generating (~4.9s per trace), 891 MB on disk.

| family | verdict | distinct variants | what carries the drift |
|---|---|---|---|
| `masquerade_fault` | passed | 3/3 | `cbStatus`, `cbStatusDiff`, `SqNum`, `sqDiff`, `stDiff` (6–7 features) |
| `random_replay` | passed | 3/3 | electrical ones in `low`/`medium`; `timestampDiff` and `timeFromLastChange` in `high` |
| `grayhole` | failed | 2/3 | only `timestampDiff`, at the edge of the noise |

The first run, before the fix described below, had also failed
`random_replay`.

### Fixed: the compiler was closing intervals

The first round of the pilot killed the `high` variant of `random_replay` in the
generator: the compiler produced `windowS = {min: 2.0, max: 2.0}` and ERENO
refused the entire process with *"The lower limit (2000) must be less than the
upper limit (2000)"*. The pair clamp preserved `min <= max`, the attack's schema
validates `min <= max`, the capability gate does not look at pairs — a
degenerate interval passed through everything the repo has and died in Java.

It was not an isolated case: sweeping every field of every attack alone, the old clamp
produced a degenerate pair in **22 distinct pairs, in 10 of the 11 registered
attacks** (`windowS`, `burst`, `gapMs`, `dropRate`, `burstInterval`,
`networkDelayMs`, `jump`, `blockLen`, `analog.deltaAbs`, ...). It hit the
E1 campaign the same way it hit the pilot.

`intent_compiler._clamp_paired` now guarantees a **strict** `min <`: when the
computed value would cross its sibling, **the sibling becomes the effective bound** and the same
intensity step is reapplied against it. No epsilon is invented — the
distance comes from the escalation policy that was already in use — and the ladder stays
monotonic (`windowS.max` now gives 6.42 / 4.05 / 2.9 at low/medium/high, against
a floor of 2.0). For integer pairs, when rounding sticks to the sibling the
value steps back one unit; if even that does not fit, the field does not move and disappears from the
`diff`, which now tells the truth about what changed.

The invariant is swept in `tests/test_intent_compiler_ranges.py`: every attack,
every effect, every intensity, **and each field compiled alone** — which is the
dangerous case, a bound pushed against a frozen sibling.

### What still stands

**`grayhole` is the fragile family, on both axes.** Its signature against normal
traffic is 0.80 and **a single feature** exceeds the 0.2 threshold (against 14
in `masquerade_fault` and 7 in `random_replay`). The variation is also thin: only
`timestampDiff` moves, and the `low` variant falls within the generator's noise —
it is the one that fails the family. The `lower_recall` effect pushes
`dropRate`/`burstDropProb` down, that is, it makes the attack more discreet; the
search space that remains is narrow.

**In `random_replay`, only the high intensity affects timing.** The `low`
and `medium` variants move 11 features, all **electrical** (`vsbARmsValue`,
`isbARmsValue`, trap areas): changing the replay window changes *which* legitimate
messages are recaptured, and with them the analog values carried. Only at
`high` — the variant that the fix above unblocked — does the drift reach
`timestampDiff` and `timeFromLastChange`, which is where a replay should show up.
Attributing the drift of `low`/`medium` to the attack's timing would be misreading
what the measurement says.

## The grayhole is squeezed from both sides

The failure of `grayhole` with `lower_recall` and 3 fields was not enough to cut
the family — it could be that the pilot ran too narrow. Two diagnostic runs
(`--label`, which separates the directories so as not to overwrite the E0 traces)
settled the question, and the answer is more interesting than "passes" or "does not pass":

| run | fields | prevalence low→high | drift low/medium/high |
|---|---|---|---|
| `lower_recall`, 3 fields | 3 | — | 0.045 / 0.073 / 0.163 |
| `lower_recall`, 11 fields | 9–11 | 0.27 → 0.41 | 0.039 / 0.084 / 0.115 |
| `increase_attack_activity` | 6 | 0.13 → **below the floor** | 0.108 / 0.063 / — |

The two directions move the prevalence in **opposite** senses, and both have
a wall:

- Pushing toward **evasion** means dropping less. More messages labeled as attack
  remain (0.23 at baseline → 0.41), but the traffic looks more and more like the
  baseline — the `low` variant falls within the generator's noise and it is the one that fails the
  family.
- Pushing toward **activity** means dropping more. At high intensity
  115 attack rows remain out of 50,112 (0.23%), and the E4 prevalence gate
  rejects the dataset: a recall computed over 115 samples is sampling noise.
  The grayhole's attack class disappears precisely when the attack becomes
  more aggressive, because its rows *are* the messages it did not drop.

And in every configuration tested — 3, 6, 9 or 11 fields, in both directions — the
drift concentrates in **a single feature**, `timestampDiff`. The family is
usable, but in a narrow band and on a single axis: with 11 fields, `medium` and `high`
are distinct from the baseline and monotonic. E1 can use it as long as it does not
count on the `low` intensity or on the activity direction.

### Caveat: three creators read fields from the wrong place (fixed on 2026-10-13)

The numbers above were measured with a JAR in which three configurable creators
called `getNestedRangeMinInt(parent, child)` — which reads `parent.child.min` —
for fields that are a **top-level** range (`blockLen`, `burstDropLen`) or
an integer **inside** an object (`burst.min`/`burst.max`). The path never
existed and the default silently kicked in:

| attack | field | effect in the old JAR |
|---|---|---|
| `inverse_replay` (uc02) | `blockLen` | default 1..1 → `randomBetween(1,1)` → generation always aborted (the "returncode=3") |
| `inverse_replay` (uc02) | `burst.{min,max}` | default 0 → the burst never fired |
| `random_replay` (uc01) | `burst.{min,max}` | same: with `burst.prob=1.0`, `bursts=0` in the creator's log (the fixed JAR, same seed: `bursts=1926`) |
| `grayhole` (uc08) | `burstDropLen.{min,max}` | always 3..7, whatever the config (the baseline coincides with the default) |

That is: in `random_replay` the entire `burst.*` block was inert, and in `grayhole`
`burstDropLen` too. Without a configured burst (`burst.prob=0`), the fixed JAR
generates the same bytes as the old one.

Re-running with 11 fields exposed two more bugs, which also brought generation
down (commit `46b0039` in ERENO):

- `delayMs` below 10 was converted from seconds **bound by bound**:
  `{min: 8, max: 80}` became 8000..80 ms. The unit is now decided once
  for the pair, by the `max`.
- `randomBetween(int, int)` is inclusive at both ends, but it refused
  `lower == upper`. A replay window smaller than one GOOSE interval (the
  `windowS.min = 0.385` of the `high` intensity) or a one-element `ttlOverride.valuesMs`
  aborted the run.

With a fixed seed, the baseline of the 12 registered attacks is byte-for-byte identical
before and after the two fixes.

## Re-run on 2026-10-13, with the fixed JAR

Same cuts as 2026-09-18 (labels `posfix*`; the old reports remain in
`outputs/pilot/`), plus `inverse_replay`, which had never generated, and a
replay cut with 11 fields, the only one in which the compiler chooses the `burst`
fields.

| cut | family | 2026-09-18 | 2026-10-13 | what changed |
|---|---|---|---|---|
| default (3 fields) | `masquerade_fault` | passed, 3/3 | passed, 3/3 | nothing — control: the creator was not touched |
| default (3 fields) | `random_replay` | passed, 3/3 | passed, 3/3 | nothing: with 3 fields the compiler chooses `delayMs`/`ethSpoof`/`windowS`, never `burst` |
| default (3 fields) | `grayhole` | failed, 2/3 | failed, 2/3 | nothing: `burstDropLen` is not among the 3 fields |
| default (3 fields) | `inverse_replay` | did not generate | passed, 3/3 | only `timestampDiff` moves |
| 11 fields | `grayhole` | failed, 2/3 | failed, 2/3 | the failing variant switched from `low` to `medium`; the drift remains only in `timestampDiff` |
| activity (6 fields) | `grayhole` | failed, 2/3 | failed, 1/3 | **worse**: with `burstDropLen` active the class disappears earlier (`medium` already has 0.47% prevalence) |
| 11 fields | `random_replay` | — | passed, 3/3 | **15 features at every intensity**, including `timestampDiff`, `timeFromLastChange`, `SqNum`, `sqDiff` |
| 11 fields | `inverse_replay` | — | passed, 3/3 | `low` moves 1 feature; `medium`/`high` move 4 (`timestampDiff`, `tDiff`, `stDiff`) |

What this does to the previous readings:

- **"In `random_replay`, only the high intensity affects timing"** was
  true of the 3-field cut, not of the family. With `burst` active,
  timing and sequence already move at `low`. The sentence can only be quoted
  together with the cut.
- **"The grayhole is squeezed from both sides"** still holds, and the wall on the
  activity side got closer: `burstDropLen`, now effective, drops more, and
  the attack class falls below the prevalence floor already at `medium`. The drift
  remains in a single feature, `timestampDiff`, in every cut.
- **`inverse_replay` is usable**, with the same narrow shape as the grayhole in the
  3-field cut (a single feature) and wider with 11.

A measurement caveat: the pilot runs **without a seed**, as on 2026-09-18, to measure the
generator's noise. A variant near the threshold can change verdict between
runs (which is what happened with `low`/`medium` of the grayhole with 11 fields).
The per-family verdict repeated; that of an isolated variant should not be
quoted on its own.

## Measured budget

**Generation:** ~4.9s per trace and ~45 MB per trace in the JAR (18 generations = 88s and
891 MB). It is the dominant cost in disk and the lowest in money.

**`intent` stage:** 6,657 input tokens and 228–259 output tokens per call,
measured on 2026-09-18 with `openai/gpt-oss-120b` over the three families
(`--llm-probe-only`). Practically all of it is the prompt: the injected capability
catalog. That is **~6.9k tokens per call against a TPM of 8,000** — one
call per minute is the account's ceiling, and the three only went through because the probe spaces them
90s apart. It is an account constraint, not a code constraint, and holds for any
campaign: one round per minute, in the best case.

The `Defender`'s share has not been measured yet — it depends on the round's
`DetectionReport` and only shows up in a full cycle, in `LoopRecord.total_tokens`.

## How to read the report

Per family: `verdict`, `reasons` (empty when it passed), `noise_floor`
(`by_feature`, `unusable_features`, the replicate pairs), `replicates` and
`variants`. Each variant carries the compiled `diff` (which fields changed, from what
value to which), the volume gate, the signature and the `drift` block with
`moved_features` — the list that supports the verdict. `budget` closes with generations,
time per trace, disk and, if `--llm-probe` ran, the intent stage's consumption.

Running in `--generator-mode cached` executes the script but pilots nothing: every
trace is the same file, the noise is zero by construction and no variant
is distinguishable. The script warns and carries on.

## Side effect of any run with the JAR

`GeneratorRunner` rewrites `generator_runtime/config/actions/` and
`generator_runtime/config/attacks/` at every generation, and those files are
versioned. After running the pilot, `git status` shows the last compiled
variant inside them — `git checkout generator_runtime/` before committing.
It is not specific to the pilot: it holds for any run in `jar` mode.
