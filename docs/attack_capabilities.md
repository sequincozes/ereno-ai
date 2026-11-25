# Intent-driven capability catalog

Reference for what each attack exposes to the intent-driven pipeline
(`--engine intent`) — the allowlist between an `IntentSpec` and the fields that
`core/intent_compiler.py` may change. Source of truth:
`config/attack_capabilities.py` + `config/capabilities/*.py`. This document
is generated from the real catalog (see the command at the end) and must not diverge
from it.

## Conventions

- **`capability_id`**: `"<attack_key>.v<N>"` — one per attack registered in
  `config/attacks_registry.py`, including the 4 `delayed_replay*` variants
  (even though `delayed_replay_double_drop` is field-for-field identical to the base
  `delayed_replay`: sharing an id would make `AttackCapability.attack_key`
  lie for three of the four specs). The id goes into the `AttackCandidate`;
  the five catalogs pruned by E0 moved to `v2`, so that a candidate
  compiled against the old catalog remains distinguishable.
- **`polarity`**: `"direct"` (default) — larger value = more intense attack.
  `"inverse"` — larger value = **less** intense attack (gaps, intervals between
  bursts). The compiler inverts the effective direction of these fields, so that
  "evade detection" never means tightening a gap.
- **Effects**: `lower_f1`, `lower_recall`, `mimic_normal_traffic` (the three
  "evasion" ones — always with the same set of fields within a capability),
  `increase_attack_activity`. `increase_resource_pressure` exists in the contract
  but **no catalog advertises it** — see its own section below.
- **`minimum`/`maximum`**: the floor/ceiling that the compiler respects. It is the only
  defense against compiling the attack out of existence — the intent-driven pipeline
  does not run `shared/validator.py` (that belongs only to the legacy loop).
- **`{min, max}` pairs**: the floor/ceiling of each field does not protect the *pair*. A
  bound pushed against the frozen sibling would stop right on top of it, and
  `min == max` passes the catalog and the attack schema but ERENO refuses
  the whole run ("The lower limit must be less than the upper limit").
  The invariant is guaranteed structurally in the compiler
  (`intent_compiler._clamp_paired`, which treats the sibling as the effective bound instead
  of touching it), not field by field here — no capability declaration
  needs to worry about it. See `docs/pilot_e0.md`.

## Attacks and fields

### `masquerade_fault` (`masquerade_fault.v2`) — 11 of 12 fields

| field | type | pol. | bounds/choices | effects |
|---|---|---|---|---|
| `fault.prob` | number | direct | [0.0, 1.0] | evasion + activity |
| `fault.durationMs.min` | integer | direct | ≥0 | evasion |
| `fault.durationMs.max` | integer | direct | ≥0 | evasion |
| `cbStatus` | integer | direct | {0, 1} | evasion |
| `incrementStNumOnFault` | boolean | direct | {False, True} | evasion |
| `sqnumMode` | string | direct | {fast, normal} | evasion |
| `analog.deltaAbs.min` | number | direct | ≥0.0 | evasion |
| `analog.deltaAbs.max` | number | direct | ≥0.0 | evasion |
| `trapArea.multiplier.min` | number | direct | ≥0.0 | evasion |
| `trapArea.multiplier.max` | number | direct | ≥0.0 | evasion |
| `trapArea.spikeProb` | number | direct | [0.0, 1.0] | evasion + activity |

### `random_replay` (`random_replay.v2`) — 10 of 15 fields

| field | type | pol. | bounds | effects |
|---|---|---|---|---|
| `windowS.{min,max}` | number | direct | [0.1, 30.0] | evasion |
| `delayMs.{min,max}` | number | direct | [1.0, 2000.0] | evasion |
| `burst.prob` | number | direct | [0.0, 1.0] | evasion + activity |
| `burst.{min,max}` | integer | direct | [1, 50] | evasion + activity |
| `burst.gapMs.{min,max}` | number | **inverse** | [0.05, 1000.0] | evasion + activity |
| `reorderProb` | number | direct | [0.0, 1.0] | evasion |

> **Recorded uncertainty**: `delayMs.{min,max}` is tagged `direct` (less
> delay ≈ timing closer to normal), but a replay with almost no
> delay is also a more obvious duplicate — plausibly `inverse`.
> Resolve with an A/B experiment (`--generator-mode jar`, compare recall)
> rather than guessing.

### `inverse_replay` (`inverse_replay.v2`) — 9 of 12 fields

Same structural family as `random_replay`. `blockLen` replaces `windowS`;
`delayMs` is an **integer** here (vs. a float in uc01); no `ethSpoof`/`reorderProb`.
`burst.gapMs.{min,max}` is also `inverse`, for the same reason.

### `injection` (`injection.v1`) — 2 of 12 fields

| field | type | pol. | bounds/choices | effects |
|---|---|---|---|---|
| `numInjectedMessages` | integer | direct | [5, 5000] | evasion + activity |
| `injectionPattern` | string | — | {random, uniform} | evasion |

The floor of 5 on `numInjectedMessages` exists because the E4 gate
(`DatasetBundle`) requires `min_attack_rows=5` by default — a smaller value
would produce a candidate that the pipeline itself would reject in PREPROCESS.

### `high_stnum` (`high_stnum.v2`) — 5 of 11 fields, evasion only

| field | type | pol. | bounds/choices | effects |
|---|---|---|---|---|
| `jump.{min,max}` | integer | direct | [1, 1000] | evasion |
| `sqnumResetProb` | number | direct | [0.0, 1.0] | evasion |
| `sqNumDelta.max` | integer | direct | [0, 100] | evasion |
| `randomizeTimestamp` | boolean | direct | {False, True} | evasion |

The two activity levers (`count.lambda`, `ttlOverride.prob`) were removed
by E0, so `high_stnum` no longer advertises `increase_attack_activity`.

### `flooding` (`flooding.v1`) — 8 fields, full coverage

| field | type | pol. | bounds/choices | effects |
|---|---|---|---|---|
| `burst.{min,max}` | integer | direct | [1, 2000] | evasion + activity |
| `gapMs.{min,max}` | number | **inverse** | [0.01, 10.0] | evasion + activity |
| `stnumEveryPacket` | boolean | direct | {False, True} | evasion |
| `sqnumStrideValues` | integer_list | direct | — | **activity only** |
| `ttlMs` | integer | direct | [1, 1000] | evasion |
| `ethSrcSpoofProb` | number | direct | [0.0, 1.0] | evasion |

`gapMs` is the main lever of the attack — a **smaller** gap between packets is
what characterizes flooding, hence `inverse`. `sqnumStrideValues` only carries
`increase_attack_activity`: scaling the list elementwise by a factor &lt;1
(evasion) produces a degenerate multiset (`[1,2,4]` × 0.5 → `[1,1,2]`, duplicating
stride 1), so the evasion direction is not safe for this field.

### `grayhole` (`grayhole.v2`) — 5 of 12 fields

| field | type | pol. | bounds | effects |
|---|---|---|---|---|
| `dropRate.{min,max}` | number | direct | [0.05, 1.0] | evasion + activity |
| `burstDropProb` | number | direct | [0.0, 1.0] | evasion + activity |
| `burstDropLen.{min,max}` | integer | direct | [1, 100] | evasion + activity |

`dropRate.{min,max}` is semantically a probability whose leaves are called
`min`/`max` — the legacy generic clamp (which matches by the substring `"prob"`
in the key) does not reach it; here the bounds are explicit, with a floor of `0.05` (not
`0.0`) so that the compiler cannot reduce the drop rate to a point
where the attack ceases to exist physically.

### `delayed_replay` / `_backoff` / `_batch_dump` / `_double_drop` (uc10)

Fields common to the 4 variants:

| field | type | pol. | bounds | effects |
|---|---|---|---|---|
| `burstInterval.{min,max}` | integer | **inverse** | [10, 5000] | evasion + activity |
| `burstMax` | integer | direct | [1, 2000] | evasion + activity |
| `selectionProb.value` | number | direct | [0.05, 1.0] | evasion + activity |
| `networkDelayMs.{min,max}` | number | direct | [1.0, 5000.0] | evasion |
| `shiftSendTimestamp` | boolean | direct | {False, True} | evasion |
| `replaceWithFake` | boolean | direct | {False, True} | evasion |

Extras per variant: `_backoff` adds `rateMultiplier` (direct,
`[1.0, 10.0]` — `1.0` is "backoff off", the correct semantic floor, not
`0.0`); `_batch_dump` adds `microGapMs` (**inverse**, `[0.1, 100.0]`).
`_double_drop` adds nothing — the variant is dispatch-only, differentiated
only by the `attackType`.

`selectionProb.value` is a probability whose leaf key is `value` — the legacy
generic clamp does not reach it.

## Excluded fields and why

| field | attack(s) | reason |
|---|---|---|
| `orderBy` | `delayed_replay*` (4) | a string with no enum documented anywhere in the repo. `FieldCapability._string_fields_need_choices` would reject the declaration without `choices` at import time. Recoverable via archaeology on the JAR bytecode (`generator_runtime/ereno-generator.jar`), as was done for `sqnumMode` — decompile the uc10 creator class and find the `switch`/`equalsIgnoreCase` over the string. |
| `randomSeed` | `injection` | a reproducibility control, not an intensity lever. |
| `stNum`, `sqNum`, `cbStatus.values`, `ttlMs`, `confRev` (9 paths) | `injection` | they only have a physical effect when `injectionPattern == "synthetic"`; the baseline uses `"random"` (clones legitimate messages, ignores these ranges). Compiling a change to them would produce a diff with no physical effect — it would corrupt the E10 policy, which would measure "no improvement" and escalate a dead lever. Enabling them requires changing the baseline (`inputs/attacks/uc05_injection.json`) to `"synthetic"`, which would also change what `--engine live --attack injection` (legacy loop) optimizes — a data decision outside the scope of this catalog. |
| `sqNumDelta.min`, `padBytes.min` | `high_stnum` | baseline `0`, dead in both directions: `decrease` does not go below a floor of `0`, and `increase` without `maximum` is multiplicative (`0 × factor = 0`). Only the `.max` of each pair enters. |
| `protectStatusChanges`, `statusChangeDropProb` | `grayhole` | the probability is only consulted with the gate on (state-change messages are dropped with it instead of `dropRate`); the gate is `false` in the baseline and was never in the catalog, so the probability was unreachable — E0 measured it as inert. |

### Excluded by the E0 audit (2026-10-13)

The catalog test guarantees that each field changes the **config**; it does not guarantee that
the config changes the **trace**. `scripts/run_e0_sensitivity.py` generates each field
alone against the reference on the same seeds, and the fields below gave a
byte-for-byte identical trace. The choice was to prune the catalog, not to fix the
generator: fixing it would change the simulator to make the attack easier to
detect, and in the case of the TTL it would create an artifact.

| field | attack(s) | cause in ERENO-2.0 |
|---|---|---|
| `count.lambda` | `random_replay`, `inverse_replay`, `high_stnum` | no creator reads it; the number of attack messages is the per-segment budget of `CreateAttackDatasetAction` (50% prevalence whatever the value). |
| `ttlOverride.{valuesMs,prob}`, `ttlMsValues` | `random_replay`, `inverse_replay`, `high_stnum`, `masquerade_fault` | the TTL is applied to the message, but `Goose.copy()` does not copy `gooseTimeAllowedtoLive` and the creator emits the copy: the CSV always comes out with 11000. Fixing it would create an artifact — the benign traffic has a constant TTL, and `getFrameLen()` adds the TTL as text, which would leak into `frameLengthDiff`. The `prob` is not inert (it consumes draws), but has no effect outside the RNG. |
| `ethSpoof.{srcProb,dstProb}` | `random_replay` | the creator never reads it (and `Goose.ethSrc` is `static`). Swapping the MAC of a replay would make it less like the publisher. |
| `padBytes.max` | `high_stnum` | the creator never reads it; padding inside a BER PDU would make a malformed frame. |
| `extraDelayMs.{min,max}`, `delayBurstProb`, `delayBurstLen.{min,max}` | `grayhole` | `GrayHoleVictimCreatorC` never reads them; the messages that survive come out with no delay. Modeling the delay is a decision for the grayhole study itself. |

**Not audited yet:** `flooding.ttlMs` and `flooding.ethSrcSpoofProb` go through
the same two mechanisms (TTL via `copy()`, static `ethSrc`) and are
suspect; E0 did not cover `flooding`.

## `increase_resource_pressure` — why no catalog advertises it

It is the physically most natural effect for `flooding`/`grayhole`, but
`DetectionReport` has no metric that measures resource pressure — the only
candidates would be `latency_ms` (measures our own Python process, not the
GOOSE bus) or something derived from `DatasetBundle.class_counts`/rows per
second (it would require a second source in `decide_feedback`, a contract change
outside the scope of this delivery). Advertising the effect without a real metric would make
`core/feedback_policy.py` "choose an arbitrary metric and pretend to measure" —
exactly what the module documents that it does not do. A campaign that asks for this
effect is rejected at the entry gate (`resolve_candidate_paths`) with an
actionable message, instead of dying late in round 1 with
`NO_OBJECTIVE_METRIC`. `increase_attack_activity` covers the physically
representable half of the request in the same fields (`burst.*`, `gapMs.*`,
`dropRate.*`) — the loss is lexical, not physical.

## Cached mode limitation

In `--generator-mode cached` the dataset served is always
`data/baseline_dataset.csv`, whose attack class is `masquerade_fake_fault`.
The E4 integration gate (`build_dataset_bundle`,
`expected_attack_label=spec.label`) requires that label to be present in the
trace — so **the 10 attacks that are not `masquerade_fault` are only
physically measurable in `--generator-mode jar`**; in cached mode the
pipeline stops at the PREPROCESS stage with a gate error, not a crash.
The tests (`tests/test_intent_loop_orchestrator.py`) synthesize a minimal CSV
per label to exercise the seven stages without Java.

The 4 `delayed_replay*` variants share `label="delayed_replay"` in the
registry — the E4 gate does not distinguish them from each other by class.

## How this document was generated

```bash
uv run python -c "
from adversarial_ids.config.attack_capabilities import ATTACK_CAPABILITY_CATALOG
from adversarial_ids.config.attacks_registry import list_attack_keys, get_attack_spec
for k in list_attack_keys():
    cap = ATTACK_CAPABILITY_CATALOG[get_attack_spec(k).intent_capability_id]
    print(cap.capability_id, len(cap.fields))
"
```

`tests/test_attack_capabilities_catalog.py` keeps this catalog honest —
any dead lever in the config, undocumented path or effect with no field
that produces it fails there before it gets here. A lever that is dead in the **trace** (the
config changes, the generator ignores it) only shows up by running the generator:
`uv run python scripts/run_e0_sensitivity.py` (needs the JAR).
