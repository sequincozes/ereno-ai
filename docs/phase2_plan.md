# Phase 2 — attack behaviors outside the catalog

Living plan. Written on 2026-09-26, after the `intent-driven-v0.2.0` tag (Phase 1,
dictated values). Decisions from this planning round are marked **[decided
2026-09-26]**.

## Goal

Today the user chooses **which** of the 11 ERENO attacks, with **which** effect and
intensity (Phase 0/1), and since Phase 1 can **dictate exact values** for that
attack's fields. What they still cannot do is describe a **behavior
that does not exist** among the 11 — "drop only the trip messages and resend the
previous state 200 ms later" does not map to any cataloged attack.

Phase 2 delivers a **programmable attack**: a generic attack in ERENO driven
by **declarative rules in JSON**. A new behavior becomes **validated data, not
new Java code**.

## Guardrail (the same as Phase 1, extended)

The LLM emits the **rules** (typed data); deterministic Python and a schema in
Java decide whether they are valid and execute them. **No Java is written by the
LLM.** It is the natural extension of the central rule: in Phase 1 the LLM proposed a value and
the catalog decided whether it existed; here the LLM proposes a rule and the schema decides
whether it is executable. An impossible behavior **fails**, it is not silently adjusted.

## Stack

Unchanged and it stays that way: agents in **Agno 2.7.2** over
**Groq**; deterministic core in Python; generator in Java (ERENO). Phase 2 does not
introduce a new agent framework.

## The reality of the two repositories

The Java work is in the sibling repo `../ERENO-2.0`, **branch `feat/journal`** (created
from `dev-ian`, which is the branch the current JAR was compiled from — see the
memory note `ereno-jar-origem`). `ereno-ai` consumes the compiled JAR at
`generator_runtime/ereno-generator.jar`.

**Build discipline (non-negotiable):** when recompiling the JAR, compare the output
against the current JAR **before** swapping it in `generator_runtime/`. The 11
existing attacks must come out byte-for-byte identical for the same configs — the
E0 pilot results and the golden's byte-stability depend on it. The tag
`intent-driven-v0.2.0` is the safe return point on the Python side; on the Java side,
note the `dev-ian` commit that generated the current JAR.

## What the 2026-09-26 investigation found

Survey of the ERENO source (`feat/journal`) and the Python side:

1. **The seed infrastructure already exists.** `ConfigLoader.RNG` is a
   `ThreadScopedRandom`; the `randomSeed` config calls `setSeed()`; and
   `IED.randomBetween` already draws from that RNG. **But 6 creators bypass the RNG with
   `Math.random()`** (22 occurrences): uc01 random_replay, uc02 inverse_replay,
   uc03 masquerade_fault, uc05 injection, uc06 high_stnum, uc08 grayhole. This is the
   cause of the nondeterminism E0 measured (two distinct `content_hash` for the
   same config). The fix is targeted, not a rebuild.
2. **Adding an attack is mechanical, but touches several points.**
   - Java: a `case` in `CreateAttackDatasetAction.createAttackDevice` (it is a fixed
     `switch` on `attackType`); an IED class + a Creator (the interface is
     `MessageCreator.generate(IED, int)`); a label in `Labels.LABELS` and a branch
     in `getLabelForSegmentName` (maps segment→label by `ucXX` prefix — the
     programmable one needs a free prefix, e.g. `uc11`); a segment in
     `config/actions/action_create_attack_dataset.json`.
   - Python: `AttackSpec` in `config/attacks_registry.py` (key, segment_name,
     label, description, intent_capability_id); baseline in
     `inputs/attacks/`; frozen model in `domain/attack_configs/`; capability
     module in `config/capabilities/`.
3. **Mutation surface = what `Goose` exposes.** Available setters:
   `setCbStatus`, `setStNum`, `setSqNum`, `setT`, `setGooseTimeAllowedtoLive`,
   `setSvPrefixCsv`, `setLabel`, `setConfRev`, `setPublisherTxTs`,
   `setSubscriberRxTs`, timestamp. The mutation DSL targets this set.
4. **`AttackConfig` already reads arbitrary JSON** via `getRaw()` (Gson's `JsonObject`)
   — a generic Creator can read a list of declarative rules without
   new parsing infrastructure.
5. **TPM constraint.** The IntentAgent prompt is at ~6.24k/8000 tokens. The
   programmable attack **cannot** enumerate fields in the catalog like the other 11
   (that is what blew the TPM in Phase 1) — it enters with a compact description of the
   rule grammar.

## Sub-phases

Each one is deliverable and verifiable in isolation.

### 2.0 — Determinism (foundation) **[DELIVERED 2026-09-26]**

Route the creators' `Math.random()` calls through `ConfigLoader.RNG`, and propagate
`IntentSpec.seed` → `GeneratorRunner` → `randomSeed` in the action config →
`ConfigLoader.setSeed`. Before, the intent's `seed` existed in the contract but did not
reach the generator.

- **Java side (`../ERENO-2.0`, `feat/journal`):** the 22
  `Math.random()` calls replaced by `ConfigLoader.RNG.nextDouble()` in 5 creators (uc01, uc02,
  uc03, uc06, uc08). uc05 was already deterministic (it uses its own `randomSeed` +
  `randomBetween`, which goes through the RNG). `CreateAttackDatasetAction` now reads
  a `randomSeed` from the top of the action config and calls `ConfigLoader.setSeed` before
  generating.
- **Python side:** `GeneratorRunner` gained `random_seed` (default `None`,
  compatible) and writes/removes `randomSeed` in the action config when selecting the
  segment; `intent_loop._build_generator` passes `intent.seed` (the round's baseline and
  variant share the seed, paired comparison).
- **DoD — verified:** the same config (masquerade_fault) run twice with
  `seed=123` gave an **identical hash** (`dc67e1cb…`, 80019 rows); with `seed=999`,
  a different hash. A smoke test on 4 of the 5 changed attacks (random_replay,
  masquerade_fault, high_stnum, grayhole) produces the attack class normally.
  The distribution is preserved by construction (`Math.random()` and
  `RNG.nextDouble()` are both uniform on [0,1)); the difference is only the RNG
  source. 1476 Python tests pass.
- **Side finding:** `inverse_replay` (uc02) fails at generation (`returncode=3`) —
  **but fails identically on the old JAR (v2.0)**, so it is a pre-existing bug, outside the
  set validated in E0, not a regression of this phase. Record as a separate item.
- **Closing items for 2.0:** commit the new JAR in `generator_runtime`
  (gitignored — swapped locally, backup in scratch); update
  `docs/pilot_e0.md` (the section "The generator is not deterministic" now has
  a caveat: it only holds without `randomSeed`).

### 2.R — Replicate mode (seed sweep) **[DELIVERED 2026-09-26]**

**Usage:** `uv run adversarial-ids --engine intent --generator-mode jar --prompt
"..." --replicates 5` (sweeps `42..46`) or `--seeds 42,101,777`. The LLM is called
once; the summary gives mean ± standard deviation of the objective metric.

**Verified (JAR + Groq, 2026-09-26):** 3 replicates (seeds 42/43/44) → **1 LLM
call**, three **distinct** `content_hash` values (the seed reaches the generator), all three
reached FEEDBACK. `masquerade_fault` gave recall 1.0 on all three (a metric not
sensitive for that config; the variance shows up in evasive configs). Built-in fix: the
intent LLM consumption now counts only in the round/replicate that actually
called the LLM — before, a round that reused the intent added the tokens again
(a latent bug also in round 2+ of a campaign). 1497 tests pass.

The text below is the original design.

---


Counterpart of 2.0. 2.0 made each seed reproducible; 2.R uses that to
run **N seeds of the same experiment** and report the objective metric as
**mean ± standard deviation**, which is what a statistical claim needs (a
single trace is a sample, not a measurement). Independent of 2.1/2.2/2.3 — it applies to
the 11 current attacks and to the programmable one.

- **Insight the architecture already offers (TPM savings):** between replicates, only
  the **generation** changes; the **intent** is the same (same prompt → same
  `IntentSpec`). So the replicate calls the LLM **once** and reuses the compiled
  `IntentSpec`, varying only the `seed` passed to `GeneratorRunner`. N
  replicates cost **1** intent call, not N — decisive with the TPM of 8,000
  (see `docs/target_values.md`/E0). It is the same intent↔generation separation that already
  underpins resumption and the feedback policy (the Red side is deterministic from
  round 2 on).
- **Where it lives:** a thin controller that, given an already-compiled `IntentSpec`,
  iterates over a list of seeds and calls the deterministic core
  (`GeneratorRunner` → `IdsEvaluator` → `DetectionReport`) per seed. Each replicate
  is its own `LoopRecord` (`seed` is already a field on it), linked by a batch id. The
  aggregation (mean ± standard deviation of the objective metric, and the interval) reads
  the ledger — no new "aggregated result" contract until one is needed.
- **Interface:** `--replicates N` (sweeps `seed, seed+1, …`) or `--seeds a,b,c`
  (explicit list) on `--engine intent`; alternatively a
  `scripts/run_replicates.py` to keep the CLI lean. To be decided in 2.R.
- **Relation to the campaign (E10):** a replicate is **one round** repeated under
  different seeds — not a multi-round campaign. The two are orthogonal:
  each round of a campaign can be replicated, but the first cut replicates only
  round 1 (the prompt's intent).
- **DoD:** `--replicates 5` over a JAR-measurable attack produces 5
  `LoopRecord`s with distinct seeds, **one** LLM call in the batch, and a summary
  with mean ± standard deviation of the objective metric. Rerunning the batch with the same list of
  seeds reproduces the same numbers (2.0 guarantees this).
- **Caution:** it only makes sense in `--generator-mode jar`. In cached mode, every trace is
  the same file and the replicates collapse — the mode must warn, as the pilot already
  does.

### 2.1 — Vertical slice: programmable attack, mutation-only DSL **[DELIVERED 2026-09-26]**

See `docs/programmable_attack.md`. Delivered end to end: `attack_key
"programmable"` (uc11) in Java (creator/IED reading `rules` as a dict of slots +
label + switch) and in Python (AttackSpec, baseline, schema, capability, playbook).
The compiler materializes the rules **with no new code**: it reuses Phase 1's
`target_values` — the capability exposes `rules.rN.op/field/value` as author-only
fields (empty effect, never swept by the heuristic) and `rules.rN.fraction` as the
intensity lever. **Verified with a real LLM:** "multiply sqNum by 3
in ~40%" → the LLM chose `programmable` and authored the rule; the JAR generated the
`programmable` class. 1537 tests pass. Scope: mutation only (selection/timing is
2.2). Documented nuance: the baseline brings 2 slots with a default rule.

The text below is the original design.

---


A new `attack_key = "programmable"`, with **one** rule type: mutation of a
GOOSE field (`set` / `add` / `scale`) on a fraction of the messages. Wired end
to end.

- **Java:** `uc11` (free prefix) — `ProgrammableIEDC` + `ProgrammableCreatorC`
  that reads `rules: [{op, field, value, fraction}]` from `AttackConfig.getRaw()`,
  applies them via `Goose`'s setters, and validates each rule against an allowlist of
  mutable fields (the Java schema); `programmable` label in `Labels`; case in the
  switch; `uc11_programmable` segment in the action config.
- **Python:** `AttackSpec("programmable", "uc11_programmable", "programmable",
  ...)`; minimal baseline in `inputs/`; `domain/attack_configs/programmable.py`
  (frozen model of the rule list); `config/capabilities/programmable.py`
  (the rule grammar as a capability — which fields, which ops, ranges);
  the `intent_compiler.py` compiler learns to materialize rules (not just
  adjust scalar fields).
- **DoD:** a prompt that describes a behavior that is **not** one of the 11 (e.g.
  "duplicate 10% of trip messages with stNum incremented") compiles to
  rules, generates a trace, passes the E4 gates and produces a `DetectionReport`. The
  loop closes.

### 2.2 — Widen the DSL: selection and timing **[DELIVERED 2026-10-01]**

**Selection** rules (a condition on a message field) and **timing** rules
(delay, drop, duplication, reordering), each type as data validated by schema
at both ends.

**Selection** came in as a `when = {field, cmp, value}` block in each slot, with
`cmp ∈ {always, eq, ne, gt, lt, gte, lte}`. The condition is read from the **original**
message, before any mutation — which message a rule targets cannot
depend on what the previous slot already changed, or the slot order would become part of the
behavior. `when` is mandatory in the schema and the baseline brings `cmp="always"`:
`target_values` applies a value to an existing path and does not create a missing key,
so an optional `when` would never be authorable, and a non-neutral default would change the
default attack.

**Timing did not get an op of its own, and that is the finding of the phase.** The written
stream is not in the order the creator emits: messages leave through a
`PriorityQueue<EthernetFrame>` ordered by `timestamp`, and the deltas
(`timestampDiff`, `tDiff`, `delay`) are computed in that order. Any buffer or
position swap in the creator would be erased by the queue. So the clock *is* the order:
with `t`/`timestamp` (in seconds) in the mutable-field allowlist, **delaying is
`add` with a positive operand and reordering is the same `add` with a negative operand**.
Only the ops that touch *cardinality* rather than the clock remained as new ops:
`drop` and `duplicate`. `set`/`scale` are refused on the time fields
(absolute clock: they relocate the message instead of shifting it) — a coherence
*between* fields of the same rule, which lives in the schema, because no
field-by-field allowlist can express it.

- **Prompt cost, and the ceiling that was blown.** The phase added ~1.6k characters to the
  prompt, and validation with a real LLM hit the account's 8000 TPM: a
  request measured at **8467 tokens**, HTTP 413. Spacing out does not help when a
  single request exceeds the limit — and of the four calls, two passed at
  8461 and 8059, that is, the ceiling had been cleared by luck. It forced pulling the
  catalog compaction (a 2.3 item) forward to now: fields identical across more than one
  attack became a shared section cited by path, effect labels
  became `ev`/`at`/`ev+at` and type labels `num`/`int`/`bool`/`str`/`int[]`.
  Instructions went from **23448 → 20063 characters**; measured afterwards: 7300, 7414 and 7508
  tokens on the same prompts. It is the same trade that `_render_effects` already made —
  say it once, with a legend, and let the gate be the authority.
- **DoD reached (JAR, seed 4242):** selection + delay landed only on the selected
  messages (minimum of `timestamp - t` exactly 0.02 on them, 0.00 on the
  others); `duplicate value=2` gave exactly 3× the rows; `drop fraction=0.4`
  left exactly 60%; and the mutation-only config reproduced just 2.1. Regression
  of the other attacks verified by CRC: of the JAR's 281 classes, only
  `ProgrammableCreatorC.class` changed. Detail in `docs/programmable_attack.md`.

### 2.3 — Prompt/TPM, validation and release **[DELIVERED 2026-10-01]**

The first two items shipped inside 2.2, out of necessity: validation with a real
LLM blew the TPM and only fit after compacting the catalog.

- ~~Compact programmable entry in the IntentAgent catalog~~ — done in 2.2
  (instructions 23448 → 20063 characters).
- ~~Validation with a real LLM (`openai/gpt-oss-120b`)~~ — done in 2.2, four
  axes, all exact.
- **Golden prompts** (`tests/golden_prompts/programmable_intents.json`): 12
  entries in **three** categories, not two. The other attacks only need
  `valid`/`invalid` because in them every impossible request dies at the intent
  gates — the impossibility is always about a single field. In the programmable one there is a
  third: `op=scale` and `field=timestamp` pass field by field and the combination
  does not exist, so the one that refuses is the schema, inside the compiler. The
  `incoherent` category pins this division of labor in a test and not only in prose.
  The five `valid` entries marked `source: "llm"` are the payloads the real model
  produced, not invented payloads — which is what makes the validation repeatable
  without spending Groq on each `pytest`.
- `docs/programmable_attack.md`; tag `intent-driven-v0.3.0`.

## Risks

- **DSL scope.** Too general is impossible to validate and to fit in the prompt;
  too narrow does not deliver "new behaviors". Mitigation: vertical slice
  2.1 with a single rule type. **[decided 2026-09-26]**
- **TPM.** Each rule type pays tokens in the IntentAgent prompt (~6.24k/8000
  today). Budget by increment; Phase 1's effect compression bought headroom,
  but it is finite.
- **Regression of the 11 attacks on recompilation.** Mitigation: byte-for-byte comparison
  of the JAR before swapping, all 11 identical.
- **New label misaligned between Java and Python** (index in `Labels` vs.
  `AttackSpec.label`). Mitigation: a test that matches the two.

## Status

- Phase 0/1 delivered; tags `intent-driven-v0.1.0` (loop + resumption) and
  `intent-driven-v0.2.0` (dictated values). 1476 tests pass without Groq/Java.
- Java branch `feat/journal` created from `dev-ian`; dev-ian's local configs in
  `git stash` (see memory note `ereno-jar-origem`).
- **2.0 (determinism) DELIVERED** — reproducible seed verified; JAR recompiled
  and swapped in `generator_runtime`.
- **2.R (replicate mode) DELIVERED** — `--replicates`/`--seeds`, 1 LLM call
  per batch, verified with JAR + Groq.
- **2.1 (programmable attack, mutation DSL) DELIVERED** — `programmable` (uc11)
  end to end, rule authoring via `target_values`, verified with JAR +
  Groq. See `docs/programmable_attack.md`.
- **2.2 (selection + timing) DELIVERED** — per-slot `when`, `t`/`timestamp` in the
  allowlist (delay and reordering are the same signed `add`), `drop`/`duplicate`
  for cardinality. Verified with the JAR on four configs **and with a real LLM on the
  four axes of the grammar** (the model chose `programmable` on its own and authored
  the expected rule in each one). The catalog compaction that 2.3 had planned came
  along, out of necessity: without it the validation did not fit in the TPM.
- **Open:** with seven paths fixed at once, the model sometimes emits the
  `target_values` as top-level tool arguments and Groq refuses the call
  (`tool_use_failed`). It is Phase 1 machinery and model serialization, not the
  grammar — see `docs/programmable_attack.md`.
- **2.3 (golden prompts + release) DELIVERED** — `programmable_intents.json` with
  12 entries in three categories (`valid`/`invalid`/`incoherent`), the five
  valid ones coming from the model's real payloads. Tag `intent-driven-v0.3.0`.
- **Phase 2 closed.** Open: the model sometimes flattens `target_values` when the
  request fixes ~7 paths (see above), and `inverse_replay` (uc02) still has the
  pre-existing generation bug, outside the phase.
