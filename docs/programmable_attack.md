# Programmable attack (uc11) — behaviors outside the catalog

The ten catalogued attacks are fixed families (replay, flooding, grayhole, …).
The **programmable attack** is the route to a behavior that is none of them: the
user describes in natural language what they want, and that becomes a **list of
rules** applied to the GOOSE messages. A new behavior is **validated
data**, not a new Java class.

The guardrail is the same as in the earlier phases, one step up: in Phase 1 the LLM
proposed a *value*; here it authors a *rule*; in both, deterministic Python
and a Java schema decide whether it is executable. No Java is written by the LLM.

## The rule

Each rule is `{op, field, value, fraction, when}`:

| field | meaning |
|---|---|
| `op` | `set`/`add`/`scale` (mutates the field), `drop` (discards the message), `duplicate` (repeats the message) |
| `field` | the field the mutation touches (`drop`/`duplicate` ignore it) |
| `value` | the operand of the mutation, or the number of **extra** copies of `duplicate` (1 to 10) |
| `fraction` | probability in [0,1] that the rule fires on the messages it targets |
| `when` | the **selection**: `{field, cmp, value}`; `cmp="always"` targets every message |

Available fields: the integers `cbStatus`, `stNum`, `sqNum`,
`gooseTimeAllowedtoLive`, `confRev`, and the time fields `t` and `timestamp` (in
**seconds**).

The rules live under `rules` as an **object** of slots (`{"r0": {...}, "r1":
{...}}`), not an array. The repo's path model is a dotted dict
(`rules.r0.when.cmp`) and has no support for array indices; a dict of slots
keeps both ends — the Python compiler and the Java `ProgrammableCreatorC` —
on the same path model. Java applies the slots in key order.

### Three axes, a single grammar

- **Mutation** — `set`/`add`/`scale` over a frame field.
- **Selection** (`when`) — the rule only acts on the messages where the condition holds.
  The condition is read from the **original** message, before any mutation: which
  message a rule targets cannot depend on what the previous slot already changed, or
  the order of the slots would become part of the behavior. The mutations, those do
  chain in slot order.
- **Timing** — `add` on the time fields, with a sign.

## Why there is no `delay` op and no `reorder` op

It is not a grammar economy: it simply could not work. The stream written by
ERENO is **not** in the order in which the creator emits it — the messages leave through a
`PriorityQueue<EthernetFrame>` whose `compareTo` orders by `timestamp`, and the
delta features (`timestampDiff`, `tDiff`, `delay`) are computed between
consecutive messages in *that* order. Buffering or swapping messages in the
creator would be erased by the queue.

So the clock **is** the order:

| the request | the rule |
|---|---|
| "delay by 20 ms" | `op=add`, `field=timestamp`, `value=0.02` |
| "take out of order / advance" | `op=add`, `field=timestamp`, `value=-0.02` |
| "drop 40%" | `op=drop`, `fraction=0.4` |
| "duplicate each message" | `op=duplicate`, `value=1` |

`set` and `scale` are **refused** on the time fields: they are absolute clocks,
and setting or scaling one relocates the message to an arbitrary point of the
capture instead of shifting it. Only `add` has meaning there. The rule belongs to the schema
(`domain/attack_configs/programmable.py`), not to the allowlist — it is a coherence
*between* two fields of the same rule, which no field-by-field allowlist
can express: the gate approves `op=scale` and `field=timestamp`
separately, and it is the schema that refuses the combination, in the compiler.

## How an intent becomes rules

The design point: the capability separates two kinds of field per slot.

- **`fraction` carries effect.** It is the only lever that the compiler's
  heuristic sweeps by intensity — because it is the only one that maps cleanly to
  evasion (less mutation → more similar to benign) and activity (more
  mutation). Baseline 0.5 to leave headroom in both directions; floor 0.05.
- **`op`/`field`/`value`/`when.*` have an empty effect.** They are the *behavior*
  itself, and the heuristic never touches them. The LLM **authors** them via `target_values`
  (Phase 1), which applies a dictated value without depending on an effect and never clamps it.

So, "delay by 20 ms the messages with the circuit breaker closed" becomes, without a new
compiler:

```json
"target_values": {
  "rules.r0.when.field": "cbStatus", "rules.r0.when.cmp": "eq",
  "rules.r0.when.value": 1,
  "rules.r0.op": "add", "rules.r0.field": "timestamp",
  "rules.r0.value": 0.02, "rules.r0.fraction": 1.0
}
```

The grammar (which ops/fields/comparators/ranges are legal) lives in the
capability; the LLM fills in the slots. A `field` outside the creator's allowlist, a
`fraction` outside [0,1] or an invented comparator is refused at the
deterministic gate — not silently corrected.

## Usage

```bash
uv run adversarial-ids --engine intent --generator-mode jar \
  --prompt "Atrase em 20 ms só as mensagens com o disjuntor fechado."
# "Delay by 20 ms only the messages with the circuit breaker closed."
```

It is only physically measurable in `--generator-mode jar` (in cached mode every trace is the
same file). The attack produces the `programmable` class in the dataset.

## The baseline ships two slots with a default rule

`inputs/attacks/uc11_programmable.json` ships with `r0` (set cbStatus=1) and `r1`
(add stNum+1), both fraction 0.5 and `when.cmp="always"` — a valid programmable
attack, detectable by default.

The `when` **must** be in the baseline, and must be inert: `target_values`
applies a value to an existing path, it does not create a missing key, so an optional
`when` could never be authored; and a `when` with a non-neutral default
would change the default attack. `cmp="always"` solves both — it ignores
`when.field`/`when.value` and reproduces exactly the pre-selection
behavior.

An intent that authors only `r0` **keeps `r1` at the default**: the resulting trace
applies the authored rule plus the baseline's `r1`. For a behavior "from scratch",
zero out the fraction of the slot you do not want (`rules.r1.fraction: 0`) or say so in the
prompt. Two slots is what allows authoring two-rule behaviors; the
cost is carrying two defaults.

## `drop` and the E4 prevalence gate

`drop` reduces the volume of the attack class: the creator emits one message fewer
per firing. With a high `fraction` this lowers the prevalence of the `programmable`
class and may hit the 1% floor of the E4 gate
(`INTENT_LOOP_MIN_ATTACK_PREVALENCE`) — which is the gate doing its job,
not a bug: an attack that is too rare does not produce a metric that means anything.
Measured: `fraction=0.4` leaves 30001 attack rows of the 50000 (60%), well
above the floor. `duplicate`, on the other hand, goes in the opposite direction and multiplies the trace — hence
the cap of 10 extra copies.

## Scope

The surface is limited to the fields with a getter/setter in `Goose` that the creator
knows how to read and write; widening it is additive (more entries in the allowlist on both
sides). What does **not** exist: rules that depend on more than one message
(windows, bursts, state between messages) — each rule decides looking only at the
current message.

## Verification

- `tests/test_programmable_attack.py` — registry, schema (dict of slots, valid
  ops, fraction in [0,1], `when` required), the op↔time-field coherence
  and the `duplicate` copy limit, the capability (only fraction with effect), the
  authoring of behavior/selection/delay via `target_values`, never-clamp,
  and determinism.
- `tests/test_intent_golden_prompts_programmable.py` — the four axes of the
  grammar as golden prompts, with the payloads that the real LLM produced, plus
  the refused requests: five at the intent gates and two only in the schema
  (the `incoherent` category, which exists because here the impossibility may lie
  in the combination of two legal fields).
- `tests/test_attack_capabilities_catalog.py` / `test_intent_loop_orchestrator.py`
  — programmable passes the catalog's six invariants and the seven stages in
  cached mode like any attack.
- **JAR (2026-09-26, Phase 2.1):** "scale sqNum×3 in 40% + set TTL=5" generated a trace
  with 50000 rows of the `programmable` class.
- **Real LLM (`openai/gpt-oss-120b`, 2026-09-26):** the prompt "reescreva o sqNum
  multiplicando por 3 em ~40%" ("rewrite the sqNum by multiplying it by 3 in ~40%") → the LLM chose `base_attack=programmable` and
  authored `{field:sqNum, op:scale, value:3, fraction:0.4}`, in 6826 tokens (under the
  TPM of 8000).
- **JAR (2026-10-01, Phase 2.2), seed 4242, four configs:**
  - *mutation only* (regression from 2.1): 50000 attack rows — the neutral `when`
    changed nothing;
  - *selection + delay* (`when cbStatus=1`, `add timestamp +0.02`, fraction 1.0):
    the rows with `cbStatus=1` ended up with a **minimum** `timestamp - t` of **exactly
    0.02** and those with `cbStatus=0` with a minimum of 0.00 — the delay fell only on the selection;
  - *duplication* (`duplicate value=2`, fraction 1.0): 150002 rows, exactly
    3× the baseline, and the zeros of `timestampDiff` jumped from 31512 to 128706
    (the copies are adjacent with the same clock);
  - *drop* (`drop`, fraction 0.4): 30001 rows, exactly the 60% that
    remain.
- **Regression of the other attacks:** compared the CRCs of the 281 classes
  `br/ufu/facom/ereno/**` between the 2.1 JAR and the 2.2 one — **only
  `ProgrammableCreatorC.class` differs**.
- **Real LLM (`openai/gpt-oss-120b`, 2026-10-01), one prompt per grammar axis.**
  In all four the model chose `base_attack=programmable` on its own:

  | prompt (abridged) | authored |
  |---|---|
  | "reescreva o sqNum só nas mensagens com o disjuntor aberto" ("rewrite the sqNum only on the messages with the circuit breaker open") | `when.field=cbStatus`, `when.cmp=eq`, `when.value=0`, `add sqNum`, `fraction=1.0` |
  | "empurre o relógio 20 ms para a frente, sem reenviar nada" ("push the clock 20 ms forward, without resending anything") | `op=add`, `field=timestamp`, `value=0.02`, `fraction=1.0` |
  | "mensagens fora de ordem: recue ~um terço delas em 30 ms" ("out-of-order messages: move ~a third of them back by 30 ms") | `op=add`, `field=timestamp`, `value=-0.03`, `fraction=0.33` |
  | "cada mensagem três vezes em vez de uma" ("each message three times instead of once") | `op=duplicate`, `value=2`, `fraction=1.0` |

  The two timing axes came out exact, with the ms→s conversion and the sign
  right, and without the model looking for a `delay`/`reorder` op that does not exist.

## Open limitation: the model sometimes flattens `target_values`

In the selection prompt — the one that fixes **seven** paths, more than any other — the
model emitted the paths as top-level arguments of the tool, next to a
`target_values: null`, and Groq refused the whole call
(`tool_use_failed: additionalProperties ... not allowed`). It happened in 2 of 3
attempts at that prompt; in the others the same prompt nested correctly and produced the
row of the table above.

It is model serialization, not the grammar: the signature of `submit_intent_spec`
types `target_values` as a dict and the other three prompts (4 to 5 paths) nested
correctly. It is also **Phase 1** machinery, not 2.2's — except that 2.2 makes it common
to fix many paths at once, which is when it shows up. An explicit warning in the
parameter description did not eliminate the case. Options, if it becomes a nuisance: a
retry on `tool_use_failed`, or fewer paths per call.

## Maintenance note

Three pairs must agree between Python and Java, and Java is always the last
line of the guardrail (if Python accepts what Java does not know, generation
fails loudly with `IllegalArgumentException`, by design):

| Python | Java |
|---|---|
| `MUTABLE_FIELDS` (`domain/attack_configs/programmable.py`) | `ProgrammableCreatorC.FIELD_ALLOWLIST` |
| `TIME_FIELDS` (idem) | `ProgrammableCreatorC.TIME_FIELDS` |
| `MAX_EXTRA_COPIES` (idem) | `ProgrammableCreatorC.MAX_EXTRA_COPIES` |

On the Python side these constants live in the **domain** and the capability
(`config/capabilities/programmable.py`) imports them — the schema needs them for
the coherence rules, and `config/` already depends on `domain/`, never the other way around.
A single list, at both Python ends.
