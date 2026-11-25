# Values dictated by the request (`target_values`)

Before, the prompt chose **which** attack and **how much** intensity; each field's
value always came from the compiler's heuristic. Now the request can state the
exact value of a field:

> *"Reduza o recall com a duração da falha entre 50 e 80 ms."* (*"Reduce the recall with the fault duration between 50 and 80 ms."*)

becomes `{"fault.durationMs.min": 50, "fault.durationMs.max": 80}` in
`IntentSpec.restrictions.target_values`, and the compiled config carries those two
numbers as they are.

The central guardrail does not change: the LLM still produces **only** the typed
intent. It now also proposes the number, but whoever decides whether the number exists is the
capability catalog, in deterministic Python.

## The single rule

**A dictated value is never silently corrected.** Either it holds exactly as
requested, or the compilation fails saying why.

This is what separates a pinned field from a heuristic field. The heuristic *negotiates*:
`_clamp_paired` moves a bound closer to its sibling to preserve `min < max`, because
its value is a means to an effect, not an end. A dictated value is an end — the
user said 80. Adjusting it to 79 would produce a config that asserts a value that
nobody asked for, and the `diff` and the justification (which the Defender reads as evidence)
would start lying about its origin.

Practical consequence: a request that is impossible **for the pipeline** fails at that stage,
with the cause on the `LoopStage`, instead of becoming an attack that nobody asked for.

## A pinned field leaves the heuristic entirely

| | heuristic field | pinned field |
|---|---|---|
| value | computed by the `intensity` step | the one from the request |
| choice of field | draw reproducible from the `seed` | always applied |
| must it carry the `desired_effect`? | yes | no |
| `{min, max}` pair clamp | yes, yields to the sibling | no, the sibling yields |
| counts toward `max_fields_changed` | yes | yes |

The two rows that deserve explanation:

- **It does not need to carry the effect.** The heuristic only chooses among fields
  *capable* of the requested effect. A dictated value is a direct instruction, stronger
  than that choice — if the request names the field and the value, the compiler applies it.
  That is why a fully dictated request also does not hit the error "the restrictions
  remove all fields": it did not remove any field, it named the fields one by
  one.
- **It counts toward the quota.** A pinned field is a changed field. If the request pins 4
  values and `max_fields_changed` is 3, the gate refuses, stating the number to use —
  instead of choosing on its own which 3 of the 4 to obey.

## Where each thing is refused

Validation is split by **what each layer can know**, not by
convenience:

| layer | decides | examples |
|---|---|---|
| contract (`domain/intent_spec.py`) | internal contradictions of the request, without knowing which attack it is about | same field pinned twice; field pinned *and* forbidden; field pinned outside `allowed_fields`; number in quotes in a list |
| gate (`config/attack_capabilities.py::validate_target_values`) | what the catalog decides without opening the baseline | path outside the allowlist; wrong type; outside `minimum`/`maximum`/`choices`; inverted `{min, max}` pair with **both** bounds pinned; more pinned than the quota |
| compiler (`core/intent_compiler.py::_assert_pinned_ranges`) | what only the baseline decides | pinned bound against the sibling that stayed at the baseline (`min=900` against `max=800`) |

The gate runs inside `IntentAgent.interpret`, so almost everything is refused
**before** any generation — the LLM receives the error and corrects it in the same call.
Only the last case reaches the GENERATOR stage.

## Two type decisions that are not obvious

- **`True` never passes as an integer.** `isinstance(True, int)` is true in
  Python; without the explicit check, `cbStatus: true` would reach the ERENO JSON.
- **`1` passes where the field expects `number`.** The request says "probability 1", not
  "1.0" — requiring the decimal point would mean requiring floating-point syntax from natural-language
  text. `80.5` in an integer field, on the other hand, is refused: there the conversion
  would change the magnitude, not the spelling.

The line between the two is this: **convert the representation, never the magnitude.**
For the same reason, `["20", "40"]` is refused instead of converted — the conversion
would preserve the number, but the asymmetry with `"0.42"` (which Pydantic refuses in a
scalar field) could not be explained to whoever reads the contract.

## Ranges

The contract has no range type because the attack config does not either:
`{min, max}` are two fields. "Between 50 and 80 ms" is **two** pinned values. Both
invariants hold:

- **strict** `min < max` — `min == max` passes every gate in this repo and ERENO
  refuses the whole execution (*"The lower limit must be less than the upper
  limit"*); it is the same invariant that the E0 pilot discovered, now applied to
  dictated values (see `docs/pilot_e0.md`);
- pinning **one** bound and leaving the other at the baseline is valid, as long as the order
  survives. If the heuristic chooses the sibling, it yields to the pinned value: the
  pinned ones are written into the config **first**, so the clamp already sees them.

## What this cost in the prompt (and why there ended up being room left over)

The injected catalog began to show each field's accepted range in square
brackets (`fault.prob — number [0..1]`, `cbStatus — integer [0|1]`) — without it the
LLM would have no way to choose a value the gate accepts. On its own, this addition
took the `intent` stage prompt from ~7.9k to **8381 tokens** (Groq's actual count),
**above** the account's `on_demand` TPM of 8,000: the first real call
died with `rate_limit_exceeded / Request too large`. The feature was
unusable with the real LLM.

The fix was not to remove the ranges — it was to cut the redundancy that was already there.
The catalog repeated the list of effects in full on each of the 103 fields
(`lower_f1, lower_recall, mimic_normal_traffic`, ~1.9k tokens on that alone), but
invariant 4 guarantees that the three evasion levers always go together — only
three sets of effects exist in the whole catalog. Replacing them with a short
label with a single legend (`· evasão`, `· atividade`, `· evasão, atividade`) (*evasion*, *activity*, *evasion, activity*)
gave back more than the ranges cost. Result measured over three real
calls: **~6.24k input tokens** — about 1.6k **below** where the prompt
was before Phase 1. The `intent` stage gained TPM headroom, it did not lose any.

The test remains: one call per minute (~6.9k in+out against 8k), see
`docs/pilot_e0.md` and the note on the account's TPM.

## Compatibility

`target_values` is additive with default `()`, like `round`/`parent_run_id` in the
`LoopRecord`: an `IntentSpec` saved before this version loads normally, and
`schema_version` remains `1`. A request without dictated values compiles exactly
as before — the heuristic is the common path, not the legacy path.

## Tests

- `tests/test_intent_target_values.py` — the three layers of refusal, the value
  arriving intact in the config, the intensity not moving a pinned field, the quota
  shared with the heuristic, the pair's sibling yielding to the pinned value, the
  list becoming JSON, the E10 ladder over an intent with pinned values, and
  the end-to-end path to the `attack_candidate.json` on disk.
- `tests/test_intent_tool_schema.py` — the `target_values` schema exposes
  scalars, not empty objects (same `list[object]` bug as the Strategist).

Validation with the real LLM (`openai/gpt-oss-120b`, 2026-09-26, three spaced-out
calls): a natural-language range ("entre 50 e 80 ms" — *"between 50 and 80 ms"*) extracted into the
two paths with `max_fields_changed=2`; a nonexistent field (probability on
`flooding`) makes the LLM fall back to the heuristic instead of inventing a path; and a value
outside the range (`fault.prob=1.5`) is proposed by the LLM and **refused by the gate**
with an actionable message — the never-clamp guarantee, confirmed end to end.
