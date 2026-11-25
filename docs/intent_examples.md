# Qualitative examples of `--engine intent`

Two examples of the intent-driven pipeline, to illustrate the path
prompt → `IntentSpec` → compiler → JAR → detector → Defender → feedback.
All of them ran with `--generator-mode jar` and `PROTOCOL_FEATURES_MODE=deltas`.
The artifacts live in `outputs/intent_loop/<run_id>/`, and the campaign
record in `outputs/loop_records.json`.

## Example 1a: high_stnum with every field dictated (2026-11-03)

```bash
uv run adversarial-ids --engine intent --generator-mode jar --rounds 3 --prompt \
  "evada o IDS com o ataque high_stnum usando salto de stNum entre 2 e 5 e probabilidade de reinício do sqNum igual a 0"
# "evade the IDS with the high_stnum attack using an stNum jump between 2 and 5 and an sqNum reset probability equal to 0"
```

Rounds `994bb292…` (1) and `0fcbbfc0…` (2). Log in
`outputs/experiments/intent_ex1_high_stnum.log`. Exit 0, 16.4k tokens in total.

- **The dictated values arrived intact.** The LLM extracted three `target_values`
  (`jump.min = 2`, `jump.max = 5`, `sqnumResetProb = 0`), and the compiler
  applied them untouched, outside the intensity ladder. There was no clamp.
- **It does not evade.** Internal training: recall 0.997. Variant against the baseline:
  recall 0.990 and F1 0.994, with `stDiff` dominating the importance (0.68). An
  stNum jump of 2 to 5 is still a jump that `stDiff` sees.
- **The escalation had nowhere to act.** The feedback raised the intensity from medium
  to high, but the three fields the compiler changes (`max_fields_changed = 3`)
  were already all fixed in the request. The round 2 config is identical to that of
  round 1, and so is the trace, byte for byte (same seed 42). That is why the
  campaign stopped at `no_improvement` (gain 0.0000) in round 2 of 3, before
  widening `max_fields_changed`. This is correct under the rules (a dictated value
  is never overwritten), but it shows a limitation of the policy: it spends
  a round on an escalation that is already known to be empty before generating the trace.
- **The Defender's plans came out generic and with priority low:**
  `network_segmentation` in round 1, and `detector_retraining` plus
  `network_segmentation` in round 2. None cites a feature, the same pattern
  as the E1 and E3-LLM plans.

## Example 1b: high_stnum with only the jump dictated (2026-11-03)

```bash
uv run adversarial-ids --engine intent --generator-mode jar --rounds 3 --prompt \
  "evada o IDS com o ataque high_stnum usando salto de stNum entre 2 e 5"
# "evade the IDS with the high_stnum attack using an stNum jump between 2 and 5"
```

Rounds `9b20557c…` (1) and `f032ffd7…` (2). Log in
`outputs/experiments/intent_ex1b_high_stnum.log`. Exit 0, 16.0k tokens.

It repeated the defect of 1a. The LLM set `max_fields_changed = 2`, the same number of
dictated fields, so the intensity again had no free field. The trace of
round 2 came out identical to that of round 1, and the campaign stopped at `no_improvement`. The
variant evades even less than that of 1a: recall 0.996 and F1 0.997. Without fixing
`sqnumResetProb = 0`, the baseline's 0.4 applies, and the sqNum resets probably also
give the attack away.

With two cases in a row it became clear that the problem is structural, not one of the
prompt. The policy now skips the intensity step when the whole quota
is in the dictated values, and goes straight to widening `max_fields_changed`
(`core/feedback_policy.py::_escalate`, see `docs/feedback_policy.md`).

## Example 1c: the 1a prompt, with the fix (2026-11-03)

Same prompt as 1a, after commit `ee6d6b84`. Rounds `9afdc23e…` (1) and
`ae8d609b…` (2). Log in `outputs/experiments/intent_ex1c_high_stnum.log`.
Exit 0, 17.3k tokens.

| round | intensity | fields | compiled diff | recall | F1 |
|---|---|---|---|---|---|
| 1 | medium | 3 | jump 20–50 → 2–5 (dictated), sqnumResetProb 0.4 → 0 (dictated) | 0.9899 | 0.9944 |
| 2 | medium | 4 | the same, plus `sqNumDelta.max` 5 → 2 (ladder) | 0.9772 | 0.9877 |

- **The ladder now moves.** Round 1 is byte for byte the same as in 1a (same
  intent, same seed). In round 2 the policy widened the fields from 3 to 4
  (cap 5) without touching the intensity. The new field came from the ladder, with the
  dictated values intact.
- **The campaign stopped at `no_improvement`, and now with a reason.** The F1 drop
  (0.0066) was below the minimum of 0.01 (`FEEDBACK_MIN_DELTA`). This time the
  variant did change; it just did not improve enough to justify
  round 3.
- **The evasion remains small,** because `stDiff` still sees any stNum
  jump. The false negatives went from 503 to 1139 in 50k.
- **The Defender's plans remain generic and with priority low.** In
  round 1: `detector_threshold_tuning` (false positives) and
  `network_segmentation` (latency). In round 2: `detector_retraining`
  (recall) and `traffic_rate_limiting` (latency).

Examples 1a and 1b remain as a record of the defect that 1c exposed.

## Example 2: programmable, selective delay (2026-11-03)

```bash
uv run adversarial-ids --engine intent --generator-mode jar --rounds 3 --prompt \
  "Atrase em 20 ms só as mensagens com o disjuntor fechado."
# "Delay by 20 ms only the messages with the circuit breaker closed."
```

Rounds `10247d66…` (1), `3b82f168…` (2) and `d0ac8093…` (3). Log in
`outputs/experiments/intent_ex2_programmable.log`. Exit 0, 23.5k tokens.

**The LLM authored the right rule on the first attempt.** There were 7 `target_values`
in `rules.r0`: `when cbStatus eq 1`, `op = add`, `field = t`, `value = 0.02`,
`fraction = 1`. The compiler applied them literally.

| round | intensity | fields | what the ladder changed | recall | F1 |
|---|---|---|---|---|---|
| 1 | medium | 7 | nothing (all 7 dictated) | 0.637 | 0.702 |
| 2 | medium | 8 | `rules.r1.fraction` 0.5 → 0.275 | 0.560 | 0.628 |
| 3 | high | 8 | `rules.r1.fraction` 0.5 → 0.1175 | 0.504 | 0.572 |

- **The full ladder showed up, already with the `ee6d6b84` fix.** With all
  7 fields dictated, round 2 widened from 7 to 8 without going through
  the intensity (before the fix, it would have repeated the trace). Round 3
  raised medium → high. The campaign stopped at `levers_exhausted`, with the cap of
  8 fields reached.
- **Caveat: the attack is not only the delay.** The `uc11_programmable` baseline
  carries a second rule, `r1` (adds 1 to stNum in 50% of the messages), which the
  prompt did not mention and which stayed active. The field the ladder touched was
  precisely its fraction. The F1 drop comes from diluting the stNum jump that
  the detector recognizes, not from the 20 ms delay having become stealthier.
- **Caveat: the detector was trained on the `uc11` baseline.** That baseline is
  a different attack (`set cbStatus = 1` plus stNum + 1), with an internal recall of
  only 0.775. The variant is therefore evaluated outside the training
  distribution.
- **For the first time, a plan cites a feature.** The `device_quarantine` of
  rounds 1 and 2 cites `stDiff` (the one with the highest importance, 0.27). The priority
  was medium in all three rounds. The rest remains generic: threshold
  tuning, dataset enrichment and network segmentation.

Example 2 shows that the DSL and the LLM produce the requested behavior
and that the ladder walks through both steps. It does **not** serve as
evidence that the delay evades, because of the leftover `r1` rule.
