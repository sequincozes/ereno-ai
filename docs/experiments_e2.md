# E2 (reduced) — transfer of the E1 variants to other detectors

Run on 2026-11-09 (`scripts/run_e2.py`), in response to a review
("is the non-robustness a property of the RF or of the problem?").

## Protocol

- Same protocol as E1: training seeds 101–103, calibration 201–202 and test 401–405;
  `protocol_evaluator` pipeline, with `PROTOCOL_FEATURES_MODE=deltas` and the
  per-budget thresholds calibrated on the calibration normal traffic. Only the
  `FrozenIds` detector changes.
- The evaluated configs are the ones **selected against the RF** by each E1
  campaign (B1–B4, all repetitions). The selection is not redone: the question
  is about transfer.
- It only uses traces from the cache (`CachedOnlyStore`): nothing is generated,
  and the JAR is not called.
- The RF enters as an integrity control: the script aborts if the RF's soft
  evasion does not reproduce the E1 `results/`. It passed on the 4 families.
- Detectors: `random_forest`, `xgboost`, `decision_tree` and `svm_linear`. For
  the SVM, the score is the decision function, so the soft evasion comes out as
  NaN and the AP and the evasion by budget are what count.

## Results (test; mean of the campaigns)

Soft evasion (B0 = reference):

| Family | Detector | B0 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|
| high_stnum | RF | 0.007 | 0.166 | 0.102 | 0.231 | 0.223 |
| | XGBoost | 0.004 | 0.227 | 0.061 | 0.278 | 0.260 |
| | DT | 0.004 | 0.190 | 0.046 | 0.290 | 0.277 |
| inverse_replay | RF | 0.001 | 0.180 | 0.027 | 0.191 | — |
| | XGBoost | 0.000 | 0.333 | 0.049 | 0.359 | — |
| | DT | 0.000 | 0.349 | 0.049 | 0.373 | — |
| masquerade_fault | RF | 0.000 | 0.515 | 0.514 | 0.537 | 0.539 |
| | XGBoost | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| | DT | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| random_replay | RF | 0.002 | 0.092 | 0.062 | 0.189 | — |
| | XGBoost | 0.000 | 0.001 | 0.000 | 0.001 | — |
| | DT | 0.000 | 0.126 | 0.102 | 0.357 | — |

AP, variant − reference (all campaigns, stratified by family):
RF −0.025 [−0.032, −0.018]; XGBoost −0.193 [−0.227, −0.158];
DT −0.314 [−0.341, −0.284]; linear SVM −0.297 [−0.320, −0.274].

Soft evasion, variant − reference: RF +0.252; XGBoost +0.397; DT +0.439.
All intervals exclude zero.

B3 − B1 (paired by repetition, stratified):

| Metric | RF | XGBoost | DT | Linear SVM |
|---|---|---|---|---|
| soft evasion | +0.049 [+0.011, +0.088] | +0.019 [−0.025, +0.056] | +0.089 [+0.016, +0.166] | — |
| evasion@10 | +0.030 [+0.018, +0.042] | +0.023 [−0.037, +0.074] | +0.025 [−0.145, +0.176] | +0.014 [−0.013, +0.041] |
| AP | −0.011 [−0.022, +0.000] | −0.078 [−0.133, −0.032] | −0.119 [−0.167, −0.068] | −0.074 [−0.113, −0.030] |

## Reading

1. **The fragility belongs to the attack space, not to the RF.** The variants
   chosen against the RF transfer to the three other detectors, almost always
   with a **larger** effect than on the RF itself. The RF, by averaging over many
   trees, is the most robust of the four. In the AP drop it is by far the one
   that loses the least.
2. **masquerade_fault is the extreme case.** The RF keeps AP = 1.0 with a soft
   evasion of ~0.5, that is, the variants get closer to the boundary but remain
   separable. XGBoost, DT and SVM, on the other hand, drop to AP 0.10–0.55, and
   XGBoost/DT give a score of ~0 to the attack messages (soft evasion 1.0).
3. **random_replay against XGBoost is the exception.** No variant transfers
   (soft evasion ≤ 0.001, AP 1.0).
4. **The LLM's advantage (B3 > B1) holds in direction, but does not always
   separate.** On AP, it separates for XGBoost, DT and SVM, with a larger effect
   than on the RF, where it was measured. On soft evasion, it separates for the
   RF and the DT, but not for XGBoost.
5. **B2 transfers little.** The ladder without feedback produces variants that
   only the simpler detectors let through.

## Caveats

- The budgets are NaN for XGBoost/DT on masquerade (and for DT on high_stnum and
  random_replay): the calibration normal traffic ties at a score of ~0, the
  threshold ends up above all the normal samples, and `is_saturated` marks the
  point as "no operating point". This is the tie *from below* (perfect
  separation in the calibration), different from the tie *from above* that
  motivated the NaN in E3. The conservative reading was kept. Soft evasion and
  AP carry the result.
- On the DT, the "soft evasion" is practically the arg-max evasion: the leaves
  are almost pure and the score is ~0/1.
- The linear SVM already starts weak on high_stnum: evasion@10 of 0.32 on the
  reference itself.
- The selection was made against the RF. An attacker that targeted each
  detector could evade more. The numbers above are a floor for the transfer,
  not a ceiling.
- The full E2 (single-feature rules) remains out of scope.
