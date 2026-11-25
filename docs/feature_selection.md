# Undersampling + feature selection (E7)

## Goal

Epic E7 of the 60-day plan is the rest of the D25-35 window that E6 explicitly
left out: "undersampling and feature selection by mutual
information", with the done criterion "configurable strategies with
ablation". Both components consume `FeatureManifest.feature_columns`
(E6) as their starting point — they are not part of the `FeaturePreprocessor`, which had
already documented this boundary since E6.

- `core/feature_selector.py::FeatureSelector` — feature selection by
  mutual information (`sklearn.feature_selection.mutual_info_classif`).
- `core/undersamplers.py` + `core/undersampling_policy.py` — class balancing of the
  train partition by random resampling without replacement.
- `domain/selection_manifest.py::SelectionManifest` — frozen contract that
  records what the two strategies decided, persisted as
  `selection_manifest.json` in the intent-driven pipeline (same DETECTOR
  stage that already persists `feature_manifest.json`).

Both strategies are `"none"` by default
(`config.settings.FEATURE_SELECTION_MODE`/`UNDERSAMPLING_MODE`) —
behavior identical to before E7 existed, until someone explicitly opts in to one
of them.

## Pipeline order

`IdsEvaluator.train_baseline` (`core/ids_evaluator.py`), after the
`FeaturePreprocessor` (E6) has already done `fit_transform`/`transform` on train/test:

1. **Feature selection first**, fit on the **whole** train split
   (`FeatureSelector.fit_transform(X_train, y_train)`, also applied to
   `X_test` via `transform`) — mutual information is more stable with more
   rows, and the minority attack class must not lose weight in the
   relevance score before even deciding which features survive.
2. **Undersampling afterwards**, only in the already-reduced feature space
   (`fit_resample(X_train, y_train)` of the resolved undersampler) — never on
   `X_test`/`y_test`, nor on any dataset passed to `evaluate_variant`:
   resampling the evaluation set would distort the very metric one
   wants to measure (the real distribution of attack vs. normal is what the detector
   has to face in production).

The model is only trained after both steps; `evaluate_variant` and
`get_shap_importances` apply `preprocessor.transform` followed by
`feature_selector.transform` (never undersampling) — the same column space
the model learned.

## `FeatureSelector` — filter-based and wrapper-based

- `strategy="none"` (default): keeps all candidate features, in the
  order the preprocessor produced. `scores` stays `{}` — the "none" strategy
  computes no score at all.
- `strategy="mutual_info"`: requires `top_k` and/or `min_score` (without a cutoff
  criterion there is no way to decide how many features to keep — it raises
  `FeatureSelectorError` at construction). It fits
  `mutual_info_classif(X_train, y_train, random_state=...)`, sorts by
  descending score with deterministic tie-breaking by column name, and
  applies `top_k` and/or `min_score` (intersection, when both are
  given). `selected_features` preserves the original order of the candidate
  columns, not the ranking order — it keeps the layout stable for the
  trained model. It raises `FeatureSelectorError` if nothing survives the
  cutoff. It is **filter-based**: it scores the data against the label without training
  any model.
- `strategy="rfe"`: **wrapper-based**. It trains a model, discards the least
  important feature, retrains, and repeats until `top_k` remain
  (`sklearn.feature_selection.RFE`). It requires `top_k` — RFE's stopping criterion is a
  number of features, not a threshold — and **refuses** `min_score`, which has
  no meaning over an elimination ranking. A `top_k` above the number
  of candidates keeps them all instead of failing: "I want the best 20 out of 12" is
  a request that can be met.

### Why RFE does not use the detector that will be trained

That would be the textbook wrapper, and it would break E8's guarantee. The five detectors
are comparable **because the pipeline upstream of them is identical**
(`docs/detectors.md`); if selection used each detector as its own
evaluator, each would train on a different feature space and the
comparison would stop measuring the detector and start measuring the selection. RFE's estimator
is fixed, cheap and independent of the detector — a tree of depth 8.
A design decision, not a limitation.

Cost: RFE trains once per eliminated feature. What makes it viable is the
estimator being cheap, not the set being small — swapping it for an ensemble
would multiply the cost by the number of features.

### The two scores are opposite

`scores` is filled in by both strategies and **the scale depends on which one**:

| strategy | what the number is | direction |
|---|---|---|
| `mutual_info` | mutual information with the label | **higher** is better |
| `rfe` | elimination ranking (`1` = kept) | **lower** is better |

Sorting `feature_scores` without looking at `feature_selection` produces an inverted
ranking in one of the two cases. It is the same discipline as
`DetectorManifest.importance_kind` (see `docs/detectors.md`): same format,
different meaning, and a field that says which.

## Undersamplers — registry by key

`core/undersamplers.py` is the third registry of the same shape (detectors in
E8, generators upstream): a `Literal` in the domain
(`domain/undersampling.py::UndersamplerKey`), one `UndersamplerSpec` per key,
and a sync check at import.

| key | what it does | needs `target_ratio` |
|---|---|---|
| `none` (default) | returns `X`/`y` unchanged, without copying | no |
| `random` | subsamples each class down to the size of the minority | no |
| `ratio` | subsamples each class down to `target_ratio ×` the minority | **yes** |

Everything is deterministic by `random_state` (the same `42` as `IdsEvaluator`,
reused — not a new, independent parameter). No strategy uses
`imbalanced-learn`: resampling without replacement and without a synthetic substitute is
simple enough for plain `numpy`/`pandas`. An undersampler that needs
neighborhoods (NearMiss, Tomek) comes in as one more key, and then the dependency is
its own.

`random` and `ratio` are the **same** class with `target_ratio` 1.0 and N —
balancing down to the minority is the special case of reducing down to a ratio. This
is description, not aspirational architecture, and there is a test that keeps the two
identical at `target_ratio=1.0`; the seam the registry opens holds for any
strategy that is not a per-class ceiling.

### Why `ratio` exists

Balancing down to the minority class in a trace whose attack prevalence can
be 1% (the E4 floor) discards ~99% of the benign rows: the detector ends up
training in a world that does not exist, and the bill is paid in false positives on
real traffic. `ratio` bounds the imbalance instead of eliminating it. With 990
benign and 10 attack rows, `random` leaves 10 benign and `ratio=5.0` leaves 50.

## Who decides the undersampler — `config` vs. `policy`

The Pre-Processor box in the architecture diagram foresees "intent-driven" and
"policy-driven" undersamplers — names that do not describe algorithms but
**who decides**. Until now nothing decided anything: the strategy was an env var read
once. `core/undersampling_policy.py::decide_undersampling` is the decider, and
it has exactly two modes:

- **`config`** — `UNDERSAMPLING_MODE` is `none`/`random`/`ratio`: the strategy
  is fixed and the observed imbalance **does not take part** in the choice (but it is
  measured and recorded anyway, for whoever reads the result later).
- **`policy`** — `UNDERSAMPLING_MODE=policy`: the rule declared in
  `UndersamplingPolicy` resolves. Below `max_imbalance_ratio` (default
  10:1) nothing is resampled; above it the majority is reduced to
  `target_ratio` (default 3.0) times the minority. Deterministic, and the reason
  is written to the manifest.

### What was deliberately left out

There is no `feedback` or `intent` decider, and the refusal is **structural**:

```python
def decide_undersampling(*, class_counts, mode, policy) -> UndersamplingDecision
```

There is no parameter through which a `DetectionReport`, a `Metrics` or an
`IntentSpec` could enter — and this is not argument economy, it is the rule. Two
reasons, and each holds independently:

1. A loop that tunes its own preprocessing until the metric improves stops
   measuring what the detector faces: the reported number becomes a
   joint property of the attack and of the balancing chosen to favor it,
   and no round is comparable with another again.
2. Letting the attacker's intent configure the defender's training is the Red Team
   weakening the detector by a path that is not the attack.

`tests/test_undersampling_policy.py` has a test that inspects the signature
and fails if such a parameter appears. It is the same figure as E10 with the sign
flipped: there the deterministic policy replaces a second LLM call;
here it replaces an optimization nobody asked for.

### Configuration

| env var | default | effect |
|---|---|---|
| `UNDERSAMPLING_MODE` | `none` | `none`/`random`/`ratio`/`policy` |
| `UNDERSAMPLING_TARGET_RATIO` | `3.0` | target ceiling of `ratio` and of the policy |
| `UNDERSAMPLING_MAX_IMBALANCE` | `10.0` | above this the policy resamples |

An unknown mode fails when the `IdsEvaluator` is **constructed**, and not in the
middle of a training run — which is where a mistyped mode would cost an entire generation to
show up. `target_ratio > max_imbalance_ratio` is also refused
at construction: the policy would be correcting for an imbalance that it
itself declares intolerable.

## Output contract — `SelectionManifest`

`domain/selection_manifest.py::SelectionManifest` (`extra="forbid"`,
`frozen=True`), sibling of `FeatureManifest` and separate from it:

- `feature_selection`/`feature_selection_top_k`/`feature_selection_min_score`
  — the strategy and the cutoff criterion/criteria used;
- `candidate_features`/`selected_features` — before/after selection;
- `feature_scores` — empty only when `feature_selection="none"`; filled in by
  `mutual_info` and by `rfe` with **opposite** scales (see above),
  and covers exactly `candidate_features`;
- `undersampling`/`undersampling_random_state` — the **resolved** strategy (
  never `"policy"`, which is a decider and not an undersampler) and the seed;
- `undersampling_decided_by`/`undersampling_decision_reason`/
  `undersampling_observed_imbalance`/`undersampling_target_ratio` — who
  chose, why, over what measure, and with what ceiling. Without them, a run with the
  policy on is indistinguishable from a hand-configured run;
- `class_counts_before`/`class_counts_after` — count per **original**
  label (not the `LabelEncoder` code), before/after
  undersampling;
- `fitted_rows_before_undersampling`/`fitted_rows_after_undersampling` —
  **train** rows, never test nor the dataset total (same anti-leakage proof as
  `FeatureManifest.fitted_rows`).

Validators enforce: `selected_features` is a subset of
`candidate_features`; `feature_scores` empty when `feature_selection="none"`
and complete when `"mutual_info"`; `mutual_info` with neither `top_k` nor `min_score`
is rejected (the manifest cannot pretend there was a cutoff without saying which);
rows after undersampling never increase, and `undersampling="none"` cannot
change the count; `ratio` requires `undersampling_target_ratio` and the other
two refuse it (recording it would describe a cut that did not happen);
`undersampling_decided_by="policy"` requires the reason and the observed imbalance
— a policy that decides without saying why is indistinguishable from a default, and
that is precisely the distinction the artifact exists to preserve; the sum of `class_counts_before`/`class_counts_after`
matches `fitted_rows_before_undersampling`/`fitted_rows_after_undersampling`
when the counts are provided.

## Persistence in the intent-driven pipeline

`agents/orchestrator/intent_loop.py::IntentLoopOrchestrator._run_detector`
persists `selection_manifest.json` in the DETECTOR stage, right after
`feature_manifest.json` — same ordering reason: if the evaluation fails at the
binary-matrix gate of the `DetectionReport` right afterwards, what
selection/undersampling decided about the training is already on disk for
diagnosis. It is not a `LoopStage` of its own, it is a DETECTOR artifact, just like the
E6 manifest.

## Ablation and reproducibility

Done criterion of E7: "configurable strategies with ablation".
`tests/test_ids_evaluator_selection.py` covers this in real integration with the
`IdsEvaluator`:

- by default (`"none"`/`"none"`), `SelectionManifest.selected_features ==
  candidate_features` and `fitted_rows_before_undersampling ==
  fitted_rows_after_undersampling` — behavior identical to before E7;
- `feature_selection="mutual_info"` with `top_k=1` drops the noise features from
  the model (`evaluator.feature_columns` shrinks) and reproduces the same
  selection with the same seed;
- `undersampling="random"` balances the train classes
  (`class_counts_after` with the same count in both classes) without ever
  changing the evaluation's `dataset_rows` (the proof that test/variant was not
  touched) and reproduces the same final count with the same seed;
- the two strategies compose: selection reduces columns, undersampling reduces
  rows of what is left, in the order described above.

`tests/test_feature_selector.py`, `tests/test_undersamplers.py` and
`tests/test_undersampling_policy.py` cover the
two components in isolation (fit/transform contract, errors, determinism,
column-order preservation); `tests/test_new_contracts.py` covers each
internal-consistency validator of the `SelectionManifest`;
`tests/test_intent_loop_orchestrator.py` checks that
`selection_manifest.json` is persisted and valid in the real loop.

## Out of scope and future work

- **RF/DT/SVM detector interface (E8)**: ~~out of scope~~ — **delivered**.
  The four registered detectors train on the feature space
  this component selected and on the rows that undersampling left;
  `DetectorManifest.trained_features`/`trained_rows` match
  `SelectionManifest.selected_features`/`fitted_rows_after_undersampling`.
  See `docs/detectors.md`.
- **Wrapper-based selection (RFE)**: ~~out of scope~~ — **delivered**, and it proved
  that the `strategy` `Literal` was indeed extensible without breaking the contract: the
  only change to `SelectionManifest` was the vocabulary and one validator per
  strategy.
- **Policy-driven undersamplers**: ~~out of scope~~ — **delivered**. The box
  now has a declared decider, and the `SelectionManifest` now records
  who chose and why.
- **Neighborhood-based undersampling strategies** (NearMiss, Tomek
  links) remain out: they come in as one more key in the registry, with whatever dependency
  they ask for.
- **Other selection criteria** (e.g.: the model's own importance) remain
  out — which, incidentally, is what RFE avoids on purpose, for the reason explained
  above.
- **Undersampler decided by the loop or by the intent**: out of scope **and
  refused by design**, not postponed. See "What was deliberately left
  out".
