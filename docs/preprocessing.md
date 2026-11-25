# Feature preprocessor (E6)

## Goal

Epic E6 of the 60-day plan asks for a "modular preprocessor" with the done
criterion "fit/transform without leakage; reproducible manifest". Before E6, the
feature preparation (dropped columns, imputation, categorical encoding)
lived embedded and private in `core/ids_evaluator.py::IdsEvaluator`, with a
real leakage: `train_baseline` fitted those decisions on the **whole**
dataset before doing the `train_test_split` — the constant-column detection
and the categorical vocabulary saw the test rows.

E6 extracts this preparation into `core/preprocessor.py::FeaturePreprocessor`, an
independent component with a `fit`/`transform` contract, reusable by
any detector under the same protocol (consumed by E8 — RF/DT/SVM, see
`docs/detectors.md`). The
`IdsEvaluator` now does the `train_test_split` on the **raw** X and only then
calls `fit` — exclusively on the train partition.

Undersampling and feature selection by mutual information (the rest of the
D25-35 window of the plan) are **epic E7**, which consumes `FeatureManifest.feature_columns`
as its starting point — they are not part of this component.

## Output contract

`domain/feature_manifest.py::FeatureManifest` (`extra="forbid"`,
`frozen=True`) is the snapshot of what a `fit` produced — reconstructible on its own
via `FeaturePreprocessor.from_manifest`, without pickle:

- `label_column`, `feature_columns` (final fitted order);
- `dropped_columns`: each dropped column and the reason (`always_drop`,
  `constant` or `cb_status`);
- `numeric_features` / `categorical_features`, an exhaustive partition of
  `feature_columns`;
- `numeric_imputation` (`zero` or `median`) + `numeric_fill_values`, one value
  per numeric feature;
- `categorical_codes`, one `{value: code}` dictionary per categorical
  feature;
- `scaler` (`none` or `standard`) + `scaler_stats` (mean/deviation per
  numeric feature, required for all of them when `scaler="standard"`);
- `fitted_rows`: how many **train** rows fed the fit — never the
  dataset total;
- `fitted_content_hash`: sha256 of the exact content that was fitted (same
  convention as `DatasetBundle.content_hash`).

Validators (`@model_validator(mode="after")`) enforce: `numeric_features` and
`categorical_features` partition `feature_columns` with no remainder and no
overlap; `dropped_columns` does not repeat a name and does not collide with
`feature_columns`; `numeric_fill_values` and `categorical_codes` cover
**exactly** the features of their type (not a subset); `scaler_stats`
covers all the `numeric_features` when `scaler="standard"`; no fitted value
(`numeric_fill_values`, `scaler_stats`) may be NaN/Infinity.

## Anti-leakage

`FeaturePreprocessor.fit` only sees the dataframe it receives — the caller is
responsible for having already separated train and test before calling `fit`.
`transform` never recomputes anything from the data it receives; it only applies
what `fit` already decided (a never-seen category → `unseen_category_code` (-1),
a column missing from the input dataframe → `missing_column_fill` (0.0), then
imputed/scaled like any other value). `FeatureManifest.fitted_rows` is
the proof, in the artifact itself, that the fit saw only the train partition —
`tests/test_intent_loop_orchestrator.py` checks this number directly
(14 of 20 rows of the cached seed, default `test_size=0.3`).

A column may qualify for more than one drop reason at the same
time (in the real ERENO dataset, 21 of the 33 dropped columns are
simultaneously `always_drop` and `constant`) — the recorded reason always follows
`always_drop` > `cb_status` > `constant`, never a combination.

Known limitation, not leakage but honest to document: the
numeric-vs-categorical type of each column comes from the dtype that `pandas.read_csv`
infers over the whole file, before any split — a column cannot
change type between `fit` and `transform`, but that specific decision
is not exclusive to the train partition.

## `PREPROCESSOR_MODE` — before/after comparison

`config.settings.PREPROCESSOR_MODE` (env var, default `"modular"`) and the
homonymous parameter of `IdsEvaluator.__init__` select **when** the `fit`
happens relative to the split — not a different preprocessing logic,
both variants call the same `FeaturePreprocessor`:

- `"modular"` (default): split on the raw X, `fit` only afterwards, on the train
  partition. No leakage.
- `"legacy"`: `fit` on the whole dataset before the split — deliberately reproduces
  the pre-E6 behavior, only to compare metrics.

On the real ERENO data (`data/baseline_dataset.csv`) the two variants
produce exactly the same metrics: all 11 categorical columns of the
protocol are already dropped by `always_drop`, so there is no categorical vocabulary
left to leak, and none of the 20 surviving numeric features is
constant only in the whole dataset but not in the train — `git diff` of
`data/iteration_history.json` after regenerating the golden confirms this: zero
difference outside the dataset's absolute path. The fix matters for
datasets where this coincidence does not hold.

## `PROTOCOL_FEATURES_MODE` — identity is not the same as semantics

The always-drop list came whole from the framework's initial commit and
treated two very different groups under the same justification
("temporal/sequence, strong derived features"). Measuring each dropped column
against the class in `data/baseline_dataset.csv` — 21 of the 33 are constant and the
measurement only makes sense for the other 12, with a ceiling of **H(class) = 0.6611 nats**
— the separation is unambiguous:

| column | MI | % of ceiling | group |
|---|---|---|---|
| `sqDiff` | 0.5367 | 81% | delta |
| `tDiff` | 0.5335 | 81% | delta |
| `SqNum` | 0.5192 | 79% | delta |
| `timeFromLastChange` | 0.4343 | 66% | delta |
| `timestampDiff` | 0.2251 | 34% | delta |
| `stDiff` | 0.1958 | 30% | delta |
| `Time` | 0.0000 | 0% | identity |
| `StNum` | 0.0002 | 0.03% | identity |
| `t`, `GooseTimestamp`, `receivedTimestamp`, `delay` | ~0.0001 | ~0% | identity |

Identifiers and absolute values carry no signal — and what they would
carry in a synthetic generator would be the generation block, not the attack.
Dropping them is right. The deltas are the opposite: they are the sequence and
timing semantics of GOOSE, and replay, flooding and grayhole **are** anomalies of
`sqDiff`/`tDiff`. Without them, the detector has no way to see anything beyond the
forged fault.

`config.settings.PROTOCOL_FEATURES_MODE` (env var) and the homonymous parameter of
`IdsEvaluator`/`FeaturePreprocessor` choose between the two regimes:

- `"drop"` (**default**): drops both groups, 33 columns. Reproduces
  exactly the inherited behavior — `data/iteration_history.json` remains
  byte-stable. 20 features remain, 18 of them electrical quantities.
- `"deltas"`: drops only the 24 identity ones. 29 remain; the 9 additional ones are
  `stDiff`, `sqDiff`, `SqNum`, `tDiff`, `timestampDiff`, `timeFromLastChange`,
  `gooseLengthDiff`, `apduSizeDiff`, `frameLengthDiff`.

The raw `SqNum` goes into the deltas group, and not the identity one, because in
GOOSE it resets when `stNum` increments — it carries state, not an index. It is the
least clear case of the group, and the asymmetry with `StNum` (0.03% of MI) is
deliberate.

### Ablation

On 8 000 balanced rows of the reference dataset, Random Forest:

| mode | features | F1 | top-4 by importance |
|---|---|---|---|
| `drop` | 20 | 1.0000 | `vsbBTrapAreaSum` .168, `vsbCTrapAreaSum` .162, `vsbATrapAreaSum` .160, `cbStatus` .130 |
| `deltas` | 26 | 1.0000 | `SqNum` .253, `timeFromLastChange` .117, `sqDiff` .114, `tDiff` .102 |

The metric does not move: `masquerade_fault` was already perfectly separable by the
analog signature alone, and it is the only physically measurable attack in
`--generator-mode cached`. **What changes is what the detector relies on** — and
this cuts both ways. `SqNum` becoming the dominant feature for detecting
a forged waveform is plausible (the masquerade injects frames and disturbs
the cadence), but it is also exactly what a generator artifact would
look like. That is why the mode is measured by ablation instead of being turned on:
the real gain is in the attacks that need the jar, and the decision belongs to the
experiment.

## Stage failure

`FeaturePreprocessor.fit` raises `PreprocessorError` (actionable message)
when the received dataframe is empty, when no feature survives the
drop, or when a categorical exceeds the cardinality ceiling (1000
distinct values — defensive against a high-cardinality column, such as
an ID or a text timestamp, turning into a megabyte-sized manifest). `transform`
raises before any `fit`. In the intent-driven pipeline, any one of
these failures marks the `detector` `LoopStage` as `failed` with the cause —
the same generic mechanism of `IntentLoopOrchestrator._stage` that the other
stages use.

`feature_manifest.json` is persisted **before** the binary-matrix gate of the
`DetectionReport` (`core/detection_reporter.py::build_detection_report`), not
after — if the evaluation fails there, the manifest of what was fitted on the
baseline is already on disk for diagnosis.

## Out of scope and future work

- **Undersampling and feature selection (E7)**: they consume
  `FeatureManifest.feature_columns` as their starting point; not implemented
  here — see `core/feature_selector.py`, `core/undersamplers.py` and
  `docs/feature_selection.md`.
- **RF/DT/SVM detector interface (E8)**: ~~out of scope~~ — **delivered**.
  `scaler="standard"` stopped being just a preparation for the future: the two
  SVMs registered in `core/detectors.py` ask for it, and `IdsEvaluator` resolves the
  scaling from the chosen detector. See `docs/detectors.md`.
- **Feature → defensive technique mapping**: mentioned as a natural extension
  in `docs/feedback_policy.md`; it is not this epic.

## Tests

- `tests/test_preprocessor.py`: basic fit/transform (imputation, encoding,
  column order), anti-leakage (a column constant only in the train, a category
  not seen, `fitted_rows`, scaling with statistics from the train only, a
  zero-deviation safeguard), errors (empty `fit`, no surviving features, cardinality
  ceiling, `transform`/`manifest` before `fit`), manifest determinism
  and round-trip via `from_manifest`/JSON.
- `tests/test_new_contracts.py`: `FeatureManifest` versioned/frozen, and
  each internal-consistency validator.
- `tests/test_intent_loop_orchestrator.py`: `feature_manifest.json`
  persisted in the DETECTOR stage, validated as `FeatureManifest`, with
  `fitted_rows` checked against the real size of the train partition.
- `tests/test_detection_reporter.py`, `tests/test_orchestrator_workflow.py`,
  `tests/test_feature_importances.py`: an `IdsEvaluator` actually trained
  (the latter on the real dataset of ~80k rows) keep passing with no
  change to the expected values.
