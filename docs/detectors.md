# Pluggable RF/DT/SVM detector (E8)

## Goal

Epic E8 is the D36-46 window of the 60-day plan: "same split/protocol for
all detectors", with the done criterion "RF/DT/SVM comparable". It is the
epic that E6 and E7 had been announcing as their consumer — both had already
made data preparation detector-agnostic; what was missing was for the detector
to stop being a `RandomForestClassifier` hardcoded in `core/ids_evaluator.py`.

- `core/detectors.py::Detector` — fit/predict adapter over a registered
  scikit-learn estimator, resolved by key.
- `domain/detector_manifest.py::DetectorManifest` — frozen contract that
  records which detector trained and under what data preparation, persisted as
  `detector_manifest.json` in the intent-driven pipeline (the same DETECTOR stage
  that already persists `feature_manifest.json` and `selection_manifest.json`).
- `core/detector_comparison.py::compare_detectors` + `domain/detector_comparison.py::DetectorComparison`
  — the **comparative report** the window's gate requires, produced by
  `scripts/compare_detectors.py`.
- `config.settings.DETECTOR_MODE` / `DETECTOR_SCALER` — the two environment
  knobs. The default is `random_forest` with no scaling: **behavior identical
  to before E8**, and the golden history stays byte-stable.

What E8 does **not** change: no split, no metric, no feature
preparation. All the comparability between detectors comes from their being fed by the
same upstream pipeline (E6 → E7), not from anything done inside the
detector.

## Registered detectors

| key | estimator | requested scaling | importance | SHAP | cost at ERENO volume |
|---|---|---|---|---|---|
| `random_forest` (default) | `RandomForestClassifier` | `none` | Gini | yes | low (`n_jobs=-1`) |
| `decision_tree` | `DecisionTreeClassifier` | `none` | Gini | yes | very low |
| `svm_linear` | `LinearSVC` (liblinear) | `standard` | `\|coef_\|` | no | medium, scales well |
| `svm_rbf` | `SVC(kernel="rbf")` | `standard` | **none** | no | **O(n²)–O(n³) in training rows** |
| `xgboost` | `XGBClassifier` | `none` | **gain** | yes | low (`n_jobs=-1`) |

`xgboost` (epic E9) depends on an **optional extra**:

```bash
uv sync --extra dev --extra dashboard --extra xgboost
```

`uv sync` syncs *exactly* the extras requested — asking only for `--extra xgboost`
removes the others from the environment. The import is lazy: `core/detectors.py`
remains importable without the package, and only constructing the detector fails, with the
install instruction in the message. `scripts/compare_detectors.py` isolates
failure per detector, so without the extra the comparison still comes out with the other
four and `xgboost` appears with `status="failed"` and the reason — a visible
omission, not a silent one.

### Why four keys and not a single `svm` key with a kernel parameter

The kernel changes both the cost and the explainability of the detector, and
both facts need to be visible in the name of what is being run. An `svm`
key that silently was linear (fast, explainable) or RBF
(slow, opaque) would make two incomparable experiments look like the same one.

### Deviations from scikit-learn defaults

The registry's hyperparameters are sklearn's own defaults, with two
deliberate exceptions (both commented in `core/detectors.py`):

- `svm_linear`: `max_iter=5000` instead of 1000 — with tens of thousands of
  rows liblinear often does not converge at the default, and a
  poorly converged model yields a metric that measures the optimizer, not the detector.
- `svm_rbf`: `cache_size=500` instead of 200 MB — the libsvm bottleneck at this
  volume is the kernel cache. It does not change the model, only the time.

`random_forest` reproduces exactly the pre-E8 hardcoded construction
(`n_estimators=100, random_state=42, n_jobs=-1`); the other registry defaults
are sklearn default values, so passing them explicitly does not
alter the resulting model.

## Scaling (E6 × E8)

`scaler="standard"` has existed in `FeaturePreprocessor` since E6, off by
default, precisely for this epic — trees are invariant to a
monotonic feature transformation, SVMs are not. E8 closes the circuit:

- `IdsEvaluator(scaler=None)` (default) **delegates to the detector**: `standard`
  for both SVMs, `none` for the trees.
- `IdsEvaluator(scaler="none"|"standard")` **forces** both sides to use the
  same scaling — which is what a controlled ablation between detectors needs.

Running an SVM without scaling is legitimate (it is a valid ablation) and so it is not an
error. But it is recorded: `DetectorManifest` carries `requires_scaling` (the
recommendation) and `resolved_scaler` (what was applied) side by side, so an
SVM that ran without scaling explains a poor metric on its own, without needing to
rerun anything.

## Comparative report

The detector registry makes RF/DT/SVM interchangeable, but each run of the
loop trains **one**. The window's exit gate asks for more than that — "same
split/protocol; **comparative report**; RF fallback preserved", with
"ranking by F1/recall/latency". That is what `scripts/compare_detectors.py`
produces:

```bash
uv run python scripts/compare_detectors.py                    # all, ranked by F1
uv run python scripts/compare_detectors.py --rank-by latency_ms
uv run python scripts/compare_detectors.py --detector random_forest decision_tree
```

```
 #  detector            F1  recall    prec    latência   escala    importância
-----------------------------------------------------------------------------
 1  decision_tree   1.0000  1.0000  1.0000      85.3ms  none      gini
 2  random_forest   1.0000  1.0000  1.0000     151.6ms  none      gini
 3  svm_rbf         0.9913  1.0000  0.9828     103.8ms  standard  none
 4  svm_linear      0.9313  0.9825  0.8851      93.7ms  standard  linear_coef
```

(Column headers are the script's literal output: latência = latency, escala = scaling, importância = importance.)

It writes `outputs/detector_comparison.json` (a valid `DetectorComparison`) and exits
with code 1 when the comparison does **not** serve as evidence for the gate: some
detector failed, or the protocol was not uniform.

Three properties that the contract guarantees, and that the table alone would not:

- **The ranking is recomputable.** A `DetectorComparison` validator reorders
  the `runs` from the `ranking_metric` and rejects the artifact if the
  declared order does not match. The chosen metric rules; the other two break
  ties (F1 → recall → latency, and finally the detector name, so that an exact
  tie does not depend on the order in which the list was built). In the example above
  that is exactly what separates `decision_tree` from `random_forest`.
- **An isolated failure does not bring down the others.** Each detector runs in its own
  try/except: an SVM that runs out of memory becomes a `failed` row with the cause,
  stays out of the ranking, and the others are still measured. A literal requirement of the
  acceptance table of the Detection layer.
- **"Same protocol" is verified, not promised.** See the next section.

## Limit of the "same protocol" guarantee

E8's comparability is literal as long as `FEATURE_SELECTION_MODE="none"`
(the default): the four detectors receive the same split, the same candidate columns
and the same training rows, and the only difference is the scaling — which
trees ignore by construction.

With `FEATURE_SELECTION_MODE=mutual_info` the guarantee becomes **conditional**. E7's
selection is fitted on the already-preprocessed `X`, that is, **already scaled**,
and `mutual_info_classif` is not strictly scale-invariant (the noise it
injects is proportional to the amplitude of each column, and the estimator is based on
k-nearest-neighbor distances). In principle, therefore, a tree (`scaler="none"`)
and an SVM (`scaler="standard"`) may receive different feature sets
while both manifests claim to have run "under the same protocol".

In practice this did not show up on the ERENO baseline — I measured RF and `svm_linear`
with `top_k=5` and the selection came out identical. But it is not a guarantee, it is
a coincidence of the data. **For a controlled comparison with selection on,
force `scaler=` to the same value on both sides**:

```bash
export DETECTOR_SCALER=standard FEATURE_SELECTION_MODE=mutual_info FEATURE_SELECTION_TOP_K=15
uv run adversarial-ids --engine live --detector svm_linear
uv run adversarial-ids --engine live --detector random_forest
```

`tests/test_ids_evaluator_detectors.py::test_forcing_the_same_scaler_pins_the_selected_features_across_detectors`
pins this remedy as a guarantee.

And the comparative report **detects** the divergence instead of leaving it as a
footnote: `DetectorComparison.protocol_consistent` compares the feature space
that each detector actually trained on, and `protocol_notes` says which ones
diverged, why, and what the remedy is. A comparison with a non-uniform protocol
comes out with a warning in the table and exit code 1 — it is still a valid
result to investigate, but it is not a concluded "fair comparison".

## Importances: same format, different meanings

`DetectionReport.top_features` keeps the format frozen by issue #15
(`{"feature": str, "importance": float}`) for the five detectors, but the
number is **not comparable across them**:

- **Gini** (RF/DT): sums to 1 over the features, measuring impurity reduction.
- **Gain** (XGBoost): the mean loss improvement that the splits on that feature
  brought. It is the trickiest of the four, because XGBoost publishes it on the
  **same attribute** (`feature_importances_`) as scikit-learn's trees: the
  code that reads both is the same branch in `_raw_importances`, and
  all that separates them is the `importance_kind`. `importance_type="gain"` is pinned
  in the registry instead of inherited because the `XGBClassifier` default is `None` —
  it resolves to "gain" today, but that is the package's default, not a commitment of its own.
- **`|coef_|`** (linear SVM): is in the unit of the already standardized space — it is only
  interpretable *because* the detector asked for `scaler="standard"`. Without scaling,
  the feature with the largest amplitude would receive the smallest coefficient and the ranking would say
  the opposite of what the decision boundary actually uses.
- **None** (RBF SVM): `top_features` comes out empty by construction.

`DetectorManifest.importance_kind` is the field that says which of the four is
in play — comparing feature rankings between detectors is only honest after
looking at that field.

Consequence for the Blue Team (E5): with `svm_rbf`, the `DefensePlan` can still
cite all the scalar metrics and the confusion matrix of the report
(see `agents/defender/tools.py::select_evidence_candidates`), but loses the
per-feature evidence. It is less evidence, not a failure — the guardrail
still holds, and no invented evidence gets through.

SHAP follows the same discipline: `shap.TreeExplainer` only knows how to read a
tree-based model — which includes XGBoost — so `get_shap_importances` returns `[]`
for the two SVMs through the
same best-effort path as when the `shap` package is absent — never a wrong
explainer applied to a model it cannot read.

## The three manifests of the DETECTOR stage

E8 closes the chain that E6 and E7 opened. Each run of the intent-driven
pipeline writes all three side by side in `outputs/intent_loop/<run_id>/`:

| artifact | epic | answers |
|---|---|---|
| `feature_manifest.json` | E6 | how the features were prepared |
| `selection_manifest.json` | E7 | which ones survived, over how many rows |
| `detector_manifest.json` | E8 | which detector learned over that space |

It is this triple that makes the D36-46 requirement auditable from the artifacts
on disk: two reports with different `model_name` only prove they ran under
the same protocol if the preparation manifests match. None of the three is a
`LoopStage` of its own — they are DETECTOR artifacts, persisted **before** the
`DetectionReport`'s binary-matrix gate, so that an evaluation that fails there
still leaves on disk what was fitted and trained.

`DetectorManifest.trained_rows`/`trained_features` follow the same
anti-leakage discipline as `FeatureManifest.fitted_rows`: they count what `fit` saw — the
training partition **after** feature selection and undersampling —
never the whole dataset nor the test partition.

## How to choose the detector

```bash
# CLI — applies to --engine live and --engine intent
uv run adversarial-ids --engine intent --prompt "..." --detector decision_tree
uv run adversarial-ids --engine live --attack random_replay --detector svm_linear

# Environment (affects every path that inherits the default from settings)
DETECTOR_MODE=svm_rbf uv run adversarial-ids --engine live ...

# Controlled scaling ablation: forces both sides to the same scaling
DETECTOR_MODE=svm_linear DETECTOR_SCALER=none uv run ...
```

```python
# Programmatically
from adversarial_ids.core.ids_evaluator import IdsEvaluator

evaluator = IdsEvaluator(detector="svm_linear")            # scaling resolved on its own
evaluator = IdsEvaluator(detector="svm_rbf", scaler="none")  # explicit ablation
evaluator = IdsEvaluator(
    detector="decision_tree", detector_hyperparameters={"max_depth": 5}
)
```

`--engine demo` **rejects** `--detector`: the golden history is an already-recorded
replay, no detector is trained there, and silently accepting the choice would
lie about what the run does.

`n_estimators` remains in the `IdsEvaluator` signature (it predates E8),
but it only applies to `random_forest` — it is that detector's hyperparameter. For the
others, use `detector_hyperparameters`; a hyperparameter that the detector does not
support is rejected at construction with `DetectorError`, the same discipline as the
`extra="forbid"` of the `domain/` contracts.

## Reproducibility and the golden fixture

`DETECTOR_MODE` and `DETECTOR_SCALER` are env vars that change the trained model —
the same latent risk that `PREPROCESSOR_MODE`/`FEATURE_SELECTION_MODE` already
had. That is why `scripts/generate_golden_history.py` **pins** the detector and
the scaling explicitly (`detector="random_forest", scaler="none"`) instead of
inheriting the default: without that, a `DETECTOR_MODE` in the shell of whoever regenerates
would rewrite `data/iteration_history.json` with another model.

All four detectors receive `random_state` (42 by default) and produce
identical metrics across runs with the same seed — covered in
`tests/test_detectors.py` and `tests/test_ids_evaluator_detectors.py`.

A caveat about `detector_manifest.json` itself: it carries
`train_duration_seconds`, which is wall-clock time and therefore **differs between
identical runs**. This is intentional (it is a cost datum, and cost is
precisely what distinguishes `svm_rbf` from the others) and harmless because the file
lives in `outputs/`, never in `data/` — no versioned fixture nor golden
comparison reads that file. If a test ever needs to compare two
run dirs byte for byte, this is the field to exclude.

## Out of scope and future work

- **Automatic detector selection**: `compare_detectors` ranks, but nothing
  *acts* on the ranking — the adversarial loop still trains the detector that
  the configuration dictated, not the winner of the last comparison. Promoting the
  winner automatically would require deciding how often to recompare, and
  that is policy, not measurement.
- **`class_weight` balancing**: exposed as an overridable hyperparameter
  in the four scikit-learn registrations (XGBoost uses
  `scale_pos_weight`, which is not exposed), but `None` by default — touching the
  default would change the `random_forest` path, which has to remain identical
  to before E8. This repository's imbalance lever is E7's
  undersampling (see `docs/feature_selection.md`).
- **Detectors outside scikit-learn**: XGBoost came in (E9) and proved the
  path — `DetectorSpec` is an extensible registry, `DetectorLike` is the
  protocol that `IdsEvaluator` consumes, and an optional dependency fits with a lazy
  import. Each new key still requires an entry in the `Literal`
  of `domain/detector_manifest.py::DetectorKey` and in the two tables next to it
  — on purpose: the detector vocabulary is a contract, not free
  configuration, and the registry validates its own sync at import (including the
  **order**, which is what caught the XGBoost entry out of place).
- **Explanation for non-tree models**: SHAP's `KernelExplainer`/`LinearExplainer`
  would cover the SVMs, but the former costs orders of magnitude more than the
  `TreeExplainer` already used. Outside this epic; today the RBF SVM simply
  offers no feature ranking.

## Tests

- `tests/test_detectors.py` — registry (keys, defaults, unsupported
  hyperparameter), fit/predict of the four through the same interface, errors
  (`predict`/`manifest` before `fit`, empty input, mismatched sizes),
  importances by kind (Gini/coef/none, truncation by `top_n`, misaligned feature
  space), per-seed determinism and manifest JSON round-trip.
- `tests/test_ids_evaluator_detectors.py` — integration with the E6/E7 pipeline:
  the four train and evaluate a variant; all share the same split and the
  same feature space; the recommended scaling reaches the
  `FeaturePreprocessor` and an explicit `scaler` overrides it; the manifest matches
  `SelectionManifest.fitted_rows_after_undersampling`; SHAP is not attempted
  on non-tree models.
- `tests/test_new_contracts.py` — `DetectorManifest` versioned/frozen and
  each internal-consistency validator (`importance_kind` × detector, SHAP
  only on trees, unregistered key, empty feature space).
- `tests/test_intent_loop_orchestrator.py` — `detector_manifest.json`
  persisted and valid in the real loop; the detector choice reaches
  `DetectionReport.model_name`, the manifest and the Defender; an SVM resolves
  `scaler="standard"` down to the `FeatureManifest`; an invalid key blows up at
  orchestrator construction, not as a stage failure.
- `tests/test_live_wiring.py` / `tests/test_cli_intent_engine.py` — the
  `--detector` reaches `run_live_workflow` and `run_intent_loop`, and the
  `demo` engine rejects the choice.
