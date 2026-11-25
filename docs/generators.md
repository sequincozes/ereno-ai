# Generators — where the trace comes from

The pipeline trains an IDS on a trace. Until now, "which generator produced this
trace" was not a question the system could answer from its artifacts: ERENO was the only possible source, the stage that generated the trace was
called `ereno` in the `LoopRecord` vocabulary, and cached mode — which always returns
the same file — looked like a normal run. The generator registry
(`core/generators.py`) makes the same move epic E8 made with the detector,
only upstream.

## The two registered backends

| key | simulator | needs Java | varies with the config |
|---|---|---|---|
| `cached` (default) | — | no | **no** |
| `jar` | ERENO | yes | yes |

`cached` serves `data/baseline_dataset.csv`, a versioned benign+attack trace.
It is what allows running the whole loop, the tests and CI without the JAR.

`jar` runs the ERENO generator (Java) over the compiled attack config.

```bash
uv run adversarial-ids --engine intent --generator-mode jar --prompt "..."
```

The keys are `cached`/`jar` for compatibility with the CLI and the env vars that
already existed. The field that says what actually produced the trace is `simulator`, in the
manifest — today there is one backend per simulator, but it need not stay that way
(ERENO via JAR and embedded ERENO would be two backends, one simulator).

## `produces_per_config_variation`, the field that exists so it cannot lie

It is the difference between **measuring the effect of an intent** and **exercising the flow**.

In `cached` the trace is the same file on every run, so the
`DetectionReport` does not move between rounds: a campaign parks at round 2
with `no_improvement`, and any reading of the kind "this intent dropped
recall" made over a cached run is a claim about nothing. This was already
documented in prose; now it is in a field a machine reads, written to
`generator_manifest.json` next to the result.

The capability comes from the **registry**, never from whoever writes the manifest, and the
domain contract refuses a manifest that disagrees with the backend it names. A
manifest claiming that `cached` varies with the config describes a generator that does not
exist — and it is exactly the claim that would make a campaign plateau pass for an
experimental result.

## The fourth manifest

`generator_manifest.json` closes the provenance chain of a round, next to the
three that already existed:

| manifest | answers |
|---|---|
| `generator_manifest.json` | where the trace came from |
| `feature_manifest.json` (E6) | how the features were prepared |
| `selection_manifest.json` (E7) | which ones survived, over how many rows |
| `detector_manifest.json` (E8) | which model learned, with which hyperparameters |

It does not repeat the `DatasetBundle`: the bundle answers *what is in the trace*
(columns, classes, count, content hash), the manifest answers *how it
was produced*. Same split as `DetectionReport` (how well the detector did) and
`DetectorManifest` (which detector it was). With both side by side, a difference
between two results is attributable to the config, the simulator or the seed; without
them, it is attributable to nothing.

`random_seed` records the seed passed to the backend (Phase 2.0). `None`
means clock RNG — two runs of the same config are not comparable byte for
byte, which is the difference between a replicate and a repetition.

## Adding a backend

The architecture diagram puts ERENO, Mininet and libiec61850 side by
side. To get in, a backend needs:

1. a class that fulfills `GeneratorLike` — `generate_dataset(attack_config,
   iteration) -> CSV path`, a single line: everything that separates one backend from
   another (subprocess, JVM, network topology) is its own internal matter;
2. one more key in `GeneratorKey` (`domain/generator_manifest.py`) and in the
   three capability tables next to it;
3. one more `GeneratorSpec` in the registry, declaring capabilities, the options it
   requires and how it is built.

Nothing else changes: not `GeneratorRunner`, not the orchestrator, not the evaluator.
The sync check runs at import — a key registered and absent from the
`Literal` would produce a manifest impossible to validate, and a key only in the
`Literal` would be a CLI option that builds nothing; both fail before
any campaign starts.

## Two honesties about the current state

**The two registered backends are the same class.** `cached` and `jar` are both
`GeneratorRunner`, and the choice between them is literally passing or not passing
`cached_dataset_path`. This is description, not aspirational architecture: the seam
that the registry opens holds for the **third** backend, which implements itself.
Same caveat that E8's `DetectorLike` makes about itself.

**The stage vocabulary was fixed along with it (`schema_version` 2).** Up to v1
`LoopStageName` called `generator` the step that **compiles** the intent into an
`AttackCandidate`, and `ereno` the one that actually generates the trace — the dashboard already
compensated by labeling the first as "Compilation". In an architecture of
pluggable backends, naming a stage after a backend is the kind of thing a
reviewer notices. In v2 they are `compiler` and `generator`, each named for what it does.

v1 ledgers are refused by `load_loop_records` with the reason and the path of the
file, not with a Pydantic enum error. They live in `outputs/`, which is
disposable (`scripts/init_state.py`), so there is no migration to write — only a
file to delete.
