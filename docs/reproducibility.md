# Reproducibility — running without Java (cached mode)


The ERENO synthetic generator is a heavy Java `.jar` that lives **outside the repo**
(gitignored). So that the flow can run without Java (including in CI), the core
offers a **cached mode**: instead of running Java, it serves an already-versioned
benign+attack dataset. The goal is to exercise the flow (agents, memory,
dashboard), not to reproduce the physical per-configuration variation.

---

## 1. Prerequisites

- [`uv`](https://docs.astral.sh/uv/) installed.
- **Java is not required** in cached mode.
- A **Groq key** is only needed to call the real Strategist agent
  (tool calling). The tests, the golden and the reset run without a key.

```bash
uv sync --extra dev
```

The repository configures `.uv-cache/` as a local cache in `uv.toml`, avoiding
dependence on the user's global cache. To validate the toolchain and record a
complete baseline, run:

```bash
uv run python scripts/verify_environment.py
```

The initial baseline and the interpretation procedure are documented in
[`docs/environment_baseline.md`](environment_baseline.md).

---

## 2. The two generator modes

Selected in `config/settings.py` (`GENERATOR_MODE`), overridable through the
environment variable `GENERATOR_MODE`:

| Mode     | What it does                                                     | Needs Java?      |
| -------- | ---------------------------------------------------------------- | ---------------- |
| `cached` | Copies `data/baseline_dataset.csv` as each iteration's dataset   | No               |
| `jar`    | Runs the real ERENO generator over the `attack_config`           | Yes (+ the `.jar`) |

**Default:** the mode automatically falls back to `cached` when the JAR
(`generator_runtime/ereno-generator.jar`) is **not** present, and to `jar`
when it is. To force it:

```bash
GENERATOR_MODE=cached uv run adversarial-ids --iterations 2
```

In cached mode the dataset is the same in every iteration — the configs, diffs, logs
and the history are still produced normally in both modes.

### How to obtain the ERENO generator (JAR)

The JAR is an uber-jar (~25 MB) kept **outside code version control**
(gitignored, because of its size and because it is a build artifact). It is needed **only**
for the `jar` mode; the `cached`/`demo` modes, the tests and the golden run without it.

Running it requires Java 11+. The prebuilt jar is attached to every
[release](https://github.com/sequincozes/ereno-ai/releases/latest) of this
repository:

```bash
curl -L -o generator_runtime/ereno-generator.jar \
  https://github.com/sequincozes/ereno-ai/releases/latest/download/ereno-generator.jar
```

Check its integrity (SHA-256):

```text
dfe25fd818178aea315b8a033963e3ab2591c1445462d6fe9c8fc63e5251658c  ereno-generator.jar
```

```bash
# Linux/macOS
sha256sum generator_runtime/ereno-generator.jar
# PowerShell
Get-FileHash generator_runtime/ereno-generator.jar -Algorithm SHA256
```

Alternatively, build it from source:

```bash
git clone -b feat/journal https://github.com/sequincozes/ereno-2.0.git ../ERENO-2.0
cd ../ERENO-2.0
./mvnw package -DskipTests        # Windows: mvnw.cmd package -DskipTests
```

Then copy `target/ERENO-1.0-SNAPSHOT-uber.jar` to
`generator_runtime/ereno-generator.jar` in this repository. Once it is in place,
`GENERATOR_MODE` automatically falls back to `jar`.

---

## 3. Versioned seeds (source of truth)

They live in `data/` (outside `outputs/`) and are **not deleted** by the reset:

| File                          | What it is                                                           |
| ----------------------------- | ------------------------------------------------------------------- |
| `data/baseline_dataset.csv`   | Benign+attack dataset served in cached mode (`class` column).       |
| `data/iteration_history.json` | **Golden** history: a stable list of `IterationRecord`.              |

The golden is consumed by the Dashboard (#19) and by the tests without needing an API,
Java or the execution of the real agents.

---

## 4. Determinism

- `IdsEvaluator` uses `random_state=42` (train/test split + Random Forest),
  so the metrics are reproducible.
- The `FakeStrategist` / `FakeAnalyst` stubs (in `tests/fakes/`) are
  **deterministic**: given the same iteration, they always return the same output.
- The golden generator (`scripts/generate_golden_history.py`) uses a **fixed
  timestamp**, so `data/iteration_history.json` stays byte-stable across
  regenerations.

Result: running the golden twice produces exactly the same file.

---

## 5. Commands

### Verify the infrastructure (no key, no Java)

```bash
uv run pytest
```

Exercises the cached mode, the schemas in `domain/`, the history persistence
(`ExperimentMemory`) and the stubs.

### Run the cached loop and regenerate the golden

```bash
uv run python scripts/generate_golden_history.py
```

Chains `FakeStrategist → cached generator → IdsEvaluator → FakeAnalyst` and
writes each step as an `IterationRecord` in `data/iteration_history.json`.

### Reset the experiment state

```bash
uv run python scripts/init_state.py            # clears outputs/ and logs/
uv run python scripts/init_state.py --dry-run  # only shows what it would do
uv run python scripts/init_state.py --keep-logs
```

Clears the generated artifacts **preserving the seeds in `data/`**. Use it before a
clean run. It is idempotent and safe: it never touches `data/`.

### Run the CLI with the real agent (needs a Groq key)

```bash
cp .env.example .env        # fill in GROQ_API_KEY
uv run adversarial-ids --model-id openai/gpt-oss-120b --iterations 2
```

Without the JAR, generation falls back to cached mode automatically.

---

## 6. How the history is persisted

`core/experiment_memory.py` (`ExperimentMemory`) accumulates `IterationRecord`s and
writes them to disk in the same format as the golden:

```json
{ "history": [ { "iteration": 0, "attack_config": { ... }, "metrics": { ... }, "..." : "..." } ] }
```

- `add_record(record)` — the Orchestrator's path (#16), with the record already typed.
- `add_iteration(...)` — adapter from loose dicts, used by the CLI;
  validates against the schemas in `domain/` before storing.
- `ExperimentMemory.load(path)` — rebuilds the memory from a saved history
  (empty memory if the file does not exist), allowing it to resume where it left off.

Since they are valid `IterationRecord`s, the histories generated by the CLI and by the golden
have the **same contract** — the Dashboard (#19) and the Strategist's memory (#8)
read either one without special handling.
