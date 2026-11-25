# Observability and UI of the intent-driven loop (E11)

Closes the D55-60 window on the operations side. The epic's definition of done is
literal — *"the user sees stage, duration, artifacts, error and outcome"* — and the
"Operation" row of the acceptance table adds *"timeout, retry, structured logs,
secret → failures have a stage/cause; prompts do not expose keys;
execution fits within the defined budget"*.

## What existed and what was missing

`LoopRecord`/`LoopStage` (E3) already carried `name`, `status`,
`duration_seconds`, `artifact_ref` and `error` per stage. Three things were missing:

1. **Nobody read them.** Both UIs only knew `IterationRecord` (the legacy
   Strategist↔Analyst loop); the intent-driven engine was not reachable from the interface.
   The CLI printed stage, artifact and error — but not the duration, which it measured
   and then threw away.
2. **Nothing was observable during execution.** The `LoopRecord` only exists after
   the campaign ends. While it runs, anyone watching sees nothing — and the UI
   papered over that gap on its own, inferring the current phase by substring
   from captured `stdout`
   (`studio/bridge.py::progress()` looked for `"=== ITERAÇÃO"` and `"concluída"`
   (*"=== ITERATION"* and *"completed"*)).
   Renaming a log message broke the progress bar without a single test turning red.
3. **`cost_usd` was declared and never populated.** One cannot claim that
   the execution fits within the budget based on a field that is always `None`.

## LoopEvent: what is happening right now

`domain/loop_event.py` is the transition contract. Four types, and the split is
structural: `run_started`/`run_finished` are about the execution (`stage=None`),
`stage_started`/`stage_finished` are about one of the seven stages. `status` and
`duration_seconds` exist only on the `*_finished` pair — a start event that
claimed a duration would be lying, and the validator refuses it instead of leaving the UI
to decide whether to believe it.

Ordering is by `sequence`, not `created_at`: two fast stages can share the same
timestamp, and the clock can go backwards on an NTP sync.

The stage vocabulary (`LoopStageName`) and the status vocabulary (`LoopStageStatus`) come
from `loop_record`. Event and record speak of the same seven stages, or the
live timeline and the history would tell different stories about the same execution
— and an orchestrator test pins exactly that equality, field by field.

### Where events are born

`IntentLoopOrchestrator._stage` is the single point: it already measured start, end,
duration, artifact and failure cause. What changed is that the three loose variables
of the round (`stages`, `run_dir`, and now the emitter) became a `_RoundContext`,
whose `record_stage` appends the `LoopStage` **and** emits the `LoopEvent` in the same
call. That is what makes it impossible to record one without announcing the other.

### Where events go

`shared/loop_event_store.py`. A sink is `(LoopEvent) -> None` and nothing more — no
abstract class, because the file one is a class (it has a path), the UI one is an
`append` and a test passes a `lambda`. `fanout` composes several and swallows each
one's exception: broken observability is a smaller problem than a pipeline
interrupted because of it.

The file sink is **always** on, per execution: running through the CLI leaves
`events.jsonl` in the run directory without anyone asking. Whoever passes an `event_sink`
to `run_intent_loop`/`IntentLoopOrchestrator` is who needs the events
*before* the end — the live UI.

**JSONL, and not the `save_json` used in the rest of the repo.** A JSON array is only readable
after the closing bracket, that is, after the execution has finished — exactly
when the file stops being useful for observing. Line by line, the UI reads the
timeline of an execution in progress, and an execution that died midway leaves on
disk everything that managed to happen. The reader skips a truncated line instead of
failing: it is precisely for that execution that one most wants to see the timeline.

## The screen

The **Loop intent-driven** page of the studio
(`interfaces/dashboard/studio/view_loop.py`). It launches the campaign from an
intent written in Portuguese — prompt, rounds, generator mode and detector — and shows
the five things the window asks for, each from its own source:

| what | from where |
|---|---|
| current stage, error | live `LoopEvent` |
| duration, artifacts | `LoopStage` of the persisted `LoopRecord` |
| outcome | the JSON files the execution left in its directory |

The outcome needs the third source because the `LoopRecord` stores the *path*
of the artifact, not its content: `loop_bridge.outcome_for` opens
`detection_report.json`, `defense_plan.json` and `defense_rules.json`. Every field
is optional — an execution that stopped at ERENO has no detection report, and
that is information, not an error.

A stage that never ran appears as **pending** instead of vanishing from the table:
in an execution that stopped at the third, what is missing is half the information.

`loop_bridge.py` is a sibling of `bridge.py` and differs from it in what matters:
that one captures `stdout` and guesses; this one registers a sink and reads a typed event.
`progress()` here is closed stages over expected ones (seven per round), with no
heuristic and without the artificial 98% ceiling the other one needs.

## Secrets

`shared/redaction.py`. The path through which a key would escape is not the prompt
we write — it is the **error** we write back: `_stage` records
`str(exc)`, and an HTTP client exception usually carries the request that
failed, authorization header included. That text goes to
`loop_records.json`, to `events.jsonl` and to the screen.

Two defenses, and the first is the one that counts:

1. **The real value.** Every env var whose name contains `API_KEY`, `SECRET`, `TOKEN`,
   `PASSWORD`, `CREDENTIAL`… has its value searched for literally in the text. It does not
   depend on the key having a recognizable format — only on it being in the environment,
   which is exactly the case when it leaked from there. Values shorter than 12
   characters are ignored: a `DEBUG_TOKEN=1` would erase every digit 1 in the
   messages.
2. **The format.** `gsk_…`, `sk-…`, `Bearer …` — catches a key that is not this
   process's.

The surrounding message survives: `401 Unauthorized para ***` (*401 Unauthorized for ****)
remains diagnosable.

**Why not in the contract.** It would be tempting to make `LoopStage`/`LoopEvent`
redact by themselves. No: the contract would start reading `os.environ`, and a
frozen model that depends on the environment stops being reproducible. Redaction lives
at the point of emission, which is where the text is born.

## Budget

`domain/run_usage.py::AgentUsage` + `agents/usage.py`. The agents record in
`last_usage` what the provider's response declared, read through `getattr` instead of
importing the `agno` type — it works the same with a real agent, a stub, or a
version of the lib in which the metrics object changed its name.

Accounting happens **before** the gates: the call was paid for even when the output is
refused right after. A budget that only counts successful calls does not measure
what was spent.

**Tokens, not dollars.** `input_tokens`/`output_tokens` are counts reported by the
provider; `cost_usd` only exists when it reports one, and the repo carries no
price table of its own — it ages silently (this project's default `MODEL_ID`
already had to change because Groq retired the model), and a wrong cost is worse
than a missing one because it looks usable for deciding. Adding a known cost
to an unknown one does not pretend the latter is zero.

`INTENT_LOOP_TOKEN_BUDGET` (env var, default `0` = off) is the campaign's cap. It is checked **between**
rounds and kept outside the E10 feedback policy on purpose: the policy decides whether it is worth continuing *scientifically*,
the cap decides whether it is possible to continue *operationally*. Mixing the two would make a
cost overrun look like a verdict on the experiment.

Two deliberate consequences:

- **the round in progress always finishes** — a half-finished `LoopRecord` is no
  cheaper, just less useful;
- **a campaign whose agents do not report usage is never interrupted** by this. Zero
  tokens is a statement about the call; `None` is the admission that it is not
  known, and comparing a budget against an invented zero would punish whoever merely lacks
  the information.

Usage appears in the footer of the CLI summary and in the execution panel in the studio —
and only when it was reported: a `0 tokens` printed because nobody counted would be
worse than its absence.

## Timeout and retry

They already existed before E11, in `GeneratorRunner`:
`GENERATOR_TIMEOUT_SECONDS`, `GENERATOR_MAX_RETRIES` and
`GENERATOR_RETRY_BACKOFF_SECONDS` (see `config/settings.py`). E11 did not
change them; what it adds is that the failure of that stage appears with stage and
cause in the timeline and on the screen, instead of only in `stdout`.

## Tests

- the `LoopEvent` contract: a stage event without a stage, an execution event with a
  stage, `*_finished` without status, a start declaring an outcome, and vocabulary equality
  with the `LoopRecord` (`tests/test_loop_events.py`);
- sinks: one valid JSON line per event, a `snapshot` that does not alias the live list,
  a `fanout` that survives a sink that blows up, a reader that skips a
  truncated line and sorts by `(round, sequence)` and not by timestamp;
- emission: the complete timeline of an execution, field-by-field equality between
  event and `LoopStage`, the `events.jsonl` written without anyone asking, the stage
  that fails naming the cause, the campaign round in each event, and a
  broken sink that does not bring down the execution (`tests/test_intent_loop_orchestrator.py`);
- UI bridge: directory derived from the artifact and not from `settings`, the three
  outcome artifacts, an execution that died early, a corrupted artifact read
  as absent, `progress()` by closed stage, current stage and failure cause
  (`tests/test_studio_loop_bridge.py`);
- the page itself, via `AppTest`: it opens without history, and shows the status, duration and
  artifact name of a saved execution, with the stages that did not run shown
  as pending;
- redaction: the real value with no recognizable format, the known patterns, the
  short value that is not a secret, and the end-to-end path — a key in the agent's
  exception does not reach the `LoopStage` or the timeline (`tests/test_redaction.py`);
- usage and budget: a sum that preserves unknown cost, a cost without tokens
  refused, tolerant reading of the provider response, aggregation in the
  `LoopRecord`, the cap that cuts the next round and the one that never cuts
  (`tests/test_run_usage.py`);
- duration and usage in the CLI summary (`tests/test_cli_intent_engine.py`).
