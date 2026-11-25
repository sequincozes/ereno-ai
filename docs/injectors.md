# Injector: intent-, model- and data-driven

The architecture diagram draws the Injector as a box with three strategies.
Until 2026-10-13 only the intent-driven one existed, and switching injectors meant
switching the whole pipeline with `--engine`. Now all three live behind a single
contract, inside the intent-driven pipeline:

```bash
# intent-driven (default, the usual behavior)
uv run adversarial-ids --engine intent --prompt "reduza o recall variando a temporização da falha"
# (prompt: "reduce the recall by varying the timing of the fault")

# model-driven: the LLM reads the Java creator of an attack and proposes a new attack
uv run adversarial-ids --engine intent --generator-mode jar \
  --injector model --injector-source inverse_replay \
  --prompt "proponha uma variante mais discreta do replay invertido"
# (prompt: "propose a more discreet variant of the inverse replay")

# data-driven: the LLM reads the profile of a trace and proposes a new attack
uv run adversarial-ids --engine intent --generator-mode jar \
  --injector data --injector-dataset outputs/intent_loop/<run>/dataset_iteration_1.csv \
  --prompt "reproduza a assinatura temporal do ataque observado"
# (prompt: "reproduce the temporal signature of the observed attack")
```

## The contract

The three strategies implement the same `interpret(prompt) -> IntentSpec` that the
orchestrator already consumed (`IntentLike`). The `IntentSpec` goes through the same
gates, is compiled by the same compiler and runs through the same pipeline —
generator, pre-processor, detector, defender, feedback. **What separates the
strategies is only the context the LLM reads before proposing the attack:**

| `--injector` | context | the LLM may return |
|---|---|---|
| `intent` | the whole capability catalog (12 attacks) | any attack in the catalog |
| `model` | the Java creator of `--injector-source` + the `programmable` catalog | only `programmable`, with an authored rule |
| `data` | the profile of `--injector-dataset` + the `programmable` catalog | only `programmable`, with an authored rule |

The pipeline's output contracts do not change: the intent-driven pipeline still
writes `LoopRecord`, and the legacy loop (`--engine live`) still writes
`IterationRecord`. The Injector unifies the **input**, not the output.

## Why model and data are pinned to programmable

"New attack" cannot mean new code. The project's central guardrail is
that no LLM output writes into the ERENO runtime — and a Java creator written
by the model would be exactly that. The programmable attack (uc11) is the way for a
new attack to exist without new code: declarative rules (mutate, drop,
repeat, delay, reorder, select whom to act on) that the compiler validates
field by field and the schema validates as a whole (see `docs/programmable_attack.md`).

Generating real Java is epic E12, outside the scope of the 60 days.

A third gate, after the two that every `IntentSpec` already goes through
(`agents/injector/gate.py`), checks that what came back is what the strategy
promises:

- `base_attack` is `programmable`;
- there is at least one `rules.` path in `target_values`. Without an authored rule, the
  programmable runs the baseline's rules — and the "new attack" that the manifest
  announces would be the default attack.

## The context of each strategy

**model** (`agents/injector/java_sources.py`): the configurable creator (suffix
`C`) of the attack, read from `ERENO_SOURCE_DIR` (default: the sibling repo
`../ERENO-2.0`; the JAR does not carry source). Package and imports are stripped — they
teach nothing about the behavior and cost TPM. The `programmable` is not
a source: reading the DSL interpreter in order to write in the DSL itself is circular.

**data** (`core/trace_profile.py`): a deterministic profile of the CSV, not the CSV.
For each numeric column, mean and standard deviation in the attack class and in `normal`, and the
separation between them — the same measure as the E0 pilot, |Δ means| / combined
deviation. The LLM sees the columns that separate the most, the constants at
different values (total separation, without becoming infinite) and, always, the columns that the
programmable's rules touch. With more than one attack class in the trace,
`--injector-label` chooses which one to profile.

Only the `programmable` catalog goes into the instructions, not all 12 attacks: the
strategy can only return `programmable`, and the whole catalog would take up ~7k
of the account's 8k tokens per minute, leaving no room for the code the model
needs to read. Measured on 2026-10-13: the `model` instructions are 9–14.5 thousand
characters depending on the creator, those of `data` ~8 thousand, against 20 thousand for `intent`.

## Artifacts

The round that calls the LLM writes, **before** the call:

- `injector_manifest.json` (`InjectorManifest`): the strategy, the kind of
  context, the `base_attack` restriction, the `subject` (the attack whose creator was
  read, or the profiled class), the sources with the **sha256** of each one, and the
  size of the instructions. The hash matters: the same path at another ERENO
  commit, or a regenerated trace, is a different context.
- `trace_profile.json` (`TraceProfile`), `data` only: what the model saw.

Before the call because, if it fails, the context is the first place to
look. The following rounds of a campaign inherit the intent from the feedback
policy (E10), do not call the LLM and do not rewrite the manifest.

The `InjectorManifest` validator refuses a manifest that does not match the
strategy: a `model` without the `programmable` restriction, or a `model`/`data`
without a source, would describe an injector that this repo does not have.

## Resuming

`--resume` only calls the LLM when it redoes a round whose intent never came to
exist. In that case, the strategy must be the same as the campaign's — the
orchestrator reads the `injector_manifest.json` of the redone round and refuses another.
Pass the same `--injector*` flags as the original execution. In the other cases the
intent comes from the ledger and the injector does not even run.

## Validated with a real LLM (2026-10-13, `--generator-mode jar`)

- **model**, on `inverse_replay`, request "variante mais discreta: só parte
  das mensagens, atraso pequeno" (*"more discreet variant: only some
  of the messages, small delay"*): `add timestamp +0.02` on 10% of the messages.
  All seven stages passed. It is an **approximation** of the replay, and the model did not
  use `duplicate` — what the creator does and the DSL does not express (choosing
  messages from a random past) was left out, as the prompt instructs.
- **data**, on an `inverse_replay` trace (signature: `timestampDiff`
  1.8 ms in the attack against 0.7 s in normal), request "reproduza a assinatura
  temporal" (*"reproduce the temporal signature"*): it needed two prompt adjustments, and the two gates did
  their job before that.
  1. It authored 10 paths and declared `max_fields_changed=2` → refused at
     the intent stage. The common block gained the counting rule next to
     `target_values`.
  2. It read the profile correctly but translated "mensagens coladas" (*"glued-together messages"*) as `set timestamp
     0` → refused by the programmable schema. The data prompt gained the
     signature → rule table (a delta near zero is `duplicate`, whose copies
     come out adjacent with the same clock).
  3. With both: `duplicate` on 20% + `add timestamp +0.04` on 50%. Intent,
     compiler, generator and detector passed.

On the third execution, what failed was the **Defender**, not the Injector: the LLM
wrote the `detection_report_ref` (a Windows path) with the backslashes escaped
twice, and the validator refused the plan. It is an E5 matter, intermittent — the model's round
went through the same stage.

## What the Injector is not

A decider. Nothing chooses the strategy from a metric — whoever runs the experiment
chooses. Same reasoning as the undersampling decider
(`docs/feature_selection.md`): what optimizes the metric does not choose its own
stage.

## Adding a strategy

1. The key in `InjectorKey` (`domain/injector_manifest.py`), with its
   `ContextKind` and `base_attack` restriction.
2. An `InjectorSpec` in the `_REGISTRY` of `agents/injector/registry.py` (the
   import refuses a registry and contract out of sync), and the branch of
   `prepare_injector` that reads the context.
3. The role in `prompts/injector_<key>.md`; the common block
   (`prompts/injector_common.md`) already applies to any strategy pinned to
   `programmable`.
