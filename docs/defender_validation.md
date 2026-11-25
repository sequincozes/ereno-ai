# Validation of the Defender agent output (E5)

## Goal

The Defender agent acts as the DEFENDER stage of the intent-driven pipeline.
It receives the `DetectionReport` produced by the DETECTOR stage and proposes a
`DefensePlan`: detection, containment and hardening actions, each tied to
real evidence from the report and to a verification method.

The LLM response is not accepted directly. Every output goes through typed
validation and deterministic rules before it is persisted in the
`LoopRecord` — the DoD of epic E5: "every action references a metric/feature
and a verification method".

## Output contract (`schema_version: 2`)

The response must respect the `DefensePlan` model, composed of:

- priority (`low`, `medium`, `high` or `critical`);
- an optional reference to the source `DetectionReport`
  (`detection_report_ref`), at the top of the plan;
- lists of detection, containment and hardening actions (at least one in
  total).

Each action (`DefenseAction`) contains:

- description;
- a named **defensive technique** (`technique`);
- at least one piece of evidence;
- a **validation test** (`validation_test`).

Each piece of evidence (`Evidence`) contains:

- the cited metric or feature;
- the cited value;
- an optional reference to the source `DetectionReport`.

### v1 and why it changed

Until the D36-46 window the contract was at v1, and the action carried a
free-text `validation_method: str`: any non-empty string passed,
including `"revalidar"` ("revalidate"). The exit gate of D47-54 is "100% of recommendations
tied to evidence **and a validation test**", and half of it was not
verifiable by any type. v2 replaces that field with `ValidationTest`.
The prose was not lost — it became `ValidationTest.procedure`.

### Techniques and buckets

`DefenseTechnique` is the closed vocabulary of defensive techniques, and lives in
`domain/defense_plan.py` (just as `DetectorKey` lives in
`domain/detector_manifest.py`): the feature→technique catalog in `config/`
declares itself in sync with it, never the other way around.

Each technique is legal in one or more buckets. Several are dual-use — a
publisher allowlist blocks the intruder *and* is a structural control —, so
forcing a single bucket would reject a correct plan. What the map prevents is the
mistake the LLM actually makes: proposing `network_segmentation` as a
*detection* action, where it measures nothing.

The prompt receives the list per bucket in `legal_techniques`, built by
`agents/defender/tools.py::legal_techniques_by_bucket` from the same map
the contract consults to reject — what is offered and what is accepted
cannot diverge.

### The validation test

`ValidationTest` answers "how will we know whether this worked?" in terms
that can be remediated:

| field | meaning |
|---|---|
| `metric` | the metric to measure again; only values of `ValidationMetric` |
| `direction` | `increase`/`decrease` demand a change; `at_least`/`at_most` are a floor/ceiling |
| `target` | the number to reach; finite, non-negative, and in [0, 1] for rates |
| `split` | optional; `null` means the same split as the report |
| `procedure` | how the measurement will be made, in one sentence |

`ValidationMetric` is a strict subset of the evidence allowlist:
`model_name` and `split` are text (there is no "increase the split"), and feature
importance was left out because it **is not a defensive result** — proving that a
feature became more important does not prove that the attack is now detected.

## Schema validation

The Pydantic model `DefensePlan` (`extra="forbid"`, `frozen=True`) rejects:

- extra fields — including the v1 `validation_method`;
- a priority outside the four allowed values;
- an action with no evidence, no technique or no validation test;
- a plan with no action in the three buckets;
- a technique outside the vocabulary, or legitimate but in the wrong bucket;
- a validation metric that is not a remediable scalar of the report;
- a target that is NaN, infinite, negative, or above 1.0 on a rate metric;
- an empty description, procedure or evidence;
- evidence that references a different run from the one declared by the plan.

## Validation against the DetectionReport

The function `agents/defender/tools.py::validate_plan_against_report` checks
whether the response is backed by the real data of the run. The allowlist of
citable evidence is built by `select_evidence_candidates`: the report's
scalar fields (`model_name`, `split`, `accuracy`, `precision`,
`recall`, `f1`, `latency_ms`), the four confusion-matrix fields under
`confusion_matrix.<field>`, and the name of each feature in `top_features`
mapped to its importance.

The following are rejected:

1. evidence (metric or feature) that does not appear in the allowlist;
2. an altered evidence value (numeric comparison tolerant of
   JSON rounding, with an explicit guard so as not to confuse `bool`
   with `int`);
3. duplicate evidence within the same action, in any of the three
   buckets;
4. a validation test whose target does not demand improvement: "raise the recall to
   0.50" when the recall is already 0.60 is a regression sold as a fix, and
   would pass any type check. Only `increase`/`decrease` are
   compared against the measured value — `at_least`/`at_most` are guards
   against regression, legitimately already satisfied when the action exists to
   *protect* a metric that is good while another is attacked;
5. a priority incompatible with `priority_from_report`;
6. a `detection_report_ref` that is present and different from this run's
   report — closes the gap of the Defender citing real evidence but
   pointing to an invented report;
7. an action whose technique answers none of the evidence it itself cites —
   the rule evaluation, described in the next section.

The split between contract and gate follows the same line as the rest of the repo: the
contract refuses what is verifiable on its own (vocabulary, type, numeric
domain, internal coherence of the plan), and the gate refuses what is only known with
the `DetectionReport` in hand (is the evidence real? does the target mean anything?).

An additional guard in `select_evidence_candidates` rejects the construction of the
allowlist itself if a feature name collides with a reserved metric
key (for example, a feature called `f1`) — it never occurs
with real ERENO data (features are CamelCase, like `TrapAreaSum`), but
it keeps the allowlist unambiguous when it does.

## Rule evaluation (`core/defense_rules.py`)

The six items above prove that the plan **does not lie**: the metric exists, the value
matches, the target demands improvement. None of them proves that the recommendation **answers** the
problem. A plan that cites `recall` and proposes `goose_authentication` used to pass
entirely — the bucket is legal, the evidence is real, the test is remediable —, and
authenticating the publisher is simply not what a low recall calls for.

`core/defense_rules.py::evaluate_plan_rules` closes this by consulting the
feature→technique catalog (`config/defense_techniques.py`, see the section "Techniques and
buckets"). An action is **grounded** when its `technique` appears
in the catalog for at least one of the evidence items it cites — at least one, not
all: a serious plan cites the metric that hurts *and* the feature that explains it, and
requiring the technique to answer both would reject precisely the best-grounded
action.

The evaluation is deterministic and pure (no LLM, same discipline as E10), and
never raises an exception because of content: it returns a `DefenseRuleReport`, and the one
that refuses is the gate. A bad plan must be readable before being refused.

### The five rules

| rule | severity | what it sees |
|---|---|---|
| `technique_not_grounded` | **blocking** | the technique is not a catalogued answer to any cited evidence |
| `evidence_not_actionable` | advisory | the action cites `model_name`/`split`: they identify the run, but there is nothing to remediate in them |
| `uncatalogued_feature` | advisory | the evidence is a real feature that the catalog does not cover yet — a catalog gap, not a defect of the plan |
| `playbook_technique_missing` | advisory | the IEC-61850 playbook for that family prescribes techniques that the plan does not propose |
| `playbook_signature_ignored` | advisory | the report highlighted a signature feature of the playbook and no action cited it |

Only the first one refuses. The two playbook rules speak of the **scenario**, not of this
round: a replay playbook asks for sequence validation even under a
`PROTOCOL_FEATURES_MODE` in which no delta reaches the report (see
`docs/preprocessing.md`). Blocking on them would turn the right answer for
*this* run into an error; silencing them would throw away the only reading the
catalog has of the scenario.

The playbook axis only turns on when the caller provides `attack_key` — the
orchestrator passes `IntentSpec.base_attack`, the same attack the compiled
intent used. Without it the report comes out with `playbook_key=None`, saying that
that axis was not evaluated instead of letting the absence of findings look like
approval. `playbook_signature_ignored` only charges features that the report
**actually** highlighted in `top_features`: charging an absent one would be charging the plan
for something the report had no way to show.

### The prompt and the gate read the same map

`techniques_by_evidence` arrives in the Defender's context with the techniques
catalogued for each key of `citable_evidence`, under the same `top_n`. Both
sides consult `config/defense_techniques.py::techniques_for_evidence`, so what
the prompt offers is literally what the gate requires — the same discipline as
`select_evidence_candidates` for evidence and `legal_techniques_by_bucket`
for buckets. Without it, the LLM would choose the technique blindly and would discover the rule
only at the refusal. `model_name` and `split` are left out of the map: showing them with an empty
list would invite citing them alone.

### The `defense_rules.json` artifact

The DEFENDER stage persists the `DefenseRuleReport` next to the plan
(`IntentLoopOrchestrator._run_defender`). `grounded_actions`/`total_actions` is,
literally, the fraction of recommendations tied to evidence that the exit gate
of the D47-54 window demands — and the advisory findings are recorded per run,
which is what makes it possible to see over a campaign which playbook the Defender
systematically ignores.

## Priority rule

The priority is computed deterministically from the F1 and the recall of the
`DetectionReport` (`agents/defender/tools.py::priority_from_report`):

- base level by F1: `high` if F1 is less than 0.50; `medium` if F1 is greater than or
  equal to 0.50 and less than 0.80; `low` if F1 is greater than or equal to 0.80;
- then, if the recall is less than 0.50, the level goes up one step in the
  order `low` → `medium` → `high` → `critical`.

The LLM receives the expected priority in the `required_priority` field, but the
response is still validated again by the code.

## Safety and scope

The agent:

- analyzes only synthetic data;
- does not access real systems;
- does not produce offensive code;
- does not invent a metric, feature, value or reference to the report;
- does not claim causality without evidence;
- produces only defensive recommendations — the `DefensePlan` has no execution
  field, and is never applied automatically.

## Stage failure

Unlike the legacy Analyst (which degrades to an absent analysis on
failure, without interrupting the loop — `AdversarialWorkflow._safe_analyze`), in the
intent-driven pipeline a Defender failure — a call error or a plan
rejected by the gate — marks the `defender` `LoopStage` as `failed` with the
cause and interrupts the pipeline at that point
(`IntentLoopOrchestrator.run`). The partial `LoopRecord` is still persisted:
the cause is never lost.

## Tests

The automated tests cover:

- the `DefensePlan` v2 contract — technique vocabulary, technique in the wrong
  bucket, target domain, NaN/infinity, traceability and the invariants
  inherited from v1 (`tests/test_defense_plan_contract.py`), plus the
  versioning/freezing/round-trip common to all contracts
  (`tests/test_new_contracts.py`);
- `priority_from_report` at the four levels and at the boundaries 0.50/0.80/recall
  0.50;
- allowlist construction, truncation by `top_n` and the name-collision
  guard;
- the seam between `ValidationMetric` and the allowlist: every validation metric
  must exist in a real `DetectionReport`, and what is left of the allowlist is
  text or feature importance;
- detection of invented evidence, altered value, duplicate evidence,
  a target that does not demand improvement and an incompatible priority;
- the JSON example in the prompt, parsed through the `DefensePlan` — an
  invalid example would teach the model to produce exactly what the gate refuses;
- detection of a divergent `detection_report_ref`;
- conversion from dict, JSON (with and without the ```` ``` ```` fence) and
  Pydantic model;
- execution without external calls using a fake agent
  (`tests/test_defender_agent.py`);
- the rule evaluation: grounding per action and per bucket, the technique foreign to the
  evidence, the two evidence items that support nothing (descriptive and outside the
  catalog), the two playbook findings and the invariants of the
  `DefenseRuleReport` itself — a count that contradicts the findings, a downgraded
  severity, a finding for an action with no location (`tests/test_defense_rules.py`);
- the gate's refusal by the grounding rule and the fraction cited in the message, plus
  `techniques_by_evidence` against the same query the gate uses
  (`tests/test_defender_tools.py`);
- the DEFENDER stage end to end in the orchestrator, including the failure
  path and the persisted `defense_rules.json`
  (`tests/test_intent_loop_orchestrator.py`).
