# Validation of the Analyst agent output

## Goal

The Analyst agent acts as the Blue Team component of the adversarial loop.
It receives the metrics produced by the Random Forest IDS, interprets
the available evidence and recommends defensive mitigations.

The LLM response is not accepted directly. Every output goes through
typed validation and deterministic rules before it is handed
to the Orchestrator or persisted in the history.

## Output contract

The response must conform to the `AnalystOutput` model, which consists of:

- iteration number;
- list of relevant features;
- defensive diagnosis;
- list of mitigations;
- severity level.

Each feature contains:

- feature name;
- importance value;
- natural-language explanation.

The allowed mitigations are:

- `threshold`;
- `feature`;
- `retrain`.

The severity must be:

- `high`;
- `medium`;
- `low`.

## Schema validation

The Pydantic model `AnalystOutput` rejects:

- extra fields;
- negative iteration;
- negative importance;
- severity outside the allowed values;
- invalid mitigation type;
- empty diagnosis or recommendation;
- responses incompatible with the contract.

## Validation against the metrics

The function `validate_output_against_metrics` checks whether the response
is supported by the real data of the iteration.

The following are rejected:

1. features that do not appear in the evidence;
2. duplicate features;
3. altered importance values;
4. severity incompatible with the F1;
5. an iteration different from the one requested;
6. missing explanation when features are available.

## Use of SHAP and feature importances

When SHAP values are available, they are used as the primary
evidence.

When SHAP is not available, the system uses
`top_feature_importances`, produced by the Random Forest, as a fallback.

The agent only interprets this evidence. It does not compute or
invent importance values.

## Severity rule

The severity is computed deterministically from the F1 of the `masquerade`
class:

- F1 below 0.50: `high`;
- F1 greater than or equal to 0.50 and below 0.80: `medium`;
- F1 greater than or equal to 0.80: `low`;
- F1 absent: `low`.

The LLM receives the expected severity in the `required_severity` field,
but the response is still validated again by the code.

## Safety and scope

The agent:

- analyzes synthetic data only;
- does not access real systems;
- does not produce offensive code;
- does not change attack configurations;
- does not invent metrics;
- does not claim causality without evidence;
- produces defensive recommendations only.

## Tests

The automated tests cover:

- the `AnalystOutput` contract;
- mitigation types;
- severity thresholds;
- SHAP priority;
- fallback to feature importances;
- detection of an invented feature;
- detection of an altered importance;
- detection of an incorrect severity;
- detection of an incorrect iteration;
- conversion of dict, JSON and Pydantic model;
- execution without external calls using a fake agent.

The manual test with Groq is available at:

```text
scripts/test_analyst_real.py
```

It is not part of the automatic suite because it makes an external call
and consumes API tokens.
