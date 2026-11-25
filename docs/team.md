# Team and responsibilities

This document records the division of work of the Laboratório Inteligente
(Smart Laboratory) project according to `docs/divisao_tarefas.html`. The
available documentation does not fully associate the contributors' names with
the identifiers M1–M4; therefore, the per-member identification is kept without
inferences.

| Member | Main responsibility | Issues and deliverables |
|---|---|---|
| M1 | Strategist / Red Team | Strategist contract, Agno agent, tools, personas, tests and strategy evolution |
| M2 | Analyst / Blue Team | Analyst contract, agent, feature explanation, mitigation and tests |
| M3 | Core and orchestration | Layout, domain, cache, evaluation, Agno Team, memory and reproducibility |
| M4 | Interface and packaging | #6, #18, #19 and #20: package, environment, CLI, dashboard and final documentation |

## Deliverables by phase

### Phase 0 — Foundation

- M3: `src/` layout, domain types, cached dataset and fixtures;
- M4: `pyproject.toml` and `.env.example` compatible with the layout;
- M1/M2: output contracts required by the agents.

### Phase 1 — Agents

- M1: Strategist that proposes changes to the configuration;
- M2: Analyst that interprets metrics and recommends mitigation;
- M3: workflow that chains the agents and persists `IterationRecord`.

In the current state, Strategist and Analyst are implemented and tested. The
workflow that is supposed to chain them remains a placeholder.

### Phase 2 — Delivery

- M4: thin CLI, Streamlit dashboard, documentation and walkthrough script;
- integration of the interfaces with the workflow stabilized by M3.

The CLI and the dashboard already share the `ExperimentRunner` contract. While
the workflow is not available, both use `CachedHistoryRunner` to demonstrate
the golden history.

## Dependencies between the work items

- M1 and M2 consume the types shared by M3;
- M3's workflow depends on the outputs of the Strategist and the Analyst;
- M4 consumes `Metrics` and `IterationRecord`;
- the CLI and the dashboard depend on the workflow's public API for real execution;
- the dashboard presents `analyst_output`, currently coming from the fixture;
- seeds and stubs allow isolated tests without Java and without all the real agents.

## Collaboration through contracts

The shared boundaries are:

| Contract | Produced by | Consumed by |
|---|---|---|
| Attack config (dict, schema varies per attack) | Shared base | Strategist, validator and generator |
| `Metrics` | IDS evaluator | Strategist, Analyst and dashboard |
| `IterationRecord` | Memory/workflow | CLI, dashboard and history |
| `ExperimentRunner` | M4 interface | CLI, dashboard and future workflow adapter |

`FakeStrategist`, `FakeAnalyst` and the golden make independent development and
testing possible. They must not be presented as real agents nor confused with
the implementations already available.
