# Equipe e responsabilidades

Este documento registra a divisão de trabalho do projeto Laboratório
Inteligente conforme `docs/divisao_tarefas.html`. A documentação disponível não
associa de forma completa os nomes dos contribuidores aos identificadores
M1–M4; por isso, a identificação por membro é mantida sem inferências.

| Membro | Responsabilidade principal | Issues e entregas |
|---|---|---|
| M1 | Strategist / Red Team | Contrato do Strategist, agente Agno, ferramentas, personas, testes e evolução da estratégia |
| M2 | Analyst / Blue Team | Contrato do Analyst, agente, explicação das features, mitigação e testes |
| M3 | Núcleo e orquestração | Layout, domínio, cache, avaliação, Agno Team, memória e reprodutibilidade |
| M4 | Interface e empacotamento | #6, #18, #19 e #20: pacote, ambiente, CLI, dashboard e documentação final |

## Entregas por fase

### Fase 0 — Fundação

- M3: layout `src/`, tipos de domínio, dataset cacheado e fixtures;
- M4: `pyproject.toml` e `.env.example` compatíveis com o layout;
- M1/M2: contratos de saída necessários aos agentes.

### Fase 1 — Agentes

- M1: Strategist que propõe mudanças na configuração;
- M2: Analyst que interpreta métricas e recomenda mitigação;
- M3: workflow que encadeia os agentes e persiste `IterationRecord`.

No estado atual, Strategist e Analyst estão implementados e testados. O
workflow que deve encadeá-los permanece como placeholder.

### Fase 2 — Entrega

- M4: CLI fina, dashboard Streamlit, documentação e roteiro;
- integração das interfaces com o workflow estabilizado pelo M3.

A CLI e o dashboard já compartilham o contrato `ExperimentRunner`. Enquanto o
workflow não está disponível, ambos usam `CachedHistoryRunner` para demonstrar
o histórico golden.

## Dependências entre os trabalhos

- M1 e M2 consomem os tipos compartilhados pelo M3;
- o workflow do M3 depende das saídas do Strategist e do Analyst;
- M4 consome `Metrics` e `IterationRecord`;
- a CLI e o dashboard dependem da API pública do workflow para execução real;
- o dashboard apresenta `analyst_output`, atualmente proveniente da fixture;
- seeds e stubs permitem testes isolados sem Java e sem todos os agentes reais.

## Colaboração por contratos

As fronteiras compartilhadas são:

| Contrato | Produzido por | Consumido por |
|---|---|---|
| Config do ataque (dict, schema variável por ataque) | Base compartilhada | Strategist, validador e gerador |
| `Metrics` | Avaliador do IDS | Strategist, Analyst e dashboard |
| `IterationRecord` | Memória/workflow | CLI, dashboard e histórico |
| `ExperimentRunner` | Interface M4 | CLI, dashboard e futuro adaptador do workflow |

`FakeStrategist`, `FakeAnalyst` e o golden possibilitam desenvolvimento e
testes independentes. Eles não devem ser apresentados como agentes reais nem
confundidos com as implementações já disponíveis.